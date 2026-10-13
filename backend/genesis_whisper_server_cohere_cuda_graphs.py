"""CUDA-Graph-Beschleunigung fuer CohereLabs/cohere-transcribe (Encoder + Greedy-Decoder).

Der eager Cohere-Pfad ist bei kurzen Clips fast nur Kernel-Launch-Overhead: ~260 ms fuer
2 s Audio, ~1 s fuer 14 s, davon >90 % im Token-fuer-Token-Decoder (HF ``generate`` mit
Python-Overhead pro Schritt). Zusaetzlich baut cuDNN fuer jede neue Encoder-Laenge einen
Conv-Plan (+400-550 ms beim ersten Auftreten jeder Laenge). Dieses Modul ersetzt beides
ohne Inductor/Triton/C++-Compiler, nur mit ``torch.cuda.CUDAGraph``:

- Decoder: ``model.generate`` wird als Instanz-Attribut ersetzt. Cross-K/V werden einmal
  pro Request in statische, auf 448 Encoder-Frames (= 35 s Audio) gepaddete Puffer
  gerechnet, der Prompt (~9 Token) laeuft eager, jeder weitere Ein-Token-Schritt ist ein
  Graph-Replay pro Batch-Bucket (1/2/4/8/16) mit Self-KV-Laenge ``DECODER_CACHE_LEN``.
  EOS/finished werden im Graph gefuehrt. Alles Ungewoehnliche (Sampling, Beams, unbekannte
  kwargs, nicht-uniforme Prompt-Maske, B > 16, > 35 s, CPU) geht an das Original-generate.
- Encoder: ``model.encoder.forward`` wird ersetzt. Features werden in 128-Frame-Buckets
  bis 3584 Frames mit Nullen gepaddet (mathematisch ein No-op fuer die gueltigen Frames:
  Pre-Encode maskiert per Laenge, Attention maskiert gepaddete Keys, das Conv-Modul nullt
  Pad-Frames, BatchNorm ist eval) und die Ausgabe auf die eager Form zurueckgeschnitten.
  Batch-Buckets <= 4, groessere Batches werden geteilt.
- Der eager Encoder (Fallback und Capture) laeuft mit deaktiviertem cuDNN: das spart den
  Plan-Aufbau pro neuer Laenge. Sicher, weil alle GPU-Phasen des Prozesses auf einem
  Host-Thread serialisiert sind (``genesis_whisper_server_gpu._gpu_executor``).

Validiert gegen eager: Transkripte byte-identisch in 190+ Faellen (zufaellige Slices aller
Testaudios, Batch 3/5/8/16, > 35 s Chunking-Pfad, mehrere Threads). Gemessen auf RTX 4090:
2-s-Clip 260 -> 42 ms, 14,4-s-Datei ~1000 -> 129 ms, Warmup ~5 s.

VRAM ist begrenzt: nur die Buckets oben sind capture-faehig, zusaetzlich gilt eine harte
Obergrenze fuer Graph-Anzahl und Graph-Speicher (statische Puffer + private Graph-Pools).
Wird sie erreicht, laeuft die betroffene Form eager weiter, statt weiter zu capturen.
Gemessen (fp16): Warmup-Satz 31 Graphen / ~550 MB; alle 89 erlaubten Graphen in
produktionsnaher Lazy-Reihenfolge ~1,3 GB (Budget-Default 1,5 GB).

Capture passiert ausschliesslich auf dem Thread, der installiert hat (im Server der
GPU-Executor-Thread). Ein Capture wird von gleichzeitigem ``synchronize``/``empty_cache``/
grossen Allokationen auf anderen Threads zerstoert; auf fremden Threads werden vorhandene
Graphen nur abgespielt, fehlende Formen laufen eager.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ENCODER_FRAME_BUCKET = 128
# 35 s (max_audio_clip_s des Modells) -> 3501 Feature-Frames -> naechster 128er-Bucket.
ENCODER_MAX_FRAMES = 3584
ENCODER_BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4)
DECODER_BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16)
# Prompt (~9-10 Token) + max_new_tokens 256 des transcribe()-Pfads, auf 64 gerundet.
DECODER_CACHE_LEN = 320
# Harte Obergrenzen. Der vollstaendige erlaubte Bucket-Satz sind 3 x 28 Encoder- plus
# 5 Decoder-Graphen (89); das Speicherbudget ist die eigentliche Bremse. Gemessen belegen
# alle 89 Graphen ~1,3 GB (statische Puffer ~0,47 GB + Pools ~0,85 GB), das Budget liegt
# knapp darueber, damit ungluecklichere Capture-Reihenfolgen nicht unbegrenzt wachsen.
DEFAULT_MAX_GRAPHS = 89
DEFAULT_MEMORY_BUDGET_MB = 1536
# Startup-/Reload-Warmup: haeufige Live-Formen. Groessere Batches capturen lazy.
WARMUP_DECODER_BATCHES: Tuple[int, ...] = (1, 2, 4)
WARMUP_ENCODER_BATCHES: Tuple[int, ...] = (1,)

# generate()-kwargs, die der transcribe()-Pfad mit fester Bedeutung uebergibt. Alles andere
# (Sampling-, Beam-, Logits-Prozessor- oder Ausgabe-Optionen) geht an das Original.
_DECODER_ACCEPTED_KWARGS = frozenset(
    {
        "decoder_input_ids",
        "decoder_attention_mask",
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "decoder_start_token_id",
        "use_cache",
    }
)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def bucket_for(value: int, buckets: Sequence[int]) -> Optional[int]:
    """Kleinster Bucket >= value, None wenn value groesser als der groesste Bucket ist."""

    for bucket in buckets:
        if value <= bucket:
            return bucket
    return None


def frame_bucket_for(frames: int, frame_bucket: int = ENCODER_FRAME_BUCKET) -> int:
    return _ceil_div(max(1, int(frames)), frame_bucket) * frame_bucket


def encoder_output_len(frames: int) -> int:
    """Zeitlaenge nach dem 8x-Subsampling (drei Stride-2-Convs mit Padding 1, Kernel 3)."""

    length = int(frames)
    for _ in range(3):
        length = _ceil_div(length, 2)
    return length


def decoder_cache_len(prompt_len: int, max_new_tokens: int) -> int:
    return _ceil_div(int(prompt_len) + int(max_new_tokens), 64) * 64


def _tensor_bytes(tensors: Iterable[Optional[torch.Tensor]]) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors if t is not None)


@contextmanager
def cudnn_disabled() -> Iterator[None]:
    """Nur ``cudnn.enabled`` umschalten (prozessweit; sicher, weil GPU-Phasen serialisiert sind)."""

    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = previous


# EIN Side-Stream pro GPU fuer den ganzen Prozess, fuer Warmup UND Capture. Der
# Caching-Allocator haelt Bloecke pro Stream (ein neuer Stream pro Capture liess je Bucket
# einen kompletten Aktivierungssatz liegen, gemessen 14 GB reserviert), und cuBLAS legt pro
# (Handle, Stream) einen nie freigegebenen Workspace an: ein neuer Stream pro Modell-Load
# leakte ~9 MB, der Default-Capture-Stream von torch.cuda.graph pinnte seinen Workspace
# zudem im ersten Graph-Pool, der dadurch nie freigegeben wurde.
_SIDE_STREAMS: Dict[int, "torch.cuda.Stream"] = {}
_SIDE_STREAMS_LOCK = threading.Lock()


def _side_stream(device: torch.device) -> "torch.cuda.Stream":
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _SIDE_STREAMS_LOCK:
        stream = _SIDE_STREAMS.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            _SIDE_STREAMS[index] = stream
        return stream


def _private_pool_bytes(pool_id: Optional[Tuple[int, int]]) -> int:
    """Reservierte Bytes eines privaten Graph-Pools (nur beim Capture aufgerufen)."""

    if pool_id is None:
        return 0
    try:
        wanted = tuple(pool_id)
        return sum(
            int(segment.get("total_size", 0))
            for segment in torch.cuda.memory_snapshot()
            if tuple(segment.get("segment_pool_id", ())) == wanted
        )
    except Exception:
        return 0


def cohere_cuda_graphs_supported(model: Any, model_precision: str) -> Tuple[bool, str]:
    """Darf auf diesem geladenen Modell ueberhaupt gecaptured werden? (bool, Grund)."""

    if not torch.cuda.is_available():
        return False, "CUDA nicht verfuegbar"
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError):
        return False, "Modell ohne Parameter"
    if parameter.device.type != "cuda":
        return False, f"Modell liegt auf '{parameter.device}'"
    precision = str(model_precision or "").strip().lower()
    if precision not in {"fp16", "bf16"}:
        # Gegen eager verifiziert sind fp16 und bf16. int8_bnb/fp8 (Quantisierungs-Kernels,
        # eigene Allokationen) und fp32 (~8 GB Modell) sind ungetestet -> eager.
        return False, f"Precision '{precision}' wird nicht unterstuetzt"
    if getattr(model, "is_quantized", False) or getattr(model, "hf_quantizer", None) is not None:
        return False, "quantisiertes Modell"
    if not parameter.is_floating_point():
        return False, "Gewichte sind nicht floating point"
    for attribute in ("encoder", "transf_decoder", "log_softmax", "generation_config"):
        if not hasattr(model, attribute):
            return False, f"Modellstruktur unbekannt (kein '{attribute}')"
    return True, ""


class _EncoderState:
    def __init__(self, batch: int, channels: int, frames: int, device, dtype) -> None:
        self.batch = batch
        self.frames = frames
        self.feat = torch.zeros(batch, channels, frames, device=device, dtype=dtype)
        self.length = torch.full((batch,), frames, device=device, dtype=torch.long)
        self.out: Optional[torch.Tensor] = None
        self.out_len: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def static_bytes(self) -> int:
        return _tensor_bytes((self.feat, self.length))

    @staticmethod
    def estimate_bytes(batch: int, channels: int, frames: int, dtype) -> int:
        return batch * channels * frames * torch.empty((), dtype=dtype).element_size() + batch * 8


class _DecoderKV:
    """Self-/Cross-KV-Puffer fuer den groessten Decoder-Bucket. Jeder Batch-Bucket nutzt die
    ersten B Zeilen als View (gleiche Strides wie ein eigener Tensor). Sicher, weil Decoder-
    Laeufe unter dem Handle-Lock nie ueberlappen; spart gegenueber eigenen Puffern pro Bucket
    die Summe 1+2+4+8+16 = 31 Zeilen zugunsten von 16 (~27 MB pro Zeile bei fp16)."""

    def __init__(self, runner: "CohereCudaGraphs", device, dtype) -> None:
        rows = runner.decoder_batch_buckets[-1]
        heads, head_dim = runner.num_heads, runner.head_dim
        layers = len(runner.layers)
        self.k_self = [torch.zeros(rows, heads, runner.cache_len, head_dim, device=device, dtype=dtype) for _ in range(layers)]
        self.v_self = [torch.zeros(rows, heads, runner.cache_len, head_dim, device=device, dtype=dtype) for _ in range(layers)]
        self.k_cross = [torch.zeros(rows, heads, runner.cross_len, head_dim, device=device, dtype=dtype) for _ in range(layers)]
        self.v_cross = [torch.zeros(rows, heads, runner.cross_len, head_dim, device=device, dtype=dtype) for _ in range(layers)]

    def nbytes(self) -> int:
        return _tensor_bytes([*self.k_self, *self.v_self, *self.k_cross, *self.v_cross])

    @staticmethod
    def estimate_bytes(runner: "CohereCudaGraphs", dtype) -> int:
        element = torch.empty((), dtype=dtype).element_size()
        rows = runner.decoder_batch_buckets[-1]
        per_layer = 2 * rows * runner.num_heads * runner.head_dim * (runner.cache_len + runner.cross_len)
        return len(runner.layers) * per_layer * element


class _DecoderState:
    def __init__(self, runner: "CohereCudaGraphs", batch: int, kv: _DecoderKV, device, dtype) -> None:
        cache_len, cross_len = runner.cache_len, runner.cross_len
        self.batch = batch
        self.cache_len = cache_len
        self.k_self = [tensor[:batch] for tensor in kv.k_self]
        self.v_self = [tensor[:batch] for tensor in kv.v_self]
        self.k_cross = [tensor[:batch] for tensor in kv.k_cross]
        self.v_cross = [tensor[:batch] for tensor in kv.v_cross]
        self.cross_mask = torch.zeros(batch, 1, 1, cross_len, device=device, dtype=dtype)
        self.tok = torch.zeros(batch, 1, device=device, dtype=torch.long)
        self.pos = torch.zeros(1, device=device, dtype=torch.long)
        self.finished = torch.zeros(batch, device=device, dtype=torch.bool)
        self.out_tok = torch.zeros(batch, device=device, dtype=torch.long)
        self.key_idx = torch.arange(cache_len, device=device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def static_bytes(self) -> int:
        # Die KV-Views gehoeren dem gemeinsamen _DecoderKV und werden dort gezaehlt.
        return _tensor_bytes((self.cross_mask, self.tok, self.pos, self.finished, self.out_tok, self.key_idx))

    @staticmethod
    def estimate_bytes(runner: "CohereCudaGraphs", batch: int, dtype) -> int:
        element = torch.empty((), dtype=dtype).element_size()
        return batch * runner.cross_len * element + batch * 32 + runner.cache_len * 8 + 8


class CohereCudaGraphs:
    """Handle fuer die installierten Cohere-CUDA-Graphen (``warmup``/``uninstall``/``stats``)."""

    def __init__(
        self,
        model: Any,
        *,
        encoder_batch_buckets: Sequence[int] = ENCODER_BATCH_BUCKETS,
        decoder_batch_buckets: Sequence[int] = DECODER_BATCH_BUCKETS,
        frame_bucket: int = ENCODER_FRAME_BUCKET,
        max_frames: int = ENCODER_MAX_FRAMES,
        cache_len: int = DECODER_CACHE_LEN,
        max_graphs: int = DEFAULT_MAX_GRAPHS,
        memory_budget_mb: Optional[float] = DEFAULT_MEMORY_BUDGET_MB,
    ) -> None:
        self.model = model
        self.encoder = model.encoder
        decoder = model.transf_decoder
        self.embedding = decoder._embedding
        self.layers = list(decoder._decoder.layers)
        self.final_layer_norm = decoder._decoder.final_layer_norm
        self.head = model.log_softmax
        self.projection = getattr(model, "encoder_decoder_proj", None)
        first_attention = self.layers[0].first_sub_layer
        self.num_heads = int(first_attention.num_heads)
        self.head_dim = int(first_attention.head_dim)
        self.hidden = int(first_attention.hidden_size)

        self.encoder_batch_buckets = tuple(sorted({int(b) for b in encoder_batch_buckets if int(b) >= 1}))
        self.decoder_batch_buckets = tuple(sorted({int(b) for b in decoder_batch_buckets if int(b) >= 1}))
        if not self.encoder_batch_buckets or not self.decoder_batch_buckets:
            raise ValueError("Cohere CUDA-Graphs brauchen mindestens einen Encoder- und Decoder-Batch-Bucket.")
        self.encoder_max_batch = self.encoder_batch_buckets[-1]
        self.frame_bucket = int(frame_bucket)
        self.max_frames = frame_bucket_for(max_frames, self.frame_bucket)
        self.cross_len = encoder_output_len(self.max_frames)
        position_limit = getattr(getattr(self.embedding, "position_embedding", None), "max_sequence_length", cache_len)
        self.cache_len = min(int(cache_len), int(position_limit))
        self.max_graphs = max(0, int(max_graphs))
        self.memory_budget_bytes = None if memory_budget_mb is None else int(float(memory_budget_mb) * 2**20)

        generation_config = model.generation_config
        eos = generation_config.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        pad = generation_config.pad_token_id
        if eos is None or pad is None:
            raise ValueError("Cohere generation_config ohne eos_token_id/pad_token_id.")
        self.eos_token_id = int(eos)
        self.pad_token_id = int(pad)

        self._lock = threading.RLock()
        self._owner_thread: Optional[int] = None
        self._installed = False
        self._capture_disabled_reason: Optional[str] = None
        self._decoder_kv: Optional[_DecoderKV] = None
        self._encoder_pool: Optional[Tuple[int, int]] = None
        self._decoder_pool: Optional[Tuple[int, int]] = None
        self._encoder_states: Dict[Tuple[int, int], _EncoderState] = {}
        self._decoder_states: Dict[int, _DecoderState] = {}
        self._pool_bytes = 0
        self._logged_reasons: set = set()
        self._had_generate_attr = False
        self._had_forward_attr = False
        self._orig_generate = None
        self._orig_encoder_forward = None
        self._counters: Dict[str, int] = {
            "decoder_graph": 0,
            "decoder_fallback": 0,
            "encoder_graph": 0,
            "encoder_eager": 0,
            "captures": 0,
            "capture_failures": 0,
            "capture_skipped": 0,
        }
        self._fallback_reasons: Dict[str, int] = {}
        self._capture_seconds = 0.0
        self._warmup_seconds: Optional[float] = None

    # ------------------------------------------------------------------ install

    def install(self) -> "CohereCudaGraphs":
        if self._installed:
            return self
        model, encoder = self.model, self.encoder
        self._had_generate_attr = "generate" in model.__dict__
        self._had_forward_attr = "forward" in encoder.__dict__
        self._orig_generate = model.generate
        self._orig_encoder_forward = encoder.forward
        runner = self

        def generate(_model_self, *args, **kwargs):
            return runner.generate(*args, **kwargs)

        def encoder_forward(input_features=None, length=None, return_dict=False, **kwargs):
            return runner.encoder_forward(input_features, length, return_dict=return_dict, **kwargs)

        model.generate = types.MethodType(generate, model)
        encoder.forward = encoder_forward
        self._owner_thread = threading.get_ident()
        self._installed = True
        return self

    def uninstall(self) -> None:
        """Originale wiederherstellen und Graphen samt privaten Pools freigeben.

        Reihenfolge ist wichtig: erst die Ausgabe-/Puffertensoren loslassen, dann die Graphen
        zuruecksetzen. Umgekehrt bleiben Bloecke im bereits freigegebenen Pool haengen und
        ``empty_cache`` gibt sie nie an den Treiber zurueck. ``empty_cache`` macht der Aufrufer.
        """

        with self._lock:
            if self._installed:
                if self._had_generate_attr:
                    self.model.generate = self._orig_generate
                else:
                    self.model.__dict__.pop("generate", None)
                if self._had_forward_attr:
                    self.encoder.forward = self._orig_encoder_forward
                else:
                    self.encoder.__dict__.pop("forward", None)
                self._installed = False
            graphs: List[torch.cuda.CUDAGraph] = []
            for state in self._encoder_states.values():
                if state.graph is not None:
                    graphs.append(state.graph)
                state.graph = None
                state.out = state.out_len = None
                state.feat = state.length = None
            for state in self._decoder_states.values():
                if state.graph is not None:
                    graphs.append(state.graph)
                state.graph = None
                for name in ("k_self", "v_self", "k_cross", "v_cross"):
                    setattr(state, name, [])
            self._encoder_states.clear()
            self._decoder_states.clear()
            self._decoder_kv = None
            for graph in graphs:
                try:
                    graph.reset()
                except Exception:
                    pass
            graphs.clear()
            self._encoder_pool = None
            self._decoder_pool = None
            self._pool_bytes = 0
            self._capture_disabled_reason = "deinstalliert"
            # Keine Referenz auf das (ggf. gleich entladene) Modell behalten, falls jemand das
            # Handle noch haelt (Logs/Stats): sonst bliebe das ganze Modell im VRAM.
            self.model = self.encoder = self.embedding = self.head = self.projection = None
            self.final_layer_norm = None
            self.layers = []
            self._orig_generate = self._orig_encoder_forward = None

    @property
    def installed(self) -> bool:
        return self._installed

    # ------------------------------------------------------------------ bookkeeping

    def graph_count(self) -> int:
        return sum(1 for s in self._encoder_states.values() if s.graph is not None) + sum(
            1 for s in self._decoder_states.values() if s.graph is not None
        )

    def static_bytes(self) -> int:
        kv_bytes = self._decoder_kv.nbytes() if self._decoder_kv is not None else 0
        return (
            kv_bytes
            + sum(s.static_bytes() for s in self._encoder_states.values())
            + sum(s.static_bytes() for s in self._decoder_states.values())
        )

    def graph_memory_bytes(self) -> int:
        return self.static_bytes() + self._pool_bytes

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "installed": self._installed,
                "graphs": self.graph_count(),
                "encoder_graphs": sorted(k for k, s in self._encoder_states.items() if s.graph is not None),
                "decoder_graphs": sorted(k for k, s in self._decoder_states.items() if s.graph is not None),
                "max_graphs": self.max_graphs,
                "static_mb": round(self.static_bytes() / 2**20, 1),
                "pool_mb": round(self._pool_bytes / 2**20, 1),
                "graph_memory_mb": round(self.graph_memory_bytes() / 2**20, 1),
                "memory_budget_mb": None if self.memory_budget_bytes is None else round(self.memory_budget_bytes / 2**20, 1),
                "capture_seconds": round(self._capture_seconds, 2),
                "warmup_seconds": None if self._warmup_seconds is None else round(self._warmup_seconds, 2),
                "capture_disabled_reason": self._capture_disabled_reason,
                "fallback_reasons": dict(self._fallback_reasons),
                **self._counters,
            }

    def _log_once(self, key: str, message: str) -> None:
        if key in self._logged_reasons:
            return
        self._logged_reasons.add(key)
        print(message, file=sys.stderr)

    def _note_fallback(self, reason: str) -> None:
        self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1

    def _capture_block_reason(self, new_state_bytes: int) -> Optional[str]:
        """None wenn ein neuer Graph gecaptured werden darf, sonst der Grund."""

        if self._capture_disabled_reason is not None:
            return self._capture_disabled_reason
        if threading.get_ident() != self._owner_thread:
            return "fremder Thread"
        if self.graph_count() >= self.max_graphs:
            return f"Graph-Limit {self.max_graphs} erreicht"
        if self.memory_budget_bytes is not None and self.graph_memory_bytes() + new_state_bytes > self.memory_budget_bytes:
            return f"Graph-Speicherbudget {self.memory_budget_bytes // 2**20} MB erreicht"
        return None

    def _may_capture(self, kind: str, new_state_bytes: int) -> bool:
        reason = self._capture_block_reason(new_state_bytes)
        if reason is None:
            return True
        self._counters["capture_skipped"] += 1
        self._log_once(
            f"skip:{reason}",
            f"[INFO] Cohere CUDA-Graph-Capture ({kind}) uebersprungen: {reason} -> diese Form laeuft eager.",
        )
        return False

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Cohere CUDA-Graph-Capture ist nur auf dem installierenden GPU-Thread erlaubt.")

    def _run_capture(self, kind: str, pool_attr: str, warm_fn, capture_fn) -> Optional[torch.cuda.CUDAGraph]:
        """Side-Stream-Warmup + Capture. Liefert den Graphen oder None bei einem Fehler."""

        self._require_owner_thread()
        started = time.perf_counter()
        try:
            device = next(self.model.parameters()).device
            with torch.cuda.device(device):
                stream = _side_stream(device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    warm_fn()
                torch.cuda.current_stream().wait_stream(stream)
                if getattr(self, pool_attr) is None:
                    setattr(self, pool_attr, torch.cuda.graph_pool_handle())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, pool=getattr(self, pool_attr), stream=stream, capture_error_mode="thread_local"
                ):
                    capture_fn()
        except Exception as exc:
            self._counters["capture_failures"] += 1
            self._capture_disabled_reason = f"Capture-Fehler ({type(exc).__name__})"
            print(
                f"[WARNUNG] Cohere CUDA-Graph-Capture ({kind}) fehlgeschlagen ({type(exc).__name__}: {exc}). "
                "Weitere Captures deaktiviert, betroffene Formen laufen eager.",
                file=sys.stderr,
            )
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return None
        finally:
            self._capture_seconds += time.perf_counter() - started
        self._counters["captures"] += 1
        return graph

    def _refresh_pool_bytes(self) -> None:
        self._pool_bytes = _private_pool_bytes(self._encoder_pool) + _private_pool_bytes(self._decoder_pool)

    def _feature_channels(self) -> int:
        encoder_config = getattr(getattr(self.model, "config", None), "encoder", None)
        if isinstance(encoder_config, dict):
            return int(encoder_config.get("feat_in", 128))
        return int(getattr(encoder_config, "feat_in", 128) or 128)

    # ------------------------------------------------------------------ encoder

    def _eager_encoder(self, input_features, length, return_dict=False, **kwargs):
        with cudnn_disabled():
            return self._orig_encoder_forward(input_features, length, return_dict=return_dict, **kwargs)

    def _encoder_graph_state(self, batch_bucket: int, frames_bucket: int, channels: int, device) -> Optional[_EncoderState]:
        key = (batch_bucket, frames_bucket)
        state = self._encoder_states.get(key)
        if state is not None and state.graph is not None:
            return state
        largest_key = (batch_bucket, self.max_frames)
        if frames_bucket < self.max_frames and largest_key not in self._encoder_states:
            # Pool zuerst mit der groessten Form dieses Batch-Buckets belegen: jede kleinere
            # Form passt danach in die freigewordenen Bloecke. Lazy in aufsteigender Reihenfolge
            # gecaptured wuchs der gemeinsame Pool sonst je Capture (gemessen 7,4 GB statt 0,5 GB).
            self._encoder_graph_state(batch_bucket, self.max_frames, channels, device)
        conv_dtype = self.encoder.pre_encode.conv[0].weight.dtype
        estimate = _EncoderState.estimate_bytes(batch_bucket, channels, frames_bucket, conv_dtype)
        if not self._may_capture(f"Encoder B={batch_bucket} F={frames_bucket}", estimate):
            return None
        state = _EncoderState(batch_bucket, channels, frames_bucket, device, conv_dtype)

        def run_once():
            with torch.inference_mode(), cudnn_disabled():
                return self._orig_encoder_forward(state.feat, state.length)

        outputs: Dict[str, Any] = {}

        def capture():
            outputs["out"], outputs["len"] = run_once()

        graph = self._run_capture(f"Encoder B={batch_bucket} F={frames_bucket}", "_encoder_pool", run_once, capture)
        if graph is None:
            return None
        state.out, state.out_len, state.graph = outputs["out"], outputs["len"], graph
        self._encoder_states[key] = state
        self._refresh_pool_bytes()
        return state

    def encoder_forward(self, input_features=None, length=None, return_dict=False, **kwargs):
        if (
            input_features is None
            or length is None
            or return_dict
            or kwargs
            or not torch.is_tensor(input_features)
            or input_features.dim() != 3
            or input_features.device.type != "cuda"
        ):
            self._counters["encoder_eager"] += 1
            return self._eager_encoder(input_features, length, return_dict=return_dict, **kwargs)
        batch, channels, frames = input_features.shape
        if batch > self.encoder_max_batch:
            return self._encoder_forward_split(input_features, length)
        frames_bucket = frame_bucket_for(frames, self.frame_bucket)
        batch_bucket = bucket_for(batch, self.encoder_batch_buckets)
        if batch_bucket is None or frames_bucket > self.max_frames:
            self._counters["encoder_eager"] += 1
            self._note_fallback("encoder_shape")
            return self._eager_encoder(input_features, length)
        with self._lock, torch.inference_mode():
            state = self._encoder_graph_state(batch_bucket, frames_bucket, channels, input_features.device)
            if state is None:
                self._counters["encoder_eager"] += 1
                self._note_fallback("encoder_no_graph")
                return self._eager_encoder(input_features, length)
            state.feat.zero_()
            state.feat[:batch, :, :frames].copy_(input_features)
            state.length.fill_(frames_bucket)
            state.length[:batch].copy_(length)
            state.graph.replay()
            out_frames = encoder_output_len(frames)
            hidden = state.out[:batch, :out_frames].clone()
            out_len = state.out_len[:batch].clone()
        self._counters["encoder_graph"] += 1
        return hidden, out_len

    def _encoder_forward_split(self, input_features, length):
        """Grosse Batches in Graph-taugliche Teile (je eigener Frame-Bucket) zerlegen und die
        Ausgaben auf die eager Laenge zurueckpaddeln; Pad-Frames maskiert der Decoder."""

        _, _, frames = input_features.shape
        out_frames = encoder_output_len(frames)
        hidden_parts, length_parts = [], []
        for start in range(0, input_features.shape[0], self.encoder_max_batch):
            part_lengths = length[start : start + self.encoder_max_batch]
            used_frames = int(part_lengths.max().item())
            hidden, part_out_len = self.encoder_forward(
                input_features[start : start + self.encoder_max_batch, :, :used_frames],
                part_lengths,
            )
            hidden_parts.append(F.pad(hidden, (0, 0, 0, out_frames - hidden.shape[1])))
            length_parts.append(part_out_len)
        return torch.cat(hidden_parts, 0), torch.cat(length_parts, 0)

    # ------------------------------------------------------------------ decoder

    def _attention(self, attention, query_input, key, value, mask):
        batch, steps, _ = query_input.shape
        query = attention._reshape(attention.query_net(query_input))
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, scale=attention.scale)
        return attention.out_projection(output.transpose(1, 2).contiguous().view(batch, steps, self.hidden))

    def _decoder_step(self, state: _DecoderState) -> None:
        """Ein Decode-Schritt auf statischen Puffern (capture-faehig: kein Host-Sync, feste Formen)."""

        batch = state.batch
        position = state.pos
        hidden = self.embedding(state.tok, position.view(1, 1).expand(batch, 1))
        negative = torch.full((), float("-inf"), device=hidden.device, dtype=hidden.dtype)
        zero = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        self_mask = torch.where(state.key_idx <= position, zero, negative).view(1, 1, 1, state.cache_len)
        for index, layer in enumerate(self.layers):
            attention = layer.first_sub_layer
            normed = layer.layer_norm_1(hidden)
            key = attention._reshape(attention.key_net(normed))
            value = attention._reshape(attention.value_net(normed))
            state.k_self[index].index_copy_(2, position, key)
            state.v_self[index].index_copy_(2, position, value)
            hidden = hidden + self._attention(attention, normed, state.k_self[index], state.v_self[index], self_mask)
            cross = layer.second_sub_layer
            hidden = hidden + self._attention(
                cross, layer.layer_norm_2(hidden), state.k_cross[index], state.v_cross[index], state.cross_mask
            )
            hidden = hidden + layer.third_sub_layer(layer.layer_norm_3(hidden))
        hidden = self.final_layer_norm(hidden)
        log_probs = self.head(hidden)[:, -1, :].float()
        next_token = torch.argmax(log_probs, dim=-1)
        next_token = torch.where(state.finished, torch.full_like(next_token, self.pad_token_id), next_token)
        state.out_tok.copy_(next_token)
        state.finished.logical_or_(next_token == self.eos_token_id)
        state.tok.copy_(next_token.view(batch, 1))
        state.pos.add_(1)

    def _decoder_graph_state(self, batch_bucket: int, device, dtype) -> Optional[_DecoderState]:
        state = self._decoder_states.get(batch_bucket)
        if state is not None and state.graph is not None:
            return state
        estimate = _DecoderState.estimate_bytes(self, batch_bucket, dtype)
        if self._decoder_kv is None:
            estimate += _DecoderKV.estimate_bytes(self, dtype)
        if not self._may_capture(f"Decoder B={batch_bucket}", estimate):
            return None
        if self._decoder_kv is None:
            self._decoder_kv = _DecoderKV(self, device, dtype)
        state = _DecoderState(self, batch_bucket, self._decoder_kv, device, dtype)
        scratch_position = state.cache_len - 1  # nie von einem echten Request erreicht

        def warm():
            for _ in range(2):
                state.pos.fill_(scratch_position)
                self._decoder_step(state)

        def capture():
            self._decoder_step(state)

        with torch.inference_mode():
            state.pos.fill_(scratch_position)
            graph = self._run_capture(f"Decoder B={batch_bucket}", "_decoder_pool", warm, capture)
            for tensor in (state.tok, state.pos, state.finished, state.out_tok):
                tensor.zero_()
        if graph is None:
            return None
        state.graph = graph
        self._decoder_states[batch_bucket] = state
        self._refresh_pool_bytes()
        return state

    def _decoder_route(self, input_features, input_ids, length, attention_mask, kwargs) -> Tuple[Optional[str], Any, int]:
        """(Fallback-Grund oder None, decoder_input_ids, max_new_tokens)."""

        decoder_ids = kwargs.get("decoder_input_ids", input_ids)
        if set(kwargs) - _DECODER_ACCEPTED_KWARGS:
            return "generate_kwargs", decoder_ids, 0
        if kwargs.get("do_sample"):
            return "sampling", decoder_ids, 0
        if int(kwargs.get("num_beams", 1) or 1) != 1:
            return "beams", decoder_ids, 0
        if kwargs.get("use_cache", True) is False:
            return "no_cache", decoder_ids, 0
        if input_ids is not None and "decoder_input_ids" in kwargs:
            return "ambiguous_ids", decoder_ids, 0
        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            return "max_new_tokens", decoder_ids, 0
        max_new_tokens = int(max_new_tokens)
        if (
            input_features is None
            or decoder_ids is None
            or length is None
            or not torch.is_tensor(input_features)
            or input_features.dim() != 3
            or not torch.is_tensor(decoder_ids)
            or decoder_ids.dim() != 2
        ):
            return "inputs", decoder_ids, max_new_tokens
        if decoder_ids.shape[0] != input_features.shape[0] or max_new_tokens < 1:
            return "inputs", decoder_ids, max_new_tokens
        if bucket_for(int(decoder_ids.shape[0]), self.decoder_batch_buckets) is None:
            return "batch", decoder_ids, max_new_tokens
        if decoder_ids.shape[1] + max_new_tokens > self.cache_len:
            return "cache_len", decoder_ids, max_new_tokens
        if encoder_output_len(int(input_features.shape[-1])) > self.cross_len:
            return "audio_len", decoder_ids, max_new_tokens
        decoder_mask = kwargs.get("decoder_attention_mask")
        if decoder_mask is not None and not bool(decoder_mask.bool().all()):
            return "decoder_mask", decoder_ids, max_new_tokens
        if attention_mask is not None and not bool(torch.as_tensor(attention_mask).bool().all()):
            return "attention_mask", decoder_ids, max_new_tokens
        if input_features.device.type != "cuda":
            return "device", decoder_ids, max_new_tokens
        return None, decoder_ids, max_new_tokens

    def generate(self, input_features=None, input_ids=None, length=None, attention_mask=None, **kwargs):
        reason, decoder_ids, max_new_tokens = self._decoder_route(input_features, input_ids, length, attention_mask, kwargs)
        if reason is None:
            parameter = next(self.model.parameters())
            batch_bucket = bucket_for(int(decoder_ids.shape[0]), self.decoder_batch_buckets)
            with self._lock, torch.inference_mode():
                state = self._decoder_graph_state(batch_bucket, input_features.device, parameter.dtype)
                if state is not None:
                    self._counters["decoder_graph"] += 1
                    return self._generate_with_graph(state, input_features, length, decoder_ids, max_new_tokens)
            reason = "decoder_no_graph"
        self._counters["decoder_fallback"] += 1
        self._note_fallback(reason)
        return self._orig_generate(
            input_features=input_features, input_ids=input_ids, length=length, attention_mask=attention_mask, **kwargs
        )

    def _generate_with_graph(self, state: _DecoderState, input_features, length, decoder_ids, max_new_tokens):
        model = self.model
        encoded, _ = model.encoder(input_features, length)
        if self.projection is not None:
            encoded = self.projection(encoded)
        batch, frames, _ = encoded.shape
        batch_bucket = state.batch
        device = encoded.device
        dtype = state.cross_mask.dtype
        if encoded.dtype != dtype:
            encoded = encoded.to(dtype)

        # Cross-K/V + Maske einmal pro Request in die statischen Puffer.
        encoder_lengths = model._infer_encoder_lengths_from_raw(length)
        valid = torch.arange(frames, device=device)[None, :] < encoder_lengths.to(device)[:, None]
        state.cross_mask.zero_()
        state.cross_mask[:, :, :, frames:] = float("-inf")
        state.cross_mask[:batch, 0, 0, :frames].masked_fill_(~valid, float("-inf"))
        for index, layer in enumerate(self.layers):
            cross = layer.second_sub_layer
            state.k_cross[index][:batch, :, :frames].copy_(cross._reshape(cross.key_net(encoded)))
            state.v_cross[index][:batch, :, :frames].copy_(cross._reshape(cross.value_net(encoded)))
            if batch_bucket > batch:
                state.k_cross[index][batch:].zero_()
                state.v_cross[index][batch:].zero_()
        if batch_bucket > batch:
            state.cross_mask[batch:].zero_()

        # Prompt-Prefill eager, gleiche Mathematik wie das Original-forward mit kv_seq_len=P.
        prompt_len = decoder_ids.shape[1]
        ids = decoder_ids
        if batch_bucket > batch:
            ids = torch.cat([ids, ids[:1].expand(batch_bucket - batch, -1)], 0)
        positions = torch.arange(prompt_len, device=device)[None, :].expand(batch_bucket, -1)
        hidden = self.embedding(ids, positions)
        causal = torch.full((prompt_len, prompt_len), float("-inf"), device=device, dtype=dtype).triu_(1)
        causal = causal.view(1, 1, prompt_len, prompt_len)
        for index, layer in enumerate(self.layers):
            attention = layer.first_sub_layer
            normed = layer.layer_norm_1(hidden)
            key = attention._reshape(attention.key_net(normed))
            value = attention._reshape(attention.value_net(normed))
            state.k_self[index][:, :, :prompt_len].copy_(key)
            state.v_self[index][:, :, :prompt_len].copy_(value)
            hidden = hidden + self._attention(attention, normed, key, value, causal)
            cross = layer.second_sub_layer
            hidden = hidden + self._attention(
                cross, layer.layer_norm_2(hidden), state.k_cross[index], state.v_cross[index], state.cross_mask
            )
            hidden = hidden + layer.third_sub_layer(layer.layer_norm_3(hidden))
        hidden = self.final_layer_norm(hidden)
        first_token = torch.argmax(self.head(hidden)[:, -1, :].float(), dim=-1)

        finished = torch.zeros(batch_bucket, device=device, dtype=torch.bool)
        finished[batch:] = True
        state.finished.copy_(finished | (first_token == self.eos_token_id))
        state.tok.copy_(first_token.view(batch_bucket, 1))
        state.pos.fill_(prompt_len)
        generated = [first_token]
        steps = 1
        while steps < max_new_tokens:
            if bool(state.finished.all()):
                break
            state.graph.replay()
            generated.append(state.out_tok.clone())
            steps += 1
        tokens = torch.stack(generated, 1)[:batch]
        return torch.cat([decoder_ids, tokens], 1)

    # ------------------------------------------------------------------ warmup

    def warmup(
        self,
        decoder_batches: Sequence[int] = WARMUP_DECODER_BATCHES,
        encoder_batches: Sequence[int] = WARMUP_ENCODER_BATCHES,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decoder-Buckets und alle Encoder-Frame-Buckets vorab capturen (groesste zuerst:
        kleinere Captures nutzen die freigewordenen Bloecke des gemeinsamen Pools; aufsteigend
        wuchs der Pool je Bucket)."""

        self._require_owner_thread()
        if not self._installed:
            raise RuntimeError("Cohere CUDA-Graphs sind nicht installiert.")
        parameter = next(self.model.parameters())
        started = time.perf_counter()
        top_frames = min(self.max_frames, frame_bucket_for(max_frames or self.max_frames, self.frame_bucket))
        channels = self._feature_channels()
        with self._lock, torch.inference_mode():
            for batch in decoder_batches:
                batch_bucket = bucket_for(int(batch), self.decoder_batch_buckets)
                if batch_bucket is not None:
                    self._decoder_graph_state(batch_bucket, parameter.device, parameter.dtype)
            for batch in encoder_batches:
                batch_bucket = bucket_for(int(batch), self.encoder_batch_buckets)
                if batch_bucket is None:
                    continue
                for frames in range(top_frames, 0, -self.frame_bucket):
                    self._encoder_graph_state(batch_bucket, frames, channels, parameter.device)
        self._warmup_seconds = time.perf_counter() - started
        return self.stats()


class CohereCudnnFreeEncoder:
    """Nur-eager-Variante: der Encoder laeuft ohne cuDNN (kein Plan-Aufbau pro neuer Laenge)."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.encoder = model.encoder
        self._had_forward_attr = False
        self._orig_encoder_forward = None
        self._installed = False

    def install(self) -> "CohereCudnnFreeEncoder":
        if self._installed:
            return self
        encoder = self.encoder
        self._had_forward_attr = "forward" in encoder.__dict__
        self._orig_encoder_forward = encoder.forward
        original = self._orig_encoder_forward

        def encoder_forward(*args, **kwargs):
            with cudnn_disabled():
                return original(*args, **kwargs)

        encoder.forward = encoder_forward
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._had_forward_attr:
            self.encoder.forward = self._orig_encoder_forward
        else:
            self.encoder.__dict__.pop("forward", None)
        self._installed = False
        self.model = self.encoder = self._orig_encoder_forward = None

    @property
    def installed(self) -> bool:
        return self._installed

    def stats(self) -> Dict[str, Any]:
        return {"installed": self._installed, "graphs": 0, "mode": "eager_cudnn_off"}


def install_cohere_cuda_graphs(model: Any, **options: Any) -> CohereCudaGraphs:
    """Graphen installieren (noch ohne Capture). Danach ``warmup()`` auf demselben Thread."""

    return CohereCudaGraphs(model, **options).install()


def install_cohere_cudnn_free_encoder(model: Any) -> CohereCudnnFreeEncoder:
    return CohereCudnnFreeEncoder(model).install()


__all__ = [
    "CohereCudaGraphs",
    "CohereCudnnFreeEncoder",
    "DECODER_BATCH_BUCKETS",
    "DECODER_CACHE_LEN",
    "DEFAULT_MAX_GRAPHS",
    "DEFAULT_MEMORY_BUDGET_MB",
    "ENCODER_BATCH_BUCKETS",
    "ENCODER_FRAME_BUCKET",
    "ENCODER_MAX_FRAMES",
    "bucket_for",
    "cohere_cuda_graphs_supported",
    "cudnn_disabled",
    "decoder_cache_len",
    "encoder_output_len",
    "frame_bucket_for",
    "install_cohere_cuda_graphs",
    "install_cohere_cudnn_free_encoder",
]

"""Local speaker segmentation: a pure-PyTorch port of pyannote's PyanNet (segmentation-3.0).

The model is the frame-level part of ``pyannote/speaker-diarization-community-1`` (its ``segmentation`` folder, the
same weights as ``pyannote/segmentation-3.0``): SincNet front end, 4-layer BiLSTM, two linear layers and a powerset
classifier over 3 local speakers with at most 2 active at once. It answers, per ~17 ms frame of a 10 s chunk, which of
up to three local speakers talks. The server uses it for the speech/overlap count of the DIA-free diarization and for
the live ``speaker_turns`` speech mask; speaker identity comes from ReDimNet (``genesis_whisper_server_vid``).

Neither pyannote.audio nor asteroid are dependencies: the network is rebuilt here with the same parameter names, and
the weights are read from the pyannote Lightning checkpoint through a restricted unpickler (see
``load_checkpoint_state``). Inference slides exactly like ``pyannote.audio.Inference`` with ``skip_aggregation``:
10 s chunks, 1 s step, the last incomplete chunk zero-padded, powerset output converted to hard multilabel.
Nothing is ever downloaded; the checkpoint comes from the ``segmentation_model_path`` setting or the local HF cache.
"""

from __future__ import annotations

import io
import math
import os
import pickle
import sys
import threading
import types
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .genesis_whisper_server_globals import current_settings, resolve_local_model_cache_path, settings_lock
from .genesis_whisper_server_gpu import shared_gpu_lease


SEGMENTATION_MODEL_NAME = "pyannote/segmentation-3.0"
SEGMENTATION_SAMPLE_RATE = 16000
SEGMENTATION_CHUNK_SECONDS = 10.0
SEGMENTATION_STEP_SECONDS = 1.0
SEGMENTATION_NUM_SPEAKERS = 3
SEGMENTATION_MAX_SIMULTANEOUS = 2
SEGMENTATION_BATCH_SIZE = 32
# HF cache layouts searched for the checkpoint, in order (repo, file inside the snapshot).
SEGMENTATION_CHECKPOINT_CANDIDATES = (
    ("pyannote/speaker-diarization-community-1", "segmentation/pytorch_model.bin"),
    ("pyannote/segmentation-3.0", "pytorch_model.bin"),
)

# SincNet layout of segmentation-3.0: (kernel, stride) of every conv/pool layer, in order. Used for the frame grid.
_SINC_FILTERS = 80
_SINC_KERNEL = 251
_FRONT_END_LAYERS = ((_SINC_KERNEL, 10), (3, 3), (5, 1), (3, 3), (5, 1), (3, 3))


# ---------------------------------------------------------------------------
# Restricted checkpoint loading
# ---------------------------------------------------------------------------

class _InertObject(dict):
    """Stand-in for every class the checkpoint names outside the allow-list.

    A pyannote checkpoint pickles Lightning/pyannote objects (``Specifications``, ``Problem``, version info, ...) next
    to the state_dict. Importing those classes would need pyannote, and resolving arbitrary globals would let a
    crafted file run any callable (``os.system`` via REDUCE). An inert stub accepts every pickle protocol operation
    (construction, ``__setstate__``, item/list appends) and does nothing else, so the file loads without executing
    code and the parts we do not need simply become dead data.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = dict.__new__(cls)
        obj._pickle_args = args
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - never runs the real constructor
        pass

    def __setstate__(self, state: Any) -> None:
        self._pickle_state = state

    def append(self, value: Any) -> None:
        self.setdefault("_pickle_items", []).append(value)

    def extend(self, values: Any) -> None:
        for value in values:
            self.append(value)

    def __call__(self, *args: Any, **kwargs: Any) -> "_InertObject":
        return _InertObject(*args)


# The only globals resolved to real objects: tensor/parameter rebuild helpers and OrderedDict (the state_dict type).
# Storage classes are resolved by torch's own loader before this unpickler is asked.
_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
}
_stub_classes: Dict[tuple[str, str], type] = {}


def _stub_class(module: str, name: str) -> type:
    key = (module, name)
    if key not in _stub_classes:
        _stub_classes[key] = type(f"Inert_{name}", (_InertObject,), {"__module__": "g3_inert." + module})
    return _stub_classes[key]


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        return _stub_class(module, name)


def _restricted_load(file: Any, **kwargs: Any) -> Any:
    return _RestrictedUnpickler(file, **kwargs).load()


def _restricted_loads(data: bytes, **kwargs: Any) -> Any:
    return _RestrictedUnpickler(io.BytesIO(data), **kwargs).load()


# torch.load(pickle_module=...) only needs Unpickler/load/loads and a __name__.
_restricted_pickle = types.ModuleType("g3_restricted_pickle")
_restricted_pickle.Unpickler = _RestrictedUnpickler  # type: ignore[attr-defined]
_restricted_pickle.load = _restricted_load  # type: ignore[attr-defined]
_restricted_pickle.loads = _restricted_loads  # type: ignore[attr-defined]


def load_checkpoint_state(path: str | os.PathLike) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """The state_dict and plain hyper-parameters of a pyannote checkpoint, without executing pickled code.

    ``weights_only=True`` refuses the file (it pickles pyannote classes); ``weights_only=False`` with the stock
    pickle would import and call whatever the file names. torch's zip loader runs our ``_RestrictedUnpickler``
    instead (it subclasses ``pickle_module.Unpickler`` and keeps storage handling), so only the allow-listed tensor
    rebuild helpers are real and every other class is an inert stub.
    """

    checkpoint = torch.load(str(path), map_location="cpu", pickle_module=_restricted_pickle, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Segmentierungs-Checkpoint hat kein erwartetes Format.")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("Segmentierungs-Checkpoint enthaelt kein state_dict.")
    tensors = {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError("Segmentierungs-Checkpoint enthaelt keine Gewichte.")
    hparams_raw = checkpoint.get("hyper_parameters")
    hparams = _plain(hparams_raw) if isinstance(hparams_raw, Mapping) else {}
    return tensors, hparams if isinstance(hparams, dict) else {}


def _plain(value: Any) -> Any:
    """Plain dict/list/scalar copy of hyper-parameters; inert stubs and anything exotic become None."""

    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items() if not str(k).startswith("_pickle")}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return None


# ---------------------------------------------------------------------------
# Network (parameter names follow pyannote.audio PyanNet / asteroid ParamSincFB)
# ---------------------------------------------------------------------------

class ParamSincFB(nn.Module):
    """asteroid_filterbanks.ParamSincFB: learnable band-pass sinc filters, cos and sin halves."""

    def __init__(self, n_filters: int = _SINC_FILTERS, kernel_size: int = _SINC_KERNEL, sample_rate: float = 16000.0,
                 min_low_hz: float = 50.0, min_band_hz: float = 50.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)
        self.half_kernel = kernel_size // 2
        self.cutoff = n_filters // 2
        # Initial values only (the checkpoint overwrites them): mel-spaced bands as asteroid initialises them.
        low_hz = 30.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(2595 * np.log10(1 + low_hz / 700), 2595 * np.log10(1 + high_hz / 700), self.cutoff + 1,
                          dtype="float32")
        hz = 700 * (10 ** (mel / 2595) - 1)
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz)).view(-1, 1))
        window = torch.hamming_window(kernel_size, periodic=False)[: self.half_kernel]
        n = 2 * math.pi * (torch.arange(-self.half_kernel, 0.0).view(1, -1) / self.sample_rate)
        self.register_buffer("window_", window, persistent=False)
        self.register_buffer("n_", n, persistent=False)

    def _bandpass(self, low: torch.Tensor, high: torch.Tensor, kind: str) -> torch.Tensor:
        band = (high - low)[:, 0]
        ft_low = torch.matmul(low, self.n_)
        ft_high = torch.matmul(high, self.n_)
        if kind == "cos":
            left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2)) * self.window_
            center = 2 * band.view(-1, 1)
            right = torch.flip(left, dims=[1])
        else:
            left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2)) * self.window_
            center = torch.zeros_like(band.view(-1, 1))
            right = -torch.flip(left, dims=[1])
        band_pass = torch.cat([left, center, right], dim=1) / (2 * band[:, None])
        return band_pass.view(self.n_filters // 2, 1, self.kernel_size)

    def filters(self) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        return torch.cat([self._bandpass(low, high, "cos"), self._bandpass(low, high, "sin")], dim=0)


class _SincEncoder(nn.Module):
    """asteroid_filterbanks.Encoder around ParamSincFB (no padding): ``conv1d.0.filterbank.*`` in the state_dict."""

    def __init__(self, stride: int, sample_rate: int) -> None:
        super().__init__()
        self.filterbank = ParamSincFB(sample_rate=sample_rate)
        self.stride = int(stride)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return F.conv1d(waveform, self.filterbank.filters(), stride=self.stride)


class SincNet(nn.Module):
    def __init__(self, sample_rate: int = SEGMENTATION_SAMPLE_RATE, stride: int = 10) -> None:
        super().__init__()
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        self.conv1d = nn.ModuleList([_SincEncoder(stride, sample_rate), nn.Conv1d(80, 60, 5), nn.Conv1d(60, 60, 5)])
        self.pool1d = nn.ModuleList([nn.MaxPool1d(3, stride=3) for _ in range(3)])
        self.norm1d = nn.ModuleList([nn.InstanceNorm1d(80, affine=True), nn.InstanceNorm1d(60, affine=True),
                                     nn.InstanceNorm1d(60, affine=True)])

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.wav_norm1d(waveforms)
        for index, (conv, pool, norm) in enumerate(zip(self.conv1d, self.pool1d, self.norm1d)):
            outputs = conv(outputs)
            if index == 0:
                outputs = torch.abs(outputs)
            outputs = F.leaky_relu(norm(pool(outputs)))
        return outputs


class PyanNet(nn.Module):
    """SincNet -> BiLSTM -> linear stack -> powerset log-probabilities (batch, frames, classes)."""

    def __init__(self, *, lstm_layers: int = 4, lstm_hidden: int = 128, linear_layers: int = 2,
                 linear_hidden: int = 128, num_classes: int = 7, sinc_stride: int = 10,
                 sample_rate: int = SEGMENTATION_SAMPLE_RATE) -> None:
        super().__init__()
        self.sincnet = SincNet(sample_rate=sample_rate, stride=sinc_stride)
        self.lstm = nn.LSTM(60, lstm_hidden, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        width = 2 * lstm_hidden
        self.linear = nn.ModuleList()
        for _ in range(linear_layers):
            self.linear.append(nn.Linear(width, linear_hidden))
            width = linear_hidden
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.sincnet(waveforms)
        outputs, _ = self.lstm(outputs.transpose(1, 2))
        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))
        return F.log_softmax(self.classifier(outputs), dim=-1)


def powerset_mapping(num_speakers: int = SEGMENTATION_NUM_SPEAKERS,
                     max_simultaneous: int = SEGMENTATION_MAX_SIMULTANEOUS) -> np.ndarray:
    """(classes, speakers) 0/1 matrix in pyannote's Powerset order: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}."""

    rows = []
    for size in range(max_simultaneous + 1):
        for subset in combinations(range(num_speakers), size):
            row = np.zeros(num_speakers, np.float32)
            row[list(subset)] = 1.0
            rows.append(row)
    return np.stack(rows)


def powerset_to_multilabel(log_probs: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Hard conversion (pyannote Powerset.to_multilabel, soft=False): one-hot argmax times the mapping."""

    hard = F.one_hot(torch.argmax(log_probs, dim=-1), mapping.shape[0]).to(mapping.dtype)
    return torch.matmul(hard, mapping)


def frame_grid(sample_rate: int = SEGMENTATION_SAMPLE_RATE) -> tuple[float, float, float]:
    """(start, step, duration) in seconds of the output frames (pyannote's ``receptive_field``)."""

    size, jump = 1, 1
    for kernel, stride in _FRONT_END_LAYERS:
        size += (kernel - 1) * jump
        jump *= stride
    return 0.0, jump / sample_rate, size / sample_rate


def num_output_frames(num_samples: int) -> int:
    frames = int(num_samples)
    for kernel, stride in _FRONT_END_LAYERS:
        frames = (frames - kernel) // stride + 1
    return frames


def build_model(state: Mapping[str, torch.Tensor], hparams: Optional[Mapping[str, Any]] = None) -> PyanNet:
    """A PyanNet shaped after the state_dict, weights loaded strictly (unknown extra buffers are ignored)."""

    hparams = hparams or {}
    lstm_layers = len({key for key in state if key.startswith("lstm.weight_ih_l") and not key.endswith("_reverse")})
    if lstm_layers == 0 or "lstm.weight_hh_l0" not in state:
        raise ValueError("Segmentierungs-Checkpoint hat kein PyanNet-LSTM.")
    lstm_hidden = int(state["lstm.weight_hh_l0"].shape[1])
    linear_keys = sorted(key for key in state if key.startswith("linear.") and key.endswith(".weight"))
    linear_hidden = int(state[linear_keys[0]].shape[0]) if linear_keys else 2 * lstm_hidden
    num_classes = int(state["classifier.weight"].shape[0])
    expected_classes = powerset_mapping().shape[0]
    if num_classes != expected_classes:
        raise ValueError(
            f"Segmentierungs-Checkpoint hat {num_classes} Klassen, erwartet {expected_classes} (Powerset 3 Sprecher)."
        )
    sincnet = hparams.get("sincnet") if isinstance(hparams.get("sincnet"), Mapping) else {}
    stride = int(sincnet.get("stride") or 10)
    sample_rate = int(hparams.get("sample_rate") or SEGMENTATION_SAMPLE_RATE)
    if sample_rate != SEGMENTATION_SAMPLE_RATE or stride != 10:
        raise ValueError("Segmentierungs-Checkpoint passt nicht zu segmentation-3.0 (16 kHz, SincNet-Schritt 10).")
    model = PyanNet(lstm_layers=lstm_layers, lstm_hidden=lstm_hidden, linear_layers=len(linear_keys),
                    linear_hidden=linear_hidden, num_classes=num_classes, sinc_stride=stride, sample_rate=sample_rate)
    own = model.state_dict()
    missing = [key for key in own if key not in state]
    if missing:
        raise ValueError(f"Segmentierungs-Checkpoint unvollstaendig: {', '.join(missing[:5])}")
    model.load_state_dict({key: state[key] for key in own}, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def _batch_bucket(size: int, maximum: int) -> int:
    bucket = 1
    while bucket < size:
        bucket *= 2
    return min(bucket, maximum)


def chunk_audio(audio: np.ndarray, sample_rate: int = SEGMENTATION_SAMPLE_RATE,
                chunk_seconds: float = SEGMENTATION_CHUNK_SECONDS,
                step_seconds: float = SEGMENTATION_STEP_SECONDS) -> np.ndarray:
    """(chunks, samples) exactly as pyannote Inference.slide cuts: full chunks, then the zero-padded remainder."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    window = int(round(chunk_seconds * sample_rate))
    step = int(round(step_seconds * sample_rate))
    total = len(samples)
    full = (total - window) // step + 1 if total >= window else 0
    chunks = [samples[i * step: i * step + window] for i in range(full)]
    if total < window or (total - window) % step > 0:
        last = samples[full * step:]
        padded = np.zeros(window, np.float32)
        padded[: len(last)] = last
        chunks.append(padded)
    return np.stack(chunks) if chunks else np.zeros((0, window), np.float32)


class SegmentationModel:
    """A loaded PyanNet on one device with the powerset mapping; ``segment`` returns the seg-3.0 chunk arrays."""

    def __init__(self, model: PyanNet, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.mapping = torch.from_numpy(powerset_mapping()).to(device)
        start, step, duration = frame_grid()
        self.frame_start, self.frame_step, self.frame_duration = start, step, duration

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike, device: torch.device | str = "cpu") -> "SegmentationModel":
        state, hparams = load_checkpoint_state(path)
        return cls(build_model(state, hparams), torch.device(device))

    def log_probs(self, chunks: np.ndarray) -> np.ndarray:
        """Raw powerset log-probabilities (chunks, frames, 7) for already cut chunks."""

        return self._run(chunks, convert=False)

    def _run(self, chunks: np.ndarray, *, convert: bool, batch_size: int = SEGMENTATION_BATCH_SIZE) -> np.ndarray:
        chunks = np.asarray(chunks, dtype=np.float32)
        outputs = []
        with torch.inference_mode():
            for first in range(0, len(chunks), batch_size):
                current = chunks[first: first + batch_size]
                # Pad to power-of-two batch shapes: the network is per-sample (instance norm, LSTM), padding rows
                # never change the real rows, and fixed shapes avoid first-shape kernel preparation on CUDA.
                bucket = _batch_bucket(len(current), batch_size) if self.device.type == "cuda" else len(current)
                batch = np.zeros((bucket, current.shape[1]), np.float32)
                batch[: len(current)] = current
                tensor = torch.from_numpy(batch).to(self.device)[:, None, :]
                result = self.model(tensor)[: len(current)]
                if convert:
                    result = powerset_to_multilabel(result, self.mapping)
                outputs.append(result.float().cpu().numpy())
        if not outputs:
            frames = num_output_frames(int(SEGMENTATION_CHUNK_SECONDS * SEGMENTATION_SAMPLE_RATE))
            width = self.mapping.shape[1] if convert else self.mapping.shape[0]
            return np.zeros((0, frames, width), np.float32)
        return np.concatenate(outputs)

    def segment(self, audio: np.ndarray) -> Dict[str, Any]:
        """seg-3.0 output of a 16 kHz mono signal, the arrays pyannote's segmentation Inference produces.

        ``data``: (chunks, 589, 3) hard multilabel activations (0/1) of the three local speakers per chunk;
        chunk ``c`` starts at ``c * chunk_step`` seconds; frame ``f`` of a chunk sits at
        ``frame_start + f * frame_step`` inside it.
        """

        chunks = chunk_audio(audio)
        data = self._run(chunks, convert=True)
        return {
            "data": data,
            "chunk_start": 0.0,
            "chunk_step": SEGMENTATION_STEP_SECONDS,
            "chunk_dur": SEGMENTATION_CHUNK_SECONDS,
            "frame_start": self.frame_start,
            "frame_step": self.frame_step,
            "frame_dur": self.frame_duration,
        }


# ---------------------------------------------------------------------------
# Discovery and the process-wide instance
# ---------------------------------------------------------------------------

def _hf_cache_root() -> Path:
    """The HF cache the ASR models use (local_model_cache_path), else HF_HUB_CACHE, else ~/.cache/huggingface/hub."""

    with settings_lock:
        configured = str(current_settings.get("local_model_cache_path", "") or "").strip()
    resolved = resolve_local_model_cache_path(configured)
    if resolved:
        return Path(resolved)
    env_cache = str(os.getenv("HF_HUB_CACHE") or "").strip()
    if env_cache:
        return Path(env_cache).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_file(cache_root: Path, repo_id: str, relative: str) -> Optional[Path]:
    repo_root = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_root / "snapshots"
    candidates: list[Path] = []
    try:
        revision = (repo_root / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision:
        candidates.append(snapshots / revision)
    if snapshots.is_dir():
        try:
            candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), reverse=True))
        except OSError:
            pass
    for snapshot in candidates:
        path = snapshot / relative
        if path.is_file():
            return path
    return None


def resolve_segmentation_checkpoint(settings: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    """The checkpoint file: ``segmentation_model_path`` when set (must exist), else the first HF cache hit."""

    if settings is None:
        with settings_lock:
            settings = dict(current_settings)
    explicit = str(settings.get("segmentation_model_path") or "").strip()
    if explicit:
        path = Path(resolve_local_model_cache_path(explicit))
        return path if path.is_file() else None
    roots = [_hf_cache_root()]
    default_root = Path.home() / ".cache" / "huggingface" / "hub"
    if default_root not in roots:
        roots.append(default_root)
    for root in roots:
        for repo_id, relative in SEGMENTATION_CHECKPOINT_CANDIDATES:
            path = _snapshot_file(root, repo_id, relative)
            if path is not None:
                return path
    return None


_segmentation_lock = threading.Lock()
_segmentation_state: Dict[str, Any] = {"model": None, "path": None, "device": None, "error": None}


def _target_device() -> torch.device:
    from .genesis_whisper_server_vid import resolve_local_device

    return resolve_local_device()


def load_segmentation_model() -> Optional[SegmentationModel]:
    """The process-wide model (loaded once, ~6 MB of weights), or None when no checkpoint is usable."""

    with _segmentation_lock:
        path = resolve_segmentation_checkpoint()
        device = _target_device()
        current = _segmentation_state.get("model")
        if current is not None and _segmentation_state.get("path") == str(path) and str(current.device) == str(device):
            return current
        if path is None:
            _segmentation_state.update({"model": None, "path": None, "device": None,
                                        "error": "Kein Segmentierungsmodell (segmentation-3.0) gefunden."})
            return None
        try:
            with shared_gpu_lease():
                model = SegmentationModel.from_checkpoint(path, device)
        except Exception as exc:
            _segmentation_state.update({"model": None, "path": None, "device": None,
                                        "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FEHLER-SEG] Segmentierungsmodell konnte nicht geladen werden: {exc}", file=sys.stderr)
            return None
        _segmentation_state.update({"model": model, "path": str(path), "device": str(device), "error": None})
        print(f"[INFO-SEG] {SEGMENTATION_MODEL_NAME} geladen aus {path} auf {device}.", file=sys.stderr)
        return model


def segmentation_available() -> bool:
    """A checkpoint exists locally (cheap; does not load the model)."""

    return resolve_segmentation_checkpoint() is not None


def segmentation_status() -> Dict[str, Any]:
    with _segmentation_lock:
        return {key: _segmentation_state.get(key) for key in ("path", "device", "error")} | {
            "loaded": _segmentation_state.get("model") is not None
        }


def segment_audio(audio: np.ndarray) -> Dict[str, Any]:
    """seg-3.0 arrays of ``audio`` with the process-wide model; RuntimeError when no model is available."""

    model = load_segmentation_model()
    if model is None:
        raise RuntimeError(_segmentation_state.get("error") or "Segmentierungsmodell nicht verfuegbar.")
    with shared_gpu_lease():
        return model.segment(audio)


__all__ = [
    "PyanNet",
    "SEGMENTATION_MODEL_NAME",
    "SegmentationModel",
    "build_model",
    "chunk_audio",
    "frame_grid",
    "load_checkpoint_state",
    "load_segmentation_model",
    "num_output_frames",
    "powerset_mapping",
    "powerset_to_multilabel",
    "resolve_segmentation_checkpoint",
    "segment_audio",
    "segmentation_available",
    "segmentation_status",
]

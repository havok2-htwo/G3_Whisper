"""Batched inference for IBM Granite Speech 5.0 2B (transformers GraniteSpeechForConditionalGeneration).

One scheduler batch becomes one processor call and one ``generate`` per sub-batch. The LLM is decoder-only, so
prompts are padded on the left; every clip gets exactly its own number of audio tokens (the processor expands
``<|audio|>`` per clip) and ``input_features_mask`` picks the valid projector outputs. The conformer encoder has no
padding mask, so a short clip next to a long one sees a little zero-padded silence: clips are sorted by length
before they are cut into sub-batches. A sub-batch holds at most GRANITE_MAX_INFERENCE_BATCH_SIZE clips and at most
GRANITE_KV_BUDGET_MB of KV cache (rows x padded length), in every speed mode.

Speed modes (setting ``granite_compile``): "off" runs eager generate with a DynamicCache; "static_cache" uses the
HF StaticCache without compiling; "compile" adds HF's torch.compile of the decode step (mode "reduce-overhead",
Inductor CUDA graphs, needs CUDA, Triton and a C++ compiler). Fast modes pad a sub-batch to GRANITE_BATCH_BUCKETS
rows and the cache to GRANITE_CACHE_LENGTH_BUCKET tokens, so the cache in ``model._cache`` is reused and compile
sees a bounded set of shapes; generation still stops at the eager token budget. A CUDA OOM runs that one sub-batch
eager; any other fast-mode failure is logged once and the process stays eager from then on. ``speed_stats`` reports
what really ran.

Keywords (the ASR glossary) end the prompt as " Keywords: a, b, c", the form of the model card. One glossary per call:
the scheduler batches only equal processing keys. Their tokens count in the KV sizing, so a long glossary makes
sub-batches smaller; a sub-batch whose cache would outgrow GRANITE_MAX_CACHE_LENGTH runs eager.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .genesis_whisper_server_chunking import combine_transcription_chunks
from .genesis_whisper_server_globals import (
    GRANITE_COMPILE_MODES,
    GRANITE_LANGUAGE_NAMES,
    current_settings,
    settings_lock,
)

SAMPLE_RATE = 16000
GRANITE_MAX_INFERENCE_BATCH_SIZE = 16
# Longer clips are cut at quiet moments and their texts joined (the model card recommends at most ~2 min).
GRANITE_MAX_ITEM_SECONDS = 60.0
# Shorter clips carry no decodable audio frame.
GRANITE_MIN_SAMPLES = SAMPLE_RATE // 10
# max_new_tokens per second of the longest clip. German needs ~5 tokens/s; fast Japanese ~11 (about 1.1 tokens per
# character), and "auto" or an unknown language may be Japanese.
GRANITE_TOKENS_PER_SECOND = 10
GRANITE_DENSE_TOKENS_PER_SECOND = 16
GRANITE_SPARSE_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})
GRANITE_TOKEN_MARGIN = 32
GRANITE_MAX_NEW_TOKENS_CAP = 1024
# Sub-batch KV sizing before the processor runs: projector tokens per audio second plus a prompt allowance
# (the plain prompt; keywords add their own count).
GRANITE_AUDIO_TOKENS_PER_SECOND = 10
GRANITE_PROMPT_TOKENS = 64
# Keyword token estimate when the tokenizer cannot count: ASCII ~3 chars per token, anything else one token per char.
GRANITE_KEYWORD_CHARS_PER_TOKEN = 3
GRANITE_KV_BUDGET_MB = 1536
# Fast modes: row and length buckets of the static cache (and of the compiled decode step).
GRANITE_BATCH_BUCKETS = (1, 2, 4, 8, 16)
GRANITE_CACHE_LENGTH_BUCKET = 256
# Covers every (batch bucket, length bucket) pair; dynamo's default of 8 would silently fall back to eager.
GRANITE_COMPILE_RECOMPILE_LIMIT = 64

PUNCTUATION_INSTRUCTION = "transcribe the speech with proper punctuation and capitalization."
KEYWORDS_PREFIX = " Keywords: "
AUDIO_TOKEN = "<|audio|>"
# Int8 (bitsandbytes) keeps the audio path and the output head in the compute dtype.
GRANITE_INT8_SKIP_MODULES = ["encoder", "projector", "language_model.lm_head"]

_state_lock = threading.Lock()


def _fresh_speed_state() -> Dict[str, Any]:
    return {
        "failure": None,
        "logged": set(),
        "last_mode": None,
        "fast_calls": 0,
        "eager_calls": 0,
        "budget_skips": 0,
        "length_skips": 0,
        "oom_fallbacks": 0,
        "truncated_rows": 0,
        "compiled_shapes": set(),
    }


_speed_state: Dict[str, Any] = _fresh_speed_state()


def _log_once(key: str, message: str) -> None:
    with _state_lock:
        if key in _speed_state["logged"]:
            return
        _speed_state["logged"].add(key)
    print(message, file=sys.stderr)


def keywords_suffix(keywords: Sequence[str] = ()) -> str:
    return f"{KEYWORDS_PREFIX}{', '.join(keywords)}" if keywords else ""


def prompt_content(language: Optional[str], keywords: Sequence[str] = ()) -> str:
    """User message of the prompt: with a language hint for the six Granite ASR languages, plain otherwise; the
    keywords, if any, at its end."""

    suffix = keywords_suffix(keywords)
    name = GRANITE_LANGUAGE_NAMES.get((language or "auto").strip().lower())
    if name is None:
        return f"{AUDIO_TOKEN}{PUNCTUATION_INSTRUCTION}{suffix}"
    return f"{AUDIO_TOKEN}The speech is in {name}. Transcribe it exactly; do not translate. {PUNCTUATION_INSTRUCTION}{suffix}"


def prompt_tokens(processor: Any, keywords: Sequence[str] = ()) -> int:
    """Prompt allowance of one row: GRANITE_PROMPT_TOKENS plus the keyword suffix, counted once per call."""

    suffix = keywords_suffix(keywords)
    if not suffix:
        return GRANITE_PROMPT_TOKENS
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        counted = len(tokenizer.encode(suffix, add_special_tokens=False))
    except Exception:
        counted = 0
    if counted <= 0:
        ascii_chars = sum(1 for char in suffix if ord(char) < 128)
        counted = math.ceil(ascii_chars / GRANITE_KEYWORD_CHARS_PER_TOKEN) + (len(suffix) - ascii_chars)
    return GRANITE_PROMPT_TOKENS + int(counted)


def build_prompt(processor: Any, language: Optional[str], keywords: Sequence[str] = ()) -> str:
    content = prompt_content(language, keywords)
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # chat_template.jinja of the model: "USER: {content}\n ASSISTANT:".
        return f"USER: {content}\n ASSISTANT:"


def ensure_left_padding(processor: Any) -> None:
    """Batched decoder-only generate needs left padding; the snapshot sets it, this keeps it that way."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", "left") != "left":
        print("[INFO] Granite: Tokenizer padding_side auf 'left' gesetzt (Batch-Generate).", file=sys.stderr)
        tokenizer.padding_side = "left"


def tokens_per_second(language: Optional[str]) -> int:
    code = (language or "auto").strip().lower()
    return GRANITE_TOKENS_PER_SECOND if code in GRANITE_SPARSE_LANGUAGES else GRANITE_DENSE_TOKENS_PER_SECOND


def max_new_tokens_for(seconds: float, language: Optional[str] = None) -> int:
    rate = tokens_per_second(language)
    return int(min(GRANITE_MAX_NEW_TOKENS_CAP, math.ceil(max(0.0, seconds) * rate) + GRANITE_TOKEN_MARGIN))


def estimated_total_tokens(seconds: float, language: Optional[str] = None, prompt: int = GRANITE_PROMPT_TOKENS) -> int:
    """Prompt plus audio plus new tokens of one row, an upper estimate made before the processor runs."""

    audio_tokens = math.ceil(max(0.0, seconds) * GRANITE_AUDIO_TOKENS_PER_SECOND)
    return int(prompt) + audio_tokens + max_new_tokens_for(seconds, language)


def bucket_batch_size(rows: int) -> int:
    for bucket in GRANITE_BATCH_BUCKETS:
        if rows <= bucket:
            return bucket
    return rows


def bucket_cache_length(tokens: int) -> int:
    return int(math.ceil(max(1, tokens) / GRANITE_CACHE_LENGTH_BUCKET) * GRANITE_CACHE_LENGTH_BUCKET)


# Largest static cache: the longest clip with the plain prompt. Longer (keyword) prompts run eager, so the fast modes
# keep their bounded set of shapes.
GRANITE_MAX_CACHE_LENGTH = bucket_cache_length(estimated_total_tokens(GRANITE_MAX_ITEM_SECONDS, "ja"))


def fits_static_cache(tokens: int) -> bool:
    return bucket_cache_length(tokens) <= GRANITE_MAX_CACHE_LENGTH


def split_long_item(samples: np.ndarray, max_seconds: float = GRANITE_MAX_ITEM_SECONDS) -> List[np.ndarray]:
    """One clip, or parts of at most ``max_seconds`` cut at the quietest moment near each even split."""

    duration = len(samples) / SAMPLE_RATE
    if duration <= max_seconds:
        return [samples]
    from .genesis_whisper_server_wxc import region_pieces

    parts = []
    for start, end in region_pieces([(0.0, duration)], max_seconds, audio=samples):
        part = samples[max(0, round(start * SAMPLE_RATE)) : round(end * SAMPLE_RATE)]
        if len(part):
            parts.append(part)
    return parts


def configured_speed_mode() -> str:
    with settings_lock:
        mode = str(current_settings.get("granite_compile", "off") or "off").strip().lower()
    return mode if mode in GRANITE_COMPILE_MODES else "off"


def _inductor_unavailable_reason(device: torch.device) -> Optional[str]:
    if device.type != "cuda":
        return "kein CUDA-Geraet"
    if os.name == "nt":
        has_cxx = shutil.which("cl") is not None
    else:
        has_cxx = any(shutil.which(cxx) for cxx in ("g++", "clang++", "c++"))
    if not has_cxx:
        return "kein C++-Compiler (cl/g++) gefunden"
    try:
        from torch.utils._triton import has_triton

        if not has_triton():
            return "Triton nicht installiert"
    except Exception:
        return "Triton nicht installiert"
    return None


def _kv_bytes_per_token(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    layers = int(getattr(text_config, "num_hidden_layers", 40))
    heads = int(getattr(text_config, "num_attention_heads", 16))
    kv_heads = int(getattr(text_config, "num_key_value_heads", heads) or heads)
    head_dim = int(getattr(text_config, "head_dim", 0) or getattr(text_config, "hidden_size", 2048) // heads)
    try:
        element_size = next(model.parameters()).element_size()
    except Exception:
        element_size = 2
    return 2 * layers * kv_heads * head_dim * element_size


def kv_cache_mb(model: Any, rows: int, tokens: int) -> float:
    return rows * tokens * _kv_bytes_per_token(model) / (1024 * 1024)


def sub_batch_sizes(
    seconds_desc: Sequence[float],
    max_rows: int,
    language: Optional[str],
    model: Any,
    fast: bool,
    prompt: int = GRANITE_PROMPT_TOKENS,
) -> List[int]:
    """Row counts of consecutive sub-batches over clips sorted longest first.

    Each sub-batch is padded to its first (longest) clip; it takes as many rows as fit GRANITE_KV_BUDGET_MB, with the
    row and length buckets of the fast modes counted (a sub-batch beyond GRANITE_MAX_CACHE_LENGTH is sized eager and
    runs eager). ``prompt`` is the prompt allowance per row (prompt_tokens). One row always goes, an oversized one
    runs eager.
    """

    sizes: List[int] = []
    start = 0
    while start < len(seconds_desc):
        tokens = estimated_total_tokens(seconds_desc[start], language, prompt)
        bucketed = fast and fits_static_cache(tokens)
        if bucketed:
            tokens = bucket_cache_length(tokens)
        size = 1
        for rows in range(min(max_rows, len(seconds_desc) - start), 0, -1):
            padded_rows = bucket_batch_size(rows) if bucketed else rows
            if kv_cache_mb(model, padded_rows, tokens) <= GRANITE_KV_BUDGET_MB:
                size = rows
                break
        sizes.append(size)
        start += size
    return sizes


def _reset_compiled_graphs() -> None:
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    try:
        from torch._inductor import cudagraph_trees

        cudagraph_trees.reset_cudagraph_trees()
    except Exception:
        pass
    with _state_lock:
        _speed_state["compiled_shapes"] = set()


def release_speed_state(model: Any, keep_compiled: bool = False) -> None:
    """Drop the persistent StaticCache of ``model``; without ``keep_compiled`` also the compiled decode step, its
    dynamo frames and the CUDA-graph pool (call before empty_cache)."""

    attributes = getattr(model, "__dict__", None) if model is not None else None
    if not attributes:
        return
    had_compiled = "_compiled_call" in attributes
    names = ("_cache",) if keep_compiled else ("_cache", "_compiled_call", "_last_compile_config")
    for name in names:
        if name in attributes:
            try:
                delattr(model, name)
            except Exception:
                pass
    if had_compiled and not keep_compiled:
        _reset_compiled_graphs()


def release_idle_cache(model: Any) -> bool:
    """Idle trim: free the StaticCache unless compiled CUDA graphs were captured on it (they would re-record)."""

    attributes = getattr(model, "__dict__", None) if model is not None else None
    if not attributes or "_cache" not in attributes or "_compiled_call" in attributes:
        return False
    release_speed_state(model, keep_compiled=True)
    return True


def _kept_cache_mb(model: Any) -> float:
    cache = getattr(model, "__dict__", {}).get("_cache") if model is not None else None
    if cache is None:
        return 0.0
    try:
        return round(kv_cache_mb(model, int(cache.max_batch_size), int(cache.max_cache_len)), 1)
    except Exception:
        return 0.0


def speed_stats(model: Any = None) -> Dict[str, Any]:
    """What the Granite speed mode really did since start: for admin stats and benchmarks."""

    requested = configured_speed_mode()
    kept = _kept_cache_mb(model)
    with _state_lock:
        return {
            "requested": requested,
            "last_mode": _speed_state["last_mode"],
            "failure": _speed_state["failure"],
            "fast_calls": _speed_state["fast_calls"],
            "eager_calls": _speed_state["eager_calls"],
            "budget_skips": _speed_state["budget_skips"],
            "length_skips": _speed_state["length_skips"],
            "oom_fallbacks": _speed_state["oom_fallbacks"],
            "truncated_rows": _speed_state["truncated_rows"],
            "compiled_shapes": len(_speed_state["compiled_shapes"]),
            "recompile_limit": GRANITE_COMPILE_RECOMPILE_LIMIT,
            "kept_cache_mb": kept,
            "kv_budget_mb": GRANITE_KV_BUDGET_MB,
            "max_cache_length": GRANITE_MAX_CACHE_LENGTH,
        }


def _effective_speed_mode(model: Any, device: torch.device) -> str:
    mode = configured_speed_mode()
    if mode == "off":
        return "off"
    with _state_lock:
        failure = _speed_state["failure"]
    if failure is not None:
        return "off"
    if mode == "compile":
        quantizer = getattr(model, "hf_quantizer", None)
        if quantizer is not None and not getattr(quantizer, "is_compileable", False):
            _log_once(
                "compile:quantized",
                "[INFO] Granite granite_compile='compile': quantisiertes Modell, transformers kompiliert nicht -> static_cache.",
            )
            return "static_cache"
        reason = _inductor_unavailable_reason(device)
        if reason is not None:
            _log_once(f"compile:{reason}", f"[INFO] Granite granite_compile='compile' nicht moeglich: {reason} -> eager.")
            return "off"
    return mode


def _raise_recompile_limit() -> None:
    try:
        config = torch._dynamo.config
        if int(config.recompile_limit) < GRANITE_COMPILE_RECOMPILE_LIMIT:
            config.recompile_limit = GRANITE_COMPILE_RECOMPILE_LIMIT
    except Exception:
        pass


def _stop_at_length(limit: int) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopAtLength(StoppingCriteria):
        """Ends generate at the eager token budget while the static cache keeps its bucketed length."""

        def __init__(self, max_length: int):
            self.max_length = max_length

        def __call__(self, input_ids: torch.Tensor, scores: Any, **kwargs: Any) -> torch.Tensor:
            done = input_ids.shape[-1] >= self.max_length
            return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_StopAtLength(limit)])


def _generate(model: Any, inputs: Any, max_new_tokens: int, mode: str, cache_length: Optional[int] = None) -> torch.Tensor:
    kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1}
    if mode != "off":
        input_length = int(inputs["input_ids"].shape[1])
        # HF sizes the static cache from max_length; the stopping criterion keeps the eager budget.
        kwargs["max_new_tokens"] = max(max_new_tokens, int(cache_length or 0) - input_length)
        kwargs["stopping_criteria"] = _stop_at_length(input_length + max_new_tokens)
        kwargs["cache_implementation"] = "static"
        kwargs["disable_compile"] = mode != "compile"
        if mode == "compile":
            from transformers import CompileConfig

            _raise_recompile_limit()
            kwargs["compile_config"] = CompileConfig(fullgraph=False, mode="reduce-overhead")
    with torch.inference_mode():
        return model.generate(**inputs, **kwargs)


def _first_rows(inputs: Any, rows: int) -> Any:
    batch = int(inputs["input_ids"].shape[0])
    if rows >= batch:
        return inputs
    return {
        key: value[:rows] if torch.is_tensor(value) and tuple(value.shape[:1]) == (batch,) else value
        for key, value in inputs.items()
    }


def _is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _record_fast_call(model: Any, mode: str, batch: int, cache_length: int) -> None:
    cache = getattr(model, "__dict__", {}).get("_cache")
    shape = (int(getattr(cache, "max_batch_size", batch)), int(getattr(cache, "max_cache_len", cache_length)))
    with _state_lock:
        _speed_state["fast_calls"] += 1
        _speed_state["last_mode"] = mode
        shapes = 0
        if mode == "compile":
            _speed_state["compiled_shapes"].add(shape)
            shapes = len(_speed_state["compiled_shapes"])
    if shapes > GRANITE_COMPILE_RECOMPILE_LIMIT:
        _log_once(
            "recompile_limit",
            f"[WARNUNG] Granite compile: mehr als {GRANITE_COMPILE_RECOMPILE_LIMIT} Formen, dynamo laeuft fuer neue eager.",
        )


def _generate_with_fallback(
    model: Any, inputs: Any, needed_new_tokens: int, mode: str, rows: int, force_eager: bool = False
) -> torch.Tensor:
    """Generate in ``mode`` with an eager fallback; ``inputs`` may carry padding rows beyond ``rows`` (fast modes).

    ``force_eager``: one sub-batch too long for the static cache (counted by the caller) runs eager in an active fast
    mode; only the cache is freed, the compiled step stays for the next sub-batch that fits.
    """

    if mode == "off":
        # A cache or compiled step left by a fast mode switched off since would hold its VRAM for nothing.
        release_speed_state(model)
    elif force_eager:
        release_speed_state(model, keep_compiled=True)
    else:
        input_length = int(inputs["input_ids"].shape[1])
        batch = int(inputs["input_ids"].shape[0])
        cache_length = bucket_cache_length(input_length + needed_new_tokens)
        cache_mb = kv_cache_mb(model, batch, cache_length)
        if cache_length > GRANITE_MAX_CACHE_LENGTH:
            with _state_lock:
                _speed_state["length_skips"] += 1
            _log_once(
                "length",
                f"[INFO] Granite StaticCache fuer {cache_length} Tokens ueber der groessten Laenge "
                f"{GRANITE_MAX_CACHE_LENGTH} (lange Keyword-Liste) -> dieser und aehnliche Batches eager.",
            )
            release_speed_state(model, keep_compiled=True)
        elif cache_mb > GRANITE_KV_BUDGET_MB:
            with _state_lock:
                _speed_state["budget_skips"] += 1
            _log_once(
                "budget",
                f"[INFO] Granite StaticCache fuer {batch}x{cache_length} Tokens ({cache_mb:.0f} MB) ueber dem Budget "
                f"{GRANITE_KV_BUDGET_MB} MB -> dieser und aehnliche Batches eager.",
            )
            # Only the cache: the compiled step stays for the next batch that fits.
            release_speed_state(model, keep_compiled=True)
        else:
            try:
                output = _generate(model, inputs, needed_new_tokens, mode, cache_length)
                _record_fast_call(model, mode, batch, cache_length)
                return output
            except Exception as exc:
                if _is_out_of_memory(exc):
                    with _state_lock:
                        _speed_state["oom_fallbacks"] += 1
                    print(
                        f"[WARNUNG] Granite granite_compile='{mode}': CUDA OOM bei {batch}x{cache_length} Tokens, "
                        "dieser Batch eager, der Modus bleibt.",
                        file=sys.stderr,
                    )
                    release_speed_state(model, keep_compiled=True)
                else:
                    with _state_lock:
                        _speed_state["failure"] = f"{mode}: {type(exc).__name__}: {exc}"[:500]
                    print(
                        f"[WARNUNG] Granite granite_compile='{mode}' fehlgeschlagen ({type(exc).__name__}: {exc}). "
                        "Bis zum Neustart eager.",
                        file=sys.stderr,
                    )
                    release_speed_state(model)
                    if mode == "compile":
                        _reset_compiled_graphs()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    output = _generate(model, _first_rows(inputs, rows), needed_new_tokens, "off")
    with _state_lock:
        _speed_state["eager_calls"] += 1
        _speed_state["last_mode"] = "off"
    return output


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _stop_token_ids(processor: Any, model: Any) -> List[int]:
    ids: set = set()
    for source in (getattr(model, "generation_config", None), getattr(processor, "tokenizer", None)):
        if source is None:
            continue
        for attribute in ("eos_token_id", "pad_token_id"):
            value = getattr(source, attribute, None)
            values = value if isinstance(value, (list, tuple)) else [value]
            ids.update(int(item) for item in values if isinstance(item, int) and not isinstance(item, bool))
    return sorted(ids)


def _count_truncated_rows(new_tokens: torch.Tensor, needed: int, stop_ids: Sequence[int]) -> int:
    """Rows that used the whole budget without an end token (their transcript is cut)."""

    if not stop_ids or new_tokens.shape[1] < needed:
        return 0
    stops = torch.tensor(list(stop_ids), dtype=new_tokens.dtype, device=new_tokens.device)
    return int((~torch.isin(new_tokens, stops).any(dim=1)).sum().item())


def _transcribe_sub_batch(
    processor: Any,
    model: Any,
    clips: Sequence[np.ndarray],
    language: Optional[str],
    mode: str,
    keywords: Sequence[str] = (),
    force_eager: bool = False,
) -> List[str]:
    device = _model_device(model)
    prompt = build_prompt(processor, language, keywords)
    rows = len(clips)
    batch_clips = list(clips)
    if mode != "off" and not force_eager:
        # Fixed row buckets keep the static cache and the compiled shapes reusable; padding rows repeat the shortest clip.
        batch_clips += [clips[-1]] * (bucket_batch_size(rows) - rows)
    # A list of 1-D float32 arrays: the feature extractor pads them itself and keeps each clip's own length.
    inputs = processor([prompt] * len(batch_clips), audio=batch_clips, device=str(device), return_tensors="pt")
    inputs = inputs.to(device)
    longest_seconds = max(len(clip) for clip in clips) / SAMPLE_RATE
    needed = max_new_tokens_for(longest_seconds, language)
    output = _generate_with_fallback(model, inputs, needed, mode, rows, force_eager)
    # Left padding: every row's prompt ends at the same column.
    new_tokens = output[:rows, inputs["input_ids"].shape[1] :]
    truncated = _count_truncated_rows(new_tokens, needed, _stop_token_ids(processor, model))
    if truncated:
        with _state_lock:
            _speed_state["truncated_rows"] += truncated
        print(
            f"[WARNUNG] Granite: {truncated} von {rows} Clips am Token-Limit {needed} abgeschnitten "
            f"(laengster Clip {longest_seconds:.1f} s, Sprache {language or 'auto'}).",
            file=sys.stderr,
        )
    texts = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [str(text).strip() for text in texts]


def transcribe_granite_batch(
    processor: Any,
    model: Any,
    audio_batch: Sequence[np.ndarray],
    language: Optional[str],
    batch_size: int,
    keywords: Sequence[str] = (),
) -> List[str]:
    """Texts in input order. One language and one glossary per call: the scheduler batches only equal processing keys."""

    keywords = tuple(keywords or ())

    waveforms = [np.asarray(audio, dtype=np.float32).reshape(-1) for audio in audio_batch]
    parts: List[Tuple[int, int, np.ndarray]] = []
    for index, waveform in enumerate(waveforms):
        if len(waveform) < GRANITE_MIN_SAMPLES:
            continue
        for order, part in enumerate(split_long_item(waveform)):
            if len(part) >= GRANITE_MIN_SAMPLES:
                parts.append((index, order, part))

    texts_by_part: Dict[Tuple[int, int], str] = {}
    if parts:
        max_rows = max(1, min(int(batch_size or 1), GRANITE_MAX_INFERENCE_BATCH_SIZE))
        ordered = sorted(parts, key=lambda part: len(part[2]), reverse=True)
        mode = _effective_speed_mode(model, _model_device(model))
        seconds = [len(part[2]) / SAMPLE_RATE for part in ordered]
        prompt = prompt_tokens(processor, keywords)
        start = 0
        for size in sub_batch_sizes(seconds, max_rows, language, model, mode != "off", prompt):
            chunk = ordered[start : start + size]
            force_eager = False
            if mode != "off":
                # A fast mode that failed on an earlier sub-batch is off now.
                mode = _effective_speed_mode(model, _model_device(model))
                if mode != "off" and not fits_static_cache(estimated_total_tokens(seconds[start], language, prompt)):
                    # Sized eager by sub_batch_sizes: no padding rows, no static cache; the compiled step stays.
                    force_eager = True
                    with _state_lock:
                        _speed_state["length_skips"] += 1
            start += size
            results = _transcribe_sub_batch(
                processor, model, [part[2] for part in chunk], language, mode, keywords, force_eager
            )
            for (index, order, _), text in zip(chunk, results):
                texts_by_part[(index, order)] = text

    texts: List[str] = []
    for index in range(len(waveforms)):
        pieces = [texts_by_part[key] for key in sorted(key for key in texts_by_part if key[0] == index)]
        texts.append(combine_transcription_chunks(pieces))
    return texts

"""Batched inference for IBM Granite Speech 5.0 2B (transformers GraniteSpeechForConditionalGeneration).

One scheduler batch becomes one processor call and one ``generate`` per sub-batch of at most
GRANITE_MAX_INFERENCE_BATCH_SIZE clips. The LLM is decoder-only, so prompts are padded on the left; every clip gets
exactly its own number of audio tokens (the processor expands ``<|audio|>`` per clip) and ``input_features_mask``
picks the valid projector outputs. The conformer encoder has no padding mask, so a short clip next to a long one
sees a little zero-padded silence: clips are sorted by length before they are cut into sub-batches.

Speed modes (setting ``granite_compile``): "off" runs eager generate with a DynamicCache; "static_cache" uses the
HF StaticCache without compiling; "compile" adds HF's torch.compile of the decode step (mode "reduce-overhead",
Inductor CUDA graphs, needs CUDA, Triton and a C++ compiler). A static cache lives in ``model._cache`` between
batches, so it is capped by GRANITE_STATIC_CACHE_BUDGET_MB; a sub-batch that would need more runs eager. Any
failure in a fast mode is logged once and the process stays eager from then on.
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
# max_new_tokens per sub-batch: generous for every supported language (German ~5, Japanese ~8 tokens/s).
GRANITE_TOKENS_PER_SECOND = 10
GRANITE_TOKEN_MARGIN = 32
GRANITE_MAX_NEW_TOKENS_CAP = 1024
# Static cache: total length rounded up to this bucket so fewer reallocations and recompiles happen.
GRANITE_CACHE_LENGTH_BUCKET = 256
GRANITE_STATIC_CACHE_BUDGET_MB = 1536

PUNCTUATION_INSTRUCTION = "transcribe the speech with proper punctuation and capitalization."
AUDIO_TOKEN = "<|audio|>"
# Int8 (bitsandbytes) keeps the audio path and the output head in the compute dtype.
GRANITE_INT8_SKIP_MODULES = ["encoder", "projector", "language_model.lm_head"]

_state_lock = threading.Lock()
_speed_state: Dict[str, Any] = {
    "failure": None,
    "logged": set(),
    "fast_calls": 0,
    "eager_calls": 0,
    "budget_skips": 0,
}


def _log_once(key: str, message: str) -> None:
    with _state_lock:
        if key in _speed_state["logged"]:
            return
        _speed_state["logged"].add(key)
    print(message, file=sys.stderr)


def prompt_content(language: Optional[str]) -> str:
    """User message of the prompt: with a language hint for the six Granite ASR languages, plain otherwise."""

    name = GRANITE_LANGUAGE_NAMES.get((language or "auto").strip().lower())
    if name is None:
        return f"{AUDIO_TOKEN}{PUNCTUATION_INSTRUCTION}"
    return f"{AUDIO_TOKEN}The speech is in {name}. Transcribe it exactly; do not translate. {PUNCTUATION_INSTRUCTION}"


def build_prompt(processor: Any, language: Optional[str]) -> str:
    content = prompt_content(language)
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


def max_new_tokens_for(seconds: float) -> int:
    return int(min(GRANITE_MAX_NEW_TOKENS_CAP, math.ceil(max(0.0, seconds) * GRANITE_TOKENS_PER_SECOND) + GRANITE_TOKEN_MARGIN))


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


def release_speed_state(model: Any) -> None:
    """Drop the persistent StaticCache and the compiled call of ``model`` (before empty_cache)."""

    for name in ("_cache", "_compiled_call", "_last_compile_config"):
        if model is not None and name in getattr(model, "__dict__", {}):
            try:
                delattr(model, name)
            except Exception:
                pass


def speed_stats() -> Dict[str, Any]:
    with _state_lock:
        return {
            "requested": configured_speed_mode(),
            "failure": _speed_state["failure"],
            "fast_calls": _speed_state["fast_calls"],
            "eager_calls": _speed_state["eager_calls"],
            "budget_skips": _speed_state["budget_skips"],
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
        reason = _inductor_unavailable_reason(device)
        if reason is not None:
            _log_once(f"compile:{reason}", f"[INFO] Granite granite_compile='compile' nicht moeglich: {reason} -> eager.")
            return "off"
    return mode


def _generate(model: Any, inputs: Any, max_new_tokens: int, mode: str) -> torch.Tensor:
    kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1}
    if mode != "off":
        kwargs["cache_implementation"] = "static"
        kwargs["disable_compile"] = mode != "compile"
        if mode == "compile":
            from transformers import CompileConfig

            kwargs["compile_config"] = CompileConfig(fullgraph=False, mode="reduce-overhead")
    with torch.inference_mode():
        return model.generate(**inputs, **kwargs)


def _generate_with_fallback(model: Any, inputs: Any, needed_new_tokens: int, device: torch.device) -> torch.Tensor:
    mode = _effective_speed_mode(model, device)
    if mode == "off":
        # A cache left by a fast mode switched off since would hold its VRAM for nothing.
        release_speed_state(model)
    else:
        input_length = int(inputs["input_ids"].shape[1])
        batch = int(inputs["input_ids"].shape[0])
        total = input_length + needed_new_tokens
        total = int(math.ceil(total / GRANITE_CACHE_LENGTH_BUCKET) * GRANITE_CACHE_LENGTH_BUCKET)
        cache_mb = batch * total * _kv_bytes_per_token(model) / (1024 * 1024)
        if cache_mb > GRANITE_STATIC_CACHE_BUDGET_MB:
            with _state_lock:
                _speed_state["budget_skips"] += 1
            _log_once(
                "budget",
                f"[INFO] Granite StaticCache fuer {batch}x{total} Tokens ({cache_mb:.0f} MB) ueber dem Budget "
                f"{GRANITE_STATIC_CACHE_BUDGET_MB} MB -> dieser und aehnliche Batches eager.",
            )
            release_speed_state(model)
            mode = "off"
        else:
            try:
                output = _generate(model, inputs, total - input_length, mode)
                with _state_lock:
                    _speed_state["fast_calls"] += 1
                return output
            except Exception as exc:
                with _state_lock:
                    _speed_state["failure"] = f"{mode}: {type(exc).__name__}: {exc}"[:500]
                print(
                    f"[WARNUNG] Granite granite_compile='{mode}' fehlgeschlagen ({type(exc).__name__}: {exc}). "
                    "Bis zum Neustart eager.",
                    file=sys.stderr,
                )
                release_speed_state(model)
                if mode == "compile":
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    output = _generate(model, inputs, needed_new_tokens, "off")
    with _state_lock:
        _speed_state["eager_calls"] += 1
    return output


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _transcribe_sub_batch(processor: Any, model: Any, clips: Sequence[np.ndarray], language: Optional[str]) -> List[str]:
    device = _model_device(model)
    prompt = build_prompt(processor, language)
    # A list of 1-D float32 arrays: the feature extractor pads them itself and keeps each clip's own length.
    inputs = processor([prompt] * len(clips), audio=list(clips), device=str(device), return_tensors="pt")
    inputs = inputs.to(device)
    needed = max_new_tokens_for(max(len(clip) for clip in clips) / SAMPLE_RATE)
    output = _generate_with_fallback(model, inputs, needed, device)
    # Left padding: every row's prompt ends at the same column.
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    texts = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [str(text).strip() for text in texts]


def transcribe_granite_batch(
    processor: Any,
    model: Any,
    audio_batch: Sequence[np.ndarray],
    language: Optional[str],
    batch_size: int,
) -> List[str]:
    """Texts in input order. One language per call: the scheduler batches only equal processing keys."""

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
        effective_batch_size = max(1, min(int(batch_size or 1), GRANITE_MAX_INFERENCE_BATCH_SIZE))
        ordered = sorted(parts, key=lambda part: len(part[2]), reverse=True)
        for start in range(0, len(ordered), effective_batch_size):
            chunk = ordered[start : start + effective_batch_size]
            results = _transcribe_sub_batch(processor, model, [part[2] for part in chunk], language)
            for (index, order, _), text in zip(chunk, results):
                texts_by_part[(index, order)] = text

    texts: List[str] = []
    for index in range(len(waveforms)):
        pieces = [texts_by_part[key] for key in sorted(key for key in texts_by_part if key[0] == index)]
        texts.append(combine_transcription_chunks(pieces))
    return texts

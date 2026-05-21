import gc
import importlib.util
import os
import shutil
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor

try:
    import torch._dynamo as dynamo

    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False

from .genesis_whisper_server_globals import (
    current_settings,
    get_effective_transcription_language,
    get_local_model_backend,
    local_model_components,
    model_load_lock,
    resolve_local_model_cache_path,
    settings_lock,
    uses_cohere_backend,
)

COHERE_MAX_INFERENCE_BATCH_SIZE = 16
LAST_LOCAL_ASR_LOAD_ERROR: Optional[str] = None


def _normalize_model_precision(model_precision: str) -> str:
    normalized_precision = (model_precision or "fp16").strip().lower()
    if normalized_precision in {"bf16", "fp16", "int8_bnb", "fp32"}:
        return normalized_precision
    return "fp16"


def _get_model_device_and_dtype(model):
    parameter = next(model.parameters())
    if not parameter.is_floating_point():
        fallback_dtype = torch.float16 if parameter.device.type == "cuda" else torch.float32
        return parameter.device, fallback_dtype
    return parameter.device, parameter.dtype


def _uses_bitsandbytes_quantization(model_precision: str) -> bool:
    return _normalize_model_precision(model_precision) == "int8_bnb"


def _require_bitsandbytes_quantization(use_gpu: bool):
    if not use_gpu:
        raise RuntimeError("INT8 bitsandbytes Quantization benoetigt ein CUDA-GPU-Geraet.")

    missing_packages = [
        package_name
        for package_name in ("bitsandbytes", "accelerate")
        if importlib.util.find_spec(package_name) is None
    ]
    if missing_packages:
        raise RuntimeError(
            "INT8 bitsandbytes Quantization benoetigt zusaetzliche Pakete: "
            f"{', '.join(missing_packages)}. Bitte requirements installieren und erneut versuchen."
        )

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(f"BitsAndBytesConfig konnte nicht aus transformers geladen werden: {exc}") from exc

    return BitsAndBytesConfig


def _resolve_local_pretrained_source(model_id: str, cache_path: str):
    if not cache_path:
        return model_id, {}

    repo_cache_dir = os.path.join(cache_path, f"models--{model_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_cache_dir, "snapshots")
    refs_main_path = os.path.join(repo_cache_dir, "refs", "main")

    snapshot_candidates = []
    if os.path.isfile(refs_main_path):
        try:
            with open(refs_main_path, "r", encoding="utf-8") as refs_file:
                snapshot_name = refs_file.read().strip()
            if snapshot_name:
                snapshot_candidates.append(os.path.join(snapshots_dir, snapshot_name))
        except OSError:
            pass

    if os.path.isdir(snapshots_dir):
        try:
            snapshot_candidates.extend(
                os.path.join(snapshots_dir, entry.name)
                for entry in os.scandir(snapshots_dir)
                if entry.is_dir()
            )
        except OSError:
            pass

    for snapshot_path in snapshot_candidates:
        if os.path.isfile(os.path.join(snapshot_path, "preprocessor_config.json")):
            return snapshot_path, {"local_files_only": True}

    return model_id, {"cache_dir": cache_path}


def _resolve_snapshot_candidates(model_id: str, cache_path: str) -> List[str]:
    if not cache_path:
        return []

    repo_cache_dir = os.path.join(cache_path, f"models--{model_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_cache_dir, "snapshots")
    refs_main_path = os.path.join(repo_cache_dir, "refs", "main")
    snapshot_candidates: List[str] = []

    if os.path.isfile(refs_main_path):
        try:
            with open(refs_main_path, "r", encoding="utf-8") as refs_file:
                snapshot_name = refs_file.read().strip()
            if snapshot_name:
                snapshot_candidates.append(os.path.join(snapshots_dir, snapshot_name))
        except OSError:
            pass

    if os.path.isdir(snapshots_dir):
        try:
            snapshot_candidates.extend(
                os.path.join(snapshots_dir, entry.name)
                for entry in os.scandir(snapshots_dir)
                if entry.is_dir()
            )
        except OSError:
            pass

    return snapshot_candidates


def _snapshot_contains_any(snapshot_path: str, filenames: List[str]) -> bool:
    return any(os.path.isfile(os.path.join(snapshot_path, filename)) for filename in filenames)


def _is_snapshot_ready_for_model(model_id: str, snapshot_path: str) -> bool:
    if not snapshot_path or not os.path.isdir(snapshot_path):
        return False

    required_files = ["config.json", "preprocessor_config.json"]
    if uses_cohere_backend(model_id):
        required_files.extend(
            [
                "configuration_cohere_asr.py",
                "modeling_cohere_asr.py",
                "processing_cohere_asr.py",
                "tokenization_cohere_asr.py",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.model",
            ]
        )
        tokenizer_files = ["tokenizer.model"]
    else:
        required_files.append("tokenizer_config.json")
        tokenizer_files = ["tokenizer.json", "vocab.json"]

    if not all(os.path.isfile(os.path.join(snapshot_path, filename)) for filename in required_files):
        return False

    if not _snapshot_contains_any(snapshot_path, tokenizer_files):
        return False

    return _snapshot_contains_any(
        snapshot_path,
        ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"],
    )


def _resolve_cached_snapshot_path(model_id: str, cache_path: str) -> Optional[str]:
    for snapshot_path in _resolve_snapshot_candidates(model_id, cache_path):
        if _is_snapshot_ready_for_model(model_id, snapshot_path):
            return snapshot_path
    return None


def _resolve_huggingface_token() -> Optional[str]:
    with settings_lock:
        settings_token = str(current_settings.get("huggingface_token", "")).strip()
    if settings_token:
        return settings_token

    env_token = str(os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    return env_token or None


def _with_huggingface_token(pretrained_args: Dict[str, Any], huggingface_token: Optional[str]) -> Dict[str, Any]:
    if not huggingface_token:
        return dict(pretrained_args)

    args_with_token = dict(pretrained_args)
    args_with_token.setdefault("token", huggingface_token)
    return args_with_token


def _format_model_load_error(model_id: str, exc: Exception, token_present: bool) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    normalized_message = message.lower()
    is_auth_error = (
        "cannot access gated repo" in normalized_message
        or "gated repo" in normalized_message
        or "401 client error" in normalized_message
        or "401 unauthorized" in normalized_message
        or "please log in" in normalized_message
    )
    if not is_auth_error:
        return message

    if token_present:
        return (
            f"{message} The configured Hugging Face token may not have access to '{model_id}'. "
            f"Make sure the same Hugging Face account accepted the model license, then retry."
        )

    return (
        f"{message} No Hugging Face token with access to '{model_id}' is currently configured. "
        f"Save one in the admin settings or set HUGGINGFACE_TOKEN/HF_TOKEN, then retry."
    )


def _resolve_cohere_pretrained_source(
    model_id: str,
    cache_path: str,
    huggingface_token: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    cached_snapshot_path = _resolve_cached_snapshot_path(model_id, cache_path)
    if cached_snapshot_path:
        return cached_snapshot_path, {"local_files_only": True}

    if not huggingface_token:
        if cache_path:
            return model_id, {"cache_dir": cache_path}
        return model_id, {}

    print(
        f"[INFO] Cohere-Snapshot fuer '{model_id}' ist lokal unvollstaendig. Starte vollstaendigen Hugging-Face-Download...",
        file=sys.stderr,
    )
    snapshot_path = snapshot_download(
        model_id,
        cache_dir=cache_path or None,
        resume_download=True,
        token=huggingface_token,
    )
    print(f"[INFO] Cohere-Snapshot bereit: '{snapshot_path}'", file=sys.stderr)
    return snapshot_path, {"local_files_only": True}


def _resolve_requested_torch_dtype(model_precision: str, use_gpu: bool) -> torch.dtype:
    normalized_precision = _normalize_model_precision(model_precision)

    if normalized_precision == "fp32":
        print("[INFO] Precision: float32 wird fuer das lokale ASR-Modell verwendet.", file=sys.stderr)
        return torch.float32

    if normalized_precision == "int8_bnb":
        if not use_gpu:
            raise RuntimeError("INT8 bitsandbytes Quantization benoetigt ein CUDA-GPU-Geraet.")
        print("[INFO] Precision: INT8 bitsandbytes Quantization wird fuer GPU verwendet.", file=sys.stderr)
        return torch.float16

    if not use_gpu:
        print(
            f"[INFO] Precision '{normalized_precision}' ist nur fuer GPU-Ladevorgaenge aktiv. CPU verwendet float32.",
            file=sys.stderr,
        )
        return torch.float32

    if normalized_precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 wurde angefordert, wird von der aktuellen CUDA-GPU aber nicht unterstuetzt.")
        print("[INFO] Precision: bfloat16 wird fuer GPU verwendet.", file=sys.stderr)
        return torch.bfloat16

    print("[INFO] Precision: float16 wird fuer GPU verwendet.", file=sys.stderr)
    return torch.float16


def _prepare_model_loading_options(device_selection: str, model_precision: str) -> Tuple[bool, Optional[str], torch.dtype, str]:
    target_device = "cpu"
    attn_implementation = "sdpa"

    use_gpu = False
    if device_selection == "auto" and torch.cuda.is_available():
        use_gpu = True
        target_device = f"cuda:{torch.cuda.current_device()}"
    elif "cuda" in device_selection and torch.cuda.is_available():
        use_gpu = True
        target_device = device_selection

    torch_dtype = _resolve_requested_torch_dtype(model_precision, use_gpu)

    if use_gpu:
        try:
            import flash_attn  # noqa: F401

            attn_implementation = "flash_attention_2"
            print("[INFO] Optimierung: Flash Attention 2 wird verwendet.", file=sys.stderr)
        except ImportError:
            print("[INFO] Optimierung: 'flash-attn' nicht gefunden. Standard 'sdpa' wird verwendet.", file=sys.stderr)

    return use_gpu, target_device, torch_dtype, attn_implementation


def _resolve_backend_attention_implementation(model_id: str, requested_attn_implementation: str) -> str:
    if uses_cohere_backend(model_id):
        if requested_attn_implementation != "eager":
            print(
                "[INFO] Cohere-Backend erzwingt attn_implementation='eager', da sdpa/flash_attention_2 derzeit nicht unterstuetzt werden.",
                file=sys.stderr,
            )
        return "eager"
    return requested_attn_implementation


def _configure_hf_dynamic_module_cache(cache_path: str) -> Optional[str]:
    if not cache_path:
        return None

    modules_cache_path = os.path.join(cache_path, "hf_modules")
    os.makedirs(modules_cache_path, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = modules_cache_path

    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
        from transformers.utils import hub as hub_utils

        hub_utils.HF_MODULES_CACHE = modules_cache_path
        dynamic_module_utils.HF_MODULES_CACHE = modules_cache_path
    except Exception as exc:
        print(f"[WARNUNG] Konnte HF_MODULES_CACHE nicht vollstaendig auf lokalen Projektpfad umbiegen: {exc}", file=sys.stderr)

    return modules_cache_path


def _cleanup_previous_model():
    if local_model_components.get("model") is None:
        return

    print("[INFO] Entlade altes lokales Modell und bereinige Speicher aggressiv...", file=sys.stderr)
    try:
        del local_model_components["model"]
        del local_model_components["processor"]
    except KeyError:
        pass
    local_model_components["model_identifier"] = None
    local_model_components["model_backend"] = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if DYNAMO_AVAILABLE:
        try:
            dynamo.reset()
            print("[INFO] torch.compile Cache erfolgreich zurueckgesetzt.", file=sys.stderr)
        except Exception as exc:
            print(f"[WARNUNG] Konnte torch.compile Cache nicht zuruecksetzen: {exc}", file=sys.stderr)


def _store_local_model_components(model, processor, model_identifier, model_backend):
    local_model_components.clear()
    local_model_components.update(
        {
            "model": model,
            "processor": processor,
            "model_identifier": model_identifier,
            "model_backend": model_backend,
        }
    )


def get_last_local_asr_load_error() -> Optional[str]:
    return LAST_LOCAL_ASR_LOAD_ERROR


def _load_auto_speech_model(pretrained_source: str, pretrained_args: Dict[str, Any], model_kwargs: Dict[str, Any]):
    try:
        return AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_source, **model_kwargs, **pretrained_args)
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        reduced_model_kwargs = dict(model_kwargs)
        reduced_model_kwargs.pop("attn_implementation", None)
        return AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_source, **reduced_model_kwargs, **pretrained_args)


def _build_model_kwargs(
    normalized_precision: str,
    use_gpu: bool,
    target_device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    model_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    if not _uses_bitsandbytes_quantization(normalized_precision):
        return model_kwargs

    BitsAndBytesConfig = _require_bitsandbytes_quantization(use_gpu)
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True,
        # Whisper large-v3-turbo produced early EOS with every encoder layer quantized.
        llm_int8_skip_modules=["model.encoder.layers.0", "proj_out"],
    )
    model_kwargs["device_map"] = {"": target_device}
    print("[INFO] bitsandbytes: load_in_8bit=True mit Transformers device_map aktiviert.", file=sys.stderr)
    return model_kwargs


def _maybe_compile_whisper_model(model, use_gpu: bool):
    if not use_gpu:
        return model

    try:
        if hasattr(torch, "compile"):
            print("[INFO] Optimierung: Versuche, das Modell mit torch.compile() zu kompilieren...", file=sys.stderr)
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            print("[INFO] Optimierung: torch.compile() erfolgreich angewendet.", file=sys.stderr)
        else:
            print("[WARNUNG] torch.compile() ist fuer diese PyTorch-Version nicht verfuegbar (Version 2.0+ erforderlich).", file=sys.stderr)
    except Exception as exc:
        print(f"[WARNUNG] torch.compile() fehlgeschlagen: {exc}. Fahre ohne Kompilierung fort.", file=sys.stderr)
    return model


def load_local_asr_model(model_id: str, device_selection: str, cache_path: str, model_precision: str = "fp16") -> bool:
    global local_model_components
    global LAST_LOCAL_ASR_LOAD_ERROR

    with model_load_lock:
        normalized_precision = _normalize_model_precision(model_precision)
        target_identifier = (model_id, device_selection, cache_path, normalized_precision)
        model_backend = get_local_model_backend(model_id)
        current_identifier = local_model_components.get("model_identifier")

        if current_identifier == target_identifier and local_model_components.get("model") is not None:
            print(
                f"[INFO] Lokales ASR-Modell '{model_id}' auf Geraet '{device_selection}' mit Precision '{normalized_precision}' (Cache: '{cache_path or 'Standard'}') ist bereits geladen.",
                file=sys.stderr,
            )
            LAST_LOCAL_ASR_LOAD_ERROR = None
            return True

        _cleanup_previous_model()

        print(
            f"[INFO] Lade lokales ASR-Modell: '{model_id}' (Backend: {model_backend}) auf Geraet '{device_selection}' mit Precision '{normalized_precision}' und Cache-Pfad: '{cache_path or 'Standard'}'...",
            file=sys.stderr,
        )
        huggingface_token: Optional[str] = None
        try:
            use_gpu, target_device, torch_dtype, attn_implementation = _prepare_model_loading_options(device_selection, normalized_precision)
            attn_implementation = _resolve_backend_attention_implementation(model_id, attn_implementation)
            resolved_cache_path = resolve_local_model_cache_path(cache_path)
            huggingface_token = _resolve_huggingface_token()
            uses_bnb_quantization = _uses_bitsandbytes_quantization(normalized_precision)

            if resolved_cache_path:
                os.makedirs(resolved_cache_path, exist_ok=True)
            if uses_cohere_backend(model_id):
                pretrained_source, pretrained_args = _resolve_cohere_pretrained_source(
                    model_id,
                    resolved_cache_path,
                    huggingface_token,
                )
            else:
                pretrained_source, pretrained_args = _resolve_local_pretrained_source(model_id, resolved_cache_path)
            pretrained_args = _with_huggingface_token(pretrained_args, huggingface_token)

            if uses_cohere_backend(model_id):
                dynamic_modules_cache = _configure_hf_dynamic_module_cache(resolved_cache_path)
                if dynamic_modules_cache:
                    print(f"[INFO] Cohere Dynamic-Module-Cache: '{dynamic_modules_cache}'", file=sys.stderr)
                processor = AutoProcessor.from_pretrained(pretrained_source, trust_remote_code=True, **pretrained_args)
                model_kwargs = _build_model_kwargs(
                    normalized_precision,
                    use_gpu,
                    target_device,
                    torch_dtype,
                    attn_implementation,
                    trust_remote_code=True,
                )
                model = _load_auto_speech_model(pretrained_source, pretrained_args, model_kwargs)
            else:
                processor = WhisperProcessor.from_pretrained(pretrained_source, **pretrained_args)
                model_kwargs = _build_model_kwargs(
                    normalized_precision,
                    use_gpu,
                    target_device,
                    torch_dtype,
                    attn_implementation,
                )
                model = WhisperForConditionalGeneration.from_pretrained(
                    pretrained_source,
                    **model_kwargs,
                    **pretrained_args,
                )
                if target_device != "cpu" and not uses_bnb_quantization:
                    model.to(target_device)
                if not uses_bnb_quantization:
                    model = _maybe_compile_whisper_model(model, use_gpu)

            model.eval()
            if uses_cohere_backend(model_id) and target_device != "cpu" and not uses_bnb_quantization:
                model.to(target_device)
            model_device, _ = _get_model_device_and_dtype(model)
            if not use_gpu and model_device.type != "cpu":
                model.to("cpu")

            _store_local_model_components(model, processor, target_identifier, model_backend)
            LAST_LOCAL_ASR_LOAD_ERROR = None
            model_device, _ = _get_model_device_and_dtype(model)
            print(f"[INFO] Modell '{model_id}' erfolgreich auf '{str(model_device)}' geladen.", file=sys.stderr)
            return True
        except Exception as exc:
            formatted_error = _format_model_load_error(model_id, exc, token_present=bool(huggingface_token))
            LAST_LOCAL_ASR_LOAD_ERROR = f"{type(exc).__name__}: {formatted_error}"
            print(f"[FEHLER] Kritisches Problem beim Laden des Modells '{model_id}': {formatted_error}", file=sys.stderr)
            traceback.print_exc()
            _store_local_model_components(None, None, None, None)
            return False


def _build_whisper_input_features(processor, audio_batch: List[np.ndarray], model_device, model_dtype):
    normalized_audio_batch = [np.asarray(audio_data, dtype=np.float32).flatten() for audio_data in audio_batch]
    input_features = processor(
        normalized_audio_batch,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
    ).input_features
    return input_features.to(model_device, dtype=model_dtype)


def _normalize_audio_batch(audio_batch: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(audio_data, dtype=np.float32).flatten() for audio_data in audio_batch]


def _transcribe_whisper_batch(processor, model, audio_batch: List[np.ndarray], language: str) -> List[str]:
    model_device, model_dtype = _get_model_device_and_dtype(model)
    input_features = _build_whisper_input_features(processor, audio_batch, model_device, model_dtype)

    generate_kwargs = {"task": "transcribe"}
    if language and language != "auto":
        generate_kwargs["language"] = language

    with torch.inference_mode():
        predicted_ids = model.generate(input_features, **generate_kwargs)

    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return [transcription.strip() for transcription in transcriptions]


def _cohere_transcribe_via_generate_api(processor, model, audio_batch: List[np.ndarray], language: str) -> List[str]:
    normalized_audio_batch = _normalize_audio_batch(audio_batch)
    model_device, model_dtype = _get_model_device_and_dtype(model)
    inputs = processor(
        normalized_audio_batch,
        sampling_rate=16000,
        return_tensors="pt",
        language=language,
        punctuation=True,
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs = inputs.to(model_device, dtype=model_dtype)

    with torch.inference_mode():
        predicted_ids = model.generate(**inputs, max_new_tokens=256)

    decode_kwargs = {"skip_special_tokens": True, "language": language}
    if audio_chunk_index is not None:
        decode_kwargs["audio_chunk_index"] = audio_chunk_index
    texts = processor.decode(predicted_ids, **decode_kwargs)
    if isinstance(texts, str):
        texts = [texts]
    return [str(text).strip() for text in texts]


def _cohere_transcribe_via_transcribe_api(processor, model, audio_batch: List[np.ndarray], language: str, batch_size: int) -> List[str]:
    if not hasattr(model, "transcribe"):
        raise AttributeError("Das geladene Cohere-Modell bietet keine transcribe()-Methode an.")

    model_device, _ = _get_model_device_and_dtype(model)
    enable_compile = model_device.type == "cuda"
    if enable_compile and os.name == "nt" and shutil.which("cl") is None:
        enable_compile = False
        print(
            "[INFO] Cohere transcribe()-Compile auf Windows deaktiviert, da 'cl' nicht gefunden wurde.",
            file=sys.stderr,
        )

    transcribe_kwargs = {
        "processor": processor,
        "audio_arrays": _normalize_audio_batch(audio_batch),
        "sample_rates": [16000] * len(audio_batch),
        "language": language,
        "punctuation": True,
        "batch_size": max(1, int(batch_size)),
        "compile": enable_compile,
    }
    if os.name != "nt":
        transcribe_kwargs["pipeline_detokenization"] = True

    texts = model.transcribe(**transcribe_kwargs)
    return [str(text).strip() for text in texts]


def _transcribe_cohere_batch(processor, model, audio_batch: List[np.ndarray], language: str, batch_size: int) -> List[str]:
    model_id = (local_model_components.get("model_identifier") or ("", "", ""))[0]
    effective_language = get_effective_transcription_language(model_id, language)
    effective_batch_size = max(1, min(int(batch_size), COHERE_MAX_INFERENCE_BATCH_SIZE))
    transcriptions: List[str] = []

    for start_index in range(0, len(audio_batch), effective_batch_size):
        sub_batch = audio_batch[start_index:start_index + effective_batch_size]
        try:
            transcriptions.extend(
                _cohere_transcribe_via_transcribe_api(
                    processor,
                    model,
                    sub_batch,
                    effective_language,
                    effective_batch_size,
                )
            )
        except Exception as exc:
            print(
                f"[WARNUNG] Cohere transcribe()-Pfad fehlgeschlagen ({type(exc).__name__}: {exc}). Fallback auf processor/generate/decode.",
                file=sys.stderr,
            )
            transcriptions.extend(_cohere_transcribe_via_generate_api(processor, model, sub_batch, effective_language))

    return transcriptions


def transcribe_local_asr(audio_data_np: np.ndarray, language: str) -> str:
    return transcribe_local_asr_batch([audio_data_np], language=language, batch_size=1)[0]


def transcribe_local_asr_batch(audio_batch: List[np.ndarray], language: str, batch_size: int) -> List[str]:
    components = local_model_components
    if components.get("model") is None or components.get("processor") is None:
        raise RuntimeError("Lokales ASR-Modell wurde vor der Nutzung nicht geladen.")

    if not audio_batch:
        return []

    processor = components["processor"]
    model = components["model"]
    model_backend = components.get("model_backend")

    if model_backend == "cohere_transcribe":
        return _transcribe_cohere_batch(processor, model, audio_batch, language=language, batch_size=batch_size)
    return _transcribe_whisper_batch(processor, model, audio_batch, language=language)

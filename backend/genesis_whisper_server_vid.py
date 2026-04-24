# genesis_whisper_server_vid.py
# V1.0 - Voice Identification Engine
#
# Stellt Funktionen zur Generierung von Stimm-Vektoren (Embeddings)
# mit pyannote.audio zur Verfügung. Lädt das Modell bei der ersten
# Anfrage (Lazy Loading).

import os
import sys
from pathlib import Path
import torch
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any

# Lade globale, Thread-sichere Variablen
from .genesis_whisper_server_chunking import extract_speech_audio
from .genesis_whisper_server_globals import (
    current_settings,
    model_load_lock as vid_model_lock,
    resolve_local_model_cache_path,
    settings_lock,
)

# Globale, Thread-sichere Komponente für das VID-Modell
vid_model_components: Dict[str, Any] = {"model": None, "inference": None}

VID_MODEL_ID = "pyannote/embedding"
HUGGING_FACE_TOKEN = None


def _resolve_huggingface_token() -> str | None:
    with settings_lock:
        settings_token = str(current_settings.get("huggingface_token", "")).strip()
    if settings_token:
        return settings_token

    load_dotenv()
    env_token = str(os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    return env_token or None


def _resolve_vid_pretrained_source() -> tuple[str, str | None]:
    with settings_lock:
        cache_path = resolve_local_model_cache_path(str(current_settings.get("local_model_cache_path", "")).strip())

    if not cache_path:
        return VID_MODEL_ID, None

    repo_cache_dir = Path(cache_path) / f"models--{VID_MODEL_ID.replace('/', '--')}"
    refs_main_path = repo_cache_dir / "refs" / "main"
    snapshots_dir = repo_cache_dir / "snapshots"
    snapshot_candidates: list[Path] = []

    if refs_main_path.is_file():
        try:
            revision = refs_main_path.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        if revision:
            snapshot_candidates.append(snapshots_dir / revision)

    if snapshots_dir.is_dir():
        try:
            snapshot_candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
        except OSError:
            pass

    for snapshot_path in snapshot_candidates:
        if (snapshot_path / "config.yaml").is_file() and (snapshot_path / "pytorch_model.bin").is_file():
            return str(snapshot_path), cache_path

    return VID_MODEL_ID, cache_path

def load_vid_model() -> bool:
    """
    Lädt das pyannote.audio Embedding-Modell, falls es noch nicht geladen ist.
    Liest den Hugging Face Token aus der .env-Datei.
    Ist Thread-sicher.
    """
    global vid_model_components, HUGGING_FACE_TOKEN
    
    with vid_model_lock:
        # Prüfen, ob das Modell bereits geladen ist
        if vid_model_components.get("model") is not None:
            return True

        print("[INFO-VID] Lade Stimmerkennungs-Modell (pyannote/embedding)...", file=sys.stderr)

        # Token fuer jeden Ladeversuch frisch aus Settings/.env lesen,
        # damit Korrekturen im UI ohne Server-Neustart greifen.
        HUGGING_FACE_TOKEN = _resolve_huggingface_token()
        if HUGGING_FACE_TOKEN:
            print("[INFO-VID] Hugging Face Token aus Settings/.env geladen.", file=sys.stderr)
        else:
            print("[WARNUNG-VID] Hugging Face Token weder in Settings noch in .env gefunden. Versuche das Cache-Modell zu laden...", file=sys.stderr)

        try:
            from pyannote.audio import Model, Inference
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            pretrained_source, cache_dir = _resolve_vid_pretrained_source()
            if pretrained_source != VID_MODEL_ID:
                print(f"[INFO-VID] Verwende lokales Cache-Modell fuer Stimmerkennung: {pretrained_source}", file=sys.stderr)
            elif cache_dir:
                print(f"[INFO-VID] Verwende Hugging-Face-Cache fuer Stimmerkennung: {cache_dir}", file=sys.stderr)

            model = Model.from_pretrained(
                pretrained_source,
                token=HUGGING_FACE_TOKEN,
                cache_dir=cache_dir,
            )
            model.to(device)
            inference = Inference(model, window="whole")

            vid_model_components["model"] = model
            vid_model_components["inference"] = inference
            
            print(f"[INFO-VID] Stimmerkennungs-Modell erfolgreich auf '{device}' geladen.", file=sys.stderr)
            return True

        except Exception as e:
            error_message = str(e)
            if "No module named 'omegaconf'" in error_message:
                error_message = (
                    "Fehlende Python-Abhaengigkeit 'omegaconf'. "
                    "Bitte die Server-venv mit requirements.txt aktualisieren."
                )
            print(f"[FEHLER-VID] Kritisches Problem beim Laden des Stimmerkennungs-Modells: {error_message}", file=sys.stderr)
            # Zurücksetzen, um einen erneuten Ladeversuch zu ermöglichen
            vid_model_components = {"model": None, "inference": None}
            return False

def generate_voice_vector(audio_data_np: np.ndarray) -> np.ndarray:
    """
    Generiert einen Stimm-Vektor (Embedding) für den gegebenen Audio-Chunk.
    
    Args:
        audio_data_np: Mono-Audio als np.ndarray in Float32 ([-1, 1]), 16 kHz.

    Returns:
        Einen Numpy-Array der Größe (512,), der den Stimm-Vektor darstellt.
    """
    speech_only_audio = extract_speech_audio(audio_data_np)
    if len(speech_only_audio) > 0:
        audio_data_np = speech_only_audio
        print(
            f"[INFO-VID] Verwende nur erkannte Sprachanteile fuer den Stimm-Vektor ({len(audio_data_np)} Samples).",
            file=sys.stderr,
        )
    else:
        print("[WARNUNG-VID] Keine klaren Sprachanteile erkannt. Verwende Original-Audio fuer den Stimm-Vektor.", file=sys.stderr)

    # Prüfe auf eine minimale Audio-Länge, um den "Kernel size"-Fehler zu verhindern.
    # 0.1 Sekunden = 1600 Samples bei 16kHz. Dies ist ein sicherer Puffer.
    MIN_SAMPLES = 1600
    if len(audio_data_np) < MIN_SAMPLES:
        print(f"[WARNUNG-VID] Audio-Chunk ist zu kurz ({len(audio_data_np)} Samples), um einen Stimm-Vektor zu generieren. Überspringe.", file=sys.stderr)
        raise ValueError(f"Audio-Chunk zu kurz. Benötigt mind. {MIN_SAMPLES} Samples, hat aber nur {len(audio_data_np)}.")

    # Lade das Modell, falls nötig. Bricht bei Fehler ab.
    if not load_vid_model():
        raise RuntimeError("Das Stimmerkennungs-Modell konnte nicht geladen werden. Prüfen Sie die Server-Logs und den Hugging Face Token.")

    inference = vid_model_components["inference"]
    model = vid_model_components["model"] # NEU: Zugriff auf das Modell selbst
    if inference is None or model is None:
         raise RuntimeError("Inference-Objekt oder Modell für Stimmerkennung nicht initialisiert.")
         
    try:
        # --- START DER KORREKTUR ---
        # Hole das Gerät direkt vom geladenen Modell, nicht vom Inference-Objekt.
        target_device = model.device
        
        # Konvertiere Numpy-Array zu einem PyTorch-Tensor und verschiebe ihn auf das richtige Gerät
        waveform = torch.from_numpy(audio_data_np).unsqueeze(0).to(target_device)
        # --- ENDE DER KORREKTUR ---

        audio_dict = {"waveform": waveform, "sample_rate": 16000}
        
        # Führe die Inferenz durch
        vektor = inference(audio_dict)
        
        # Vektor hat die Form (1, 512), wir geben ihn als (512,) zurück
        return vektor.flatten()

    except Exception as e:
        print(f"[FEHLER-VID] Bei der Generierung des Stimm-Vektors ist ein Fehler aufgetreten: {e}", file=sys.stderr)
        raise RuntimeError(f"Fehler bei der Vektor-Generierung: {e}")

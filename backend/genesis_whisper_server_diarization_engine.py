# genesis_whisper_server_diarization_engine.py
# V1.0 - Speaker Diarization Engine
#
# Stellt Funktionen zur Sprecher-Diarisierung mit pyannote.audio zur Verfügung.
# Lädt das Modell bei der ersten Anfrage (Lazy Loading) und unterstützt GPU.

import os
import sys
import torch
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from collections import defaultdict

# Lade globale, Thread-sichere Variablen
from .genesis_whisper_server_globals import model_load_lock as diarization_model_lock
from .genesis_whisper_server_globals import current_task_status, task_status_lock
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Globale, Thread-sichere Komponente für das Diarisierungs-Modell
diarization_pipeline: Optional[Any] = None
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
HUGGING_FACE_TOKEN = None

class LiveStatusProgressHook(ProgressHook):
    """Ein benutzerdefinierter Hook, der den globalen Task-Status für die UI aktualisiert."""
    def on_update(self, description: str, total: int, completed: int):
        """Wird von der pyannote-Pipeline aufgerufen, um den Fortschritt zu melden."""
        super().on_update(description, total, completed)
        with task_status_lock:
            progress_percent = (completed / total * 100) if total > 0 else 0
            current_task_status["task_name"] = "Diarisierung"
            current_task_status["progress"] = round(progress_percent, 2)
            current_task_status["details"] = f"Schritt: {description} ({completed}/{total})"

def load_diarization_model() -> bool:
    """
    Lädt die pyannote.audio Diarisierungs-Pipeline, falls sie noch nicht geladen ist.
    Liest den Hugging Face Token aus der .env-Datei.
    Ist Thread-sicher.
    """
    global diarization_pipeline, HUGGING_FACE_TOKEN
    
    with diarization_model_lock:
        if diarization_pipeline is not None:
            return True

        print(f"[INFO-DIAR] Lade Sprecher-Diarisierungs-Modell ({DIARIZATION_MODEL_ID})...", file=sys.stderr)

        if HUGGING_FACE_TOKEN is None:
            load_dotenv()
            HUGGING_FACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
            if not HUGGING_FACE_TOKEN:
                print("[FEHLER-DIAR] Hugging Face Token nicht in .env gefunden. Diarisierung ist nicht möglich.", file=sys.stderr)
                return False

        try:
            from pyannote.audio import Pipeline
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # KORREKTUR: Verwende die globalen Variablen DIARIZATION_MODEL_ID und HUGGING_FACE_TOKEN
            # und den korrekten Parameter 'token' für die neue pyannote-Version.
            pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL_ID, 
                token=HUGGING_FACE_TOKEN
            )
            pipeline.to(device)

            diarization_pipeline = pipeline
            
            print(f"[INFO-DIAR] Sprecher-Diarisierungs-Modell erfolgreich auf '{device}' geladen.", file=sys.stderr)
            return True

        except Exception as e:
            print(f"[FEHLER-DIAR] Kritisches Problem beim Laden des Diarisierungs-Modells: {e}", file=sys.stderr)
            diarization_pipeline = None
            return False

def diarize_audio(
    audio_data_np: np.ndarray, 
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> Dict[str, list]:
    """
    Führt die Sprecher-Diarisierung für den gegebenen Audio-Chunk durch.

    Args:
        audio_data_np: Mono-Audio als np.ndarray in Float32 ([-1, 1]), 16 kHz.
        num_speakers: Die exakte Anzahl der Sprecher.
        min_speakers: Die minimale Anzahl an Sprechern.
        max_speakers: Die maximale Anzahl an Sprechern.

    Returns:
        Ein Dictionary, das jedem Sprecher eine Liste von Zeitsegmenten zuordnet.
        z.B. {"SPEAKER_00": [{"start": 0.5, "end": 1.2}, ...]}
    """
    if not load_diarization_model():
        raise RuntimeError("Das Diarisierungs-Modell konnte nicht geladen werden. Prüfen Sie die Server-Logs und den Hugging Face Token.")

    if diarization_pipeline is None:
         raise RuntimeError("Diarisierungs-Pipeline nicht initialisiert.")
         
    try:
        # Konvertiere Numpy-Array zu einem PyTorch-Tensor
        waveform = torch.from_numpy(audio_data_np).unsqueeze(0)
        audio_dict = {"waveform": waveform, "sample_rate": 16000}

        # Baue die optionalen Parameter zusammen
        pipeline_kwargs = {}
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            pipeline_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            pipeline_kwargs["max_speakers"] = max_speakers
        
        # Führe die Pipeline mit dem Live-Status-Hook aus
        with LiveStatusProgressHook() as hook:
            diarization_result = diarization_pipeline(audio_dict, hook=hook, **pipeline_kwargs)
        
        # Formatiere das Ergebnis in das gewünschte JSON-Format
        speaker_turns = defaultdict(list)
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_turns[speaker].append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3)
            })
            
        return dict(speaker_turns)

    except Exception as e:
        print(f"[FEHLER-DIAR] Bei der Diarisierung ist ein Fehler aufgetreten: {e}", file=sys.stderr)
        raise RuntimeError(f"Fehler bei der Diarisierung: {e}")

# genesis_whisper_server_vid.py
# V1.0 - Voice Identification Engine
#
# Stellt Funktionen zur Generierung von Stimm-Vektoren (Embeddings)
# mit pyannote.audio zur Verfügung. Lädt das Modell bei der ersten
# Anfrage (Lazy Loading).

import os
import sys
import torch
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any

# Lade globale, Thread-sichere Variablen
from .genesis_whisper_server_chunking import extract_speech_audio
from .genesis_whisper_server_globals import model_load_lock as vid_model_lock

# Globale, Thread-sichere Komponente für das VID-Modell
vid_model_components: Dict[str, Any] = {"model": None, "inference": None}

VID_MODEL_ID = "pyannote/embedding"
HUGGING_FACE_TOKEN = None

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

        # Token aus .env laden
        if HUGGING_FACE_TOKEN is None:
            load_dotenv()
            HUGGING_FACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
            if not HUGGING_FACE_TOKEN:
                print("[FEHLER-VID] Hugging Face Token nicht in .env gefunden. Stimmerkennung ist nicht möglich.", file=sys.stderr)
                return False

        try:
            from pyannote.audio import Model, Inference
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = Model.from_pretrained(VID_MODEL_ID, use_auth_token=HUGGING_FACE_TOKEN)
            model.to(device)
            inference = Inference(model, window="whole")

            vid_model_components["model"] = model
            vid_model_components["inference"] = inference
            
            print(f"[INFO-VID] Stimmerkennungs-Modell erfolgreich auf '{device}' geladen.", file=sys.stderr)
            return True

        except Exception as e:
            print(f"[FEHLER-VID] Kritisches Problem beim Laden des Stimmerkennungs-Modells: {e}", file=sys.stderr)
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

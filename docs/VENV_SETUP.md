# Lokaler venv-Workflow

Dieses Projekt nutzt jetzt einen lokalen Python-Ordner `venv` direkt im Repo und benoetigt keinen Anaconda-Workflow mehr.

## Erstinstallation

1. `start.bat` unter Windows oder `bash ./start.sh` unter Linux/Unix ausfuehren
2. wenn `.\venv` noch fehlt, richtet das Setup die virtuelle Umgebung automatisch ein
3. danach werden Python-Abhaengigkeiten aus `requirements.txt` und Frontend-Abhaengigkeiten installiert bzw. aktualisiert
4. anschliessend startet der Server direkt

## Hinweise

- Die virtuelle Umgebung liegt in `.\venv`
- Falls du lokale PyTorch-Wheels verwenden willst, setze vor dem Setup:
  - `set TORCH_WHEEL_DIR=X:\DEIN\WHEEL\ORDNER`
- Wenn `TORCH_WHEEL_DIR` nicht gesetzt ist, installiert das Setup den PyTorch-Stack fuer CUDA 12.8 per pip
- Das Frontend wird weiterhin separat ueber `frontend\package.json` verwaltet
- Wenn `requirements.txt` geaendert wurde, installiert `start.bat` die Python-Abhaengigkeiten beim naechsten Start erneut
- Voice Vectors verwenden ausschliesslich das fest gepinnte ReDimNet2-B6-LM-Modell (`vb2+vox2+cnc2_v0`, 192-D). Beim ersten Embedding wird Release `v1.0.0` lazy in den konfigurierten Modellcache geladen, per SHA-256 `287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868` verifiziert und einmal aufgewaermt; dafuer ist beim Erstlauf Internetzugriff erforderlich
- `pyannote.audio` und `omegaconf` werden im Whisper-`venv` nicht mehr fuer Voice Embeddings benoetigt. Bestehende 512-D-Profile sind nicht kompatibel und muessen aus Referenzaudio als 192-D-ReDimNet2-Profile neu erzeugt werden
- Fuer gated Hugging Face Modelle wie `CohereLabs/cohere-transcribe-03-2026` muss ein gueltiger Token entweder in den Admin Settings gespeichert oder als `HUGGINGFACE_TOKEN` / `HF_TOKEN` gesetzt sein
- Beim ersten Cohere-Ladevorgang kann der Server fehlende Snapshot-Dateien wie `tokenizer.model` automatisch nachladen; auf Windows kann dabei eine Hugging-Face-Warnung zu fehlender Symlink-Unterstuetzung erscheinen, was nur die Cache-Effizienz betrifft
- Der v2-Modus `diarization` benoetigt einen erreichbaren G3_DIA-Server. URL und write-only API-Key koennen in der Admin-UI oder ueber `DIA_SERVER_BASE_URL` und `DIA_SERVER_API_KEY` gesetzt werden; alle anderen v2-Modi arbeiten ohne DIA
- Falls Whisper und DIA dieselbe GPU verwenden, kann derselbe Pfad in beiden Prozessen als `GENESIS_GPU_LEASE_PATH` gesetzt werden. Beide Prozesse muessen auf dasselbe beschreibbare Lock-Verzeichnis zugreifen koennen

## Wichtige Dateien

- `start.bat`: erstellt bei Bedarf `venv`, installiert alles und startet den Server unter Windows
- `start.sh`: startet den Server unter Linux/Unix und ruft bei Bedarf vorher `install.sh` auf
- `install.sh`: fuehrt Setup und Dependency-Installation unter Linux/Unix aus
- `requirements.txt`: Python-Abhaengigkeiten fuer den Server

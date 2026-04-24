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
- Fuer Voice Vectors via `pyannote/embedding` muss die lokale `venv` vollstaendig sein; insbesondere `omegaconf` muss installiert sein und ist deshalb explizit Teil von `requirements.txt`
- Fuer gated Hugging Face Modelle wie `CohereLabs/cohere-transcribe-03-2026` muss ein gueltiger Token entweder in den Admin Settings gespeichert oder als `HUGGINGFACE_TOKEN` / `HF_TOKEN` gesetzt sein
- Beim ersten Cohere-Ladevorgang kann der Server fehlende Snapshot-Dateien wie `tokenizer.model` automatisch nachladen; auf Windows kann dabei eine Hugging-Face-Warnung zu fehlender Symlink-Unterstuetzung erscheinen, was nur die Cache-Effizienz betrifft

## Wichtige Dateien

- `start.bat`: erstellt bei Bedarf `venv`, installiert alles und startet den Server unter Windows
- `start.sh`: startet den Server unter Linux/Unix und ruft bei Bedarf vorher `install.sh` auf
- `install.sh`: fuehrt Setup und Dependency-Installation unter Linux/Unix aus
- `requirements.txt`: Python-Abhaengigkeiten fuer den Server

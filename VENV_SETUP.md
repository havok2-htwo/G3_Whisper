# Lokaler venv-Workflow

Dieses Projekt nutzt jetzt einen lokalen Python-Ordner `venv` direkt im Repo und benoetigt keinen Anaconda-Workflow mehr.

## Erstinstallation

1. `genesis2_whisper_server.bat` ausfuehren
2. wenn `.\venv` noch fehlt, richtet das Skript automatisch Python, Pakete und Frontend ein
3. danach startet der Server direkt

## Hinweise

- Die virtuelle Umgebung liegt in `.\venv`
- Falls du lokale PyTorch-Wheels verwenden willst, setze vor dem Setup:
  - `set TORCH_WHEEL_DIR=X:\DEIN\WHEEL\ORDNER`
- Wenn `TORCH_WHEEL_DIR` nicht gesetzt ist, installiert das Setup den PyTorch-Stack fuer CUDA 12.8 per pip
- Das Frontend wird weiterhin separat ueber `frontend\package.json` verwaltet

## Wichtige Dateien

- `genesis2_whisper_server.bat`: erstellt bei Bedarf `venv`, installiert alles und startet den Server
- `requirements.txt`: Python-Abhaengigkeiten fuer den Server

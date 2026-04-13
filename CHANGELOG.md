# Changelog

Dieses Dokument wird bewusst knapp gehalten und bei kuenftigen Arbeiten fortgeschrieben.

## Unreleased

- Root-Python-Dateien in das neue Paket `backend/` verschoben und die Startpfade auf `python -m backend.genesis_whisper_server` umgestellt.
- Startskripte auf `start.bat` und `start.sh` als neue Standard-Einstiegspunkte umgestellt.
- Linux-Pfad mit separatem `install.sh` fuer Setup/Build und angepasstem `start.sh` fuer den Serverstart ergänzt.
- Installer-Logik fuer Frontend-Abhaengigkeiten vervollstaendigt: `npm install` laeuft jetzt auch dann neu, wenn `frontend/package.json` oder `frontend/package-lock.json` seit dem letzten Install geaendert wurden; ausserdem gibt es klarere Hinweise fuer fehlendes `npm` und fehlendes `ffmpeg`.
- README und neue `API_DOCUMENTATION.md` auf den aktuellen Stand von Admin-Key-Workflow, OpenAI-kompatiblen `/v1/*`-Routen, Queue-/Benchmark-Daten und energiebasierter Chunk-Erkennung gezogen.
- Admin authentication redesigned to be admin-key-only, including persistent hashed key storage, `/api/admin/keys` rotation, a public landing page on `/`, and a temporary startup admin key shown by the launch script.
- Modell-Loading fuer lokalen ASR-Pfad so umgestellt, dass der Single-GPU-/CPU-Betrieb nicht mehr indirekt `accelerate` ueber `device_map` erzwingt.
- Benchmark und API geben bei Ladefehlern jetzt die konkrete Loader-Ursache weiter.
- Speech-only-Vorfilterung fuer den optionalen Stimmvektor dokumentiert und aktiv im Runtime-Pfad verankert.
- Benchmark-/Batch-Pfad mit `torch.compile(...)` repariert, indem Modell- und Processor-Checks auf explizite `is None`-Pruefungen umgestellt wurden.
- Benchmark- und Metrikwerte im Admin-Frontend auf deutsches Zahlenformat und besseres Umbruchverhalten umgestellt.
- `librosa` als explizite Python-Abhaengigkeit aufgenommen, damit lokale ASR-/Benchmark-Pfade im `venv`-Workflow sauber starten.
- Cohere-ASR-Loader auf `attn_implementation=\"eager\"` umgestellt und Dynamic-Module-Cache fuer `trust_remote_code` in den Projektordner verlegt.
- Cohere-`transcribe()` auf Windows so abgesichert, dass ohne `cl.exe` der optionale Compile-Pfad automatisch deaktiviert wird.

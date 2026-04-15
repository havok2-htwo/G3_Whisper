# Changelog

Dieses Dokument wird bewusst knapp gehalten und bei kuenftigen Arbeiten fortgeschrieben.

## Unreleased

- **Fix:** `pyannote/embedding` wird im Cache Manager nicht mehr faelschlich als unvollstaendig erkannt; als vollstaendiger Snapshot gelten jetzt `config.yaml` plus `pytorch_model.bin`.
- **Fix:** Der Voice-Vector-Loader nutzt fuer `pyannote/embedding` jetzt den korrekten pyannote-Parameter `token=` und bevorzugt einen vollstaendigen lokalen Snapshot aus `local_model_cache_path`.
- **Fix:** `omegaconf` wurde als explizite Python-Abhaengigkeit aufgenommen, weil der Runtime-Load von `pyannote/embedding` sonst trotz installiertem `pyannote.audio` mit `voice_vector=null` scheitern kann.
- **Fix:** Fehlermeldungen fuer fehlgeschlagene Voice-Vector-Generierung unterscheiden jetzt besser zwischen Token-/Cache-Problemen und fehlenden Python-Abhaengigkeiten.
- **Neu:** Hugging Face Token kann nun optional direkt in der Server-UI (Settings) konfiguriert werden, was ein manuelles Pflegen der `.env`-Datei nicht mehr zwingend erfordert.
- **Neu:** Das `pyannote/embedding` Modell fuer die Stimmerkennung ist nun explizit im Cache Manager sichtbar und kann dort ueber die UI heruntergeladen, aktualisiert oder gelöscht werden.
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

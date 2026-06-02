# Changelog

Dieses Dokument wird bewusst knapp gehalten und bei kuenftigen Arbeiten fortgeschrieben.

## 2026-06-02

- **Neu:** Startup-Warmup: Das konfigurierte ASR-Modell wird beim Start eager geladen und mit `testaudio/Testaudio_02.wav` aufgewaermt (statt lazy beim ersten Request); danach wird der CUDA-Cache getrimmt, sodass die erste echte Anfrage warm ist. Best-effort: Ein Warmup-Fehler blockiert den Start nie.
- **Neu:** Idle-VRAM-Trim: Der Batch-Worker gibt den reservierten CUDA-Pool jetzt frei, sobald die Queue nach einem Burst leerlaeuft (nicht pro Batch -> kein Churn unter Last), sodass das Leerlauf-VRAM auf den Modell-Floor faellt und Platz fuer andere GPU-Tenants bleibt.
- **Neu:** `start.bat` setzt vor dem Start `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256` (weniger Fragmentierung des reservierten Pools; `expandable_segments` wird auf Windows ignoriert).
- **Neu:** Experimentelle Precision-Option `fp8` -> `FineGrainedFP8Config` mit bf16-Compute, gated auf das importierbare HF-Paket `kernels` (sonst Fallback auf bf16, geloggt). Das `kernels`-Paket wird bewusst nicht automatisch installiert.
- **Fix:** fp16-`masked_fill`-Guard: Skalar-Maskenwerte, die fp16 uebersteigen (z.B. das hartcodierte `-1e9` im Cohere-ASR-Modell), werden auf die `finfo`-Grenze der Tensor-dtype geklemmt. Damit laeuft `int8_bnb` (fp16-Compute) ohne den `c10::Half`-Overflow-Crash; in-range-Werte sowie fp32/bf16 bleiben unveraendert. `bf16` bleibt am robustesten, `int8_bnb` spart am meisten Gewichts-VRAM.

## Unreleased

- **Fix:** Gated Hugging Face token from Admin Settings or `HUGGINGFACE_TOKEN` / `HF_TOKEN` is now reused during direct local ASR model loading, not only during explicit cache downloads.
- **Fix:** Der Cohere-Ladepfad erkennt unvollstaendige lokale Snapshots jetzt korrekt als `partial`, fuehrt bei Bedarf einen vollstaendigen `snapshot_download(...)` aus und laedt erst danach aus dem fertigen lokalen Snapshot.
- **Fix:** Der Cache Manager markiert `CohereLabs/cohere-transcribe-03-2026` nicht mehr vorschnell als `ready`; fuer `ready` muessen jetzt Remote-Code-Dateien, `tokenizer.model` und Modellgewichte vollstaendig lokal vorliegen.
- **Doku:** `README.md`, `API_DOCUMENTATION.md` und `VENV_SETUP.md` an aktuelle Defaults, Admin-Model-Endpoints und den gated-Cohere-Workflow angepasst.
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

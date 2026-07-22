# Changelog

Dieses Dokument wird bewusst knapp gehalten und bei kuenftigen Arbeiten fortgeschrieben.

## 2026-06-02

- **Neu:** Startup-Warmup: Das konfigurierte ASR-Modell wird beim Start eager geladen und mit `testaudio/Testaudio_02.wav` aufgewaermt (statt lazy beim ersten Request); danach wird der CUDA-Cache getrimmt, sodass die erste echte Anfrage warm ist. Best-effort: Ein Warmup-Fehler blockiert den Start nie.
- **Neu:** Idle-VRAM-Trim: Der Batch-Worker gibt den reservierten CUDA-Pool jetzt frei, sobald die Queue nach einem Burst leerlaeuft (nicht pro Batch -> kein Churn unter Last), sodass das Leerlauf-VRAM auf den Modell-Floor faellt und Platz fuer andere GPU-Tenants bleibt.
- **Neu:** `start.bat` setzt vor dem Start `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256` (weniger Fragmentierung des reservierten Pools; `expandable_segments` wird auf Windows ignoriert).
- **Neu:** Experimentelle Precision-Option `fp8` -> `FineGrainedFP8Config` mit bf16-Compute, gated auf das importierbare HF-Paket `kernels` (sonst Fallback auf bf16, geloggt). Das `kernels`-Paket wird bewusst nicht automatisch installiert.
- **Fix:** fp16-`masked_fill`-Guard: Skalar-Maskenwerte, die fp16 uebersteigen (z.B. das hartcodierte `-1e9` im Cohere-ASR-Modell), werden auf die `finfo`-Grenze der Tensor-dtype geklemmt. Damit laeuft `int8_bnb` (fp16-Compute) ohne den `c10::Half`-Overflow-Crash; in-range-Werte sowie fp32/bf16 bleiben unveraendert. `bf16` bleibt am robustesten, `int8_bnb` spart am meisten Gewichts-VRAM.

## Unreleased

- **Breaking Embedding-Migration bei kompatiblem Legacy-Shape:** Alle oeffentlichen Stimmvektor-Pfade nutzen jetzt ausschliesslich ReDimNet2-B6 LM `vb2+vox2+cnc2_v0` mit 192 Dimensionen und L2-Normalisierung. Release `v1.0.0`, der mit dessen `agg_gnorm`-Checkpoint kompatible Source-Commit `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035` und Checkpoint-SHA256 `287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868` sind fest gepinnt. `/transcribe/?voice_ident=true` behaelt seine Response-Felder, liefert aber 192 statt 512 Werte. Bestehende 512-D-Profile muessen aus Referenzaudio neu erzeugt werden; alte Cache-Dateien werden nicht automatisch geloescht.
- **Entfernt:** Der aktive `pyannote/embedding`-Loader, sein Cache-Manager-Eintrag, das ungenutzte lokale Diarisierungsmodul sowie die Whisper-Abhaengigkeiten `pyannote.audio` und `omegaconf` entfallen. Historische Eintraege weiter unten dokumentieren weiterhin den frueheren Stand.
- **Neu:** Versionierte Multipart-Route `POST /v2/audio/process` mit `schema_version: "2.0"` und den Modi `embedding`, `transcript`, `transcript_embedding` und `diarization`. Einheitliche Responses enthalten Request-ID, Status, Modus, Modellmetadaten, Laufzeiten, Ergebnis und Warnungen. Die v2-Embedding-Version bietet weder Modellwahl noch `embedding_space_id`.
- **Neu:** DIA wird ausschliesslich im Modus `diarization` ueber die additive G3_DIA-v2-API aufgerufen. Whisper streamt den urspruenglichen Upload, nutzt Exclusive-Turns fuer ASR, Standard-/Overlap-Daten fuer sichere ReDimNet-Fenster und gibt Sprecher-Timecodes, Original-DIA-IDs, Overlap-Markierung und bekannte/unbekannte/unresolved Zuordnung zurueck. Das interne 256-D-DIA-Modell wird weder ausgegeben noch fuer Identitaetsmatching verwendet.
- **Neu:** Diarisierungsrequests akzeptieren optional `expected_speakers` (`1..64`, exakt) und eindeutige bekannte Sprecher-IDs mit beliebig vielen gueltigen 192-D-Profilvektoren innerhalb des 16-MiB-JSON-Limits. Ungueltige Dimensionen, nicht-endliche Werte und Nullvektoren liefern HTTP 422. Hungarian-Matching erzwingt globale Eins-zu-eins-Zuordnung; schwache Evidenz bleibt unresolved. Unbekannte/unresolved Sprecher liefern deterministisch maximal 64 bereinigte Vektoren (Prototype plus bis zu 63 zeitcodierte Repraesentanten).
- **Neu:** ReDimNet-Sprecherwolken verwenden ueberlappungsfreie DIA-Bereiche, 200-ms-Wechselrand, mindestens zwei Sekunden saubere Sprache, Drei-Sekunden-Zielfenster, Audioqualitaetspruefungen, Cosinus-Komponenten sowie Median/MAD-Ausreisserbereinigung. Qualitaetszustaende sind `ready`, `low_support`, `mixed_cluster` und `insufficient_clean_speech`.
- **Neu:** Exakter Unicode-Token-Wiederholungsfilter fuer `/transcribe/`, `/v1/audio/transcriptions`, alle textliefernden v2-Modi und den Admin-Benchmark. Einzelwort-Loops werden ab fuenf, 2-32-Token-Muster ab drei direkten Wiederholungen auf das erste Original reduziert; Diarisierung filtert nur innerhalb eines Sprecher-Turns. Opt-out: `X-G3-Repetition-Filter: off`.
- **Neu:** Whisper-Admin-UI und Settings unterstuetzen DIA-Server-URL, write-only API-Key, explizites Loeschen und einen Verbindungstest gegen `GET /v2/capabilities`. Gespeicherte Werte haben Vorrang vor `DIA_SERVER_BASE_URL` / `DIA_SERVER_API_KEY`; Settings-Updates sind Partial-Merges. Der DIA-Key wird nur als `X-API-Key` gesendet und weder zurueckgegeben noch geloggt.
- **Betrieb:** ReDimNet wird lazy einmal geladen/aufgewaermt, auf CUDA in FP16 ohne `torch.compile` betrieben und batched ausgefuehrt. DIA-, ASR- und ReDim-Phasen laufen seriell; `GENESIS_GPU_LEASE_PATH` aktiviert optional eine gemeinsame dateibasierte GPU-Lease fuer Whisper und DIA.
- **Fix:** Der ffmpeg-Fallback dekodiert Uploads jetzt ueber eine seekbare Tempdatei statt `pipe:0`; dadurch funktionieren insbesondere MP4/M4A/MOV-Dateien mit `moov` am Dateiende. Public API und Admin-Benchmark vermeiden dabei eine zusaetzliche Vollkopie des Uploads im RAM und fuehren den Decode ausserhalb des Async-Request-Loops aus.
- **Fix:** Der ffmpeg-Fallback respektiert nun die angeforderte Ziel-Samplerate; Normalisierung und PCM-Konvertierung vermeiden unnoetige Vollgroessen-Kopien bei langen Aufnahmen.
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

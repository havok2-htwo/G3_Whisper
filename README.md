# G3_WHISPER

## Zweck

`G3_WHISPER` ist ein lokaler Transkriptions-Server auf Basis von FastAPI + React/Vite.
Er stellt eine einfache Upload-API fuer Audio- und Video-Dateien bereit, besitzt ein geschuetztes Admin-Dashboard und unterstuetzt aktuell zwei lokale ASR-Pfade:

- Whisper-Modelle ueber Hugging Face `transformers`
- `CohereLabs/cohere-transcribe-03-2026`

Optional kann zusaetzlich ein Stimmvektor auf Basis von `pyannote/embedding` erzeugt werden.

Diese Datei ist die zentrale fachliche und technische Referenz fuer dieses Repository. Wenn Code und README auseinanderlaufen, gilt: Die README muss im selben Arbeitsgang angepasst werden, bis sie den aktuellen Zustand wieder korrekt beschreibt.

## Pflegepflicht

Bei jeder inhaltlichen Arbeit am Projekt gilt:

1. Code, Skripte, Konfiguration und UI aendern.
2. Danach pruefen, ob `README.md` angepasst oder ergaenzt werden muss.
3. Danach `CHANGELOG.md` aktualisieren.

Das gilt auch dann, wenn die Aenderung klein wirkt. Die README ist hier bewusst "source of truth" fuer Menschen und KI-Agenten.

## Aktueller Status

Aktiv ist ein einzelner lokaler Server, der:

- das React-Frontend und die API ueber denselben Port ausliefert
- lokale ASR-Modelle laedt
- Whisper-Requests ueber ein internes Batch-Queue-System verarbeitet
- im Adminbereich Settings, Queue-Zustand, Historie und Benchmark bereitstellt
- optional einen Stimmvektor generiert

Nicht aktiv sind alte Gradio-/OpenAI-/Voxtral-/Nebenserver-Pfade. Solche Altlasten wurden weitgehend nach [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete) verschoben.

## Kernfunktionen

- Transkription ueber `POST /transcribe/`
- Admin-Login ueber signiertes HTTP-Only Session-Cookie
- Modellwechsel und Runtime-Settings im Adminpanel
- Batch-Queue fuer Whisper
- Benchmark im Adminpanel inklusive:
  - Laufzeit
  - Chunks
  - Gesamt-Audiosekunden
  - RTF
  - Peak-VRAM
  - Transkriptanzeige
- Optionale Stimmvektor-Erzeugung
- Pause-/VAD-basierte Segmentierung fuer Whisper
- Speech-only-Vorfilterung fuer den Stimmvektor

## Architektur

### Backend

Das Backend ist in mehrere Python-Dateien aufgeteilt:

- [genesis_whisper_server.py](x:/dev/G3_WHISPER/genesis_whisper_server.py)
  - erstellt die FastAPI-App
  - bindet API und Admin-Routen ein
  - definiert den Lifespan-Startup/Shutdown
  - liefert das gebaute Frontend aus
- [genesis_whisper_server_api.py](x:/dev/G3_WHISPER/genesis_whisper_server_api.py)
  - ungeschuetzte Upload-API
  - Dispatch auf lokalen ASR-Pfad
  - optionale Stimmvektor-Generierung
- [genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/genesis_whisper_server_admin.py)
  - geschuetzte Admin-Endpunkte
  - Settings, Stats, Queue, Benchmark
- [genesis_whisper_server_local_asr_engine.py](x:/dev/G3_WHISPER/genesis_whisper_server_local_asr_engine.py)
  - lokales Laden und Inferenz fuer Whisper und Cohere
- [genesis_whisper_server_batching.py](x:/dev/G3_WHISPER/genesis_whisper_server_batching.py)
  - Queueing und Batch-Verarbeitung fuer Whisper
- [genesis_whisper_server_chunking.py](x:/dev/G3_WHISPER/genesis_whisper_server_chunking.py)
  - Sprachsegmenterkennung
  - Whisper-Chunking
  - Speech-only-Extraktion fuer Voice-ID
- [genesis_whisper_server_audio.py](x:/dev/G3_WHISPER/genesis_whisper_server_audio.py)
  - Einlesen und Normalisieren von Audio/Video-Dateien
- [genesis_whisper_server_vid.py](x:/dev/G3_WHISPER/genesis_whisper_server_vid.py)
  - optionaler Stimmvektor ueber `pyannote/embedding`
- [genesis_whisper_server_auth.py](x:/dev/G3_WHISPER/genesis_whisper_server_auth.py)
  - Admin-Authentifizierung per Cookie + HMAC-Signatur
- [genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/genesis_whisper_server_storage.py)
  - Laden/Speichern der Settings
  - JSONL-Logging
- [genesis_whisper_server_globals.py](x:/dev/G3_WHISPER/genesis_whisper_server_globals.py)
  - zentrale Konstanten, Modelllisten, Runtime-State

### Frontend

Das Frontend liegt unter [`frontend`](x:/dev/G3_WHISPER/frontend) und nutzt:

- React 18
- TypeScript
- Vite

Das Build-Ziel ist [`frontend/dist`](x:/dev/G3_WHISPER/frontend/dist). Dieses Verzeichnis wird von FastAPI direkt ausgeliefert.

## Aktive Laufzeitlogik

### Transkription

Der Standardpfad ist `POST /transcribe/`.

Eingaben:

- `file`: Audio- oder Video-Datei
- `engine`: derzeit nur `local`
- `voice_ident`: `true` oder `false`

Verhalten:

- Wenn `voice_ident=false`, darf der Request in den Whisper-Batch-Worker laufen.
- Wenn `voice_ident=true`, wird der Request bewusst seriell unter GPU-Lock abgearbeitet.
- Wenn das aktive Modell ein Whisper-Modell ist, wird Audio vor der Batch-Inferenz in Sprachsegmente zerlegt.
- Wenn das aktive Modell Cohere ist, verarbeitet der lokale ASR-Pfad das Audio aktuell als ganzes Item.

### Stimmvektor

Der Stimmvektor ist optional.

Wichtige Regeln:

- Er wird nur erzeugt, wenn `voice_ident=true`.
- Vor der Embedding-Bildung werden nach Moeglichkeit nur erkannte Sprachanteile verwendet.
- Wenn keine Sprache sicher erkannt wird, faellt der Code als Fallback einmal auf das Original-Audio zurueck.

Der Stimmvektor ist also inzwischen bewusst "speech-first" und nicht "silence-first".

### Whisper-Chunking

Whisper nutzt keine stumpfe reine 30s-Zerschneidung.

Stattdessen:

- erst Sprachsegmente erkennen
- grosse Stille moeglichst auslassen
- Segmente mit Padding versehen
- nur bei Bedarf in groessere Whisper-kompatible Teilstuecke weiter zerlegen
- zwischen Teilstuecken Overlap setzen

Erkennungspfad:

- bevorzugt `webrtcvad`
- sonst energie-basierter Fallback

### Benchmark

Im Adminpanel gibt es einen Benchmark-Workflow. Dort kann eine Audio- oder Video-Datei hochgeladen werden und die Anzahl der Wiederholungen eingestellt werden.

Der Benchmark zeigt:

- Anzahl der Durchlaeufe
- Anzahl der Chunks pro Run
- Gesamtzahl der Chunks
- Audiosekunden pro Run
- Gesamt-Audiosekunden
- Gesamtzeit
- Durchschnittszeit pro Run
- RTF
- verwendete Batch-Historie
- Peak-VRAM
- erzeugtes Transkript

Zahlenanzeige im Dashboard:

- Benchmark- und Metrikwerte werden im Frontend im deutschen Zahlenformat dargestellt.
- Beispiel: `78,578` bedeutet `achtundsiebzig Komma fuenfhundertachtundsiebzig`, nicht `78.578` als Tausenderwert.

## Aktive Modelle

Die aktuelle Modellliste wird zentral in [genesis_whisper_server_globals.py](x:/dev/G3_WHISPER/genesis_whisper_server_globals.py) gepflegt.

Aktuell vorhanden:

- `CohereLabs/cohere-transcribe-03-2026`
- `openai/whisper-large-v3-turbo`
- `openai/whisper-large-v3`
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base`
- `openai/whisper-tiny`

Hinweis:

- Die `openai/whisper-*` Namen sind Hugging-Face-Modell-IDs.
- Es wird keine OpenAI-Cloud-API mehr genutzt.
- Der `engine`-Wert `openai` wird nicht mehr unterstuetzt.

## Optimierungen

Der lokale ASR-Pfad nutzt bereits mehrere Optimierungen:

- GPU-Dtype-Optimierung, bevorzugt `bfloat16`, sonst `float16`
- `torch.compile(...)`, sofern verfuegbar und sinnvoll
- `sdpa` als Attention-Standard
- `flash_attention_2` nur, wenn `flash_attn` installiert ist
- Lazy Loading fuer Modelle
- Batch-Queue fuer Whisper

Backend-spezifische Ausnahme:

- Das Cohere-ASR-Modell wird derzeit bewusst mit `attn_implementation="eager"` geladen, weil die aktuelle Transformers-Integration fuer dieses Modell `sdpa` bzw. `flash_attention_2` noch nicht sauber unterstuetzt.
- Auf Windows wird der interne Cohere-`transcribe()`-Compile-Pfad deaktiviert, wenn kein `cl.exe` gefunden wird.

Aktuell nicht als fest integrierter Standard enthalten:

- explizites Triton-for-Windows-Setup
- verpflichtendes Flash-Attention-Setup
- expliziter GPU-Warmup direkt nach Modell-Ladung

Wichtige Laufzeitentscheidung:

- Der Modell-Loader ist fuer den Standardbetrieb nicht mehr von `accelerate` als Pflichtabhaengigkeit abhaengig.
- Modelle werden im Single-GPU-/CPU-Pfad direkt auf das Zielgeraet bewegt, statt `device_map` zu erzwingen.

## Authentifizierung

Der Adminbereich nutzt eine leichte eigene Session-Loesung und kein grosses Fremd-Auth-Framework.

Prinzip:

- Username und Passwort-Hash kommen aus der `.env`
- bei erfolgreichem Login wird ein signiertes Token erzeugt
- das Token landet als HTTP-Only Cookie im Browser
- geschuetzte Routen verwenden `require_admin`

Hash-Formate:

- `pbkdf2_sha256$...`
- `sha256$...`
- `plain$...`
- oder roher SHA256-Hash als Rueckfall

## `.env`

Die wichtigsten Umgebungsvariablen:

- `HUGGINGFACE_TOKEN`
  - benoetigt fuer `pyannote/embedding`
  - kann auch fuer gated Hugging-Face-Modelle relevant sein
- `GENESIS_ADMIN_USERNAME`
- `GENESIS_ADMIN_PASSWORD_HASH`
- `GENESIS_SESSION_SECRET`

Wenn die drei `GENESIS_ADMIN_*` bzw. Session-Variablen fehlen, ist der Admin-Login nicht sauber konfiguriert.

## Persistente Daten und Logs

Der Server legt Laufzeitdaten unter [`logs`](x:/dev/G3_WHISPER/logs) ab.

Wichtige Dateien:

- `logs/genesis_whisper_settings.json`
  - persistierte Server-Settings
- `logs/transcription_log.jsonl`
  - JSONL-Log der Transkriptionsanfragen

## Default-Settings

Die aktuellen Standardwerte kommen aus [genesis_whisper_server_storage.py](x:/dev/G3_WHISPER/genesis_whisper_server_storage.py):

- `local_model`: `openai/whisper-base`
- `local_gpu_device`: `auto`
- `local_model_cache_path`: leer
- `transcription_language`: `auto`
- `batch_wait_time_ms`: `250`
- `batch_max_segments`: `8`
- `batch_max_audio_seconds`: `120.0`

## Admin-Endpunkte

Aktive Admin-Routen:

- `POST /api/admin/login`
- `POST /api/admin/logout`
- `GET /api/admin/session`
- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `GET /api/admin/stats`
- `GET /api/admin/queue`
- `POST /api/admin/benchmark`

## API-Endpunkte

Aktive offene Route:

- `POST /transcribe/`

Frontend-Auslieferung:

- `GET /`
- `GET /{full_path:path}`

SPA-Verhalten:

- statische Dateien werden direkt ausgeliefert
- unbekannte Frontend-Pfade fallen auf `index.html` zurueck
- Pfade unter `/api/` werden nicht versehentlich an das Frontend durchgereicht

## Start und Betrieb

### Ein-Klick-Start

Der normale Einstieg ist:

```bat
genesis2_whisper_server.bat
```

Das Startskript erledigt:

- lokales `venv` unter [`venv`](x:/dev/G3_WHISPER/venv) anlegen, falls noch nicht vorhanden
- `pip`, `setuptools`, `wheel` aktualisieren
- PyTorch installieren
- Python-Abhaengigkeiten aus [requirements.txt](x:/dev/G3_WHISPER/requirements.txt) installieren oder bei geaenderter Datei nachziehen
- `frontend/node_modules` bei Bedarf per `npm install` erzeugen
- Frontend bei fehlendem Build mit `npm run build` bauen
- danach den Server starten

Hinweis:

- Das Projekt verwendet bewusst `venv` und nicht `.venv`.
- Der Batch-Start ist der bevorzugte Weg fuer normale Nutzung.

### Manuelle Frontend-Kommandos

```powershell
cd frontend
npm install
npm run build
```

### Manuelle Python-Abhaengigkeiten

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

## Wichtige Repo-Bereiche

Aktiv relevant:

- [`frontend`](x:/dev/G3_WHISPER/frontend)
- [`logs`](x:/dev/G3_WHISPER/logs)
- [`models`](x:/dev/G3_WHISPER/models)
- [`testaudio`](x:/dev/G3_WHISPER/testaudio)
- die aktiven `genesis_whisper_server_*.py` Dateien im Root
- [genesis2_whisper_server.bat](x:/dev/G3_WHISPER/genesis2_whisper_server.bat)
- [requirements.txt](x:/dev/G3_WHISPER/requirements.txt)

Nicht als aktive Laufzeitquelle betrachten:

- [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete)

Dort liegen bewusst Altdateien, Backup-/Nebenpfade und ausgesonderte Hilfsskripte.

## Vorhandene Hilfsskripte

### Passwort-Hash

[genesis_whisper_password_hash.py](x:/dev/G3_WHISPER/genesis_whisper_password_hash.py)

Zweck:

- Erzeugung eines Passwort-Hashes fuer die `.env`

### Lasttest

[genesis_whisper_batch_load_test.py](x:/dev/G3_WHISPER/genesis_whisper_batch_load_test.py)

Zweck:

- schickt viele parallele Requests an den laufenden Server
- hilft beim Messen von Durchsatz, Batch-Verhalten und Stabilitaet

## Dateiformate

Der Server akzeptiert Audio- und, ueber Dekodierung, auch Video-Dateien als Quelle. Die genaue Robustheit haengt von den installierten lokalen Audio-/Codec-Abhaengigkeiten ab.

Der Admin-Benchmark akzeptiert ebenfalls Audio oder Video und nutzt denselben Audio-Ladepfad.

## Bekannte Besonderheiten

- `python-multipart` wird fuer FastAPI-Form-Uploads benoetigt und ist Teil von [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- `librosa` wird von Teilen des lokalen ASR-Stacks benoetigt und ist deshalb explizit Teil von [requirements.txt](x:/dev/G3_WHISPER/requirements.txt).
- Der Server wurde auf FastAPI-Lifespan umgestellt. Alte Event-Handler-Aufrufe sollten nicht wieder eingefuehrt werden.
- Wenn ein Modell nicht geladen werden kann, geben API und Benchmark den konkreteren Loader-Fehler weiter statt nur einer generischen Meldung.
- Bei mit `torch.compile(...)` umhuellten Modellen keine Truthiness-Pruefungen wie `if not model` verwenden. Stattdessen immer explizit `is None` pruefen.
- Das Admin-Frontend nutzt fuer Metriken bewusst deutsches Zahlenformat, damit RTF-, Sekunden- und VRAM-Werte eindeutig lesbar bleiben.
- Das Cohere-Modell nutzt `trust_remote_code`; die dafuer benoetigten Dynamic Modules werden im Projekt unter [`models/hf_modules`](x:/dev/G3_WHISPER/models/hf_modules) gecacht statt im globalen User-Cache.
- Wenn auf Windows kein `cl.exe` verfuegbar ist, laeuft Cohere weiter ohne den optionalen internen Compile-Pfad.
- `__pycache__` kann lokal erneut auftauchen. Das ist normal und kein Quellcode.
- [genesis_whisper_server_diarization_engine.py](x:/dev/G3_WHISPER/genesis_whisper_server_diarization_engine.py) liegt im Repo, ist aber aktuell nicht Teil des aktiven Hauptpfads.

## Arbeitsregeln fuer kuenftige Aenderungen

- Keine alten OpenAI-/Voxtral-/Gradio-Pfade reaktivieren, wenn das nicht ausdruecklich verlangt wird.
- `README.md` nach jeder relevanten Aenderung gegen den realen Code abgleichen.
- `CHANGELOG.md` nach jeder relevanten Aenderung fortschreiben.
- Neue Runtime-Pfade, Dateien oder Endpunkte muessen in dieser README dokumentiert werden.
- Wenn etwas nur probeweise entfernt wird, zuerst nach [`marked_for_delete`](x:/dev/G3_WHISPER/marked_for_delete) verschieben statt sofort hart loeschen.

## Schnelluebersicht fuer neue Sessions

Wenn jemand das Repo schnell verstehen muss, sind die wichtigsten Einstiege:

1. [README.md](x:/dev/G3_WHISPER/README.md)
2. [genesis_whisper_server.py](x:/dev/G3_WHISPER/genesis_whisper_server.py)
3. [genesis_whisper_server_api.py](x:/dev/G3_WHISPER/genesis_whisper_server_api.py)
4. [genesis_whisper_server_admin.py](x:/dev/G3_WHISPER/genesis_whisper_server_admin.py)
5. [genesis_whisper_server_local_asr_engine.py](x:/dev/G3_WHISPER/genesis_whisper_server_local_asr_engine.py)
6. [genesis_whisper_server_batching.py](x:/dev/G3_WHISPER/genesis_whisper_server_batching.py)

# Client-Migration auf API v2 und ReDimNet2 192-D

Stand: 23. Juli 2026

Dieses Dokument ist die Implementierungsaufgabe fuer alle Clients, die auf die aktuelle
G3_WHISPER-Version umgestellt werden. Fuer Felddefinitionen und vollstaendige
Beispielantworten gilt ergaenzend die [API-Dokumentation](API_DOCUMENTATION.md).

## Zielzustand

- Neue Integrationen verwenden `POST /v2/audio/process`.
- Alle oeffentlichen Stimmvektoren stammen ausschliesslich aus ReDimNet2-B6 LM
  `vb2+vox2+cnc2_v0-lm`, sind L2-normalisiert und haben exakt 192 Werte.
- Es gibt clientseitig keine Modellwahl und kein zu sendendes `embedding_space_id`.
- Vorhandene 512-D-Profile werden nicht konvertiert, aufgefuellt, gekuerzt oder mit
  192-D-Vektoren verglichen. Sie muessen aus Referenzaudio neu erzeugt werden.
- DIA wird ausschliesslich bei `mode: "diarization"` verwendet. Seine internen
  256-D-Vektoren sind kein oeffentliches Profilformat.

Der Modellraum von API v2 ist fest gepinnt:

| Eigenschaft | Wert |
| --- | --- |
| Modell | `ReDimNet2-B6` |
| Variante | `vb2+vox2+cnc2_v0-lm` |
| Dimension | `192` |
| Normalisierung | `l2` |
| Sample-Rate | `16000` Hz |
| Release | `v1.0.0` |
| Source-Commit | `2a8d15f65b1dfb5d73fede2f11ee42bcccca3035` |
| Checkpoint-SHA-256 | `287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868` |

Ein spaeterer Modellraumwechsel darf nicht still innerhalb von v2 erfolgen, sondern
benoetigt eine neue API- beziehungsweise Embedding-Version.

## Breaking Change gegenueber alten Clients

| Alt | Aktuell |
| --- | --- |
| 512-D-Stimmvektor | 192-D-ReDimNet2-Vektor |
| Altes Profil weiterverwenden | Profil aus Referenzaudio neu erzeugen |
| Optionales Embedding als zwingender Erfolg | Bei kombiniertem Modus darf es `null` sein |
| Beliebige Zusatzfelder im Request | Strikte v2-Feldvalidierung |
| Client waehlt/benennt Modellraum | Der Server bindet v2 fest an ReDimNet2-B6 |

Die Legacy-Route `/transcribe/` mit dem Multipart-Formfeld `voice_ident=true` behaelt ihre
bisherigen Feldnamen. `voice_vector` enthaelt jedoch jetzt 192 statt 512 Werte und kann
bei ungeeignetem Sprachmaterial `null` sein. `/v1/audio/transcriptions` bleibt eine reine
Transkriptionsschnittstelle ohne Stimmvektor.

## Verbindlicher v2-Request

Der Request ist `multipart/form-data` mit genau einem Audio-/Video-Part `file` und einem
UTF-8-JSON-Part `request`. Maschinenclients senden den Whisper-Key als `X-API-Key`, sobald
auf dem Server Client-Keys konfiguriert sind.

Auf oberster JSON-Ebene sind ausschliesslich diese Felder erlaubt:

- `schema_version`: immer exakt `"2.0"`
- `mode`: `embedding`, `transcript`, `transcript_embedding` oder `diarization`
- `diarization`: nur bei `mode: "diarization"`

Insbesondere gehoert `language` **nicht** in das v2-Request-JSON. Die Sprache kommt aus der
Serverkonfiguration. Ein Client, der beispielsweise
`{"schema_version":"2.0","mode":"transcript_embedding","language":"de"}` sendet,
erhaelt HTTP 422. `language` wird nur von der OpenAI-kompatiblen Legacy-Route
`/v1/audio/transcriptions` als Kompatibilitaetsfeld akzeptiert.

Minimaler kombinierter Request:

```bash
curl -sS http://127.0.0.1:7861/v2/audio/process \
  -H "X-API-Key: $WHISPER_API_KEY" \
  -F 'file=@Sprache.m4a' \
  -F 'request={"schema_version":"2.0","mode":"transcript_embedding"};type=application/json'
```

Die exakte Wiederholungsbereinigung ist standardmaessig aktiv. Nur wenn ein Client
bewusst den ungefilterten Text benoetigt, sendet er zusaetzlich:

```http
X-G3-Repetition-Filter: off
```

## Verhalten je Modus

### `embedding`

Der Server erzeugt genau einen Recording-Level-Vektor und fuehrt weder ASR noch DIA aus.
Bei Mehrsprecher-Audio ist das bewusst ein Mischvektor und keine Sprecheridentifikation.

Erfolg:

```json
{
  "status": "completed",
  "mode": "embedding",
  "result": {
    "embedding": {
      "vector": ["exakt 192 endliche Zahlen"]
    }
  }
}
```

Wenn keine geeignete Sprache fuer ein Embedding vorhanden ist, liefert dieser reine
Embedding-Modus HTTP 422, weil kein unabhaengiges Teilergebnis existiert.

### `transcript`

Der Server liefert `result.transcript.text`. Es wird weder ein Embedding erzeugt noch DIA
aufgerufen.

### `transcript_embedding`

Der Server transkribiert zuerst und versucht danach, genau einen Recording-Level-Vektor
zu erzeugen. Ein fehlendes Embedding darf ein bereits erzeugtes Transkript nicht
verwerfen.

Normaler Erfolg:

```json
{
  "status": "completed",
  "mode": "transcript_embedding",
  "result": {
    "transcript": { "text": "Hallo Welt." },
    "embedding": { "vector": ["exakt 192 endliche Zahlen"] }
  },
  "warnings": []
}
```

Kurze, leise oder qualitativ ungeeignete Sprache:

```json
{
  "status": "partial",
  "mode": "transcript_embedding",
  "result": {
    "transcript": { "text": "Hallo Welt." },
    "embedding": null
  },
  "warnings": [
    {
      "code": "VOICE_EMBEDDING_UNAVAILABLE",
      "message": "Keine qualitativ geeigneten Sprachfenster fuer ein Stimmembedding gefunden."
    }
  ]
}
```

Das ist HTTP 200 und ein verwertbarer Teilerfolg. Der Client muss den Text speichern und
anzeigen, `embedding: null` akzeptieren und darf den Request nicht pauschal als Fehler
behandeln.

### `diarization`

Nur dieser Modus ruft G3_DIA auf. Der optionale Requestteil lautet:

```json
{
  "schema_version": "2.0",
  "mode": "diarization",
  "diarization": {
    "expected_speakers": 5,
    "speaker_refinement": "conservative",
    "unknown_speaker_audio": true,
    "known_speakers": [
      {
        "id": "person-17",
        "embeddings": [
          ["exakt 192 endliche Zahlen"]
        ]
      }
    ]
  }
}
```

Zulaessige DIA-Felder:

- `expected_speakers`: optional, exakte Anzahl `1..64`; darf nicht kleiner als die Zahl
  der bekannten Profile sein. Ohne Wert ermittelt DIA die Anzahl automatisch.
- `known_speakers`: optional, maximal 64 eindeutige IDs; pro ID mindestens ein
  ReDimNet2-192-D-Vektor. Der gesamte JSON-Part ist auf 16 MiB begrenzt.
- `speaker_refinement`: `off` (Standard), `shadow` oder `conservative`. `off` deaktiviert
  nur die zusaetzliche ReDimNet-Korrektur von DIA-Labels, nicht die Diarisierung.
- `unknown_speaker_audio`: echtes JSON-Boolean, Standard `false`. `true` fordert
  Hoerproben fuer geeignete unbekannte beziehungsweise unresolved Stimmen an.

Jeder Profilvektor muss exakt 192 endliche Zahlen und eine von null verschiedene Norm
enthalten. 512-D- und 256-D-Vektoren, `NaN`, Unendlich, Nullvektoren, doppelte IDs oder
leere Embedding-Listen sind HTTP-422-Fehler.

Der Client liest je Transkriptsegment mindestens:

- `start_ms`, `end_ms`, `text`, `overlap`
- `speaker_id`: bekannte Anwendungs-ID oder stabile unbekannte ID
- `diarization_speaker_id`: unveraenderliches urspruengliches DIA-Label
- `speaker_kind`: `known`, `unknown` oder `unresolved`
- optional `refined_diarization_speaker_id`, wenn Refinement angefordert wurde

`unknown_speakers` und `unresolved_speakers` koennen jeweils maximal 64 Vektoreintraege
enthalten: einen Prototype und bis zu 63 diverse, zeitcodierte Repraesentanten. Wenn
`unknown_speaker_audio: true` gesetzt wurde und sichere Quellbereiche existieren, enthaelt
der Sprecher zusaetzlich:

```json
{
  "audio": {
    "mime_type": "audio/mpeg",
    "encoding": "base64",
    "data": "BASE64_OHNE_DATA_URL_PREFIX",
    "duration_ms": 30000,
    "snippets": [
      {
        "start_ms": 12000,
        "end_ms": 18000,
        "duration_ms": 6000,
        "centrality": 0.934121
      }
    ]
  }
}
```

Die MP3-Hoerprobe ist maximal 30 Sekunden lang und besteht nur aus geeigneten
Sprecherabschnitten von jeweils mindestens fuenf Sekunden. Das Feld kann trotz Anfrage
fehlen; dann bleiben die Vektoren verwendbar und `warnings` erklaert den Grund. Base64 wird
mit dem angegebenen MIME-Type dekodiert, nicht als UTF-8-Text behandelt.

## Response- und Fehlermodell

Jede erfolgreiche v2-Antwort besitzt mindestens:

```text
schema_version, request_id, status, mode, audio, models,
timings_ms, result, warnings
```

`status` ist `completed` oder bei nutzbaren Ergebnissen mit Warnungen `partial`. Ein
Client darf `partial` deshalb nicht mit einem fehlgeschlagenen Request gleichsetzen.
`X-Request-ID` und `request_id` dienen der Diagnose.

Fehler besitzen `status: "failed"`, `result: null` und ein strukturiertes `error` mit
`code`, `message` und `retryable`. Relevante HTTP-Statuscodes:

- `400`: Multipart/JSON/Audio nicht lesbar
- `401`: Whisper-API-Key fehlt oder ist ungueltig
- `413`: Request-JSON groesser als 16 MiB
- `422`: ungueltiger Request, Profilvektor oder unzureichende Sprache im reinen
  `embedding`-Modus
- `502`: DIA-Authentifizierung, Upstream oder Response-Vertrag fehlerhaft
- `503`: DIA nicht konfiguriert oder benoetigtes lokales Modell nicht verfuegbar
- `504`: DIA-Timeout
- `500`: unerwarteter lokaler Fehler

Client-Logging soll mindestens `request_id`, HTTP-Status, `error.code`, `mode` und
`timings_ms` enthalten, aber niemals API-Keys oder eingebettete MP3-Base64-Daten.

Die Phasenwerte in `timings_ms` sind Wall-Time-Werte und koennen Queue- sowie gemeinsame
GPU-Lock-Wartezeit enthalten. Sie duerfen nicht als reine CUDA-Inferenzzeit interpretiert
werden. Der Client sollte erfolgreiche Requests nicht wegen einer einzelnen langsamen
Phase erneut senden.

## Datenmigration im Client

1. Die fest codierte beziehungsweise validierte Vektordimension auf `192` setzen.
2. Gespeicherte 512-D-Profile als inkompatibel markieren und nicht mehr an v2 senden.
3. Wenn Referenzaudio vorhanden ist, jedes Profil ueber `mode: "embedding"` neu erzeugen.
4. Wenn kein Referenzaudio vorhanden ist, eine erneute Sprecheraufnahme anfordern.
5. Neue 192-D-Profile erst nach lokaler Pruefung auf 192 endliche Werte und Nicht-Null-Norm
   speichern.
6. Alte Profile erst loeschen, wenn die Neuregistrierung fachlich bestaetigt wurde; bis
   dahin klar versioniert und vom Matching ausgeschlossen aufbewahren.

Der Client darf interne Metadaten wie `api_version: 2`, `dimension: 192` und
`model_id: "ReDimNet2-B6"` in der eigenen Datenbank speichern. Diese Metadaten werden
jedoch nicht als `embedding_space_id` oder Modellwahl an die v2-API gesendet.

## Robustes Client-Verhalten bei gemischtem Rollout

Bei einem vollstaendig aktualisierten Server ist fuer `transcript_embedding` kein Retry
noetig: HTTP 200 plus `embedding: null` erhaelt das Transkript.

Falls ein Client waehrend eines Rolling Updates noch alte Serverversionen erreichen kann,
darf er ausschliesslich als Uebergangsschutz bei einem HTTP 422 nach
`transcript_embedding` genau einmal denselben Audioblob mit `mode: "transcript"` senden.
Der gerettete Text wird als Ergebnis ohne Embedding markiert. Dieser Fallback gilt nicht
fuer Dimensionsfehler, Authentifizierungsfehler, DIA-Fehler oder allgemeine automatische
Retries und kann entfernt werden, sobald alle Server den HTTP-200-Teilerfolg beherrschen.

## Referenztypen fuer TypeScript-Clients

```ts
type V2Mode =
  | "embedding"
  | "transcript"
  | "transcript_embedding"
  | "diarization";

type VoiceEmbedding = number[]; // Laufzeitvalidierung: exakt 192, endlich, Norm > 0

interface KnownSpeaker {
  id: string;
  embeddings: VoiceEmbedding[];
}

interface DiarizationOptions {
  expected_speakers?: number;
  known_speakers?: KnownSpeaker[];
  speaker_refinement?: "off" | "shadow" | "conservative";
  unknown_speaker_audio?: boolean;
}

interface V2Request {
  schema_version: "2.0";
  mode: V2Mode;
  diarization?: DiarizationOptions;
}

type TranscriptEmbeddingResult = {
  transcript: { text: string };
  embedding: { vector: VoiceEmbedding } | null;
};
```

Der Request-Builder muss `diarization` fuer alle Modi ausser `diarization` vollstaendig
weglassen. Er darf keine unbekannten Top-Level-Felder durch Objekt-Spreads einschleusen.

## Abnahmekriterien fuer jeden Client

- Ein `transcript`-Request enthaelt nur `schema_version` und `mode` im JSON-Part.
- `language` wird nicht an `/v2/audio/process` gesendet.
- `embedding` und `transcript_embedding` akzeptieren exakt 192 Werte und keine 512/256.
- HTTP 200 mit `status: "partial"` und `embedding: null` behaelt das Transkript.
- Vorhandene 512-D-Profile werden nie still transformiert oder an v2 gesendet.
- Ein DIA-Test mit fuenf erwarteten und vier bekannten Stimmen kann eine unbekannte Stimme
  samt ihren Vektoren darstellen.
- `unknown_speaker_audio: true` kann vorhandene MP3-Hoerproben abspielen und toleriert ein
  fehlendes `audio`-Feld mit Warnung.
- Der Client zeigt bekannte IDs, unbekannte IDs, Zeitcodes und Overlap korrekt an.
- Fehler verwenden die strukturierte Servermeldung und die `request_id`; Secrets und
  Base64-Audio landen nicht im Log.
- Legacy-Unterstuetzung erwartet bei `voice_ident=true` ebenfalls 192 Werte oder `null`.

## Nicht implementieren

- Keine clientseitige Auswahl zwischen ReDimNet, pyannote oder anderen Embedding-Modellen.
- Kein `embedding_space_id` im v2-Request.
- Keine Dimensionskonvertierung zwischen 512, 256 und 192.
- Kein direkter Client-Aufruf an G3_DIA; der Whisper-Server orchestriert DIA.
- Keine clientseitige Diarisierung aus kurzen Embedding-Fenstern.
- Keine Hotword-, Glossar- oder Prompt-Biasing-Felder fuer Cohere.

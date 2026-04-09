# Architektur-Blueprint: FastAPI Backend mit React/Vite Frontend und Custom Auth

Diese Dokumentation beschreibt, wie das "Drumrum" (Server, Frontend, Auth, Umgebungs-Variablen) im aktuellen `G3_WHISPER` Projekt gelöst wurde. Sie dient als **Anleitung für deinen KI-Agenten**, um eine identische Architektur für einen neuen **vLLM**-Server aufzubauen.

---

## 1. Environment & Secrets (`.env`)

Die Konfiguration der Zugangsdaten und Tokens erfolgt über eine `.env` Datei im Root-Verzeichnis.
Für ein neues vLLM Projekt sollten die Schlüsseln konzeptionell übernommen werden (z.B. mit angepasstem Prefix):

```env
HUGGINGFACE_TOKEN=hf_...
GENESIS_ADMIN_USERNAME=admin
GENESIS_ADMIN_PASSWORD_HASH=pbkdf2_sha256$...
GENESIS_SESSION_SECRET=CustomSecretKeyForHMAC
```

Das Backend lädt diese Werte via `python-dotenv` (falls installiert) oder liest sie direkt über `os.getenv()`.

---

## 2. Authentifizierung (Custom Token & Cookie)

Wir haben eine gewichtsgünstige, **eigene "JWT"-artige Implementierung** gebastelt, anstatt gigantische Auth-Frameworks einzubinden. Diese Architektur ist robust und portabel (findest du im Repository in der Datei `genesis_whisper_server_auth.py`).

### Funktionsweise:
1. **Passwörter:** Hashes werden im Format `pbkdf2_sha256$iterations$salt$hash` hinterlegt und mit `hashlib.pbkdf2_hmac` abgeglichen.
2. **Session-Token-Erstellung:** Wenn der Admin-Login erfolgreich ist (`/api/admin/login`), wird ein Token generiert:
   - Es wird ein Payload geschnürt (z.B. `{"sub": username, "exp": timestamp}`).
   - Das Payload wird base64-encodiert.
   - Es wird eine **HMAC-SHA256 Signatur** generiert (mit dem `GENESIS_SESSION_SECRET` als Schlüssel).
   - Token-Format: `base64_payload.signature`
3. **Cookie-Speicherung:** Das Token wird **als HTTP-Only Cookie** über `response.set_cookie(...)` gesetzt, damit es vor dem Zugriff durch JavaScript (XSS-Schutz) sicher ist.
4. **Schutz der Routen:** Für geschützte API-Routen wird die FastAPI Dependency `Depends(require_admin)` verwendet. Diese prüft bei jedem Request das Vorhandensein des Cookies und validiert die HMAC-Signatur.

---

## 3. Frontend-Architektur (React + Vite)

Das Frontend liegt strikt separiert im Unterordner `./frontend`.
- **Technologie:** React 18, TypeScript, gebaut mit **Vite**.
- **Entwicklung:** Während der Entwicklung läuft das Frontend separat (z.B. über `npm run dev`) und schickt Anfragen an das Python-Backend.
- **Produktion (Deployment):** Mit `npm run build` (bzw. `tsc -b && vite build`) wird das Frontend in den Ordner `./frontend/dist` transpiliert, wo es optimal für die Bereitstellung vorbereitet wird.

---

## 4. Backend (FastAPI) & Static File Serving

Das Herzstück bildet FastAPI, wobei hier API-Endpoints und das fertige React-Frontend ("Single Page Application") geschickt kombiniert – also über denselben Port – ausgeliefert werden:

1. **Struktur:** Zur besseren Lesbarkeit sind die Logik-Dateien aufgeteilt:
   - `..._server.py`: Haupt-Startpunkt (`uvicorn.run(...)`) und das Mounting/Verknüpfen des Frontends.
   - `..._server_api.py`: Die ungeschützten Funktionen (im aktuellen Projekt `/transcribe`, für vLLM z.B. `/v1/completions` oder Streaming Endpoints).
   - `..._server_admin.py`: Router für geschützte Administrations-Routen (Modell wechseln, Logs einsehen) mit Schutz durch `Depends(require_admin)`.

2. **Auslieferung des Frontends (SPA Routing):**
   Damit das React-Frontend vom FastAPI gehostet wird – auch wenn Nutzer bei einem Reload im Frontend direkt auf tieferliegenden Pfaden aufschlagen (React Router) –, ist eine geniale **"Catch-All"** Route im Haupt-Server implementiert:

   ```python
   from fastapi.responses import HTMLResponse, FileResponse
   from pathlib import Path

   FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

   @app.get("/")
   async def serve_frontend_index():
       # Zeigt das root Template (Einstieg in die React App)
       return FileResponse(FRONTEND_DIST_DIR / "index.html")

   @app.get("/{full_path:path}")
   async def serve_frontend_assets(full_path: str):
       # API Routen auslassen, diese wurden in anderen Routern (wie im Auth) bereits registriert!
       # Falls ein /api/ Zugriff hier durchkommt, heißt das, die Route gibt es nicht: 404.
       if full_path.startswith("api/"):
           return HTMLResponse(status_code=404, content="Not Found")

       asset_path = FRONTEND_DIST_DIR / full_path
       # Liefere statische Assets (JS Dateien, CSS, Bilder) aus falls gefunden
       if asset_path.exists() and asset_path.is_file():
           return FileResponse(asset_path)
       
       # WICHTIG: Falls die Datei nicht gefunden wird, handelt es sich um einen Frontend-Pfad
       # in der Single Page App. Wir retournieren daher immer die index.html,
       # damit das UI (z.B. react-router) das Rendering der Seite übernimmt.
       return FileResponse(FRONTEND_DIST_DIR / "index.html")
   ```

---

## 🚀 Arbeits-Anweisung für den zukünftigen vLLM KI-Agenten

Wenn der Agent die bestehende Architektur in einem neuen Projekt (z.B. `G3_VLLM`) aufsetzen soll, schicke ihm diese Checkliste:

1. **Authentifizierung:** Baue die Authentifizierung (als `server_auth.py`) exakt so nach (inkl. PBKDF2 Hashing und sicherer HMAC-Session Signatur). Vergiss nicht, das Token in ein `HTTPOnly` Cookie zu legen! Verzichte auf unnötige externe Auth-Frameworks.
2. **API Backend:** Definiere die ungeschützten FastAPI Endpoints für vLLM (Generierung von Text, Chat-Completion API) in einer dedizierten Datei (z.B. `server_api.py`).
3. **Admin Backend:** Setze alle Settings und Configurations des vLLMs (wie Model-Name, GPU Allocation, Prefix-Caching) in einen geteilten Backend-State (`settings_lock` und `current_settings`) und gestatte Änderungen **nur** über geschützte Endpoints in einer separaten `server_admin.py`.
4. **Environment:** Alle wichtigen Zugangsdaten, Hash-Salts und Konfigurationen müssen aus einer `.env` geladen werden.
5. **Frontend Setup:** Das Frontend MUSS in einem Verzeichnis (`./frontend`) liegen, welches am einfachsten mit `npx create-vite@latest frontend --template react-ts` initiiert wird. Nutze React 18 / TypeScript.
6. **Delivery:** Hänge die statischen Dateien nach dem Build-Prozess (`dist/`) mit dem oben genannten Catch-All Python-Snippet via FastAPI ein. Das Projekt bleibt damit ein Single-Binary / Multi-Purpose Setup.

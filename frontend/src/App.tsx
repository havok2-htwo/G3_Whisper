import { FormEvent, startTransition, useEffect, useState } from "react";

import {
  AdminOption,
  AdminSettings,
  BenchmarkResponse,
  QueueResponse,
  SettingsResponse,
  StatsResponse,
  getQueue,
  getSession,
  getSettings,
  getStats,
  login,
  logout,
  runBenchmark,
  saveSettings,
} from "./api";


type AuthState = "loading" | "logged_out" | "logged_in";

type HistoryEntry = {
  timestamp?: string;
  source_ip?: string;
  engine?: string;
  model_id?: string;
  transcription_language?: string;
  total_duration_ms?: number;
  transcription_duration_ms?: number;
  transcript?: string;
  batched?: boolean;
  segment_count?: number;
};

type BatchEntry = {
  batch_id?: string;
  timestamp?: string;
  batch_size?: number;
  audio_seconds?: number;
  duration_ms?: number;
  status?: string;
};

type BenchmarkWorkflow = "whisper_chunk_queue" | "cohere_audio_batch" | string;

const NUMBER_LOCALE = "de-DE";

const emptySettings: AdminSettings = {
  local_model: "",
  local_gpu_device: "",
  local_model_cache_path: "",
  transcription_language: "auto",
  batch_wait_time_ms: 250,
  batch_max_segments: 8,
  batch_max_audio_seconds: 120,
};


function formatValue(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  const maximumFractionDigits = Number.isInteger(value) ? 0 : 2;
  return `${new Intl.NumberFormat(NUMBER_LOCALE, { maximumFractionDigits }).format(value)}${suffix}`;
}

function formatFixed(value: number | null | undefined, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${new Intl.NumberFormat(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)}${suffix}`;
}

function formatVram(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  if (value >= 1024) {
    return `${new Intl.NumberFormat(NUMBER_LOCALE, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value / 1024)} GB`;
  }
  return `${new Intl.NumberFormat(NUMBER_LOCALE, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value)} MB`;
}

function describeBenchmarkWorkflow(workflow: BenchmarkWorkflow) {
  if (workflow === "cohere_audio_batch") {
    return "Cohere Whole-Audio Batch";
  }
  if (workflow === "whisper_chunk_queue") {
    return "Whisper Chunk-Queue";
  }
  return workflow;
}


export default function App() {
  const [authState, setAuthState] = useState<AuthState>("loading");
  const [username, setUsername] = useState("");
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [globalError, setGlobalError] = useState("");
  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  const [settingsOptions, setSettingsOptions] = useState<{ models: AdminOption[]; devices: AdminOption[]; languages: AdminOption[] }>({
    models: [],
    devices: [],
    languages: [],
  });
  const [loadedModelIdentifier, setLoadedModelIdentifier] = useState<string[] | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [saveBusy, setSaveBusy] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);
  const [benchmarkRepeatCount, setBenchmarkRepeatCount] = useState(1);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkMessage, setBenchmarkMessage] = useState("");
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);

  async function loadDashboard() {
    const [settingsResponse, statsResponse, queueResponse] = await Promise.all([getSettings(), getStats(), getQueue()]);
    startTransition(() => {
      applySettings(settingsResponse);
      setStats(statsResponse);
      setQueue(queueResponse);
      setGlobalError("");
    });
  }

  function applySettings(payload: SettingsResponse) {
    setSettingsForm(payload.settings);
    setSettingsOptions(payload.options);
    setLoadedModelIdentifier(payload.loaded_model_identifier);
  }

  async function refreshOperationalData() {
    const [statsResponse, queueResponse] = await Promise.all([getStats(), getQueue()]);
    startTransition(() => {
      setStats(statsResponse);
      setQueue(queueResponse);
    });
  }

  async function bootstrapSession() {
    try {
      const session = await getSession();
      startTransition(() => {
        setAuthState("logged_in");
        setUsername(session.username);
        setLoginError("");
      });
      await loadDashboard();
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        startTransition(() => {
          setAuthState("logged_out");
          setUsername("");
        });
        return;
      }

      startTransition(() => {
        setAuthState("logged_out");
        setGlobalError(error instanceof Error ? error.message : "Dashboard konnte nicht geladen werden.");
      });
    }
  }

  useEffect(() => {
    void bootstrapSession();
  }, []);

  useEffect(() => {
    if (authState !== "logged_in") {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshOperationalData().catch((error) => {
        if (error instanceof Error && error.message === "unauthorized") {
          startTransition(() => {
            setAuthState("logged_out");
            setUsername("");
          });
          return;
        }
        setGlobalError(error instanceof Error ? error.message : "Polling fehlgeschlagen.");
      });
    }, 5000);

    return () => window.clearInterval(intervalId);
  }, [authState]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginBusy(true);
    setLoginError("");

    try {
      const response = await login(loginUsername, loginPassword);
      startTransition(() => {
        setAuthState("logged_in");
        setUsername(response.username);
        setLoginPassword("");
      });
      await loadDashboard();
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Login fehlgeschlagen.");
    } finally {
      setLoginBusy(false);
    }
  }

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // Logout sollte die lokale Session-Ansicht trotzdem zuruecksetzen.
    } finally {
      startTransition(() => {
        setAuthState("logged_out");
        setUsername("");
      });
    }
  }

  async function handleSaveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaveBusy(true);
    setSaveMessage("");

    try {
      const response = await saveSettings(settingsForm);
      startTransition(() => {
        applySettings(response);
        setSaveMessage(
          response.model_reloaded
            ? `Einstellungen gespeichert. Modell-Reload: ${response.model_loaded ? "erfolgreich" : "fehlgeschlagen"}.`
            : "Einstellungen gespeichert.",
        );
      });
      await refreshOperationalData();
    } catch (error) {
      setSaveMessage(error instanceof Error ? error.message : "Speichern fehlgeschlagen.");
    } finally {
      setSaveBusy(false);
    }
  }

  async function handleRunBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!benchmarkFile) {
      setBenchmarkMessage("Bitte zuerst eine Audio- oder Video-Datei auswaehlen.");
      return;
    }

    setBenchmarkBusy(true);
    setBenchmarkMessage("");
    setBenchmarkResult(null);

    try {
      const result = await runBenchmark(benchmarkFile, benchmarkRepeatCount);
      startTransition(() => {
        setBenchmarkResult(result);
        setBenchmarkMessage(
          `Benchmark abgeschlossen: ${formatValue(result.repeat_count)} Lauf/Laeufe in ${formatValue(result.total_wall_time_ms, " ms")}.`,
        );
      });
      await refreshOperationalData();
    } catch (error) {
      setBenchmarkMessage(error instanceof Error ? error.message : "Benchmark fehlgeschlagen.");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  function updateSetting<K extends keyof AdminSettings>(key: K, value: AdminSettings[K]) {
    setSettingsForm((current) => ({
      ...current,
      [key]: value,
    }));
  }

  if (authState === "loading") {
    return (
      <main className="shell centered">
        <section className="panel hero-panel">
          <span className="eyebrow">GENESIS Transcription Server</span>
          <h1>Admin-Dashboard wird vorbereitet</h1>
          <p>Session wird geprueft und Betriebsdaten werden geladen.</p>
        </section>
      </main>
    );
  }

  if (authState === "logged_out") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Privater Zugriff</span>
            <h1>GENESIS Transcription Admin</h1>
            <p>
              Das Dashboard zeigt Queue-Zustand, Batch-Auslastung und die aktive lokale ASR-Konfiguration.
              Der Zugriff ist auf den Admin-Login begrenzt.
            </p>
          </div>

          <form className="login-form" onSubmit={handleLogin}>
            <label>
              <span>Benutzername</span>
              <input value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)} required />
            </label>

            <label>
              <span>Passwort</span>
              <input
                type="password"
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
                required
              />
            </label>

            <button type="submit" disabled={loginBusy}>
              {loginBusy ? "Login laeuft..." : "Einloggen"}
            </button>

            {(loginError || globalError) && <p className="message error">{loginError || globalError}</p>}
          </form>
        </section>
      </main>
    );
  }

  const history = (stats?.history ?? []) as HistoryEntry[];
  const recentBatches = (queue?.recent_batches ?? []) as BatchEntry[];

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <span className="eyebrow">GESCHUETZTER ADMIN-BEREICH</span>
          <h1>ASR-Batching und Serverstatus</h1>
          <p>
            Eingeloggt als <strong>{username}</strong>. Das Dashboard steuert Modellwahl, Batch-Fenster und die
            Sicht auf die letzten lokalen Transkriptionslaeufe.
          </p>
        </div>
        <div className="hero-actions">
          <button className="secondary-button" onClick={() => void refreshOperationalData()}>
            Status aktualisieren
          </button>
          <button className="secondary-button" onClick={() => void loadDashboard()}>
            Alles neu laden
          </button>
          <button className="secondary-button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </section>

      {globalError && <p className="message error">{globalError}</p>}

      <section className="stats-grid">
        <article className="stat-card">
          <span>Anfragen</span>
          <strong>{stats?.summary.total_requests ?? 0}</strong>
          <small>letzte protokollierte Requests</small>
        </article>
        <article className="stat-card">
          <span>Durchschnitt Gesamt</span>
          <strong>{formatValue(stats?.summary.avg_total_duration_ms, " ms")}</strong>
          <small>inklusive Queue und Nachbearbeitung</small>
        </article>
        <article className="stat-card">
          <span>Queue</span>
          <strong>{queue?.queue_size ?? 0}</strong>
          <small>wartende Queue-Items</small>
        </article>
        <article className="stat-card">
          <span>Batch Worker</span>
          <strong>{queue?.worker_running ? "aktiv" : "gestoppt"}</strong>
          <small>{queue?.last_error ? `Letzter Fehler: ${queue.last_error}` : "kein aktueller Fehler"}</small>
        </article>
      </section>

      <div className="content-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Konfiguration</span>
              <h2>ASR-Modell, Sprache und Batch-Grenzen</h2>
            </div>
            <div className="loaded-model">
              <span>Aktiv geladen:</span>
              <strong>{loadedModelIdentifier ? loadedModelIdentifier.join(" | ") : "noch kein Modell geladen"}</strong>
            </div>
          </div>

          <form className="settings-form" onSubmit={handleSaveSettings}>
            <label>
              <span>ASR-Modell</span>
              <select
                value={settingsForm.local_model}
                onChange={(event) => updateSetting("local_model", event.target.value)}
              >
                {settingsOptions.models.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Geraet</span>
              <select
                value={settingsForm.local_gpu_device}
                onChange={(event) => updateSetting("local_gpu_device", event.target.value)}
              >
                {settingsOptions.devices.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Sprache</span>
              <select
                value={settingsForm.transcription_language}
                onChange={(event) => updateSetting("transcription_language", event.target.value)}
              >
                {settingsOptions.languages.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="full-width">
              <span>Cache-Pfad</span>
              <input
                value={settingsForm.local_model_cache_path}
                onChange={(event) => updateSetting("local_model_cache_path", event.target.value)}
              />
            </label>

            <label>
              <span>Wait Time (ms)</span>
              <input
                type="number"
                min={0}
                value={settingsForm.batch_wait_time_ms}
                onChange={(event) => updateSetting("batch_wait_time_ms", Number(event.target.value))}
              />
            </label>

            <label>
              <span>Max Segmente</span>
              <input
                type="number"
                min={1}
                value={settingsForm.batch_max_segments}
                onChange={(event) => updateSetting("batch_max_segments", Number(event.target.value))}
              />
            </label>

            <label>
              <span>Max Audio (s)</span>
              <input
                type="number"
                min={1}
                step="0.5"
                value={settingsForm.batch_max_audio_seconds}
                onChange={(event) => updateSetting("batch_max_audio_seconds", Number(event.target.value))}
              />
            </label>

            <div className="form-actions full-width">
              <button type="submit" disabled={saveBusy}>
                {saveBusy ? "Speichert..." : "Einstellungen speichern"}
              </button>
              {saveMessage && <p className="message">{saveMessage}</p>}
            </div>
          </form>

          <div className="panel-divider" />

          <div className="section-heading benchmark-heading">
            <div>
              <span className="eyebrow">Benchmark</span>
              <h2>Audio durch die aktuelle Pipeline jagen</h2>
              <p className="section-copy">
                Der Benchmark nutzt die aktuell gespeicherten Server-Settings und feuert die Wiederholungen parallel
                durch die aktive Batch-/Chunk-Logik.
              </p>
            </div>
          </div>

          <form className="benchmark-form" onSubmit={handleRunBenchmark}>
            <label className="full-width">
              <span>Audiodatei</span>
              <input
                type="file"
                accept="audio/*,video/*"
                onChange={(event) => setBenchmarkFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <label>
              <span>Wiederholungen (parallel)</span>
              <input
                type="number"
                min={1}
                max={64}
                value={benchmarkRepeatCount}
                onChange={(event) => setBenchmarkRepeatCount(Math.max(1, Math.min(64, Number(event.target.value) || 1)))}
              />
            </label>

            <div className="form-actions full-width">
              <button type="submit" disabled={benchmarkBusy}>
                {benchmarkBusy ? "Benchmark laeuft..." : "Benchmark starten"}
              </button>
              {benchmarkMessage && (
                <p className={`message ${benchmarkResult ? "" : "error"}`.trim()}>{benchmarkMessage}</p>
              )}
            </div>
          </form>

          {benchmarkResult && (
            <div className="benchmark-results">
              <div className="benchmark-grid">
                <article className="benchmark-card">
                  <span>Workflow</span>
                  <strong>{describeBenchmarkWorkflow(benchmarkResult.workflow)}</strong>
                </article>
                <article className="benchmark-card">
                  <span>RTF</span>
                  <strong>{formatFixed(benchmarkResult.rtf, 3)}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Zeit gesamt</span>
                  <strong>{formatValue(benchmarkResult.total_wall_time_ms, " ms")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Zeit / Run</span>
                  <strong>{formatFixed(benchmarkResult.avg_wall_time_per_run_ms, 2, " ms")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Chunks / Run</span>
                  <strong>{benchmarkResult.chunks_per_run}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Gesamt-Chunks</span>
                  <strong>{benchmarkResult.total_chunks}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Audio / Run</span>
                  <strong>{formatFixed(benchmarkResult.audio_seconds, 3, " s")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Gesamt-Audio</span>
                  <strong>{formatFixed(benchmarkResult.total_audio_seconds, 3, " s")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Batches benutzt</span>
                  <strong>{benchmarkResult.batches_used}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Peak VRAM</span>
                  <strong>{formatVram(benchmarkResult.peak_vram_reserved_mb)}</strong>
                </article>
                <article className="benchmark-card">
                  <span>VRAM alloziert</span>
                  <strong>{formatVram(benchmarkResult.peak_vram_allocated_mb)}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Text konsistent</span>
                  <strong>{benchmarkResult.transcripts_match ? "ja" : "nein"}</strong>
                </article>
              </div>

              <div className="benchmark-meta">
                <div>
                  <span>Datei</span>
                  <strong>{benchmarkResult.file_name}</strong>
                </div>
                <div>
                  <span>Modell</span>
                  <strong>{benchmarkResult.model_id}</strong>
                </div>
                <div>
                  <span>Sprache</span>
                  <strong>{benchmarkResult.transcription_language}</strong>
                </div>
                <div>
                  <span>Wiederholungen</span>
                  <strong>{benchmarkResult.repeat_count}</strong>
                </div>
              </div>

              <div className="benchmark-transcript-wrap">
                <span className="benchmark-transcript-label">Transkript</span>
                <pre className="benchmark-transcript">{benchmarkResult.transcript || "Kein Text vorhanden."}</pre>
              </div>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Batch Queue</span>
              <h2>Laufender Zustand</h2>
            </div>
          </div>

          <div className="queue-metrics">
            <div>
              <span>Aktiver Batch</span>
              <strong>{queue?.active_batch_id ?? "kein aktiver Batch"}</strong>
            </div>
            <div>
              <span>Aktive Batch-Groesse</span>
              <strong>{queue?.active_batch_size ?? 0}</strong>
            </div>
            <div>
              <span>Aktive Audio-Summe</span>
              <strong>{formatValue(queue?.active_batch_audio_seconds, " s")}</strong>
            </div>
            <div>
              <span>Letzter Batch</span>
              <strong>{queue?.last_batch_duration_ms ? `${queue.last_batch_duration_ms} ms` : "n/a"}</strong>
            </div>
            <div>
              <span>Verarbeitete Batches</span>
              <strong>{queue?.total_batches_processed ?? 0}</strong>
            </div>
            <div>
              <span>Verarbeitete Items</span>
              <strong>{queue?.total_segments_processed ?? 0}</strong>
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Zeit</th>
                  <th>Batch</th>
                  <th>Segmente</th>
                  <th>Audio</th>
                  <th>Dauer</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {recentBatches.length === 0 && (
                  <tr>
                    <td colSpan={6}>Noch keine Batch-Historie vorhanden.</td>
                  </tr>
                )}
                {recentBatches.map((entry, index) => (
                  <tr key={`${entry.batch_id ?? "batch"}-${index}`}>
                    <td>{entry.timestamp ?? "n/a"}</td>
                    <td>{entry.batch_id ?? "n/a"}</td>
                    <td>{entry.batch_size ?? 0}</td>
                    <td>{formatValue(entry.audio_seconds, " s")}</td>
                    <td>{formatValue(entry.duration_ms, " ms")}</td>
                    <td>{entry.status ?? "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Historie</span>
            <h2>Letzte Transkriptionen</h2>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Zeit</th>
                <th>IP</th>
                <th>Modell</th>
                <th>Sprache</th>
                <th>Gesamt</th>
                <th>Transkription</th>
                <th>Batch</th>
                <th>Segmente</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 && (
                <tr>
                  <td colSpan={9}>Noch keine Transkriptionshistorie vorhanden.</td>
                </tr>
              )}
              {history.map((entry, index) => (
                <tr key={`${entry.timestamp ?? "row"}-${index}`}>
                  <td>{entry.timestamp ?? "n/a"}</td>
                  <td>{entry.source_ip ?? "n/a"}</td>
                  <td>{entry.model_id ?? entry.engine ?? "n/a"}</td>
                  <td>{entry.transcription_language ?? "n/a"}</td>
                  <td>{formatValue(entry.total_duration_ms, " ms")}</td>
                  <td>{formatValue(entry.transcription_duration_ms, " ms")}</td>
                  <td>{entry.batched ? "ja" : "nein"}</td>
                  <td>{entry.segment_count ?? 1}</td>
                  <td className="transcript-cell">{entry.transcript ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

import { FormEvent, startTransition, useEffect, useState } from "react";

import {
  AdminKeyMetadata,
  AdminOption,
  AdminSettings,
  BenchmarkResponse,
  ManagedModel,
  QueueResponse,
  SettingsResponse,
  StatsResponse,
  deleteModel,
  downloadModel,
  getKeys,
  getModels,
  getQueue,
  getSettings,
  getStats,
  rotateAdminKey,
  runBenchmark,
  saveSettings,
} from "./api";

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
type DashboardHistoryPoint = {
  queueSize: number;
  pendingBufferSize: number;
  activeBatchSize: number;
  meanTotalDurationMs: number;
  meanTranscriptionDurationMs: number;
  batchRealtime: number;
};

const ADMIN_KEY_STORAGE = "genesis_admin_key";
const NUMBER_LOCALE = "de-DE";
const DASHBOARD_POLL_SECONDS = 5;
const DASHBOARD_HISTORY_POINTS = 72;
const DASHBOARD_HISTORY_SECONDS = DASHBOARD_POLL_SECONDS * DASHBOARD_HISTORY_POINTS;
const MODEL_STATUS_POLL_MS = 2000;

const emptySettings: AdminSettings = {
  local_model: "",
  local_gpu_device: "",
  local_model_cache_path: "",
  transcription_language: "auto",
  batch_wait_time_ms: 1000,
  batch_max_segments: 16,
  batch_max_audio_seconds: 60.0,
  huggingface_token: "",
};

function readStoredAdminKey() {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE) || sessionStorage.getItem(ADMIN_KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function writeStoredAdminKey(value: string) {
  try {
    localStorage.setItem(ADMIN_KEY_STORAGE, value);
    sessionStorage.setItem(ADMIN_KEY_STORAGE, value);
  } catch {}
}

function clearStoredAdminKey() {
  try {
    localStorage.removeItem(ADMIN_KEY_STORAGE);
    sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {}
}


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

function formatDateTime(value?: string | null) {
  if (!value) {
    return "n/a";
  }
  return new Intl.DateTimeFormat(NUMBER_LOCALE, {
    year: "2-digit",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatModelSize(model: ManagedModel) {
  if (model.size_on_disk_gb !== null && model.size_on_disk_gb !== undefined) {
    return `${formatFixed(model.size_on_disk_gb, model.size_on_disk_gb >= 10 ? 1 : 2)} GB`;
  }
  if (model.approx_size_gb !== null && model.approx_size_gb !== undefined) {
    return `~${formatFixed(model.approx_size_gb, model.approx_size_gb >= 10 ? 1 : 2)} GB`;
  }
  return "n/a";
}

function formatModelStatus(status: string) {
  if (status === "ready") {
    return "Ready";
  }
  if (status === "downloading") {
    return "Downloading";
  }
  if (status === "partial") {
    return "Partial";
  }
  if (status === "error") {
    return "Error";
  }
  return "Missing";
}

function resolveModelPath(model: ManagedModel) {
  return model.local_path || model.cache_path || model.storage_root;
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

function formatRealtimeFactor(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  if (value === 0) {
    return "0x";
  }
  if (value >= 10) {
    return `${value.toFixed(0)}x`;
  }
  if (value >= 1) {
    return `${value.toFixed(1)}x`;
  }
  return `${value.toFixed(2)}x`;
}

function computeNiceScaleMax(value: number) {
  const safeValue = Math.max(1, Number(value || 0));
  const magnitude = 10 ** Math.floor(Math.log10(safeValue));
  const normalized = safeValue / magnitude;

  if (normalized <= 1) {
    return 1 * magnitude;
  }
  if (normalized <= 2) {
    return 2 * magnitude;
  }
  if (normalized <= 5) {
    return 5 * magnitude;
  }
  return 10 * magnitude;
}

function computeQueueScaleMax(value: number) {
  const safeValue = Math.max(1, Math.ceil(Number(value || 0)));

  if (safeValue <= 4) {
    return 4;
  }
  if (safeValue <= 8) {
    return 8;
  }
  if (safeValue <= 16) {
    return 16;
  }
  if (safeValue <= 32) {
    return 32;
  }
  if (safeValue <= 64) {
    return 64;
  }

  return Math.ceil(safeValue / 32) * 32;
}

function computeBatchRealtime(entry?: BatchEntry | null) {
  const audioSeconds = Number(entry?.audio_seconds ?? 0);
  const durationMs = Number(entry?.duration_ms ?? 0);

  if (!Number.isFinite(audioSeconds) || !Number.isFinite(durationMs) || audioSeconds <= 0 || durationMs <= 0) {
    return 0;
  }

  return audioSeconds / (durationMs / 1000);
}

function resolveOptionLabel(options: AdminOption[], value: string | null | undefined, fallback = "n/a") {
  if (!value) {
    return fallback;
  }

  return options.find((option) => option.value === value)?.label || value;
}

function DashboardSparkline({ history }: { history: DashboardHistoryPoint[] }) {
  const visibleHistory = history.slice(-DASHBOARD_HISTORY_POINTS);
  const realtimePoints = visibleHistory.map((entry) => Number(entry.batchRealtime || 0));
  const queuePoints = visibleHistory.map((entry) => Number(entry.queueSize || 0));
  const currentRealtime = realtimePoints.length > 0 ? realtimePoints[realtimePoints.length - 1] : 0;
  const currentQueue = queuePoints.length > 0 ? queuePoints[queuePoints.length - 1] : 0;
  const scaleMax = computeNiceScaleMax(Math.max(1, ...realtimePoints));
  const queueScaleMax = computeQueueScaleMax(Math.max(1, ...queuePoints));
  const width = 160;
  const height = 228;
  const leftPad = 42;
  const rightPad = 42;
  const topPad = 12;
  const bottomPad = 26;
  const chartWidth = width - leftPad - rightPad;
  const chartHeight = height - topPad - bottomPad;
  const tickValues = [scaleMax, scaleMax * 0.75, scaleMax * 0.5, scaleMax * 0.25, 0];
  const queueTickValues = [queueScaleMax, queueScaleMax * 0.75, queueScaleMax * 0.5, queueScaleMax * 0.25, 0];
  const getY = (value: number) => topPad + chartHeight - (Math.max(0, value) / scaleMax) * chartHeight;
  const getQueueY = (value: number) => topPad + chartHeight - (Math.max(0, value) / queueScaleMax) * chartHeight;
  const realtimePath = realtimePoints
    .map((value, index) => {
      const x = leftPad + (realtimePoints.length <= 1 ? 0 : (index / (realtimePoints.length - 1)) * chartWidth);
      const y = getY(value);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  const queuePath = queuePoints
    .map((value, index) => {
      const x = leftPad + (queuePoints.length <= 1 ? 0 : (index / (queuePoints.length - 1)) * chartWidth);
      const y = getQueueY(value);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="sparkline-shell">
      <svg className="graph" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Whisper queue and batch throughput history">
        {tickValues.map((tickValue) => {
          const y = getY(tickValue);
          return (
            <g key={tickValue}>
              <line x1={leftPad} y1={y} x2={width - rightPad} y2={y} className="graph-grid-line" />
              <text x={leftPad - 6} y={y + 3} textAnchor="end" className="graph-axis-label">
                {formatRealtimeFactor(tickValue)}
              </text>
            </g>
          );
        })}
        {queueTickValues.map((tickValue) => {
          const y = getQueueY(tickValue);
          return (
            <g key={`queue-${tickValue}`}>
              <text x={width - rightPad + 6} y={y + 3} textAnchor="start" className="graph-axis-label graph-axis-label-batch">
                {Math.round(tickValue)}
              </text>
            </g>
          );
        })}
        {scaleMax >= 1 ? (
          <line
            x1={leftPad}
            y1={getY(1)}
            x2={width - rightPad}
            y2={getY(1)}
            className="graph-reference-line"
          />
        ) : null}
        {queuePath ? <path d={queuePath} fill="none" stroke="#ffc16c" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" opacity="0.5" /> : null}
        {realtimePath ? <path d={realtimePath} fill="none" stroke="#95f2c7" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" opacity="0.65" /> : null}
        <text x={leftPad} y={height - 4} textAnchor="start" className="graph-timeline-label">-{DASHBOARD_HISTORY_SECONDS}s</text>
        <text x={width - rightPad} y={height - 4} textAnchor="end" className="graph-timeline-label">now</text>
      </svg>
      <div className="graph-caption">
        <span className="graph-legend">
          <span className="graph-legend-item"><span className="graph-legend-dot" style={{ backgroundColor: "#95f2c7" }} />Batch Realtime</span>
          <span className="graph-legend-item"><span className="graph-legend-dot" style={{ backgroundColor: "#ffc16c" }} />Queue</span>
        </span>
        <strong>{formatRealtimeFactor(currentRealtime)} | {Math.round(currentQueue)}</strong>
      </div>
    </div>
  );
}

async function copyTextToClipboard(value: string) {
  if (!value) {
    return;
  }
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = value;
  textArea.setAttribute("readonly", "readonly");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  textArea.setSelectionRange(0, textArea.value.length);
  document.execCommand("copy");
  document.body.removeChild(textArea);
}


export default function App() {
  const [adminKey, setAdminKey] = useState(() => readStoredAdminKey());
  const [adminKeyInput, setAdminKeyInput] = useState(() => readStoredAdminKey());
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [adminMetadata, setAdminMetadata] = useState<AdminKeyMetadata | null>(null);
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<(AdminKeyMetadata & { token: string }) | null>(null);
  const [actionMessage, setActionMessage] = useState("");
  const [globalError, setGlobalError] = useState("");
  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  const [settingsOptions, setSettingsOptions] = useState<{ models: AdminOption[]; devices: AdminOption[]; languages: AdminOption[] }>({
    models: [],
    devices: [],
    languages: [],
  });
  const [managedModels, setManagedModels] = useState<ManagedModel[]>([]);
  const [loadedModelIdentifier, setLoadedModelIdentifier] = useState<string[] | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [dashboardHistory, setDashboardHistory] = useState<DashboardHistoryPoint[]>([]);
  const [saveBusy, setSaveBusy] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);
  const [benchmarkRepeatCount, setBenchmarkRepeatCount] = useState(1);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkMessage, setBenchmarkMessage] = useState("");
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [modelActionId, setModelActionId] = useState<string | null>(null);
  const [modelActionKind, setModelActionKind] = useState<"refresh" | "download" | "delete" | null>(null);
  const hasDownloadingModels = managedModels.some((model) => model.status === "downloading");

  function persistAdminKey(value: string) {
    writeStoredAdminKey(value);
    setAdminKey(value);
    setAdminKeyInput(value);
  }

  function clearPersistedAdminKey(nextMessage = "") {
    clearStoredAdminKey();
    startTransition(() => {
      setAdminKey("");
      setAdminKeyInput("");
      setAdminMetadata(null);
      setNewlyCreatedKey(null);
      setStats(null);
      setQueue(null);
      setDashboardHistory([]);
      setSettingsForm(emptySettings);
      setManagedModels([]);
      setLoadedModelIdentifier(null);
      setSettingsOptions({ models: [], devices: [], languages: [] });
      setSaveMessage("");
      setBenchmarkMessage("");
      setBenchmarkResult(null);
      setModelActionId(null);
      setModelActionKind(null);
      setActionMessage(nextMessage);
      setGlobalError(nextMessage);
      setAuthError("");
    });
  }

  function handleUnauthorized(message = "The admin key is invalid or expired.") {
    clearPersistedAdminKey(message);
  }

  async function loadDashboard(currentAdminKey = adminKey) {
    if (!currentAdminKey) {
      return;
    }

    try {
      const [settingsResponse, statsResponse, queueResponse, keysResponse] = await Promise.all([
        getSettings(currentAdminKey),
        getStats(currentAdminKey),
        getQueue(currentAdminKey),
        getKeys(currentAdminKey),
      ]);
      startTransition(() => {
        applySettings(settingsResponse);
        setStats(statsResponse);
        setQueue(queueResponse);
        setAdminMetadata(keysResponse.admin_key);
        setGlobalError("");
        setAuthError("");
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      startTransition(() => {
        setGlobalError(error instanceof Error ? error.message : "The dashboard could not be loaded.");
      });
    }
  }

  function applySettings(payload: SettingsResponse) {
    setSettingsForm(payload.settings);
    setSettingsOptions(payload.options);
    setManagedModels(payload.models ?? []);
    setLoadedModelIdentifier(payload.loaded_model_identifier);
  }

  async function refreshOperationalData() {
    if (!adminKey) {
      return;
    }

    try {
      const [statsResponse, queueResponse, keysResponse] = await Promise.all([
        getStats(adminKey),
        getQueue(adminKey),
        getKeys(adminKey),
      ]);
      startTransition(() => {
        setStats(statsResponse);
        setQueue(queueResponse);
        setAdminMetadata(keysResponse.admin_key);
        setGlobalError("");
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      setGlobalError(error instanceof Error ? error.message : "Live polling failed.");
    }
  }

  useEffect(() => {
    if (!adminKey) {
      return;
    }
    void loadDashboard(adminKey);
  }, [adminKey]);

  useEffect(() => {
    if (!adminKey) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshOperationalData();
    }, DASHBOARD_POLL_SECONDS * 1000);

    return () => window.clearInterval(intervalId);
  }, [adminKey]);

  useEffect(() => {
    if (!adminKey || !stats || !queue) {
      return;
    }

    const nextPoint: DashboardHistoryPoint = {
      queueSize: queue.queue_size ?? 0,
      pendingBufferSize: queue.pending_buffer_size ?? 0,
      activeBatchSize: queue.active_batch_size ?? 0,
      meanTotalDurationMs: stats.summary.avg_total_duration_ms ?? 0,
      meanTranscriptionDurationMs: stats.summary.avg_transcription_duration_ms ?? 0,
      batchRealtime: computeBatchRealtime((queue.recent_batches?.[0] ?? null) as BatchEntry | null),
    };

    setDashboardHistory((current) => [...current.slice(-(DASHBOARD_HISTORY_POINTS - 1)), nextPoint]);
  }, [adminKey, queue, stats]);

  async function handleOpenAdmin() {
    const candidate = adminKeyInput.trim();
    if (!candidate) {
      return;
    }

    setAuthBusy(true);
    setAuthError("");
    setActionMessage("");
    setGlobalError("");

    try {
      await Promise.all([getSettings(candidate), getKeys(candidate)]);
      persistAdminKey(candidate);
      setActionMessage("Admin key accepted. Loading dashboard.");
    } catch (error) {
      setAuthError(
        error instanceof Error && error.message === "unauthorized"
          ? "The entered admin key is not valid."
          : error instanceof Error
            ? error.message
            : "The admin key could not be verified.",
      );
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleRotateAdminKey() {
    if (!adminKey) {
      return;
    }

    setActionMessage("");
    setGlobalError("");

    try {
      const response = await rotateAdminKey(adminKey);
      persistAdminKey(response.key.token);
      startTransition(() => {
        setAdminMetadata(response.keys.admin_key);
        setNewlyCreatedKey(response.key);
        setActionMessage("Admin key rotated successfully. The new key is already stored locally.");
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      setGlobalError(error instanceof Error ? error.message : "The admin key could not be rotated.");
    }
  }

  async function handleCopyAdminKey() {
    try {
      await copyTextToClipboard(adminKey);
      setActionMessage("Admin key copied to the clipboard.");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Copying failed.");
    }
  }

  async function handleCopyNewKey() {
    if (!newlyCreatedKey?.token) {
      return;
    }
    try {
      await copyTextToClipboard(newlyCreatedKey.token);
      setActionMessage("New admin key copied to the clipboard.");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Copying failed.");
    }
  }

  async function handleSaveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!adminKey) {
      return;
    }
    setSaveBusy(true);
    setSaveMessage("");

    try {
      const response = await saveSettings(adminKey, settingsForm);
      startTransition(() => {
        applySettings(response);
        setSaveMessage(
          response.model_reloaded
            ? `Settings saved. Model reload ${response.model_loaded ? "succeeded" : "failed"}.`
            : "Settings saved.",
        );
      });
      await refreshOperationalData();
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      setSaveMessage(error instanceof Error ? error.message : "Saving settings failed.");
    } finally {
      setSaveBusy(false);
    }
  }

  async function refreshManagedModels(options?: { silent?: boolean; storagePath?: string }) {
    if (!adminKey) {
      return;
    }

    const silent = options?.silent ?? false;
    const storagePath = options?.storagePath ?? managedModels[0]?.storage_root ?? settingsForm.local_model_cache_path;

    try {
      const response = await getModels(adminKey, storagePath);
      startTransition(() => {
        setManagedModels(response.models ?? []);
        if (!silent) {
          setActionMessage("Model cache refreshed for the selected path.");
        }
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      if (!silent) {
        setGlobalError(error instanceof Error ? error.message : "Model cache refresh failed.");
      }
    }
  }

  async function handleRefreshModels() {
    setModelActionId("__refresh__");
    setModelActionKind("refresh");
    setActionMessage("");
    setGlobalError("");

    try {
      await refreshManagedModels({ storagePath: settingsForm.local_model_cache_path });
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  async function handleDownloadModel(model: ManagedModel) {
    if (!adminKey) {
      return;
    }

    setModelActionId(model.id);
    setModelActionKind("download");
    setActionMessage("");
    setGlobalError("");

    try {
      const response = await downloadModel(
        adminKey,
        model.id,
        settingsForm.local_model_cache_path,
        settingsForm.huggingface_token,
      );
      startTransition(() => {
        setManagedModels(response.models ?? []);
        setActionMessage(`Download queued for ${model.label}.`);
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      setGlobalError(error instanceof Error ? error.message : "Model download failed.");
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  async function handleDeleteModel(model: ManagedModel) {
    if (!adminKey) {
      return;
    }

    setModelActionId(model.id);
    setModelActionKind("delete");
    setActionMessage("");
    setGlobalError("");

    try {
      const response = await deleteModel(adminKey, model.id, settingsForm.local_model_cache_path);
      startTransition(() => {
        setManagedModels(response.models ?? []);
        setActionMessage(
          response.removed
            ? `Cached files removed for ${model.label}.`
            : `No cached files found for ${model.label} in the selected path.`,
        );
      });
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      setGlobalError(error instanceof Error ? error.message : "Model deletion failed.");
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  useEffect(() => {
    if (!adminKey || !hasDownloadingModels) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshManagedModels({ silent: true });
    }, MODEL_STATUS_POLL_MS);

    return () => window.clearInterval(intervalId);
  }, [adminKey, hasDownloadingModels, managedModels, settingsForm.local_model_cache_path]);

  async function handleRunBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!adminKey) {
      return;
    }
    if (!benchmarkFile) {
      setBenchmarkMessage("Please select an audio or video file first.");
      return;
    }

    setBenchmarkBusy(true);
    setBenchmarkMessage("");
    setBenchmarkResult(null);

    try {
      const result = await runBenchmark(adminKey, benchmarkFile, benchmarkRepeatCount);
      startTransition(() => {
        setBenchmarkResult(result);
        setBenchmarkMessage(
          `Benchmark finished: ${formatValue(result.repeat_count)} run(s) in ${formatValue(result.total_wall_time_ms, " ms")}.`,
        );
      });
      await refreshOperationalData();
    } catch (error) {
      if (error instanceof Error && error.message === "unauthorized") {
        handleUnauthorized();
        return;
      }
      setBenchmarkMessage(error instanceof Error ? error.message : "Benchmark failed.");
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

  if (!adminKey) {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Private Access</span>
            <h1>GENESIS Whisper Admin</h1>
            <p>
              The public transcription API stays open, while the private dashboard for queue control, benchmarks,
              and history is protected by the admin key.
            </p>
          </div>

          <div className="login-form">
            <label>
              <span>Admin Key</span>
              <input
                value={adminKeyInput}
                onChange={(event) => setAdminKeyInput(event.target.value)}
                placeholder="genesis_admin_..."
              />
            </label>

            <button type="button" onClick={() => void handleOpenAdmin()} disabled={!adminKeyInput.trim() || authBusy}>
              {authBusy ? "Checking..." : "Open Dashboard"}
            </button>

            <p className="message">
              Use the persistent admin key or the temporary startup key that is briefly shown during server launch.
            </p>

            {(authError || globalError) && <p className="message error">{authError || globalError}</p>}
          </div>
        </section>
      </main>
    );
  }

  const history = (stats?.history ?? []) as HistoryEntry[];
  const recentBatches = (queue?.recent_batches ?? []) as BatchEntry[];
  const latestBatchRealtime = computeBatchRealtime(recentBatches[0]);
  const configuredModelLabel = resolveOptionLabel(settingsOptions.models, settingsForm.local_model, "No model configured");
  const configuredDeviceLabel = resolveOptionLabel(settingsOptions.devices, settingsForm.local_gpu_device, "Auto");
  const configuredLanguageLabel = resolveOptionLabel(settingsOptions.languages, settingsForm.transcription_language, "Auto");
  const loadedModelLabel = loadedModelIdentifier ? loadedModelIdentifier.join(" | ") : "No model loaded";

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy-block">
          <span className="eyebrow">Powered by SONS</span>
          <h1>G3 Whisper Admin</h1>
          <p className="hero-copy">
            Manage model routing, batch-window tuning, queue visibility, and protected operator access from one
            private Whisper control surface.
          </p>
          <div className="hero-actions">
            <button type="button" className="secondary-button" onClick={() => void loadDashboard()}>
              Refresh
            </button>
            <button type="button" className="secondary-button" onClick={() => void handleRotateAdminKey()}>
              Rotate Admin Key
            </button>
            <button type="button" className="ghost-button" onClick={() => clearPersistedAdminKey()}>
              Logout
            </button>
          </div>
        </div>

        <div className="status-grid">
          <div className="status-pill">
            <span>Configured Model</span>
            <strong>{configuredModelLabel}</strong>
          </div>
          <div className="status-pill">
            <span>Loaded Model</span>
            <strong>{loadedModelLabel}</strong>
          </div>
          <div className="status-pill">
            <span>GPU</span>
            <strong>{configuredDeviceLabel}</strong>
          </div>
          <div className="status-pill">
            <span>Language</span>
            <strong>{configuredLanguageLabel}</strong>
          </div>
        </div>
      </section>

      {actionMessage && <p className="message">{actionMessage}</p>}
      {globalError && <p className="message error">{globalError}</p>}

      <section className="panel-grid admin-overview-grid">
        <section className="panel stack">
          <p className="eyebrow">Dashboard</p>
          <h2>Live Queue</h2>
          <div className="metric-grid whisper-dashboard-metrics">
            <div className="metric-card">
              <span>Queue</span>
              <strong>{queue?.queue_size ?? 0}</strong>
            </div>
            <div className="metric-card">
              <span>Queued Segments</span>
              <strong>{queue?.pending_buffer_size ?? 0}</strong>
            </div>
            <div className="metric-card">
              <span>Active Batch</span>
              <strong>{queue?.active_batch_size ?? 0} / {settingsForm.batch_max_segments || "-"}</strong>
            </div>
            <div className="metric-card">
              <span>Mean Total</span>
              <strong>{formatValue(stats?.summary.avg_total_duration_ms, " ms")}</strong>
            </div>
            <div className="metric-card">
              <span>Mean ASR</span>
              <strong>{formatValue(stats?.summary.avg_transcription_duration_ms, " ms")}</strong>
            </div>
            <div className="metric-card">
              <span>Batch Realtime</span>
              <strong>{latestBatchRealtime > 0 ? formatRealtimeFactor(latestBatchRealtime) : "n/a"}</strong>
            </div>
          </div>

          <DashboardSparkline history={dashboardHistory} />

          <div className="queue-insight-grid">
            <div className="insight-card">
              <span>Worker</span>
              <strong>{queue?.worker_running ? "Running" : "Stopped"}</strong>
            </div>
            <div className="insight-card">
              <span>Requests</span>
              <strong>{stats?.summary.total_requests ?? 0}</strong>
            </div>
            <div className="insight-card">
              <span>Processed Segments</span>
              <strong>{queue?.total_segments_processed ?? 0}</strong>
            </div>
          </div>

          <div className="queue-summary-card">
            <div className="queue-summary-head">
              <div>
                <h3>Current Batch</h3>
                <p className="muted">Live worker snapshot for the active batch and the most recent runtime state.</p>
              </div>
            </div>
            {queue?.active_batch_id ? (
              <div className="queue-detail-grid">
                <div>
                  <span>Batch ID</span>
                  <strong>{queue.active_batch_id}</strong>
                </div>
                <div>
                  <span>Batch Size</span>
                  <strong>{queue.active_batch_size ?? 0}</strong>
                </div>
                <div>
                  <span>Audio</span>
                  <strong>{formatValue(queue.active_batch_audio_seconds, " s")}</strong>
                </div>
                <div>
                  <span>Started</span>
                  <strong>{formatDateTime(queue.active_batch_started_at)}</strong>
                </div>
              </div>
            ) : (
              <p className="muted">No batch is currently running.</p>
            )}
            <p className="dashboard-note">
              {queue?.last_error ? `Last error: ${queue.last_error}` : "No current worker error."}
            </p>
          </div>
        </section>

        <div className="stack side-widget-stack">
        <section className="panel stack">
          <p className="eyebrow">Admin Key</p>
          <h2>Dashboard Access</h2>

          {newlyCreatedKey && (
            <div className="key-token-card">
              <div className="key-card-head">
                <div>
                  <strong>{newlyCreatedKey.label}</strong>
                  <p>The server returns the rotated key only once in plain text.</p>
                </div>
                <button type="button" className="secondary-button" onClick={() => void handleCopyNewKey()}>
                  Copy
                </button>
              </div>
              <div className="key-token-value mono">{newlyCreatedKey.token}</div>
            </div>
          )}

          <div className="metric-grid compact-metrics">
            <div className="metric-card">
              <span>Name</span>
              <strong>{adminMetadata?.label || "Master Admin Key"}</strong>
            </div>
            <div className="metric-card">
              <span>Created</span>
              <strong>{formatDateTime(adminMetadata?.created_at)}</strong>
            </div>
            <div className="metric-card">
              <span>Last Used</span>
              <strong>{formatDateTime(adminMetadata?.last_used_at)}</strong>
            </div>
            <div className="metric-card">
              <span>Browser Token</span>
              <strong>{adminKey ? "Stored" : "Missing"}</strong>
            </div>
          </div>

          <div className="key-token-card">
            <div className="key-card-head">
              <div>
                <strong>Current Browser Key</strong>
                <p>This token is sent as <code>X-Admin-Key</code> with every protected admin request.</p>
              </div>
              <button type="button" className="secondary-button" onClick={() => void handleCopyAdminKey()}>
                Copy
              </button>
            </div>
            <div className="key-token-value mono">{adminKey}</div>
          </div>

          <div className="button-row">
            <button type="button" onClick={() => void handleRotateAdminKey()}>
              Rotate Admin Key
            </button>
            <button type="button" className="secondary-button" onClick={() => void handleCopyAdminKey()}>
              Copy Current Key
            </button>
          </div>
        </section>
        <section className="panel stack benchmark-widget">
          <p className="eyebrow">Benchmark</p>
          <h2>Run Audio Through the Active Pipeline</h2>
          <p className="section-copy">
            The benchmark uses the saved server settings and fires repeated runs through the currently active
            batch and chunk pipeline.
          </p>

          <form className="benchmark-form" onSubmit={handleRunBenchmark}>
            <label className="full-width">
              <span>Audio File</span>
              <input
                type="file"
                accept="audio/*,video/*"
                onChange={(event) => setBenchmarkFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <label>
              <span>Repeats (parallel)</span>
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
                {benchmarkBusy ? "Benchmark running..." : "Run Benchmark"}
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
                  <span>Total Time</span>
                  <strong>{formatValue(benchmarkResult.total_wall_time_ms, " ms")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Time / Run</span>
                  <strong>{formatFixed(benchmarkResult.avg_wall_time_per_run_ms, 2, " ms")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Chunks / Run</span>
                  <strong>{benchmarkResult.chunks_per_run}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Total Chunks</span>
                  <strong>{benchmarkResult.total_chunks}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Audio / Run</span>
                  <strong>{formatFixed(benchmarkResult.audio_seconds, 3, " s")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Total Audio</span>
                  <strong>{formatFixed(benchmarkResult.total_audio_seconds, 3, " s")}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Batches Used</span>
                  <strong>{benchmarkResult.batches_used}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Peak VRAM</span>
                  <strong>{formatVram(benchmarkResult.peak_vram_reserved_mb)}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Allocated VRAM</span>
                  <strong>{formatVram(benchmarkResult.peak_vram_allocated_mb)}</strong>
                </article>
                <article className="benchmark-card">
                  <span>Text Match</span>
                  <strong>{benchmarkResult.transcripts_match ? "yes" : "no"}</strong>
                </article>
              </div>

              <div className="benchmark-meta">
                <div>
                  <span>File</span>
                  <strong>{benchmarkResult.file_name}</strong>
                </div>
                <div>
                  <span>Model</span>
                  <strong>{benchmarkResult.model_id}</strong>
                </div>
                <div>
                  <span>Language</span>
                  <strong>{benchmarkResult.transcription_language}</strong>
                </div>
                <div>
                  <span>Repeats</span>
                  <strong>{benchmarkResult.repeat_count}</strong>
                </div>
              </div>

              <div className="benchmark-transcript-wrap">
                <span className="benchmark-transcript-label">Transcript</span>
                <pre className="benchmark-transcript">{benchmarkResult.transcript || "No transcript available."}</pre>
              </div>
            </div>
          )}
        </section>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Settings</span>
              <h2>ASR Model, Language, and Batch Limits</h2>
            </div>
            <div className="loaded-model">
              <span>Loaded Model</span>
              <strong>{loadedModelLabel}</strong>
            </div>
          </div>

          <form className="settings-form" onSubmit={handleSaveSettings}>
            <label>
              <span>ASR Model</span>
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
              <span>Device</span>
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
              <span>Language</span>
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
              <span>Cache Path</span>
              <input
                value={settingsForm.local_model_cache_path}
                onChange={(event) => updateSetting("local_model_cache_path", event.target.value)}
              />
            </label>

            <label className="full-width">
              <span>Hugging Face Token</span>
              <input
                type="password"
                placeholder="hf_... (optional)"
                value={settingsForm.huggingface_token}
                onChange={(event) => updateSetting("huggingface_token", event.target.value)}
              />
            </label>
            <p className="field-note full-width">
              Needed for gated Hugging Face models like <code>pyannote/embedding</code>. Save the settings to persist
              it for runtime use; the current typed value is also sent with manual model downloads from the Cache
              Manager.
            </p>

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
              <span>Max Segments</span>
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
                {saveBusy ? "Saving..." : "Save Settings"}
              </button>
              {saveMessage && <p className="message">{saveMessage}</p>}
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Queue</span>
              <h2>Batch Details</h2>
            </div>
          </div>

          <div className="queue-metrics">
            <div>
              <span>Active Batch</span>
              <strong>{queue?.active_batch_id ?? "No active batch"}</strong>
            </div>
            <div>
              <span>Active Batch Size</span>
              <strong>{queue?.active_batch_size ?? 0}</strong>
            </div>
            <div>
              <span>Active Audio</span>
              <strong>{formatValue(queue?.active_batch_audio_seconds, " s")}</strong>
            </div>
            <div>
              <span>Last Batch</span>
              <strong>{queue?.last_batch_duration_ms ? `${queue.last_batch_duration_ms} ms` : "n/a"}</strong>
            </div>
            <div>
              <span>Processed Batches</span>
              <strong>{queue?.total_batches_processed ?? 0}</strong>
            </div>
            <div>
              <span>Processed Segments</span>
              <strong>{queue?.total_segments_processed ?? 0}</strong>
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Batch</th>
                  <th>Segments</th>
                  <th>Audio</th>
                  <th>Duration</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {recentBatches.length === 0 && (
                  <tr>
                    <td colSpan={6}>No batch history recorded yet.</td>
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
            <span className="eyebrow">Models</span>
            <h2>Cache Manager</h2>
            <p className="section-copy">
              Download or remove the supported ASR models directly in the cache path currently typed in the
              settings form. Downloaded sizes reflect local cache usage; other values are rough estimates.
            </p>
          </div>
          <div className="button-row">
            <button
              type="button"
              className="secondary-button"
              onClick={() => void handleRefreshModels()}
              disabled={modelActionId === "__refresh__"}
            >
              {modelActionId === "__refresh__" && modelActionKind === "refresh" ? "Refreshing..." : "Refresh Models"}
            </button>
          </div>
        </div>

        <div className="model-path-card">
          <span>Selected Cache Path</span>
          <strong className="mono">{settingsForm.local_model_cache_path || "Default Hugging Face cache"}</strong>
        </div>

        <div className="table-wrap">
          <table className="model-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Backend</th>
                <th>Size</th>
                <th>Status</th>
                <th>Path</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {managedModels.length === 0 && (
                <tr>
                  <td colSpan={6}>No supported model entries are available.</td>
                </tr>
              )}
              {managedModels.map((model) => {
                const rowBusy = modelActionId === model.id;
                const isDownloading = model.status === "downloading";
                const canDelete = Boolean(model.cache_path);
                return (
                  <tr key={model.id}>
                    <td>
                      <strong>{model.label}</strong>
                      <div className="muted mono model-id">{model.id}</div>
                    </td>
                    <td>{model.backend === "cohere_transcribe" ? "Cohere" : model.backend === "pyannote" ? "Pyannote" : "Whisper"}</td>
                    <td>{formatModelSize(model)}</td>
                    <td className="model-status-cell">
                      <strong>{formatModelStatus(model.status)}</strong>
                      {model.error && <span className="model-status-error">{model.error}</span>}
                    </td>
                    <td className="mono model-path-cell">{resolveModelPath(model)}</td>
                    <td className="model-actions-cell">
                      <div className="table-actions">
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => void handleDownloadModel(model)}
                          disabled={isDownloading || rowBusy}
                        >
                          {isDownloading
                            ? "Downloading..."
                            : rowBusy && modelActionKind === "download"
                              ? "Starting..."
                              : model.status === "ready"
                                ? "Download Again"
                                : "Download"}
                        </button>
                        <button
                          type="button"
                          className="ghost-button danger-button"
                          onClick={() => void handleDeleteModel(model)}
                          disabled={!canDelete || isDownloading || rowBusy}
                        >
                          {rowBusy && modelActionKind === "delete" ? "Deleting..." : "Delete"}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">History</span>
            <h2>Latest Transcriptions</h2>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>IP</th>
                <th>Model</th>
                <th>Language</th>
                <th>Total</th>
                <th>Transcription</th>
                <th>Batch</th>
                <th>Segments</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 && (
                <tr>
                  <td colSpan={9}>No transcription history recorded yet.</td>
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
                  <td>{entry.batched ? "yes" : "no"}</td>
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

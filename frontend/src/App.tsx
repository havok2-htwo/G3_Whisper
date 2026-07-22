import { FormEvent, startTransition, useEffect, useState } from "react";

import {
  AdminOption,
  AdminSettings,
  ApiKeyInfo,
  BenchmarkResponse,
  CreatedApiKey,
  ManagedModel,
  QueueResponse,
  SettingsResponse,
  StatsResponse,
  changePassword,
  createApiKey,
  deleteApiKey,
  deleteDiaApiKey,
  deleteModel,
  downloadModel,
  getModels,
  getQueue,
  getSettings,
  getStats,
  listApiKeys,
  login,
  logout,
  runBenchmark,
  saveSettings,
  testDiaConnection,
  whoami,
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

type AuthState = "loading" | "login" | "change" | "ready";

const NUMBER_LOCALE = "de-DE";
const DASHBOARD_POLL_SECONDS = 5;
const DASHBOARD_HISTORY_POINTS = 72;
const DASHBOARD_HISTORY_SECONDS = DASHBOARD_POLL_SECONDS * DASHBOARD_HISTORY_POINTS;
const MODEL_STATUS_POLL_MS = 2000;

const emptySettings: AdminSettings = {
  local_model: "",
  local_gpu_device: "",
  local_model_precision: "fp16",
  local_model_cache_path: "",
  transcription_language: "auto",
  batch_wait_time_ms: 1000,
  batch_max_segments: 16,
  batch_max_audio_seconds: 60.0,
  huggingface_token: "",
  dia_server_base_url: "",
  dia_api_key: "",
  dia_api_key_configured: false,
  dia_api_key_source: "none",
  dia_server_base_url_effective: "",
  dia_server_base_url_source: "none",
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
  const [authState, setAuthState] = useState<AuthState>("loading");
  const [currentUser, setCurrentUser] = useState("");
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");

  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwError, setPwError] = useState("");
  const [pwMessage, setPwMessage] = useState("");

  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [newKeyAlias, setNewKeyAlias] = useState("");
  const [createdKey, setCreatedKey] = useState<CreatedApiKey | null>(null);
  const [apiKeyBusy, setApiKeyBusy] = useState(false);

  const [actionMessage, setActionMessage] = useState("");
  const [globalError, setGlobalError] = useState("");
  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  const [settingsOptions, setSettingsOptions] = useState<{ models: AdminOption[]; devices: AdminOption[]; precisions: AdminOption[]; languages: AdminOption[] }>({
    models: [],
    devices: [],
    precisions: [],
    languages: [],
  });
  const [managedModels, setManagedModels] = useState<ManagedModel[]>([]);
  const [loadedModelIdentifier, setLoadedModelIdentifier] = useState<string[] | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [dashboardHistory, setDashboardHistory] = useState<DashboardHistoryPoint[]>([]);
  const [saveBusy, setSaveBusy] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [diaActionBusy, setDiaActionBusy] = useState<"test" | "clear" | null>(null);
  const [diaActionMessage, setDiaActionMessage] = useState("");
  const [diaActionSucceeded, setDiaActionSucceeded] = useState<boolean | null>(null);
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);
  const [benchmarkRepeatCount, setBenchmarkRepeatCount] = useState(1);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkMessage, setBenchmarkMessage] = useState("");
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [modelActionId, setModelActionId] = useState<string | null>(null);
  const [modelActionKind, setModelActionKind] = useState<"refresh" | "download" | "delete" | null>(null);
  const hasDownloadingModels = managedModels.some((model) => model.status === "downloading");
  const isReady = authState === "ready";

  function resetDashboardState() {
    startTransition(() => {
      setStats(null);
      setQueue(null);
      setDashboardHistory([]);
      setSettingsForm(emptySettings);
      setManagedModels([]);
      setLoadedModelIdentifier(null);
      setSettingsOptions({ models: [], devices: [], precisions: [], languages: [] });
      setApiKeys([]);
      setCreatedKey(null);
      setSaveMessage("");
      setDiaActionBusy(null);
      setDiaActionMessage("");
      setDiaActionSucceeded(null);
      setBenchmarkMessage("");
      setBenchmarkResult(null);
      setModelActionId(null);
      setModelActionKind(null);
      setActionMessage("");
      setGlobalError("");
    });
  }

  // Centralized reaction to auth failures raised by any admin call.
  function handleApiError(error: unknown, fallback: string): boolean {
    const message = error instanceof Error ? error.message : "";
    if (message === "unauthorized") {
      resetDashboardState();
      setAuthState("login");
      setAuthError("Your session has expired. Please sign in again.");
      return true;
    }
    if (message === "password_change_required") {
      setAuthState("change");
      return true;
    }
    setGlobalError(error instanceof Error ? error.message : fallback);
    return false;
  }

  async function loadDashboard() {
    try {
      const [settingsResponse, statsResponse, queueResponse, keysResponse] = await Promise.all([
        getSettings(),
        getStats(),
        getQueue(),
        listApiKeys(),
      ]);
      startTransition(() => {
        applySettings(settingsResponse);
        setStats(statsResponse);
        setQueue(queueResponse);
        setApiKeys(keysResponse.keys);
        setGlobalError("");
      });
    } catch (error) {
      handleApiError(error, "The dashboard could not be loaded.");
    }
  }

  function applySettings(payload: SettingsResponse) {
    setSettingsForm({ ...emptySettings, ...payload.settings, dia_api_key: "" });
    setSettingsOptions(payload.options);
    setManagedModels(payload.models ?? []);
    setLoadedModelIdentifier(payload.loaded_model_identifier);
  }

  async function refreshOperationalData() {
    if (!isReady) {
      return;
    }
    try {
      const [statsResponse, queueResponse, keysResponse] = await Promise.all([getStats(), getQueue(), listApiKeys()]);
      startTransition(() => {
        setStats(statsResponse);
        setQueue(queueResponse);
        setApiKeys(keysResponse.keys);
        setGlobalError("");
      });
    } catch (error) {
      handleApiError(error, "Live polling failed.");
    }
  }

  // Bootstrap: check the session on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const me = await whoami();
        if (cancelled) return;
        setCurrentUser(me.username);
        setAuthState(me.must_change_password ? "change" : "ready");
      } catch {
        if (!cancelled) setAuthState("login");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isReady) {
      return;
    }
    void loadDashboard();
  }, [authState]);

  useEffect(() => {
    if (!isReady) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void refreshOperationalData();
    }, DASHBOARD_POLL_SECONDS * 1000);
    return () => window.clearInterval(intervalId);
  }, [authState]);

  useEffect(() => {
    if (!isReady || !stats || !queue) {
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
  }, [authState, queue, stats]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setLoginPassword("");
      setAuthState(me.must_change_password ? "change" : "ready");
    } catch (error) {
      setAuthError(
        error instanceof Error && error.message === "unauthorized"
          ? "Invalid username or password."
          : error instanceof Error
            ? error.message
            : "Login failed.",
      );
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleChangePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPwError("");
    setPwMessage("");
    if (pwNew.length < 4) {
      setPwError("The new password must be at least 4 characters.");
      return;
    }
    if (pwNew !== pwConfirm) {
      setPwError("The new password and its confirmation do not match.");
      return;
    }
    setPwBusy(true);
    try {
      await changePassword(pwCurrent, pwNew);
      setPwCurrent("");
      setPwNew("");
      setPwConfirm("");
      setPwMessage("Password updated.");
      setAuthState("ready");
    } catch (error) {
      setPwError(error instanceof Error ? error.message : "Password change failed.");
    } finally {
      setPwBusy(false);
    }
  }

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // ignore — clear locally regardless
    }
    resetDashboardState();
    setLoginPassword("");
    setAuthState("login");
  }

  async function handleCreateApiKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setApiKeyBusy(true);
    setActionMessage("");
    setGlobalError("");
    try {
      const created = await createApiKey(newKeyAlias.trim());
      setCreatedKey(created);
      setNewKeyAlias("");
      setActionMessage(`API key "${created.alias}" created. Copy it now — it is shown only once.`);
      const keysResponse = await listApiKeys();
      setApiKeys(keysResponse.keys);
    } catch (error) {
      handleApiError(error, "The API key could not be created.");
    } finally {
      setApiKeyBusy(false);
    }
  }

  async function handleDeleteApiKey(keyId: string, alias: string) {
    setActionMessage("");
    setGlobalError("");
    try {
      await deleteApiKey(keyId);
      setCreatedKey((current) => (current?.id === keyId ? null : current));
      setActionMessage(`API key "${alias}" deleted.`);
      const keysResponse = await listApiKeys();
      setApiKeys(keysResponse.keys);
    } catch (error) {
      handleApiError(error, "The API key could not be deleted.");
    }
  }

  async function handleCopyCreatedKey() {
    if (!createdKey?.token) {
      return;
    }
    try {
      await copyTextToClipboard(createdKey.token);
      setActionMessage("API key copied to the clipboard.");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "Copying failed.");
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
            ? `Settings saved. Model reload ${response.model_loaded ? "succeeded" : "failed"}.`
            : "Settings saved.",
        );
      });
      await refreshOperationalData();
    } catch (error) {
      if (handleApiError(error, "Saving settings failed.")) {
        return;
      }
      setSaveMessage(error instanceof Error ? error.message : "Saving settings failed.");
    } finally {
      setSaveBusy(false);
    }
  }

  async function handleTestDiaConnection() {
    setDiaActionBusy("test");
    setDiaActionMessage("");
    setDiaActionSucceeded(null);
    setGlobalError("");

    try {
      const response = await testDiaConnection(settingsForm.dia_server_base_url, settingsForm.dia_api_key);
      setDiaActionMessage(`${response.message} (${response.base_url})`);
      setDiaActionSucceeded(true);
    } catch (error) {
      if (handleApiError(error, "DIA connection test failed.")) {
        return;
      }
      setDiaActionMessage(error instanceof Error ? error.message : "DIA connection test failed.");
      setDiaActionSucceeded(false);
    } finally {
      setDiaActionBusy(null);
    }
  }

  async function handleClearDiaApiKey() {
    if (settingsForm.dia_api_key) {
      updateSetting("dia_api_key", "");
      setDiaActionMessage("The unsaved DIA API key was discarded.");
      setDiaActionSucceeded(true);
      return;
    }
    if (settingsForm.dia_api_key_source !== "settings") {
      return;
    }
    if (!window.confirm("Delete the saved DIA API key? This does not remove an environment variable.")) {
      return;
    }

    setDiaActionBusy("clear");
    setDiaActionMessage("");
    setDiaActionSucceeded(null);
    setGlobalError("");
    try {
      const response = await deleteDiaApiKey();
      applySettings(response);
      setDiaActionMessage(
        response.environment_fallback_active
          ? "Saved DIA API key deleted. DIA_SERVER_API_KEY remains active as the environment fallback."
          : response.removed
            ? "Saved DIA API key deleted."
            : "No saved DIA API key was present.",
      );
      setDiaActionSucceeded(true);
    } catch (error) {
      if (handleApiError(error, "The DIA API key could not be deleted.")) {
        return;
      }
      setDiaActionMessage(error instanceof Error ? error.message : "The DIA API key could not be deleted.");
      setDiaActionSucceeded(false);
    } finally {
      setDiaActionBusy(null);
    }
  }

  async function refreshManagedModels(options?: { silent?: boolean; storagePath?: string }) {
    if (!isReady) {
      return;
    }

    const silent = options?.silent ?? false;
    const storagePath = options?.storagePath ?? managedModels[0]?.storage_root ?? settingsForm.local_model_cache_path;

    try {
      const response = await getModels(storagePath);
      startTransition(() => {
        setManagedModels(response.models ?? []);
        if (!silent) {
          setActionMessage("Model cache refreshed for the selected path.");
        }
      });
    } catch (error) {
      if (handleApiError(error, "Model cache refresh failed.")) {
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
    setModelActionId(model.id);
    setModelActionKind("download");
    setActionMessage("");
    setGlobalError("");

    try {
      const response = await downloadModel(model.id, settingsForm.local_model_cache_path, settingsForm.huggingface_token);
      startTransition(() => {
        setManagedModels(response.models ?? []);
        setActionMessage(`Download queued for ${model.label}.`);
      });
    } catch (error) {
      handleApiError(error, "Model download failed.");
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  async function handleDeleteModel(model: ManagedModel) {
    setModelActionId(model.id);
    setModelActionKind("delete");
    setActionMessage("");
    setGlobalError("");

    try {
      const response = await deleteModel(model.id, settingsForm.local_model_cache_path);
      startTransition(() => {
        setManagedModels(response.models ?? []);
        setActionMessage(
          response.removed
            ? `Cached files removed for ${model.label}.`
            : `No cached files found for ${model.label} in the selected path.`,
        );
      });
    } catch (error) {
      handleApiError(error, "Model deletion failed.");
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  useEffect(() => {
    if (!isReady || !hasDownloadingModels) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void refreshManagedModels({ silent: true });
    }, MODEL_STATUS_POLL_MS);
    return () => window.clearInterval(intervalId);
  }, [authState, hasDownloadingModels, managedModels, settingsForm.local_model_cache_path]);

  async function handleRunBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!benchmarkFile) {
      setBenchmarkMessage("Please select an audio or video file first.");
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
          `Benchmark finished: ${formatValue(result.repeat_count)} run(s) in ${formatValue(result.total_wall_time_ms, " ms")}.`,
        );
      });
      await refreshOperationalData();
    } catch (error) {
      if (handleApiError(error, "Benchmark failed.")) {
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

  if (authState === "loading") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <p className="message">Loading...</p>
        </section>
      </main>
    );
  }

  if (authState === "login") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Private Access</span>
            <h1>GENESIS Whisper Admin</h1>
            <p>
              The public transcription API stays open until you create an API key. The private dashboard for queue
              control, benchmarks, and history is protected by a username and password login.
            </p>
          </div>

          <form className="login-form" onSubmit={handleLogin}>
            <label>
              <span>Username</span>
              <input
                value={loginUsername}
                autoComplete="username"
                onChange={(event) => setLoginUsername(event.target.value)}
                placeholder="admin"
              />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                value={loginPassword}
                autoComplete="current-password"
                onChange={(event) => setLoginPassword(event.target.value)}
                placeholder="admin"
              />
            </label>

            <button type="submit" disabled={!loginUsername.trim() || !loginPassword || authBusy}>
              {authBusy ? "Signing in..." : "Sign In"}
            </button>

            <p className="message">Default credentials: admin / admin. You must change the password on first login.</p>
            {authError && <p className="message error">{authError}</p>}
          </form>
        </section>
      </main>
    );
  }

  if (authState === "change") {
    return (
      <main className="shell centered">
        <section className="panel login-panel">
          <div className="hero-copy">
            <span className="eyebrow">Security</span>
            <h1>Set a New Password</h1>
            <p>
              You are signed in as <strong>{currentUser || "admin"}</strong>. Choose a new password before continuing
              to the dashboard.
            </p>
          </div>

          <form className="login-form" onSubmit={handleChangePassword}>
            <label>
              <span>Current Password</span>
              <input
                type="password"
                value={pwCurrent}
                autoComplete="current-password"
                onChange={(event) => setPwCurrent(event.target.value)}
              />
            </label>
            <label>
              <span>New Password</span>
              <input
                type="password"
                value={pwNew}
                autoComplete="new-password"
                onChange={(event) => setPwNew(event.target.value)}
              />
            </label>
            <label>
              <span>Confirm New Password</span>
              <input
                type="password"
                value={pwConfirm}
                autoComplete="new-password"
                onChange={(event) => setPwConfirm(event.target.value)}
              />
            </label>

            <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm}>
              {pwBusy ? "Saving..." : "Save New Password"}
            </button>

            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>
              Cancel & Sign Out
            </button>

            {pwMessage && <p className="message">{pwMessage}</p>}
            {pwError && <p className="message error">{pwError}</p>}
          </form>
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
  const configuredPrecisionLabel = resolveOptionLabel(settingsOptions.precisions, settingsForm.local_model_precision, "FP16");
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
            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>
              Logout
            </button>
          </div>
        </div>

        <div className="status-grid">
          <div className="status-pill">
            <span>Signed in as</span>
            <strong>{currentUser || "admin"}</strong>
          </div>
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
          <div className="status-pill">
            <span>Precision</span>
            <strong>{configuredPrecisionLabel}</strong>
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
          <p className="eyebrow">Account</p>
          <h2>Admin Access</h2>
          <div className="metric-grid compact-metrics">
            <div className="metric-card">
              <span>Username</span>
              <strong>{currentUser || "admin"}</strong>
            </div>
            <div className="metric-card">
              <span>Session</span>
              <strong>Active (cookie)</strong>
            </div>
          </div>

          <form className="login-form" onSubmit={handleChangePassword}>
            <label>
              <span>Current Password</span>
              <input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} />
            </label>
            <label>
              <span>New Password</span>
              <input type="password" value={pwNew} autoComplete="new-password" onChange={(event) => setPwNew(event.target.value)} />
            </label>
            <label>
              <span>Confirm New Password</span>
              <input type="password" value={pwConfirm} autoComplete="new-password" onChange={(event) => setPwConfirm(event.target.value)} />
            </label>
            <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm}>
              {pwBusy ? "Saving..." : "Change Password"}
            </button>
            {pwMessage && <p className="message">{pwMessage}</p>}
            {pwError && <p className="message error">{pwError}</p>}
          </form>
        </section>

        <section className="panel stack">
          <p className="eyebrow">API Keys</p>
          <h2>Public API Access</h2>
          <p className="section-copy">
            While no key exists, <code>POST /transcribe/</code> is open to everyone. As soon as one key exists, callers
            must send a valid <code>X-API-Key</code> header. Usage (processed audio seconds) is tracked per key.
          </p>

          {createdKey && (
            <div className="key-token-card">
              <div className="key-card-head">
                <div>
                  <strong>{createdKey.alias}</strong>
                  <p>Copy this key now — it is shown only once.</p>
                </div>
                <button type="button" className="secondary-button" onClick={() => void handleCopyCreatedKey()}>
                  Copy
                </button>
              </div>
              <div className="key-token-value mono">{createdKey.token}</div>
            </div>
          )}

          <form className="benchmark-form" onSubmit={handleCreateApiKey}>
            <label className="full-width">
              <span>Alias</span>
              <input
                value={newKeyAlias}
                onChange={(event) => setNewKeyAlias(event.target.value)}
                placeholder="e.g. Key fuer Projekt X"
              />
            </label>
            <div className="form-actions full-width">
              <button type="submit" disabled={apiKeyBusy || !newKeyAlias.trim()}>
                {apiKeyBusy ? "Creating..." : "Create API Key"}
              </button>
            </div>
          </form>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Alias</th>
                  <th>Created</th>
                  <th>Audio (s)</th>
                  <th>Requests</th>
                  <th>Last Used</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {apiKeys.length === 0 && (
                  <tr>
                    <td colSpan={6}>No API keys — the public API is currently open.</td>
                  </tr>
                )}
                {apiKeys.map((key) => (
                  <tr key={key.id}>
                    <td>{key.alias}</td>
                    <td>{formatDateTime(key.created_at)}</td>
                    <td>{formatFixed(key.usage.total_seconds_processed, 1)}</td>
                    <td>{key.usage.request_count}</td>
                    <td>{formatDateTime(key.usage.last_used_at)}</td>
                    <td>
                      <button
                        type="button"
                        className="ghost-button danger-button"
                        onClick={() => void handleDeleteApiKey(key.id, key.alias)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
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
              <h2>ASR, DIA, Language, and Batch Limits</h2>
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

            <label>
              <span>Precision</span>
              <select
                value={settingsForm.local_model_precision}
                onChange={(event) => updateSetting("local_model_precision", event.target.value)}
              >
                {settingsOptions.precisions.map((option) => (
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
              Used for gated ASR models such as Cohere Transcribe. Save the settings to persist it; the current typed
              value is also sent with manual model downloads from the Cache Manager.
            </p>

            <div className="dia-settings full-width">
              <div className="settings-subheading">
                <div>
                  <span className="eyebrow">DIA Integration</span>
                  <h3>Diarization Server</h3>
                </div>
                <span className={`connection-status ${settingsForm.dia_api_key_configured ? "configured" : ""}`}>
                  {settingsForm.dia_api_key_configured
                    ? `API key: ${settingsForm.dia_api_key_source}`
                    : "No API key configured"}
                </span>
              </div>

              <div className="dia-settings-grid">
                <label className="full-width">
                  <span>DIA Server URL</span>
                  <input
                    type="url"
                    placeholder={settingsForm.dia_server_base_url_effective || "http://dia:7864"}
                    value={settingsForm.dia_server_base_url}
                    onChange={(event) => updateSetting("dia_server_base_url", event.target.value)}
                  />
                </label>
                {settingsForm.dia_server_base_url_source === "environment" && !settingsForm.dia_server_base_url && (
                  <p className="field-note full-width">
                    Using <code>{settingsForm.dia_server_base_url_effective}</code> from <code>DIA_SERVER_BASE_URL</code>.
                  </p>
                )}

                <label className="full-width">
                  <span>DIA API Key</span>
                  <input
                    type="password"
                    autoComplete="new-password"
                    placeholder={settingsForm.dia_api_key_configured ? "Enter a new key to replace the configured key" : "Optional"}
                    value={settingsForm.dia_api_key}
                    onChange={(event) => updateSetting("dia_api_key", event.target.value)}
                  />
                </label>
                <p className="field-note full-width">
                  Write-only: the server never returns the saved key. Leaving this field empty preserves the current
                  key. Environment keys remain managed through <code>DIA_SERVER_API_KEY</code>.
                </p>

                <div className="dia-actions full-width">
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={diaActionBusy !== null}
                    onClick={() => void handleTestDiaConnection()}
                  >
                    {diaActionBusy === "test" ? "Testing..." : "Test DIA Connection"}
                  </button>
                  <button
                    type="button"
                    className="ghost-button danger-button"
                    disabled={
                      diaActionBusy !== null
                      || (!settingsForm.dia_api_key && settingsForm.dia_api_key_source !== "settings")
                    }
                    onClick={() => void handleClearDiaApiKey()}
                  >
                    {diaActionBusy === "clear"
                      ? "Deleting..."
                      : settingsForm.dia_api_key
                        ? "Discard Entered Key"
                        : "Delete Saved Key"}
                  </button>
                  {diaActionMessage && (
                    <p className={`message dia-message ${diaActionSucceeded === false ? "error" : ""}`}>
                      {diaActionMessage}
                    </p>
                  )}
                </div>
              </div>
            </div>

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
                    <td>{model.backend === "cohere_transcribe" ? "Cohere" : "Whisper"}</td>
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

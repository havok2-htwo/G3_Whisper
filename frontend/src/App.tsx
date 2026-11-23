import { FormEvent, KeyboardEvent, ReactNode, startTransition, useEffect, useRef, useState } from "react";

import {
  ACTIVE_MODEL_CHANGED,
  AdminOption,
  AdminSettings,
  AudioProcessMode,
  AudioProcessRequest,
  AudioProcessResponse,
  ApiKeyInfo,
  BenchmarkResponse,
  HistoryEntry,
  ManagedModel,
  QueueResponse,
  SettingsPreset,
  SettingsResponse,
  SettingsSaveResponse,
  SpeakerRefinementMode,
  StatsResponse,
  UnknownSpeakerAudio,
  changePassword,
  createApiKey,
  deleteApiKey,
  deleteModel,
  deletePreset,
  downloadHistoryAudio,
  downloadModel,
  getAuthSetup,
  getModels,
  getPerf,
  getQueue,
  getSettings,
  getStats,
  listApiKeys,
  listPresets,
  login,
  logout,
  processAudioV2,
  retryHistoryAudio,
  runBenchmark,
  savePreset,
  saveSettings,
  selectModel,
  whoami,
} from "./api";
import { ApiKeysPanel } from "./ApiKeysPanel";
import { PerfPanel, clearPerfHistory, fmt, gpuTitle, usePerfFeed, usePerfLatest } from "./PerfPanel";
import { KeepEdits, mergeServerSettings } from "./settingsForm";
import { ToastHost, toast } from "./Toasts";
import { useVisiblePolling } from "./usePolling";

type BatchEntry = {
  batch_id?: string;
  timestamp?: string;
  batch_size?: number;
  audio_seconds?: number;
  duration_ms?: number;
  status?: string;
  error?: string;
  unique_request_count?: number;
  fast_path?: boolean;
};

type JobEntry = {
  request_id?: string;
  status?: string;
  is_long?: boolean;
  total_chunks?: number;
  batch_count?: number;
  queue_wait_ms?: number | null;
  duration_ms?: number;
  finished_at?: string;
};

type BenchmarkWorkflow = "whisper_chunk_queue" | "cohere_audio_batch" | string;
type AuthState = "loading" | "login" | "change" | "ready";
type HistoryActionKind = "retry" | "download";

const TABS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "playground", label: "Playground" },
  { id: "models", label: "Models" },
  { id: "settings", label: "Settings" },
  { id: "benchmark", label: "Benchmark" },
  { id: "history", label: "History" },
  { id: "api-keys", label: "API keys" },
  { id: "account", label: "Account" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function tabFromHash(): TabId {
  const id = window.location.hash.replace(/^#/, "");
  return (TABS.find((tab) => tab.id === id)?.id ?? "dashboard") as TabId;
}

const PIPELINE_MODES: Array<{ value: AudioProcessMode; label: string; description: string }> = [
  { value: "embedding", label: "Voice embedding", description: "One normalized 192-D ReDimNet2 voice vector." },
  { value: "transcript", label: "Transcript", description: "ASR transcript with repetition filtering." },
  {
    value: "transcript_embedding",
    label: "Transcript + embedding",
    description: "ASR transcript plus one mixed 192-D voice vector.",
  },
  {
    value: "diarization",
    label: "Full diarization",
    description: "Speaker separation (seg-3.0 + ReDimNet2), ASR, speaker profile matching, and unknown voices.",
  },
];

const PUBLIC_ROUTES = "POST /transcribe/, POST /v1/audio/transcriptions, GET /v1/models and POST /v2/audio/process";
const NUMBER_LOCALE = "en-US";
const OPERATIONAL_POLL_MS = 5000;
const API_KEYS_POLL_MS = 10000;
const MODEL_STATUS_POLL_MS = 2000;
const MIN_PASSWORD_LENGTH = 8;
const DASH = "—";

const emptySettings: AdminSettings = {
  local_model: "",
  local_gpu_device: "",
  local_model_precision: "fp16",
  local_model_cache_path: "",
  transcription_language: "auto",
  batch_wait_time_ms: 1000,
  batch_max_segments: 16,
  batch_max_audio_seconds: 60.0,
  scheduler_long_job_min_chunks: 5,
  scheduler_max_parallel_long_jobs: 2,
  scheduler_first_chunk_fast_path: true,
  cuda_memory_trim_after_batch: true,
  cohere_cuda_graphs: true,
  cohere_language_detection: true,
  language_detection_model: "openai/whisper-base",
  redimnet_max_batch_size: 8,
  debug_retain_history_audio: false,
  huggingface_token: "",
  huggingface_token_set: false,
  segmentation_model_path: "",
  speaker_turns_enabled: true,
};

// Keys that are not edited through the settings form (or only through the token field).
const NON_FORM_KEYS = new Set<string>(["model_profiles", "huggingface_token", "huggingface_token_set", "local_model"]);

// --------------------------------------------------------------------------------------------------- formatting

function formatValue(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return DASH;
  }
  const maximumFractionDigits = Number.isInteger(value) ? 0 : 2;
  return `${new Intl.NumberFormat(NUMBER_LOCALE, { maximumFractionDigits }).format(value)}${suffix}`;
}

function formatFixed(value: number | null | undefined, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return DASH;
  }
  return `${new Intl.NumberFormat(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)}${suffix}`;
}

/** ms: 1 decimal below 10 ms, integer up to 9,999 ms, then seconds with 1 decimal. */
function formatMs(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return DASH;
  }
  if (value < 10) {
    return formatFixed(value, 1, " ms");
  }
  if (value < 10000) {
    return formatFixed(value, 0, " ms");
  }
  return formatFixed(value / 1000, 1, " s");
}

/** MiB (torch peak numbers) as GB with one decimal, MB below 1 GB. */
function formatMib(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return DASH;
  }
  return value >= 1024 ? fmt(value / 1024, 1, "GB") : fmt(value, 0, "MB");
}

function formatBytes(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return DASH;
  }
  if (value >= 1024 * 1024 * 1024) {
    return formatFixed(value / (1024 * 1024 * 1024), 1, " GB");
  }
  if (value >= 1024 * 1024) {
    return formatFixed(value / (1024 * 1024), 1, " MB");
  }
  if (value >= 1024) {
    return formatFixed(value / 1024, 1, " KB");
  }
  return `${formatValue(value)} B`;
}

const pad2 = (n: number) => String(n).padStart(2, "0");

/** "2026-11-23 21:04" (local time); with seconds where they matter. Server wall-clock strings pass through. */
function formatDateTime(value?: string | null, withSeconds = false) {
  if (!value) {
    return DASH;
  }
  const wall = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(:\d{2})?$/.exec(value);
  if (wall) {
    return `${wall[1]} ${wall[2]}${withSeconds && wall[3] ? wall[3] : ""}`;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  const base = `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())} ${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
  return withSeconds ? `${base}:${pad2(date.getSeconds())}` : base;
}

function formatRealtimeFactor(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return DASH;
  }
  return fmt(value, value >= 10 ? 0 : value >= 1 ? 1 : 2, "x");
}

function describeDebugAudioReason(reason?: string | null) {
  if (reason === "file_too_large") {
    return "Not retained: file exceeds the 256 MB limit.";
  }
  if (reason === "storage_quota") {
    return "Not retained: debug-audio storage quota reached.";
  }
  if (reason === "capture_failed") {
    return "Not retained: capture failed.";
  }
  if (reason === "disabled") {
    return "Audio retention was disabled.";
  }
  return reason ? `Not retained: ${reason.replace(/_/g, " ")}.` : "Audio was not retained.";
}

function formatModelSize(model: ManagedModel) {
  if (model.size_on_disk_gb !== null && model.size_on_disk_gb !== undefined) {
    return model.size_on_disk_gb < 1
      ? formatFixed(model.size_on_disk_gb * 1024, 0, " MB")
      : formatFixed(model.size_on_disk_gb, 1, " GB");
  }
  if (model.approx_size_gb !== null && model.approx_size_gb !== undefined) {
    return model.approx_size_gb < 1
      ? `~${formatFixed(model.approx_size_gb * 1024, 0, " MB")}`
      : `~${formatFixed(model.approx_size_gb, 1, " GB")}`;
  }
  return DASH;
}

function modelStatusBadge(status: string): { text: string; tone: string } {
  if (status === "ready") {
    return { text: "Cached", tone: "ok" };
  }
  if (status === "downloading") {
    return { text: "Downloading", tone: "warn" };
  }
  if (status === "partial") {
    return { text: "Partial", tone: "warn" };
  }
  if (status === "error") {
    return { text: "Failed", tone: "danger" };
  }
  return { text: "Not cached", tone: "plain" };
}

function formatBackendLabel(backend: string) {
  if (backend === "cohere_transcribe") {
    return "Cohere";
  }
  return backend === "parakeet" ? "Parakeet (ONNX)" : "Whisper";
}

function resolveModelPath(model: ManagedModel) {
  return model.local_path || model.cache_path || model.storage_root;
}

function describeBenchmarkWorkflow(workflow: BenchmarkWorkflow) {
  if (workflow === "cohere_audio_batch") {
    return "Cohere whole-audio batch";
  }
  if (workflow === "parakeet_audio_batch") {
    return "Parakeet whole-audio batch";
  }
  if (workflow === "whisper_chunk_queue") {
    return "Whisper chunk queue";
  }
  return workflow;
}

function parseKnownSpeakerProfiles(rawValue: string) {
  if (new TextEncoder().encode(rawValue).byteLength > 16 * 1024 * 1024) {
    throw new Error("Known-speaker JSON must not exceed 16 MB.");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawValue || "[]");
  } catch {
    throw new Error("Known speakers must be valid JSON.");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("Known speakers must be a JSON array.");
  }
  if (parsed.length > 64) {
    throw new Error("At most 64 known speakers are allowed.");
  }

  const ids = new Set<string>();
  return parsed.map((entry, speakerIndex) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`Known speaker ${speakerIndex + 1} must be an object.`);
    }
    const profile = entry as Record<string, unknown>;
    const unsupportedKeys = Object.keys(profile).filter((key) => key !== "id" && key !== "embeddings");
    if (unsupportedKeys.length > 0) {
      throw new Error(`Known speaker ${speakerIndex + 1} contains unsupported field "${unsupportedKeys[0]}".`);
    }
    const id = typeof profile.id === "string" ? profile.id.trim() : "";
    if (!id) {
      throw new Error(`Known speaker ${speakerIndex + 1} needs a non-empty id.`);
    }
    if (new TextEncoder().encode(id).byteLength > 128) {
      throw new Error(`Known speaker id "${id}" exceeds 128 UTF-8 bytes.`);
    }
    if (ids.has(id)) {
      throw new Error(`Known speaker id "${id}" occurs more than once.`);
    }
    ids.add(id);

    if (!Array.isArray(profile.embeddings) || profile.embeddings.length === 0) {
      throw new Error(`Known speaker "${id}" needs at least one embedding.`);
    }
    const embeddings = profile.embeddings.map((candidate, embeddingIndex) => {
      if (!Array.isArray(candidate) || candidate.length !== 192) {
        throw new Error(`Embedding ${embeddingIndex + 1} for "${id}" must contain exactly 192 values.`);
      }
      if (candidate.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
        throw new Error(`Embedding ${embeddingIndex + 1} for "${id}" contains a non-finite value.`);
      }
      const vector = candidate as number[];
      if (Math.max(...vector.map((value) => Math.abs(value))) <= 0) {
        throw new Error(`Embedding ${embeddingIndex + 1} for "${id}" must have a non-zero norm.`);
      }
      return vector;
    });
    return { id, embeddings };
  });
}

function describeModel(model: Record<string, unknown> | undefined) {
  if (!model) {
    return DASH;
  }
  const id = typeof model.id === "string" ? model.id : typeof model.model === "string" ? model.model : "configured";
  const dimension = typeof model.dimension === "number" ? ` · ${model.dimension}-D` : "";
  return `${id}${dimension}`;
}

function resolveEmbeddingDimension(response: AudioProcessResponse) {
  const directVector = response.result.embedding?.vector;
  if (Array.isArray(directVector)) {
    return directVector.length;
  }
  const modelDimension = response.models.embedding?.dimension;
  return typeof modelDimension === "number" ? modelDimension : null;
}

function formatTimecode(milliseconds: number) {
  const totalMilliseconds = Math.max(0, Math.round(milliseconds));
  const hours = Math.floor(totalMilliseconds / 3_600_000);
  const minutes = Math.floor((totalMilliseconds % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMilliseconds % 60_000) / 1000);
  const millis = totalMilliseconds % 1000;
  return `${pad2(hours)}:${pad2(minutes)}:${pad2(seconds)}.${String(millis).padStart(3, "0")}`;
}

function speakerAudioSource(audio: UnknownSpeakerAudio) {
  return `data:${audio.mime_type};base64,${audio.data}`;
}

function resolveOptionLabel(options: AdminOption[], value: string | null | undefined, fallback = DASH) {
  if (!value) {
    return fallback;
  }
  return options.find((option) => option.value === value)?.label || value;
}

function errorText(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}

function isAbort(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}

// --------------------------------------------------------------------------------------------------- small parts

function Card({ label, value, sub, title }: { label: string; value: ReactNode; sub?: ReactNode; title?: string }) {
  return (
    <div className="card">
      <span className="card-label">{label}</span>
      <span className="card-value" title={title}>{value}</span>
      {sub !== undefined && sub !== null && sub !== "" && <span className="card-sub">{sub}</span>}
    </div>
  );
}

function Chip({ label, children, title, onClick }: { label: string; children: ReactNode; title?: string; onClick?: () => void }) {
  const content = (
    <>
      <span className="card-label">{label}</span>
      <span className="value" title={title}>{children}</span>
    </>
  );
  return onClick ? (
    <button type="button" className="chip chip-button" onClick={onClick} title={title}>{content}</button>
  ) : (
    <span className="chip">{content}</span>
  );
}

type NumberFieldProps = {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number | string;
  hint?: ReactNode;
  wide?: boolean;
  onChange: (value: number) => void;
};

/** Empty means "not set" (NaN): it is never sent as 0 and blocks saving. */
function NumberField({ label, value, min, max, step, hint, wide, onChange }: NumberFieldProps) {
  const empty = Number.isNaN(value);
  const outOfRange = !empty && ((min !== undefined && value < min) || (max !== undefined && value > max));
  const invalid = empty || outOfRange;
  return (
    <label className={wide ? "span-2" : undefined}>
      <span>{label}</span>
      <input
        type="number"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        value={empty ? "" : value}
        aria-invalid={invalid || undefined}
        onChange={(event) => onChange(event.target.value === "" ? Number.NaN : Number(event.target.value))}
      />
      {invalid && (
        <span className="field-error">
          {empty ? "Required." : `Allowed: ${min ?? "…"} to ${max ?? "…"}.`}
        </span>
      )}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

function CheckField({ label, checked, onChange, hint, disabled, tone }: {
  label: string; checked: boolean; onChange: (checked: boolean) => void; hint?: ReactNode; disabled?: boolean; tone?: "warn";
}) {
  return (
    <label className={`check-field span-2 ${tone === "warn" ? "check-warn" : ""}`.trim()}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <span>
        <span className="check-label">{label}</span>
        {hint && <span className="hint">{hint}</span>}
      </span>
    </label>
  );
}

type ModelPickerProps = {
  models: ManagedModel[];
  locked: boolean;
  selectingModelId: string | null;
  startingDownloadId: string | null;
  onLoad: (model: ManagedModel) => void;
  onDownload: (model: ManagedModel) => void;
};

function ModelPicker({ models, locked, selectingModelId, startingDownloadId, onLoad, onDownload }: ModelPickerProps) {
  const backends = Array.from(new Set(models.map((model) => model.backend)));
  if (models.length === 0) {
    return <div className="state"><span className="spinner" />Loading models…</div>;
  }
  return (
    <div className="model-picker">
      {backends.map((backend) => (
        <div className="model-picker-group" key={backend}>
          <h3>{formatBackendLabel(backend)}</h3>
          {models
            .filter((model) => model.backend === backend)
            .map((model) => {
              const isDownloading = model.status === "downloading";
              const isSelecting = selectingModelId === model.id;
              const status = modelStatusBadge(model.status);
              return (
                <div className={`model-picker-row ${model.active ? "active" : ""}`} key={model.id}>
                  <div className="model-picker-name">
                    <strong>{model.label}</strong>
                    <span className="mono hint">{model.id}</span>
                    {model.error && <span className="field-error">{model.error}</span>}
                  </div>
                  <div className="model-picker-markers">
                    <span className={`badge ${status.tone}`}>{status.text}</span>
                    {model.active && <span className="badge accent">Active</span>}
                    {model.loaded && <span className="badge ok">Loaded</span>}
                    {model.active && !model.loaded && <span className="badge plain">Not loaded</span>}
                  </div>
                  <div className="model-picker-action">
                    {model.status === "ready" ? (
                      !(model.active && model.loaded) && (
                        <button
                          type="button"
                          className="secondary-button small-button"
                          onClick={() => onLoad(model)}
                          disabled={locked}
                          aria-busy={isSelecting || undefined}
                        >
                          {isSelecting ? "Loading…" : "Load"}
                        </button>
                      )
                    ) : (
                      <button
                        type="button"
                        className="secondary-button small-button"
                        onClick={() => onDownload(model)}
                        disabled={isDownloading || startingDownloadId === model.id}
                        aria-busy={isDownloading || startingDownloadId === model.id || undefined}
                      >
                        {isDownloading ? "Downloading…" : startingDownloadId === model.id ? "Starting…" : "Download"}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
        </div>
      ))}
    </div>
  );
}

// --------------------------------------------------------------------------------------------------- app

export default function App() {
  const [authState, setAuthState] = useState<AuthState>("loading");
  const [bootError, setBootError] = useState("");
  const [setupRequired, setSetupRequired] = useState(false);
  const [currentUser, setCurrentUser] = useState("");
  const [sessionExpiresAt, setSessionExpiresAt] = useState<string | null>(null);
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [tab, setTab] = useState<TabId>(tabFromHash);

  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwError, setPwError] = useState("");

  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[] | null>(null);
  const [apiKeysError, setApiKeysError] = useState<string | null>(null);

  const [settingsForm, setSettingsForm] = useState<AdminSettings>(emptySettings);
  // What the server sent last. The form differs from it exactly where the admin has unsaved edits.
  const settingsBaseline = useRef<AdminSettings>(emptySettings);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [settingsError, setSettingsError] = useState("");
  const [tokenDraft, setTokenDraft] = useState("");
  const [tokenClear, setTokenClear] = useState(false);
  const [settingsOptions, setSettingsOptions] = useState<SettingsResponse["options"]>({
    models: [],
    devices: [],
    precisions: [],
    languages: [],
    language_detection_models: [],
  });
  const [modelFields, setModelFields] = useState<Record<string, string[]>>({});
  const [serviceFields, setServiceFields] = useState<string[]>([]);
  const [managedModels, setManagedModels] = useState<ManagedModel[]>([]);
  const [loadedModelIdentifier, setLoadedModelIdentifier] = useState<string[] | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [liveError, setLiveError] = useState("");
  const [saveBusy, setSaveBusy] = useState(false);
  const [presets, setPresets] = useState<SettingsPreset[]>([]);
  const [presetSelected, setPresetSelected] = useState("");
  const [presetSaveName, setPresetSaveName] = useState("");
  const [presetBusy, setPresetBusy] = useState(false);
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);
  const [benchmarkRepeatInput, setBenchmarkRepeatInput] = useState("1");
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [pipelineMode, setPipelineMode] = useState<AudioProcessMode>("diarization");
  const [pipelineExpectedSpeakers, setPipelineExpectedSpeakers] = useState("");
  const [pipelineKnownSpeakers, setPipelineKnownSpeakers] = useState("[]");
  const [pipelineSpeakerRefinement, setPipelineSpeakerRefinement] = useState<SpeakerRefinementMode>("off");
  const [pipelineUnknownSpeakerAudio, setPipelineUnknownSpeakerAudio] = useState(false);
  const [pipelineApiKey, setPipelineApiKey] = useState("");
  const [pipelineBusy, setPipelineBusy] = useState(false);
  const [pipelineDragActive, setPipelineDragActive] = useState(false);
  const [pipelineError, setPipelineError] = useState("");
  const [pipelineResult, setPipelineResult] = useState<AudioProcessResponse | null>(null);
  const [pipelineWallTimeMs, setPipelineWallTimeMs] = useState<number | null>(null);
  const [pipelineJsonExpanded, setPipelineJsonExpanded] = useState(false);
  const [modelActionId, setModelActionId] = useState<string | null>(null);
  const [modelActionKind, setModelActionKind] = useState<"refresh" | "download" | "delete" | null>(null);
  const [selectingModelId, setSelectingModelId] = useState<string | null>(null);
  const [historyActions, setHistoryActions] = useState<Record<string, HistoryActionKind>>({});
  const [historyActionErrors, setHistoryActionErrors] = useState<Record<string, string>>({});
  const hasDownloadingModels = managedModels.some((model) => model.status === "downloading");
  const isReady = authState === "ready";
  const modelSelectRunning = selectingModelId !== null;
  const activeModelOption = settingsOptions.models.find((option) => option.value === settingsForm.local_model);
  const activeModelFields = activeModelOption ? (modelFields[activeModelOption.backend] ?? []) : [];
  const modelPresets = presets.filter(
    (preset) => preset.model_id == null || preset.model_id === settingsForm.local_model,
  );
  // The saved cache path: listing, download and delete never follow an unsaved (maybe mistyped) path.
  const savedCachePath = settingsBaseline.current.local_model_cache_path;

  // ---- dirty tracking of the settings form ----
  const formKeys = (Object.keys(settingsForm) as Array<keyof AdminSettings>).filter((key) => !NON_FORM_KEYS.has(key));
  const dirtyKeys = formKeys.filter((key) => {
    const a = settingsForm[key];
    const b = settingsBaseline.current[key];
    return !(a === b || (Number.isNaN(a as number) && Number.isNaN(b as number)));
  });
  const invalidKeys = formKeys.filter((key) => typeof settingsForm[key] === "number" && Number.isNaN(settingsForm[key] as number));
  const dirtyCount = dirtyKeys.length + (tokenDraft.trim() ? 1 : 0) + (tokenClear ? 1 : 0);

  const perfLatest = usePerfLatest();
  usePerfFeed((signal) => getPerf(signal), isReady);

  // ---- routing ----
  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function goToTab(next: TabId) {
    if (window.location.hash !== `#${next}`) {
      window.location.hash = next;
    }
    setTab(next);
  }

  function onTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const index = TABS.findIndex((entry) => entry.id === tab);
    let next = -1;
    if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
    if (event.key === "ArrowLeft") next = (index - 1 + TABS.length) % TABS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = TABS.length - 1;
    if (next < 0) return;
    event.preventDefault();
    goToTab(TABS[next].id);
    window.requestAnimationFrame(() => document.getElementById(`tab-${TABS[next].id}`)?.focus());
  }

  // ---- session ----
  function resetDashboardState() {
    clearPerfHistory();
    toast.clear();
    startTransition(() => {
      setStats(null);
      setQueue(null);
      setLiveError("");
      settingsBaseline.current = emptySettings;
      setSettingsForm(emptySettings);
      setSettingsLoaded(false);
      setSettingsError("");
      setTokenDraft("");
      setTokenClear(false);
      setPresets([]);
      setPresetSelected("");
      setPresetSaveName("");
      setManagedModels([]);
      setLoadedModelIdentifier(null);
      setSettingsOptions({ models: [], devices: [], precisions: [], languages: [], language_detection_models: [] });
      setModelFields({});
      setServiceFields([]);
      setApiKeys(null);
      setApiKeysError(null);
      setBenchmarkError("");
      setBenchmarkResult(null);
      setBenchmarkFile(null);
      setPipelineMode("diarization");
      setPipelineExpectedSpeakers("");
      setPipelineKnownSpeakers("[]");
      setPipelineSpeakerRefinement("off");
      setPipelineUnknownSpeakerAudio(false);
      setPipelineApiKey("");
      setPipelineBusy(false);
      setPipelineDragActive(false);
      setPipelineError("");
      setPipelineResult(null);
      setPipelineWallTimeMs(null);
      setPipelineJsonExpanded(false);
      setModelActionId(null);
      setModelActionKind(null);
      setSelectingModelId(null);
      setHistoryActions({});
      setHistoryActionErrors({});
      setPwCurrent("");
      setPwNew("");
      setPwConfirm("");
      setPwError("");
    });
  }

  function expireSession() {
    resetDashboardState();
    setAuthState("login");
    void refreshSetupFlag();
    toast.warn("Session expired. Please log in again.");
  }

  /** Session errors from any admin call: true when handled (then the caller stops). Otherwise a toast. */
  function handleApiError(error: unknown, what: string, quiet = false): boolean {
    if (isAbort(error)) {
      return true;
    }
    const message = errorText(error, "");
    if (message === "unauthorized") {
      expireSession();
      return true;
    }
    if (message === "password_change_required") {
      setAuthState("change");
      return true;
    }
    if (!quiet) {
      toast.error(message ? `${what}: ${message}` : `${what}.`);
    }
    return false;
  }

  async function refreshSetupFlag() {
    try {
      setSetupRequired((await getAuthSetup()).setup_required);
    } catch {
      setSetupRequired(false);
    }
  }

  async function bootstrap() {
    setBootError("");
    setAuthState("loading");
    try {
      const me = await whoami();
      setCurrentUser(me.username);
      setSessionExpiresAt(me.expires_at ?? null);
      setAuthState(me.must_change_password ? "change" : "ready");
    } catch (error) {
      if (errorText(error, "") === "unauthorized") {
        await refreshSetupFlag();
        setAuthState("login");
        return;
      }
      setBootError(errorText(error, "The server did not answer."));
    }
  }

  useEffect(() => {
    void bootstrap();
  }, []);

  // ---- data loading ----
  function describeSelectedModel(payload: SettingsResponse) {
    const selected = payload.options.models.find((option) => option.value === payload.settings.local_model);
    return selected?.label ?? payload.settings.local_model;
  }

  function applySettings(payload: SettingsResponse, keepEdits?: KeepEdits) {
    const next = { ...emptySettings, ...payload.settings };
    const baseline = settingsBaseline.current;
    settingsBaseline.current = next;
    setSettingsForm((current) => mergeServerSettings(current, baseline, next, keepEdits, payload.service_fields));
    setSettingsOptions(payload.options);
    setManagedModels(payload.models ?? []);
    setLoadedModelIdentifier(payload.loaded_model_identifier);
    setModelFields(payload.model_fields ?? {});
    setServiceFields(payload.service_fields ?? []);
    setSettingsLoaded(true);
    setSettingsError("");
  }

  async function loadSettings(keepEdits?: KeepEdits) {
    try {
      const response = await getSettings();
      const modelChanged = keepEdits && settingsLoaded && response.settings.local_model !== settingsBaseline.current.local_model;
      startTransition(() => applySettings(response, keepEdits));
      if (modelChanged) {
        toast.info(
          `${describeSelectedModel(response)} was selected in another session. The form now shows its settings; only service-wide edits were kept.`,
        );
      }
    } catch (error) {
      if (!handleApiError(error, "Could not load the settings", true)) {
        setSettingsError(errorText(error, "Could not load the settings."));
      }
    }
  }

  async function reloadPresets() {
    try {
      const response = await listPresets();
      setPresets(response.presets);
    } catch (error) {
      handleApiError(error, "Could not load the presets");
    }
  }

  async function refreshOperationalData(signal?: AbortSignal) {
    if (!isReady) {
      return;
    }
    const [statsResult, queueResult] = await Promise.allSettled([getStats(signal), getQueue(signal)]);
    if (signal?.aborted) {
      return;
    }
    for (const result of [statsResult, queueResult]) {
      if (result.status === "rejected" && handleApiError(result.reason, "", true)) {
        return;
      }
    }
    startTransition(() => {
      if (statsResult.status === "fulfilled") setStats(statsResult.value);
      if (queueResult.status === "fulfilled") setQueue(queueResult.value);
      const failed = [statsResult, queueResult].find((result) => result.status === "rejected") as PromiseRejectedResult | undefined;
      setLiveError(failed ? errorText(failed.reason, "Live data unavailable.") : "");
    });
  }

  async function refreshApiKeys(signal?: AbortSignal) {
    try {
      const response = await listApiKeys(signal);
      if (!signal?.aborted) {
        setApiKeys(response.keys);
        setApiKeysError(null);
      }
    } catch (error) {
      if (!handleApiError(error, "Could not load the API keys", true)) {
        setApiKeysError(errorText(error, "Could not load the API keys."));
      }
    }
  }

  useEffect(() => {
    if (!isReady) {
      return;
    }
    void loadSettings();
    void reloadPresets();
    void refreshApiKeys();
  }, [authState]);

  useVisiblePolling((signal) => refreshOperationalData(signal), OPERATIONAL_POLL_MS, isReady);
  useVisiblePolling((signal) => refreshApiKeys(signal), API_KEYS_POLL_MS, isReady && tab === "api-keys");
  useVisiblePolling(
    async (signal) => {
      try {
        const response = await getModels(savedCachePath, signal);
        if (!signal.aborted) setManagedModels(response.models ?? []);
      } catch (error) {
        handleApiError(error, "Could not refresh the model list", true);
      }
    },
    MODEL_STATUS_POLL_MS,
    isReady && hasDownloadingModels,
  );

  // ---- auth actions ----
  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setLoginPassword("");
      toast.clear();
      const fresh = await whoami().catch(() => null);
      setSessionExpiresAt(fresh?.expires_at ?? null);
      setAuthState(me.must_change_password ? "change" : "ready");
    } catch (error) {
      setAuthError(errorText(error, "") === "unauthorized" ? "Invalid username or password." : errorText(error, "Login failed."));
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleChangePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPwError("");
    if (pwNew.length < MIN_PASSWORD_LENGTH) {
      setPwError(`The new password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
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
      const fresh = await whoami().catch(() => null);
      setSessionExpiresAt(fresh?.expires_at ?? null);
      setSetupRequired(false);
      toast.ok("Password changed.");
      setAuthState("ready");
    } catch (error) {
      if (errorText(error, "") === "unauthorized") {
        expireSession();
        return;
      }
      setPwError(errorText(error, "The password could not be changed."));
    } finally {
      setPwBusy(false);
    }
  }

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // ignore: clear locally regardless
    }
    resetDashboardState();
    setLoginPassword("");
    void refreshSetupFlag();
    setAuthState("login");
  }

  // ---- presets ----
  function handleLoadPreset() {
    const preset = modelPresets.find((entry) => entry.name === presetSelected);
    if (!preset) {
      return;
    }
    // Into the form only: Save changes applies it through the normal validated update.
    setSettingsForm((current) => ({ ...current, ...preset.values }));
    toast.info(`Preset "${preset.name}" loaded into the form. Review it and save the changes to apply it.`);
    if (preset.note) {
      toast.warn(preset.note);
    }
  }

  async function handleSavePreset() {
    const name = presetSaveName.trim();
    if (!name) {
      return;
    }
    // Presets of other models are not listed here, saving would silently take that preset over.
    if (presets.some((entry) => entry.name === name && !modelPresets.includes(entry))) {
      toast.error(`The name "${name}" is already used by a preset of another model.`);
      return;
    }
    if (modelPresets.some((entry) => entry.name === name && entry.built_in)) {
      toast.error(`"${name}" is a built-in preset and cannot be overwritten.`);
      return;
    }
    if (modelPresets.some((entry) => entry.name === name) && !window.confirm(`Overwrite the saved preset "${name}"?`)) {
      return;
    }
    setPresetBusy(true);
    try {
      const response = await savePreset(name, settingsForm);
      await reloadPresets();
      setPresetSelected(response.preset.name);
      setPresetSaveName("");
      toast.ok(`Preset "${response.preset.name}" saved from the current form.`);
    } catch (error) {
      handleApiError(error, "Could not save the preset");
    } finally {
      setPresetBusy(false);
    }
  }

  async function handleDeletePreset() {
    const preset = modelPresets.find((entry) => entry.name === presetSelected);
    if (!preset || preset.built_in) {
      return;
    }
    if (!window.confirm(`Delete preset "${preset.name}"?\n\nThe saved values are removed; the current settings stay.`)) {
      return;
    }
    setPresetBusy(true);
    try {
      await deletePreset(preset.name);
      setPresetSelected("");
      await reloadPresets();
      toast.ok(`Preset "${preset.name}" deleted.`);
    } catch (error) {
      handleApiError(error, "Could not delete the preset");
    } finally {
      setPresetBusy(false);
    }
  }

  useEffect(() => {
    setPresetSelected("");
  }, [settingsForm.local_model]);

  // ---- settings ----
  function describeSaveResult(response: SettingsSaveResponse) {
    if (!response.model_reloaded) {
      return "Settings saved.";
    }
    if (response.model_loaded) {
      return "Settings saved. The model was reloaded.";
    }
    return `Settings saved, but the model reload failed: ${response.load_error ?? "no cause reported."}`;
  }

  async function handleSaveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (invalidKeys.length > 0) {
      toast.error("Some fields are empty or out of range. Fix them before saving.");
      return;
    }
    setSaveBusy(true);
    try {
      const response = await saveSettings(settingsForm, { draft: tokenDraft, clear: tokenClear });
      startTransition(() => {
        applySettings(response);
        setTokenDraft("");
        setTokenClear(false);
      });
      if (response.model_reloaded && !response.model_loaded) {
        toast.error(describeSaveResult(response));
      } else {
        toast.ok(describeSaveResult(response));
      }
      await refreshOperationalData();
    } catch (error) {
      if (errorText(error, "") === ACTIVE_MODEL_CHANGED) {
        await reloadFormForActiveModel();
        return;
      }
      handleApiError(error, "Could not save the settings");
    } finally {
      setSaveBusy(false);
    }
  }

  function handleDiscardSettings() {
    setSettingsForm(settingsBaseline.current);
    setTokenDraft("");
    setTokenClear(false);
  }

  async function handleReloadSettings() {
    if (dirtyCount > 0 && !window.confirm("Reload the settings from the server?\n\nUnsaved changes are discarded.")) {
      return;
    }
    setTokenDraft("");
    setTokenClear(false);
    await loadSettings();
    await reloadPresets();
  }

  async function reloadFormForActiveModel() {
    try {
      const response = await getSettings();
      startTransition(() => {
        // The model settings in the form belong to the old model, service-wide edits are not tied to a model.
        applySettings(response, "service");
      });
      toast.error(
        `Nothing was saved: ${describeSelectedModel(response)} was selected in another session. The form now shows its settings; review them and save again.`,
      );
    } catch (error) {
      handleApiError(error, "Could not reload the settings");
    }
  }

  function updateSetting<K extends keyof AdminSettings>(key: K, value: AdminSettings[K]) {
    setSettingsForm((current) => ({ ...current, [key]: value }));
  }

  // ---- models ----
  async function handleRefreshModels() {
    setModelActionId("__refresh__");
    setModelActionKind("refresh");
    try {
      const response = await getModels(savedCachePath);
      setManagedModels(response.models ?? []);
    } catch (error) {
      handleApiError(error, "Could not refresh the model list");
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  async function handleDownloadModel(model: ManagedModel) {
    if (model.status === "ready" && !window.confirm(`Download "${model.label}" again?\n\nThe cached files are fetched once more.`)) {
      return;
    }
    setModelActionId(model.id);
    setModelActionKind("download");
    try {
      // A token typed but not yet saved is sent along; otherwise the server uses the saved one.
      const response = await downloadModel(model.id, savedCachePath, tokenDraft.trim() || undefined);
      setManagedModels(response.models ?? []);
      toast.info(`Download of ${model.label} started.`);
    } catch (error) {
      handleApiError(error, `Could not download ${model.label}`);
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  async function handleSelectModel(model: ManagedModel) {
    // Retrying the active model changes no setting, so the form keeps every edit.
    const hasUnsavedModelEdits = !model.active && dirtyKeys.some((key) => !serviceFields.includes(key));
    if (
      hasUnsavedModelEdits
      && !window.confirm(`Load "${model.label}"?\n\nThe unsaved model settings in the Settings tab are discarded.`)
    ) {
      return;
    }

    setSelectingModelId(model.id);
    try {
      const response = await selectModel(model.id);
      startTransition(() => {
        // The target's own values replace the model settings, service-wide edits are not tied to a model.
        applySettings(response, model.active ? "all" : "service");
      });
      toast.ok(`${model.label} is loaded.`);
      await refreshOperationalData();
    } catch (error) {
      if (handleApiError(error, `Could not load ${model.label}`)) {
        return;
      }
      // On a failed load the server restored the old selection, the picker has to show what really runs now.
      try {
        applySettings(await getSettings(), "all");
      } catch {
        // The selection error above is the one the admin needs.
      }
    } finally {
      setSelectingModelId(null);
    }
  }

  async function handleDeleteModel(model: ManagedModel) {
    if (
      !window.confirm(
        `Delete cached model "${model.id}"?\n\nIts files are removed from the cache; it must be downloaded again before it can be loaded.`,
      )
    ) {
      return;
    }
    setModelActionId(model.id);
    setModelActionKind("delete");
    try {
      const response = await deleteModel(model.id, savedCachePath);
      setManagedModels(response.models ?? []);
      toast.ok(
        response.removed ? `Cached files of ${model.label} removed.` : `No cached files of ${model.label} were found.`,
      );
    } catch (error) {
      handleApiError(error, `Could not delete ${model.label}`);
    } finally {
      setModelActionId(null);
      setModelActionKind(null);
    }
  }

  // ---- history ----
  function beginHistoryAction(historyId: string, action: HistoryActionKind) {
    setHistoryActions((current) => ({ ...current, [historyId]: action }));
    setHistoryActionErrors((current) => {
      const next = { ...current };
      delete next[historyId];
      return next;
    });
  }

  function finishHistoryAction(historyId: string) {
    setHistoryActions((current) => {
      const next = { ...current };
      delete next[historyId];
      return next;
    });
  }

  function reportHistoryActionError(historyId: string, error: unknown, fallback: string) {
    if (handleApiError(error, fallback, true)) {
      return;
    }
    setHistoryActionErrors((current) => ({ ...current, [historyId]: errorText(error, fallback) }));
  }

  async function handleRetryHistoryEntry(entry: HistoryEntry) {
    if (!entry.history_id || entry.debug_audio?.status !== "available" || !entry.retry_mode) {
      return;
    }
    beginHistoryAction(entry.history_id, "retry");
    try {
      await retryHistoryAudio(entry.history_id);
      toast.ok("Retry finished; it is listed as a new history entry.");
      await refreshOperationalData();
    } catch (error) {
      reportHistoryActionError(entry.history_id, error, "The retained audio could not be retried.");
    } finally {
      finishHistoryAction(entry.history_id);
    }
  }

  async function handleDownloadHistoryEntry(entry: HistoryEntry) {
    if (!entry.history_id || entry.debug_audio?.status !== "available") {
      return;
    }
    beginHistoryAction(entry.history_id, "download");
    try {
      const download = await downloadHistoryAudio(entry.history_id, entry.debug_audio.filename);
      const objectUrl = URL.createObjectURL(download.blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = download.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    } catch (error) {
      reportHistoryActionError(entry.history_id, error, "The retained audio could not be downloaded.");
    } finally {
      finishHistoryAction(entry.history_id);
    }
  }

  // ---- playground + benchmark ----
  function selectPipelineFile(file: File | null) {
    setBenchmarkFile(file);
    setPipelineError("");
    setPipelineResult(null);
    setPipelineWallTimeMs(null);
    setPipelineJsonExpanded(false);
    setBenchmarkError("");
    setBenchmarkResult(null);
  }

  async function handleRunPipeline(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!benchmarkFile) {
      setPipelineError("Choose or drop an audio or video file first.");
      return;
    }

    let request: AudioProcessRequest = { schema_version: "2.0", mode: pipelineMode };
    try {
      if (pipelineMode === "diarization") {
        const knownSpeakers = parseKnownSpeakerProfiles(pipelineKnownSpeakers);
        const expectedRaw = pipelineExpectedSpeakers.trim();
        let expectedSpeakers: number | undefined;
        if (expectedRaw) {
          if (!/^\d+$/.test(expectedRaw)) {
            throw new Error("Expected speakers must be a whole number from 1 to 64.");
          }
          expectedSpeakers = Number(expectedRaw);
          if (expectedSpeakers < 1 || expectedSpeakers > 64) {
            throw new Error("Expected speakers must be between 1 and 64.");
          }
          if (knownSpeakers.length > expectedSpeakers) {
            throw new Error("Expected speakers cannot be lower than the number of known speakers.");
          }
        }
        request = {
          ...request,
          diarization: {
            ...(expectedSpeakers === undefined ? {} : { expected_speakers: expectedSpeakers }),
            known_speakers: knownSpeakers,
            speaker_refinement: pipelineSpeakerRefinement,
            ...(pipelineUnknownSpeakerAudio ? { unknown_speaker_audio: true } : {}),
          },
        };
      }
      if (new TextEncoder().encode(JSON.stringify(request)).byteLength > 16 * 1024 * 1024) {
        throw new Error("The request JSON must not exceed 16 MB.");
      }
    } catch (error) {
      setPipelineError(errorText(error, "The request is invalid."));
      return;
    }

    setPipelineBusy(true);
    setPipelineError("");
    setPipelineResult(null);
    setPipelineWallTimeMs(null);
    setPipelineJsonExpanded(false);
    const startedAt = performance.now();
    try {
      const response = await processAudioV2(benchmarkFile, request, pipelineApiKey);
      const wallTimeMs = Math.round(performance.now() - startedAt);
      startTransition(() => {
        setPipelineResult(response);
        setPipelineWallTimeMs(wallTimeMs);
      });
      toast.ok(`Finished with status ${response.status} in ${formatMs(response.timings_ms.total)} server time.`);
      await refreshOperationalData();
    } catch (error) {
      setPipelineWallTimeMs(Math.round(performance.now() - startedAt));
      if (handleApiError(error, "", true)) {
        return;
      }
      setPipelineError(errorText(error, "The request failed."));
    } finally {
      setPipelineBusy(false);
    }
  }

  const benchmarkRepeatCount = /^\d+$/.test(benchmarkRepeatInput) ? Number(benchmarkRepeatInput) : Number.NaN;
  const benchmarkRepeatInvalid = !(benchmarkRepeatCount >= 1 && benchmarkRepeatCount <= 64);

  async function handleRunBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!benchmarkFile) {
      setBenchmarkError("Choose an audio or video file first.");
      return;
    }
    if (benchmarkRepeatInvalid) {
      setBenchmarkError("Repeats must be a whole number from 1 to 64.");
      return;
    }
    setBenchmarkBusy(true);
    setBenchmarkError("");
    setBenchmarkResult(null);
    try {
      const result = await runBenchmark(benchmarkFile, benchmarkRepeatCount);
      setBenchmarkResult(result);
      toast.ok(`Benchmark finished: ${formatValue(result.repeat_count)} run(s) in ${formatMs(result.total_wall_time_ms)}.`);
      await refreshOperationalData();
    } catch (error) {
      if (handleApiError(error, "", true)) {
        return;
      }
      setBenchmarkError(errorText(error, "The benchmark failed."));
    } finally {
      setBenchmarkBusy(false);
    }
  }

  // =========================================================================================== render: auth states

  if (authState === "loading") {
    return (
      <div className="shell login-shell">
        <ToastHost />
        {bootError ? (
          <section className="panel login-panel">
            <div className="banner danger" role="alert">
              <span>Could not reach the server: {bootError}</span>
              <button type="button" className="secondary-button small-button" onClick={() => void bootstrap()}>
                Retry
              </button>
            </div>
          </section>
        ) : (
          <div className="state"><span className="spinner" />Loading…</div>
        )}
      </div>
    );
  }

  if (authState === "login") {
    return (
      <div className="shell login-shell">
        <ToastHost />
        <main className="panel login-panel">
          <div className="titles">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1>G3 Whisper</h1>
          </div>
          <form className="form-stack" onSubmit={handleLogin}>
            <label>
              <span>Username</span>
              <input value={loginUsername} autoComplete="username" onChange={(event) => setLoginUsername(event.target.value)} />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                value={loginPassword}
                autoComplete="current-password"
                onChange={(event) => setLoginPassword(event.target.value)}
              />
            </label>
            {authError && <div className="banner danger" role="alert">{authError}</div>}
            <div className="form-actions">
              <button type="submit" disabled={!loginUsername.trim() || !loginPassword || authBusy} aria-busy={authBusy || undefined}>
                {authBusy ? "Logging in…" : "Log in"}
              </button>
            </div>
            {setupRequired && (
              <p className="hint">First start: log in with admin / admin. You must choose a new password right after.</p>
            )}
          </form>
        </main>
      </div>
    );
  }

  const passwordForm = (submitLabel: string, extra?: ReactNode) => (
    <form className="form-stack" onSubmit={handleChangePassword}>
      <label>
        <span>Current password</span>
        <input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} />
      </label>
      <label>
        <span>New password</span>
        <input
          type="password"
          value={pwNew}
          autoComplete="new-password"
          minLength={MIN_PASSWORD_LENGTH}
          onChange={(event) => setPwNew(event.target.value)}
        />
        <span className="hint">At least {MIN_PASSWORD_LENGTH} characters.</span>
      </label>
      <label>
        <span>Confirm new password</span>
        <input type="password" value={pwConfirm} autoComplete="new-password" onChange={(event) => setPwConfirm(event.target.value)} />
      </label>
      {pwError && <div className="banner danger" role="alert">{pwError}</div>}
      <div className="form-actions button-row">
        <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm} aria-busy={pwBusy || undefined}>
          {pwBusy ? "Saving…" : submitLabel}
        </button>
        {extra}
      </div>
    </form>
  );

  if (authState === "change") {
    return (
      <div className="shell login-shell">
        <ToastHost />
        <main className="panel login-panel">
          <div className="titles">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1>Set a new password</h1>
            <p className="dim">
              You are logged in as <strong>{currentUser || "admin"}</strong>. Choose a new password to continue.
            </p>
          </div>
          {passwordForm(
            "Change password",
            <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Log out</button>,
          )}
        </main>
      </div>
    );
  }

  // =========================================================================================== render: dashboard

  const history = stats?.history ?? [];
  const recentBatches = (queue?.recent_batches ?? []) as BatchEntry[];
  const recentJobs = (queue?.recent_jobs ?? []) as JobEntry[];
  const queueJobs = queue?.jobs ?? [];
  const queuedChunks = (queue?.queue_size ?? 0) + (queue?.pending_buffer_size ?? 0);
  const selectedPreset = modelPresets.find((preset) => preset.name === presetSelected);
  const configuredLanguageLabel =
    activeModelOption && !activeModelFields.includes("transcription_language")
      ? "Not used by this model"
      : resolveOptionLabel(settingsOptions.languages, settingsBaseline.current.transcription_language, "Auto");
  // The server identifies a loaded model as (model id, device, cache path, precision).
  const [loadedModelId, loadedDevice, , loadedPrecision] = loadedModelIdentifier ?? [];
  const loadedModelOption = settingsOptions.models.find((option) => option.value === loadedModelId);
  const loadedModelLabel = loadedModelId ? (loadedModelOption?.label ?? loadedModelId) : "None";
  const loadedModelUsesPrecision = loadedModelOption
    && modelFields[loadedModelOption.backend]?.includes("local_model_precision");
  const loadedModelSub = loadedModelId
    ? [
        resolveOptionLabel(settingsOptions.devices, loadedDevice, "Auto"),
        loadedModelUsesPrecision ? resolveOptionLabel(settingsOptions.precisions, loadedPrecision) : null,
      ].filter(Boolean).join(" · ")
    : "Load a model in the Models tab.";

  const serviceGpus = (perfLatest?.gpus ?? []).filter((gpu) => gpu.used_by_service);
  const gpuChip = perfLatest ? (serviceGpus.length ? serviceGpus.map(gpuTitle).join(", ") : "CPU") : DASH;

  let stateBadge: { text: string; tone: string; title?: string };
  if (modelSelectRunning) {
    stateBadge = { text: "Loading", tone: "warn" };
  } else if (queue && !queue.worker_running) {
    stateBadge = { text: "Error", tone: "danger", title: queue.last_error ?? "The batch worker is not running." };
  } else if (!settingsLoaded) {
    stateBadge = { text: DASH, tone: "plain" };
  } else if (loadedModelId) {
    stateBadge = { text: "Ready", tone: "ok" };
  } else {
    stateBadge = { text: "Idle · model unloaded", tone: "plain" };
  }

  const keyCount = apiKeys?.length ?? 0;
  const apiKeyRows = (apiKeys ?? []).map((key) => ({
    id: key.id,
    alias: key.alias,
    hint: key.hint ?? null,
    created_at: key.created_at,
    last_used_at: key.usage.last_used_at,
    requests: key.usage.request_count,
    usage: key.usage.total_seconds_processed,
  }));
  const baseUrl = window.location.origin;

  const tokenSet = settingsBaseline.current.huggingface_token_set === true;
  const tokenPlaceholder = tokenClear
    ? "Will be removed on save"
    : tokenSet
      ? `Saved: ${settingsBaseline.current.huggingface_token} · leave empty to keep`
      : "hf_… (optional)";

  const queueTable = (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Request</th>
            <th>State</th>
            <th className="num">Chunks</th>
            <th className="num">Batches</th>
            <th className="num">Queue wait</th>
          </tr>
        </thead>
        <tbody>
          {queueJobs.length === 0 && (
            <tr>
              <td colSpan={5} className="empty">No request is scheduled right now.</td>
            </tr>
          )}
          {queueJobs.map((job) => (
            <tr key={job.request_id}>
              <td><span className="mono">{job.request_id}</span></td>
              <td>
                <span className={`badge ${job.state === "active" ? "ok" : "plain"}`}>{job.state}</span>
                {job.is_long ? <span className="hint"> long</span> : null}
              </td>
              <td className="num">
                {job.completed_chunks} / {job.total_chunks}
                {job.inflight_chunks > 0 ? <span className="hint"> +{job.inflight_chunks} in batch</span> : null}
              </td>
              <td className="num">{job.batch_count}</td>
              <td className="num">{formatMs(job.queue_wait_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className="shell">
      <ToastHost />
      <header className="app-header">
        <div className="titles">
          <p className="eyebrow">G3 GPU service · Admin</p>
          <h1>G3 Whisper</h1>
          <div className="status-strip">
            <span className={`badge ${stateBadge.tone}`} title={stateBadge.title}>{stateBadge.text}</span>
            <Chip label="Model" title={loadedModelId ?? undefined}>{loadedModelLabel}</Chip>
            <Chip label="GPU">{gpuChip}</Chip>
            <Chip label="API" title="Show the API keys" onClick={() => goToTab("api-keys")}>
              {apiKeys === null ? DASH : keyCount > 0 ? (
                <span className="badge ok">Protected</span>
              ) : (
                <span className="badge warn">Open</span>
              )}
            </Chip>
          </div>
        </div>
        <div className="actions">
          <a className="button-link ghost" href="/docs" target="_blank" rel="noreferrer">API docs</a>
          <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Log out</button>
        </div>
      </header>

      <nav className="tab-bar" role="tablist" aria-label="Sections">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            id={`tab-${entry.id}`}
            type="button"
            role="tab"
            className="tab-button"
            aria-selected={tab === entry.id}
            aria-controls={`panel-${entry.id}`}
            tabIndex={tab === entry.id ? 0 : -1}
            onClick={() => goToTab(entry.id)}
            onKeyDown={onTabKeyDown}
          >
            {entry.label}
            {entry.id === "settings" && dirtyCount > 0 && <span className="tab-dot" aria-label="unsaved changes" />}
          </button>
        ))}
      </nav>

      <main role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="tab-panel">
        {/* ------------------------------------------------------------------------------ Dashboard */}
        {tab === "dashboard" && (
          <>
            <div className="grid stat-row">
              <Card label="Model" value={loadedModelLabel} title={loadedModelId ?? undefined} sub={loadedModelSub} />
              <Card
                label="State"
                value={<span className={`badge ${stateBadge.tone}`} title={stateBadge.title}>{stateBadge.text}</span>}
                sub={queue ? `Queue: ${formatValue(queuedChunks)} chunk${queuedChunks === 1 ? "" : "s"}` : DASH}
              />
              <Card
                label="Requests"
                value={stats ? formatValue(stats.summary.total_requests) : DASH}
                sub={stats?.summary.avg_total_duration_ms != null ? `Mean total ${formatMs(stats.summary.avg_total_duration_ms)}` : "Since start"}
              />
              <Card
                label="Errors"
                value={stats ? formatValue(stats.summary.failed_batches ?? 0) : DASH}
                sub="Failed batches since start"
              />
            </div>

            {liveError && <div className="banner danger" role="status">Live data unavailable: {liveError}</div>}

            <PerfPanel />

            <section className="panel" aria-labelledby="live-title">
              <div className="panel-head">
                <h2 id="live-title">Live queue</h2>
                <span className="hint">Every 5 s · ASR work is batched per model; long recordings share batches round-robin.</span>
              </div>
              <div className="grid grid-auto">
                <Card label="Active jobs" value={formatValue(queue?.active_jobs)} />
                <Card label="Waiting long jobs" value={formatValue(queue?.waiting_jobs)} />
                <Card label="Queued chunks" value={formatValue(queue ? queuedChunks : null)} />
                <Card
                  label="Active batch"
                  value={queue ? `${queue.active_batch_size ?? 0} / ${settingsBaseline.current.batch_max_segments || DASH}` : DASH}
                  sub={queue?.active_batch_fast_path ? "Fast path" : undefined}
                />
                <Card label="Mean ASR" value={formatMs(stats?.summary.avg_transcription_duration_ms)} />
                <Card label="Last batch" value={formatMs(queue?.last_batch_duration_ms ?? null)} />
              </div>

              <h3>Current batch</h3>
              {queue?.active_batch_id ? (
                <dl className="kv-grid">
                  <div><dt>Batch</dt><dd className="mono">{queue.active_batch_id}</dd></div>
                  <div><dt>Segments</dt><dd>{queue.active_batch_size ?? 0}</dd></div>
                  <div><dt>Audio</dt><dd>{formatFixed(queue.active_batch_audio_seconds, 1, " s")}</dd></div>
                  <div><dt>Started</dt><dd>{formatDateTime(queue.active_batch_started_at, true)}</dd></div>
                </dl>
              ) : (
                <p className="dim">No batch is running.</p>
              )}
              {queue?.last_error && <div className="banner danger">Last worker error: {queue.last_error}</div>}

              <h3>Scheduled jobs</h3>
              {queueTable}

              <h3>Recent batches</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Batch</th>
                      <th className="num">Segments</th>
                      <th className="num">Requests</th>
                      <th className="num">Audio</th>
                      <th className="num">Duration</th>
                      <th className="num">Realtime</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentBatches.length === 0 && (
                      <tr>
                        <td colSpan={8} className="empty">No batch has run yet. Send a request or run the benchmark.</td>
                      </tr>
                    )}
                    {recentBatches.map((entry, index) => (
                      <tr key={`${entry.batch_id ?? "batch"}-${index}`}>
                        <td className="nowrap">{formatDateTime(entry.timestamp, true)}</td>
                        <td><span className="mono">{entry.batch_id ?? DASH}</span></td>
                        <td className="num">{entry.batch_size ?? 0}</td>
                        <td className="num">
                          {formatValue(entry.unique_request_count)}
                          {entry.fast_path ? <span className="hint"> fast</span> : null}
                        </td>
                        <td className="num">{formatFixed(entry.audio_seconds, 1, " s")}</td>
                        <td className="num">{formatMs(entry.duration_ms)}</td>
                        <td className="num">
                          {entry.status === "ok" && entry.audio_seconds && entry.duration_ms
                            ? formatRealtimeFactor(entry.audio_seconds / (entry.duration_ms / 1000))
                            : DASH}
                        </td>
                        <td>
                          <span className={`badge ${entry.status === "ok" ? "ok" : "danger"}`} title={entry.error}>
                            {entry.status === "ok" ? "OK" : entry.status ?? DASH}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3>Recent jobs</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Finished</th>
                      <th>Request</th>
                      <th className="num">Chunks</th>
                      <th className="num">Batches</th>
                      <th className="num">Queue wait</th>
                      <th className="num">Total</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentJobs.length === 0 && (
                      <tr>
                        <td colSpan={7} className="empty">No job has finished yet.</td>
                      </tr>
                    )}
                    {recentJobs.map((entry, index) => (
                      <tr key={`${entry.request_id ?? "job"}-${index}`}>
                        <td className="nowrap">{formatDateTime(entry.finished_at, true)}</td>
                        <td>
                          <span className="mono">{entry.request_id ?? DASH}</span>
                          {entry.is_long ? <span className="hint"> long</span> : null}
                        </td>
                        <td className="num">{entry.total_chunks ?? 0}</td>
                        <td className="num">{entry.batch_count ?? 0}</td>
                        <td className="num">{formatMs(entry.queue_wait_ms ?? null)}</td>
                        <td className="num">{formatMs(entry.duration_ms)}</td>
                        <td>
                          <span className={`badge ${entry.status === "completed" || entry.status === "ok" ? "ok" : entry.status === "cancelled" ? "plain" : "danger"}`}>
                            {entry.status ?? DASH}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="panel" aria-labelledby="system-title">
              <div className="panel-head">
                <h2 id="system-title">System</h2>
              </div>
              <dl className="kv-grid">
                <div><dt>Batch worker</dt><dd>{queue ? (queue.worker_running ? "Running" : "Stopped") : DASH}</dd></div>
                <div><dt>Processed batches</dt><dd>{formatValue(queue?.total_batches_processed)}</dd></div>
                <div><dt>Processed segments</dt><dd>{formatValue(queue?.total_segments_processed)}</dd></div>
                <div><dt>Configured device</dt><dd>{resolveOptionLabel(settingsOptions.devices, settingsBaseline.current.local_gpu_device, "Auto")}</dd></div>
                <div><dt>Language</dt><dd>{configuredLanguageLabel}</dd></div>
                <div><dt>Model cache</dt><dd className="mono">{savedCachePath || "Default Hugging Face cache"}</dd></div>
              </dl>
            </section>
          </>
        )}

        {/* ------------------------------------------------------------------------------ Playground */}
        {tab === "playground" && (
          <section className="panel" aria-labelledby="playground-title">
            <div className="panel-head">
              <h2 id="playground-title">Playground</h2>
              <span className="badge accent plain">{PIPELINE_MODES.find((entry) => entry.value === pipelineMode)?.label}</span>
              <p className="hint">
                Drop one file, choose a mode, and inspect the complete response of <code>POST /v2/audio/process</code>.
                Your login is used unless you type a client key.
              </p>
            </div>

            <form className="form-stack" onSubmit={handleRunPipeline}>
              <div
                className={`drop-zone ${pipelineDragActive ? "drag-active" : ""} ${benchmarkFile ? "has-file" : ""}`.trim()}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setPipelineDragActive(true);
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "copy";
                  setPipelineDragActive(true);
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setPipelineDragActive(false);
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setPipelineDragActive(false);
                  selectPipelineFile(event.dataTransfer.files?.[0] ?? null);
                }}
              >
                <input
                  id="pipeline-file"
                  className="drop-input"
                  type="file"
                  accept="audio/*,video/*"
                  onChange={(event) => selectPipelineFile(event.target.files?.[0] ?? null)}
                />
                <label htmlFor="pipeline-file" className="drop-label">
                  <strong>{benchmarkFile ? benchmarkFile.name : "Drop audio or video here"}</strong>
                  <span className="hint">
                    {benchmarkFile ? `${formatBytes(benchmarkFile.size)} · click to replace` : "or click to choose a file"}
                  </span>
                </label>
              </div>

              <div className="form-grid">
                <label>
                  <span>Mode</span>
                  <select
                    value={pipelineMode}
                    onChange={(event) => {
                      setPipelineMode(event.target.value as AudioProcessMode);
                      setPipelineError("");
                      setPipelineResult(null);
                    }}
                  >
                    {PIPELINE_MODES.map((mode) => (
                      <option key={mode.value} value={mode.value}>{mode.label}</option>
                    ))}
                  </select>
                  <span className="hint">{PIPELINE_MODES.find((entry) => entry.value === pipelineMode)?.description}</span>
                </label>

                <label>
                  <span>Client API key (optional)</span>
                  <input
                    type="password"
                    autoComplete="off"
                    value={pipelineApiKey}
                    onChange={(event) => setPipelineApiKey(event.target.value)}
                    placeholder="Empty: use your login"
                  />
                  <span className="hint">Sent as Authorization: Bearer. A wrong key fails even while you are logged in.</span>
                </label>

                {pipelineMode === "diarization" && (
                  <>
                    <label>
                      <span>Expected speakers (optional)</span>
                      <input
                        type="number"
                        min={1}
                        max={64}
                        step={1}
                        value={pipelineExpectedSpeakers}
                        onChange={(event) => setPipelineExpectedSpeakers(event.target.value)}
                        placeholder="Automatic"
                      />
                    </label>

                    <label>
                      <span>Speaker refinement</span>
                      <select
                        value={pipelineSpeakerRefinement}
                        onChange={(event) => setPipelineSpeakerRefinement(event.target.value as SpeakerRefinementMode)}
                      >
                        <option value="off">Off (existing behavior)</option>
                        <option value="shadow">Shadow (diagnostics only)</option>
                        <option value="conservative">Conservative (apply safe changes)</option>
                      </select>
                      <span className="hint">
                        Shadow reports proposed label corrections without changing the transcript. Conservative applies
                        only changes that pass all safety checks and otherwise rolls back the whole refinement pass.
                      </span>
                    </label>

                    <CheckField
                      label="Include playable MP3 samples for unknown and unresolved speakers"
                      checked={pipelineUnknownSpeakerAudio}
                      onChange={setPipelineUnknownSpeakerAudio}
                    />

                    <label className="span-2">
                      <span>Known speakers (JSON array)</span>
                      <textarea
                        className="mono"
                        value={pipelineKnownSpeakers}
                        onChange={(event) => setPipelineKnownSpeakers(event.target.value)}
                        spellCheck={false}
                        placeholder={'[{"id":"person-17","embeddings":[[/* 192 values */]]}]'}
                      />
                      <span className="hint">
                        Each id must be unique. Every profile needs one or more finite, non-zero 192-D ReDimNet2
                        embeddings. Leave the array empty when no known profiles are available.
                      </span>
                    </label>
                  </>
                )}
              </div>

              {pipelineError && <div className="banner danger" role="alert">{pipelineError}</div>}

              <div className="form-actions button-row">
                <button type="submit" disabled={pipelineBusy || !benchmarkFile} aria-busy={pipelineBusy || undefined}>
                  {pipelineBusy ? "Processing…" : "Run"}
                </button>
              </div>
            </form>

            {pipelineResult && (
              <div className="result-stack">
                <div className="grid grid-auto">
                  <Card
                    label="Status"
                    value={<span className={`badge ${pipelineResult.status === "completed" ? "ok" : "warn"}`}>{pipelineResult.status}</span>}
                  />
                  <Card label="Mode" value={PIPELINE_MODES.find((entry) => entry.value === pipelineResult.mode)?.label ?? pipelineResult.mode} />
                  <Card label="Server total" value={formatMs(pipelineResult.timings_ms.total)} />
                  <Card label="Client wall time" value={formatMs(pipelineWallTimeMs)} />
                  <Card label="Audio" value={formatFixed(pipelineResult.audio.duration_ms / 1000, 1, " s")} />
                  <Card
                    label="Embedding"
                    value={resolveEmbeddingDimension(pipelineResult) === null ? DASH : `${resolveEmbeddingDimension(pipelineResult)}-D`}
                  />
                  <Card label="Warnings" value={pipelineResult.warnings.length} />
                </div>

                <div>
                  <h3>Phase timings</h3>
                  <dl className="kv-grid">
                    {Object.entries(pipelineResult.timings_ms).map(([phase, duration]) => (
                      <div key={phase}>
                        <dt>{phase.replace(/_/g, " ")}</dt>
                        <dd>{formatMs(duration)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>

                <div>
                  <h3>Models</h3>
                  {Object.keys(pipelineResult.models).length === 0 ? (
                    <p className="dim">No model metadata returned.</p>
                  ) : (
                    <dl className="kv-grid">
                      {Object.entries(pipelineResult.models).map(([role, model]) => (
                        <div key={role}>
                          <dt>{role}</dt>
                          <dd className="mono">{describeModel(model)}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </div>

                {pipelineResult.result.speaker_counts && (
                  <div>
                    <h3>Speakers</h3>
                    <dl className="kv-grid">
                      {Object.entries(pipelineResult.result.speaker_counts).map(([label, value]) => (
                        <div key={label}>
                          <dt>{label.replace(/_/g, " ")}</dt>
                          <dd>{value ?? "auto"}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                )}

                {[
                  ...(pipelineResult.result.unknown_speakers ?? []),
                  ...(pipelineResult.result.unresolved_speakers ?? []),
                ].some((speaker) => speaker.audio) && (
                  <div>
                    <h3>Unknown-speaker MP3 samples</h3>
                    <div className="speaker-audio-grid">
                      {[
                        ...(pipelineResult.result.unknown_speakers ?? []),
                        ...(pipelineResult.result.unresolved_speakers ?? []),
                      ].filter((speaker) => speaker.audio).map((speaker) => (
                        <article className="card" key={`${speaker.speaker_kind}-${speaker.diarization_speaker_id}`}>
                          <span className="card-label">{speaker.speaker_kind}</span>
                          <strong>{speaker.speaker_id}</strong>
                          <span className="hint">
                            Original label: {speaker.diarization_speaker_id} ·{" "}
                            {formatFixed((speaker.audio?.duration_ms ?? 0) / 1000, 1, " s")} ·{" "}
                            {speaker.audio?.snippets.length ?? 0} snippets
                          </span>
                          {speaker.audio && (
                            <audio controls preload="none" src={speakerAudioSource(speaker.audio)}>
                              Your browser does not support embedded MP3 playback.
                            </audio>
                          )}
                        </article>
                      ))}
                    </div>
                  </div>
                )}

                {pipelineResult.result.speaker_refinement && (
                  <div>
                    <h3>Speaker refinement</h3>
                    <dl className="kv-grid">
                      <div><dt>Mode</dt><dd>{pipelineResult.result.speaker_refinement.mode}</dd></div>
                      <div><dt>Status</dt><dd>{pipelineResult.result.speaker_refinement.status.replace(/_/g, " ")}</dd></div>
                      <div><dt>Eligible windows</dt><dd>{pipelineResult.result.speaker_refinement.eligible_windows}</dd></div>
                      <div><dt>Proposed turns</dt><dd>{pipelineResult.result.speaker_refinement.proposed_turns}</dd></div>
                      <div><dt>Applied turns</dt><dd>{pipelineResult.result.speaker_refinement.applied_turns}</dd></div>
                      <div>
                        <dt>Reassigned audio</dt>
                        <dd>{formatFixed(pipelineResult.result.speaker_refinement.reassigned_duration_ms / 1000, 1, " s")}</dd>
                      </div>
                      <div><dt>Processing time</dt><dd>{formatMs(pipelineResult.result.speaker_refinement.processing_ms)}</dd></div>
                      <div><dt>Rollback</dt><dd>{pipelineResult.result.speaker_refinement.rollback_reason ?? "None"}</dd></div>
                    </dl>

                    {pipelineResult.result.speaker_refinement.changes.length > 0 && (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Time</th>
                              <th>Speaker label</th>
                              <th>Support</th>
                              <th>Similarity evidence</th>
                              <th>Outcome</th>
                            </tr>
                          </thead>
                          <tbody>
                            {pipelineResult.result.speaker_refinement.changes.map((change) => (
                              <tr key={`${change.turn_index}-${change.start_ms}-${change.end_ms}`}>
                                <td className="mono nowrap">
                                  {formatTimecode(change.start_ms)}–{formatTimecode(change.end_ms)}
                                </td>
                                <td>
                                  <strong>{change.from_speaker_id}</strong>
                                  <span className="cell-sub">→ {change.to_speaker_id}</span>
                                </td>
                                <td>
                                  <strong>
                                    {change.supporting_windows} windows · {formatFixed(change.evidence_duration_ms / 1000, 1, " s")}
                                  </strong>
                                  <span className="cell-sub">
                                    {formatFixed(change.evidence_coverage * 100, 1, "%")} coverage ·{" "}
                                    {formatFixed(change.weighted_vote_share * 100, 1, "%")} vote
                                  </span>
                                </td>
                                <td className="mono">
                                  <strong>target {formatFixed(change.target_cosine, 3)} · own {formatFixed(change.own_cosine, 3)}</strong>
                                  <span className="cell-sub">
                                    gain {formatFixed(change.similarity_gain, 3)} · runner-up {formatFixed(change.runner_up_margin, 3)}
                                  </span>
                                </td>
                                <td>
                                  <span className={`badge ${change.applied ? "ok" : "plain"}`}>{change.applied ? "Applied" : "Proposed"}</span>
                                  {change.short_turn_exception && <span className="cell-sub">short-turn exception</span>}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {pipelineResult.result.speaker_refinement.changes_truncated && (
                      <p className="hint">Only the first 100 refinement changes are shown.</p>
                    )}
                  </div>
                )}

                {pipelineResult.result.transcript && (
                  <div>
                    <h3>
                      Transcript
                      {pipelineResult.result.transcript.segments
                        ? ` · ${pipelineResult.result.transcript.segments.length} speaker segments`
                        : ""}
                    </h3>
                    <pre className="transcript">{pipelineResult.result.transcript.text || "No transcript available."}</pre>
                  </div>
                )}

                {pipelineResult.result.transcript?.segments && pipelineResult.result.transcript.segments.length > 0 && (
                  <div>
                    <h3>Speaker turns</h3>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Time</th>
                            <th>Speaker</th>
                            <th>Kind</th>
                            <th>Overlap</th>
                            <th>Text</th>
                          </tr>
                        </thead>
                        <tbody>
                          {pipelineResult.result.transcript.segments.map((segment, index) => (
                            <tr key={`${segment.start_ms}-${segment.end_ms}-${index}`}>
                              <td className="mono nowrap">{formatTimecode(segment.start_ms)}–{formatTimecode(segment.end_ms)}</td>
                              <td>
                                <strong>{segment.speaker_id}</strong>
                                <span className="cell-sub">Original label: {segment.diarization_speaker_id}</span>
                                {segment.refined_diarization_speaker_id && (
                                  <span
                                    className={`cell-sub ${
                                      segment.refined_diarization_speaker_id !== segment.diarization_speaker_id ? "refined" : ""
                                    }`}
                                  >
                                    Refined label: {segment.refined_diarization_speaker_id}
                                  </span>
                                )}
                              </td>
                              <td>{segment.speaker_kind}</td>
                              <td>{segment.overlap ? "Yes" : "No"}</td>
                              <td className="text-cell">{segment.text}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {pipelineResult.warnings.length > 0 && (
                  <div className="banner warn">
                    <div>
                      <strong>Warnings</strong>
                      <ul className="plain-list">
                        {pipelineResult.warnings.map((warning, index) => (
                          <li key={index}>{typeof warning === "string" ? warning : JSON.stringify(warning)}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                <details onToggle={(event) => setPipelineJsonExpanded(event.currentTarget.open)}>
                  <summary className="dim">Full JSON response · request {pipelineResult.request_id}</summary>
                  {pipelineJsonExpanded && <pre>{JSON.stringify(pipelineResult, null, 2)}</pre>}
                </details>
              </div>
            )}
          </section>
        )}

        {/* ------------------------------------------------------------------------------ Models */}
        {tab === "models" && (
          <>
            <section className="panel" aria-labelledby="picker-title">
              <div className="panel-head">
                <h2 id="picker-title">Active model</h2>
                <span className="dim">Loaded: <strong>{loadedModelLabel}</strong>{loadedModelId ? ` · ${loadedModelSub}` : ""}</span>
                <p className="hint">
                  The service runs one ASR model. Load switches it right away and brings that model's saved settings
                  along; presets and settings in the Settings tab always belong to the active model.
                </p>
              </div>
              <ModelPicker
                models={managedModels}
                locked={modelSelectRunning || saveBusy}
                selectingModelId={selectingModelId}
                startingDownloadId={modelActionKind === "download" ? modelActionId : null}
                onLoad={(model) => void handleSelectModel(model)}
                onDownload={(model) => void handleDownloadModel(model)}
              />
            </section>

            <section className="panel" aria-labelledby="cache-title">
              <div className="panel-head">
                <h2 id="cache-title">Model cache</h2>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => void handleRefreshModels()}
                  disabled={modelActionId === "__refresh__"}
                  aria-busy={modelActionId === "__refresh__" || undefined}
                >
                  {modelActionId === "__refresh__" ? "Refreshing…" : "Refresh"}
                </button>
                <p className="hint">
                  Downloads and deletes use the saved cache path <span className="mono">{savedCachePath || "(default Hugging Face cache)"}</span>.
                  Sizes on disk are measured; the others are estimates. The active model cannot be deleted; load another one first.
                </p>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Backend</th>
                      <th className="num">Size</th>
                      <th>Status</th>
                      <th>Path</th>
                      <th className="actions"><span className="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {managedModels.length === 0 && (
                      <tr>
                        <td colSpan={6} className="empty">No model list yet. Use Refresh, or check the cache path in Settings.</td>
                      </tr>
                    )}
                    {managedModels.map((model) => {
                      const rowBusy = modelActionId === model.id;
                      const isDownloading = model.status === "downloading";
                      const canDelete = Boolean(model.cache_path) && !model.active && !modelSelectRunning;
                      const status = modelStatusBadge(model.status);
                      return (
                        <tr key={model.id}>
                          <td>
                            <strong>{model.label}</strong>
                            {model.active && <> <span className="badge accent">Active</span></>}
                            <span className="cell-sub mono">{model.id}</span>
                          </td>
                          <td className="nowrap">{formatBackendLabel(model.backend)}</td>
                          <td className="num nowrap">{formatModelSize(model)}</td>
                          <td>
                            <span className={`badge ${status.tone}`}>{status.text}</span>
                            {model.error && <span className="cell-sub field-error">{model.error}</span>}
                          </td>
                          <td className="path-cell"><span className="mono">{resolveModelPath(model)}</span></td>
                          <td className="actions">
                            <button
                              type="button"
                              className="secondary-button"
                              onClick={() => void handleDownloadModel(model)}
                              disabled={isDownloading || rowBusy}
                              aria-busy={isDownloading || (rowBusy && modelActionKind === "download") || undefined}
                            >
                              {isDownloading
                                ? "Downloading…"
                                : rowBusy && modelActionKind === "download"
                                  ? "Starting…"
                                  : model.status === "ready"
                                    ? "Download again"
                                    : "Download"}
                            </button>
                            <button
                              type="button"
                              className="danger-button"
                              onClick={() => void handleDeleteModel(model)}
                              disabled={!canDelete || isDownloading || rowBusy}
                              aria-busy={(rowBusy && modelActionKind === "delete") || undefined}
                              title={model.active ? "The active model cannot be deleted." : undefined}
                            >
                              {rowBusy && modelActionKind === "delete" ? "Deleting…" : "Delete"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {/* ------------------------------------------------------------------------------ Settings */}
        {tab === "settings" && (
          <form className="settings-page" onSubmit={handleSaveSettings}>
            {settingsError && (
              <div className="banner danger" role="alert">
                <span>Could not load the settings: {settingsError}</span>
                <button type="button" className="secondary-button small-button" onClick={() => void loadSettings()}>Retry</button>
              </div>
            )}
            {!settingsLoaded && !settingsError && <div className="state"><span className="spinner" />Loading settings…</div>}

            {settingsLoaded && (
              <fieldset className="settings-lock" disabled={modelSelectRunning}>
                <section className="panel" aria-labelledby="presets-title">
                  <div className="panel-head">
                    <h2 id="presets-title">Presets</h2>
                    <p className="hint">
                      Only presets of the active model ({activeModelOption?.label ?? DASH}) are listed. Built-in presets
                      exist for Cohere Transcribe (by GPU memory) and Parakeet; Whisper models have none. Loading fills the
                      form below; Save changes applies it. A preset stores model, device, precision, batching and VRAM
                      settings only, never the cache path, the language or the Hugging Face token.
                    </p>
                  </div>
                  <div className="form-grid">
                    <label>
                      <span>Preset</span>
                      <select value={selectedPreset ? presetSelected : ""} onChange={(event) => setPresetSelected(event.target.value)}>
                        <option value="">Choose a preset…</option>
                        {modelPresets.map((preset) => (
                          <option key={preset.name} value={preset.name}>
                            {preset.built_in ? `${preset.name} (built-in)` : preset.name}
                          </option>
                        ))}
                      </select>
                      {selectedPreset?.description && <span className="hint">{selectedPreset.description}</span>}
                      {selectedPreset?.note && <span className="field-error">{selectedPreset.note}</span>}
                    </label>
                    <div className="button-row align-end">
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={handleLoadPreset}
                        disabled={!selectedPreset || presetBusy || saveBusy}
                      >
                        Load into form
                      </button>
                      <button
                        type="button"
                        className="danger-button"
                        onClick={() => void handleDeletePreset()}
                        disabled={!selectedPreset || selectedPreset.built_in || presetBusy}
                      >
                        Delete
                      </button>
                    </div>
                    <label>
                      <span>Save the current form as</span>
                      <input
                        value={presetSaveName}
                        maxLength={64}
                        placeholder="Preset name"
                        onChange={(event) => setPresetSaveName(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            void handleSavePreset();
                          }
                        }}
                      />
                    </label>
                    <div className="button-row align-end">
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => void handleSavePreset()}
                        disabled={!presetSaveName.trim() || presetBusy}
                        aria-busy={presetBusy || undefined}
                      >
                        {presetBusy ? "Saving…" : "Save as preset"}
                      </button>
                    </div>
                  </div>
                </section>

                <section className="panel" aria-labelledby="model-settings-title">
                  <div className="panel-head">
                    <h2 id="model-settings-title">Model settings</h2>
                    <p className="hint">Settings of {activeModelOption?.label ?? "the active model"}. Each model keeps its own values.</p>
                  </div>
                  <div className="form-grid">
                    {activeModelFields.includes("local_gpu_device") && (
                      <label>
                        <span>Device</span>
                        <select value={settingsForm.local_gpu_device} onChange={(event) => updateSetting("local_gpu_device", event.target.value)}>
                          {settingsOptions.devices.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                        <span className="hint">Named like nvidia-smi.</span>
                      </label>
                    )}
                    {activeModelFields.includes("transcription_language") && (
                      <label>
                        <span>Language</span>
                        <select
                          value={settingsForm.transcription_language}
                          onChange={(event) => updateSetting("transcription_language", event.target.value)}
                        >
                          {settingsOptions.languages.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                      </label>
                    )}
                    {activeModelFields.includes("local_model_precision") && (
                      <label>
                        <span>Precision</span>
                        <select
                          value={settingsForm.local_model_precision}
                          onChange={(event) => updateSetting("local_model_precision", event.target.value)}
                        >
                          {settingsOptions.precisions.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                      </label>
                    )}
                    {activeModelFields.includes("batch_wait_time_ms") && (
                      <NumberField
                        label="Batch wait time (ms)"
                        min={0}
                        value={settingsForm.batch_wait_time_ms}
                        onChange={(value) => updateSetting("batch_wait_time_ms", value)}
                        hint="How long the worker waits for more chunks before it starts a batch."
                      />
                    )}
                    {activeModelFields.includes("batch_max_segments") && (
                      <NumberField
                        label="Max segments per batch"
                        min={1}
                        value={settingsForm.batch_max_segments}
                        onChange={(value) => updateSetting("batch_max_segments", value)}
                      />
                    )}
                    {activeModelFields.includes("batch_max_audio_seconds") && (
                      <NumberField
                        label="Max audio per batch (s)"
                        min={1}
                        step="0.5"
                        value={settingsForm.batch_max_audio_seconds}
                        onChange={(value) => updateSetting("batch_max_audio_seconds", value)}
                      />
                    )}
                    {activeModelFields.includes("scheduler_long_job_min_chunks") && (
                      <NumberField
                        label="Long job from (chunks)"
                        min={1}
                        value={settingsForm.scheduler_long_job_min_chunks}
                        onChange={(value) => updateSetting("scheduler_long_job_min_chunks", value)}
                      />
                    )}
                    {activeModelFields.includes("scheduler_max_parallel_long_jobs") && (
                      <NumberField
                        label="Max parallel long jobs"
                        min={0}
                        value={settingsForm.scheduler_max_parallel_long_jobs}
                        onChange={(value) => updateSetting("scheduler_max_parallel_long_jobs", value)}
                        hint="0 = unlimited."
                      />
                    )}
                    {activeModelFields.includes("redimnet_max_batch_size") && (
                      <label>
                        <span>ReDimNet max batch (voice ID)</span>
                        <select
                          value={settingsForm.redimnet_max_batch_size}
                          onChange={(event) => updateSetting("redimnet_max_batch_size", Number(event.target.value))}
                        >
                          {[8, 4, 2, 1].map((size) => (
                            <option key={size} value={size}>{size}</option>
                          ))}
                        </select>
                        <span className="hint">
                          Largest CUDA batch of 3 s windows for speaker clouds, sentence verification and embeddings.
                          The reserved VRAM pool grows with it (roughly 0.35 GB per window); larger batches make long
                          diarizations faster. Applies on save, no restart needed.
                        </span>
                      </label>
                    )}
                    {activeModelFields.includes("language_detection_model") && (
                      <label>
                        <span>Language detection model</span>
                        <select
                          value={settingsForm.language_detection_model}
                          disabled={!settingsForm.cohere_language_detection}
                          onChange={(event) => updateSetting("language_detection_model", event.target.value)}
                        >
                          {(settingsOptions.language_detection_models ?? []).map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                        <span className="hint">
                          Downloaded into the model cache on first use. Base is enough for most audio; Small and Large v3
                          Turbo are steadier on very short clips for more VRAM. Applies on save, never reloads the ASR model.
                        </span>
                      </label>
                    )}
                    {activeModelFields.includes("scheduler_first_chunk_fast_path") && (
                      <CheckField
                        label="First-chunk fast path"
                        checked={settingsForm.scheduler_first_chunk_fast_path}
                        onChange={(value) => updateSetting("scheduler_first_chunk_fast_path", value)}
                        hint={
                          'A request that has not been served yet gets its first chunk into a small batch of its own instead of riding along in a full batch of a long recording. Batches are always shared round-robin between active requests; short requests (below "Long job from") never wait for a long-job slot.'
                        }
                      />
                    )}
                    {activeModelFields.includes("cuda_memory_trim_after_batch") && (
                      <CheckField
                        label="Release idle VRAM after batches"
                        checked={settingsForm.cuda_memory_trim_after_batch}
                        onChange={(value) => updateSetting("cuda_memory_trim_after_batch", value)}
                        hint="Releases the CUDA allocator pool once every GPU phase (ASR batches, diarization, embeddings, warmup) has been idle for 5 s, so the idle footprint stays near the model size on a shared GPU. The next request reserves its pool again (a one-off delay of well under a second). Turn it off only on a GPU this service has to itself."
                      />
                    )}
                    {activeModelFields.includes("cohere_language_detection") && (
                      <CheckField
                        label="Detect the language (Cohere, language Auto)"
                        checked={settingsForm.cohere_language_detection}
                        onChange={(value) => updateSetting("cohere_language_detection", value)}
                        hint="Cohere cannot detect a language itself. With Auto, a small Whisper model picks one of the 14 Cohere languages per request before transcription: per clip, per ~25 s chunk of a long file, per speaker in diarization. An explicit language (setting or request) always wins. Off, or when the detection model cannot be loaded, Auto transcribes as German."
                      />
                    )}
                    {activeModelFields.includes("cohere_cuda_graphs") && (
                      <CheckField
                        label="Cohere CUDA graphs"
                        checked={settingsForm.cohere_cuda_graphs}
                        onChange={(value) => updateSetting("cohere_cuda_graphs", value)}
                        hint="Replays the Cohere encoder and greedy decoder as captured CUDA graphs: short clips run several times faster with identical transcripts, for about 0.5 GB more VRAM (hard cap 1.5 GB) and a few seconds of extra warmup. fp16/bf16 on CUDA only. Takes effect on the next model (re)load."
                      />
                    )}
                  </div>
                </section>

                <section className="panel" aria-labelledby="service-settings-title">
                  <div className="panel-head">
                    <h2 id="service-settings-title">Service-wide settings</h2>
                    <p className="hint">These apply to the whole service, whichever model is active.</p>
                  </div>
                  <div className="form-grid">
                    {serviceFields.includes("local_model_cache_path") && (
                      <label className="span-2">
                        <span>Model cache path</span>
                        <input
                          className="mono"
                          value={settingsForm.local_model_cache_path}
                          placeholder="Empty: default Hugging Face cache"
                          onChange={(event) => updateSetting("local_model_cache_path", event.target.value)}
                        />
                      </label>
                    )}
                    {serviceFields.includes("huggingface_token") && (
                      <label className="span-2">
                        <span>Hugging Face token</span>
                        <div className="input-with-action">
                          <input
                            type="password"
                            autoComplete="new-password"
                            value={tokenDraft}
                            disabled={tokenClear}
                            placeholder={tokenPlaceholder}
                            onChange={(event) => setTokenDraft(event.target.value)}
                          />
                          {(tokenSet || tokenClear) && (
                            <button
                              type="button"
                              className="ghost-button"
                              onClick={() => {
                                setTokenDraft("");
                                setTokenClear((current) => !current);
                              }}
                            >
                              {tokenClear ? "Keep" : "Clear"}
                            </button>
                          )}
                        </div>
                        <span className="hint">
                          Needed for gated models such as Cohere Transcribe (the default). Accept its license on{" "}
                          <a href="https://huggingface.co/CohereLabs/cohere-transcribe-03-2026" target="_blank" rel="noreferrer">
                            Hugging Face
                          </a>{" "}
                          with the account that owns the token. The saved token is never shown again; a newly typed one is
                          also used for downloads before you save.
                        </span>
                      </label>
                    )}
                    {serviceFields.includes("segmentation_model_path") && (
                      <label className="span-2">
                        <span>Segmentation checkpoint (diarization)</span>
                        <input
                          className="mono"
                          type="text"
                          placeholder="Empty: pyannote segmentation-3.0 from the local Hugging Face cache"
                          value={settingsForm.segmentation_model_path}
                          onChange={(event) => updateSetting("segmentation_model_path", event.target.value)}
                        />
                        <span className="hint">
                          Path to a <code>pytorch_model.bin</code> of pyannote segmentation-3.0 (about 6 MB). Empty searches
                          the model cache; nothing is ever downloaded. Without a usable checkpoint diarization requests
                          answer 503 MODEL_UNAVAILABLE.
                        </span>
                      </label>
                    )}
                    {serviceFields.includes("speaker_turns_enabled") && (
                      <CheckField
                        label="Offer live speaker turns"
                        checked={settingsForm.speaker_turns_enabled}
                        onChange={(value) => updateSetting("speaker_turns_enabled", value)}
                        hint={
                          <>
                            Transcript requests may carry <code>speaker_turns</code> (live speakers in Genesis3). Offered
                            in <code>/v2/capabilities</code> only while the segmentation checkpoint loads.
                          </>
                        }
                      />
                    )}
                    {serviceFields.includes("debug_retain_history_audio") && (
                      <CheckField
                        tone="warn"
                        label="Keep the original audio for history debugging"
                        checked={settingsForm.debug_retain_history_audio}
                        onChange={(value) => updateSetting("debug_retain_history_audio", value)}
                        hint="Sensitive audio data: keeps the byte-identical upload only while its item stays in the visible history. Turning this off deletes every retained recording right after saving."
                      />
                    )}
                  </div>
                </section>
              </fieldset>
            )}

            {settingsLoaded && (
              <div className={`save-bar ${dirtyCount > 0 ? "dirty" : ""}`} role="region" aria-label="Save settings">
                <span className="save-bar-status">
                  {dirtyCount > 0
                    ? `${dirtyCount} unsaved change${dirtyCount === 1 ? "" : "s"}${invalidKeys.length ? ` · ${invalidKeys.length} invalid` : ""}`
                    : "All changes saved"}
                </span>
                <div className="button-row">
                  <button type="button" className="ghost-button" onClick={handleDiscardSettings} disabled={dirtyCount === 0 || saveBusy}>
                    Discard
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void handleReloadSettings()} disabled={saveBusy}>
                    Reload
                  </button>
                  <button
                    type="submit"
                    disabled={saveBusy || modelSelectRunning || dirtyCount === 0 || invalidKeys.length > 0}
                    aria-busy={saveBusy || undefined}
                  >
                    {saveBusy ? "Saving…" : "Save changes"}
                  </button>
                </div>
              </div>
            )}
          </form>
        )}

        {/* ------------------------------------------------------------------------------ Benchmark */}
        {tab === "benchmark" && (
          <section className="panel" aria-labelledby="benchmark-title">
            <div className="panel-head">
              <h2 id="benchmark-title">Parallel ASR benchmark</h2>
              <p className="hint">
                Runs the file several times in parallel through the active ASR batch and chunk pipeline and reports the
                realtime factor and the peak VRAM of this process. The file is shared with the Playground.
              </p>
            </div>
            <form className="form-grid" onSubmit={handleRunBenchmark}>
              <label>
                <span>Audio or video file</span>
                <input type="file" accept="audio/*,video/*" onChange={(event) => selectPipelineFile(event.target.files?.[0] ?? null)} />
                <span className="hint">{benchmarkFile ? `${benchmarkFile.name} · ${formatBytes(benchmarkFile.size)}` : "No file chosen."}</span>
              </label>
              <label>
                <span>Repeats (parallel)</span>
                <input
                  type="number"
                  min={1}
                  max={64}
                  step={1}
                  value={benchmarkRepeatInput}
                  aria-invalid={benchmarkRepeatInvalid || undefined}
                  onChange={(event) => setBenchmarkRepeatInput(event.target.value)}
                />
                {benchmarkRepeatInvalid ? <span className="field-error">A whole number from 1 to 64.</span> : <span className="hint">1 to 64.</span>}
              </label>
              {benchmarkError && <div className="banner danger span-2" role="alert">{benchmarkError}</div>}
              <div className="form-actions button-row span-2">
                <button type="submit" disabled={benchmarkBusy || !benchmarkFile || benchmarkRepeatInvalid} aria-busy={benchmarkBusy || undefined}>
                  {benchmarkBusy ? "Running…" : "Run benchmark"}
                </button>
              </div>
            </form>

            {benchmarkResult && (
              <div className="result-stack">
                <div className="grid grid-auto">
                  <Card label="Realtime factor" value={formatRealtimeFactor(benchmarkResult.rtf)} sub="Audio seconds per wall second" />
                  <Card label="Total time" value={formatMs(benchmarkResult.total_wall_time_ms)} />
                  <Card label="Time per run" value={formatMs(benchmarkResult.avg_wall_time_per_run_ms)} />
                  <Card label="Audio per run" value={formatFixed(benchmarkResult.audio_seconds, 1, " s")} />
                  <Card label="Total audio" value={formatFixed(benchmarkResult.total_audio_seconds, 1, " s")} />
                  <Card label="Chunks per run" value={formatValue(benchmarkResult.chunks_per_run)} />
                  <Card label="Total chunks" value={formatValue(benchmarkResult.total_chunks)} />
                  <Card label="Batches used" value={formatValue(benchmarkResult.batches_used)} />
                  <Card label="Peak VRAM reserved" value={formatMib(benchmarkResult.peak_vram_reserved_mb)} />
                  <Card label="Peak VRAM allocated" value={formatMib(benchmarkResult.peak_vram_allocated_mb)} />
                  <Card
                    label="Transcripts"
                    value={<span className={`badge ${benchmarkResult.transcripts_match ? "ok" : "warn"}`}>{benchmarkResult.transcripts_match ? "Identical" : "Differ"}</span>}
                  />
                </div>
                <dl className="kv-grid">
                  <div><dt>Workflow</dt><dd>{describeBenchmarkWorkflow(benchmarkResult.workflow)}</dd></div>
                  <div><dt>File</dt><dd>{benchmarkResult.file_name}</dd></div>
                  <div><dt>Model</dt><dd className="mono">{benchmarkResult.model_id}</dd></div>
                  <div><dt>Language</dt><dd>{benchmarkResult.transcription_language}</dd></div>
                  <div><dt>Repeats</dt><dd>{benchmarkResult.repeat_count}</dd></div>
                </dl>
                <div>
                  <h3>Transcript</h3>
                  <pre className="transcript">{benchmarkResult.transcript || "No transcript available."}</pre>
                </div>
              </div>
            )}
          </section>
        )}

        {/* ------------------------------------------------------------------------------ History */}
        {tab === "history" && (
          <section className="panel" aria-labelledby="history-title">
            <div className="panel-head">
              <h2 id="history-title">Latest transcriptions</h2>
              <p className="hint">
                The last 25 requests. Transcription and embedding are wall-clock times and include waiting for the queue
                and the shared GPU lock. Original audio is kept only while "Keep the original audio" is on (Settings).
              </p>
            </div>
            <div className="table-wrap">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Client</th>
                    <th>Model</th>
                    <th>Language</th>
                    <th className="num">Total</th>
                    <th className="num" title="Wall time including queue and shared GPU-lock waits.">Transcription</th>
                    <th className="num" title="Wall time including queue and shared GPU-lock waits.">Embedding</th>
                    <th className="num">Segments</th>
                    <th>Text</th>
                    <th>Audio</th>
                  </tr>
                </thead>
                <tbody>
                  {!stats && (
                    <tr>
                      <td colSpan={10} className="empty"><span className="spinner" /> Loading…</td>
                    </tr>
                  )}
                  {stats && history.length === 0 && (
                    <tr>
                      <td colSpan={10} className="empty">No transcription yet. Send a request or try the Playground.</td>
                    </tr>
                  )}
                  {history.map((entry) => {
                    const action = historyActions[entry.history_id];
                    const audioAvailable = entry.debug_audio?.status === "available";
                    return (
                      <tr key={entry.history_id}>
                        <td className="nowrap">
                          {formatDateTime(entry.timestamp, true)}
                          {entry.retry_of && (
                            <span className="cell-sub" title={entry.retry_of}>Retry of {entry.retry_of.slice(0, 8)}</span>
                          )}
                        </td>
                        <td className="nowrap">{entry.source_ip ?? DASH}</td>
                        <td>
                          <span className="mono">{entry.model_id ?? entry.engine ?? DASH}</span>
                          {entry.mode && <span className="cell-sub">{entry.mode}{entry.batched ? " · batched" : ""}</span>}
                        </td>
                        <td>{entry.transcription_language ?? DASH}</td>
                        <td className="num">{formatMs(entry.total_duration_ms)}</td>
                        <td className="num">{formatMs(entry.transcription_duration_ms)}</td>
                        <td className="num">{formatMs(entry.voice_vector_duration_ms)}</td>
                        <td className="num">{entry.segment_count ?? 1}</td>
                        <td className="text-cell">{entry.transcript || <span className="dim">{DASH}</span>}</td>
                        <td className="audio-cell">
                          {audioAvailable ? (
                            <>
                              <span className="cell-sub">
                                {entry.debug_audio?.filename || "Original audio"} · {formatBytes(entry.debug_audio?.size_bytes)}
                              </span>
                              <div className="button-row">
                                {entry.retry_mode && (
                                  <button
                                    type="button"
                                    className="secondary-button"
                                    disabled={Boolean(action)}
                                    aria-busy={action === "retry" || undefined}
                                    onClick={() => void handleRetryHistoryEntry(entry)}
                                  >
                                    {action === "retry" ? "Retrying…" : "Retry"}
                                  </button>
                                )}
                                <button
                                  type="button"
                                  className="ghost-button"
                                  disabled={Boolean(action)}
                                  aria-busy={action === "download" || undefined}
                                  onClick={() => void handleDownloadHistoryEntry(entry)}
                                >
                                  {action === "download" ? "Downloading…" : "Download"}
                                </button>
                              </div>
                            </>
                          ) : (
                            <span className="hint">{describeDebugAudioReason(entry.debug_audio?.reason)}</span>
                          )}
                          {historyActionErrors[entry.history_id] && (
                            <span className="field-error" role="alert">{historyActionErrors[entry.history_id]}</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ------------------------------------------------------------------------------ API keys */}
        {tab === "api-keys" && (
          <ApiKeysPanel
            keys={apiKeys === null ? null : apiKeyRows}
            loadError={apiKeysError}
            usageLabel="Audio (s)"
            formatUsage={(value) => formatFixed(value, 1)}
            baseUrl={baseUrl}
            exampleCurl={(key) =>
              `curl -H "Authorization: Bearer ${key}" -F "file=@audio.wav" -F "model=whisper-1" ${baseUrl}/v1/audio/transcriptions`
            }
            routesText={PUBLIC_ROUTES}
            onCreate={async (alias) => {
              try {
                return await createApiKey(alias);
              } catch (error) {
                if (handleApiError(error, "", true)) throw new Error("Session expired.");
                throw error;
              }
            }}
            onRevoke={async (id) => {
              try {
                await deleteApiKey(id);
              } catch (error) {
                if (handleApiError(error, "", true)) throw new Error("Session expired.");
                throw error;
              }
            }}
            onChanged={() => void refreshApiKeys()}
          />
        )}

        {/* ------------------------------------------------------------------------------ Account */}
        {tab === "account" && (
          <div className="grid grid-2">
            <section className="panel" aria-labelledby="account-title">
              <div className="panel-head">
                <h2 id="account-title">Account</h2>
              </div>
              <dl className="kv-grid">
                <div><dt>Signed in as</dt><dd>{currentUser || DASH}</dd></div>
                <div><dt>Session expires</dt><dd>{formatDateTime(sessionExpiresAt)}</dd></div>
              </dl>
              <p className="hint">
                The admin login is separate from client API keys. Changing the password ends every other session of this user.
              </p>
              <div className="button-row">
                <button type="button" className="ghost-button" onClick={() => void handleLogout()}>Log out</button>
              </div>
            </section>
            <section className="panel" aria-labelledby="password-title">
              <div className="panel-head">
                <h2 id="password-title">Change password</h2>
              </div>
              {passwordForm("Change password")}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

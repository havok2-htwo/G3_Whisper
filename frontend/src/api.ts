export type AdminOption = {
  label: string;
  value: string;
};

export type AdminSettings = {
  local_model: string;
  local_gpu_device: string;
  local_model_precision: string;
  local_model_cache_path: string;
  transcription_language: string;
  batch_wait_time_ms: number;
  batch_max_segments: number;
  batch_max_audio_seconds: number;
  cuda_memory_trim_after_batch: boolean;
  debug_retain_history_audio: boolean;
  huggingface_token: string;
  dia_server_base_url: string;
  dia_api_key: string;
  dia_api_key_configured: boolean;
  dia_api_key_source: "settings" | "environment" | "none";
  dia_server_base_url_effective: string;
  dia_server_base_url_source: "settings" | "environment" | "none";
};

export type HistoryRetryMode = "transcript" | "transcript_embedding" | null;

export type HistoryDebugAudio = {
  status: "available" | "not_retained";
  reason?: string | null;
  filename?: string | null;
  content_type?: string | null;
  size_bytes?: number | null;
  capture_duration_ms?: number | null;
};

export type HistoryEntry = {
  history_id: string;
  timestamp?: string;
  source_ip?: string;
  engine?: string;
  model_id?: string;
  transcription_language?: string;
  mode?: string | null;
  total_duration_ms: number | null;
  transcription_duration_ms: number | null;
  voice_vector_duration_ms: number | null;
  transcript?: string;
  batched?: boolean;
  segment_count?: number;
  retry_of: string | null;
  retry_mode: HistoryRetryMode;
  debug_audio: HistoryDebugAudio | null;
};

export type HistoryRetryResponse = {
  ok: boolean;
  history_id: string;
  retry_of: string;
  mode: Exclude<HistoryRetryMode, null>;
  status: "completed" | "partial";
  timings_ms: Record<string, number>;
  transcript: string;
  warnings: Array<{ code: string; message: string }>;
};

export type DiaConnectionTestResponse = {
  ok: boolean;
  base_url: string;
  status_code: number;
  message: string;
};

export type ManagedModel = {
  id: string;
  label: string;
  backend: string;
  status: string;
  local_path: string | null;
  cache_path: string | null;
  storage_root: string;
  approx_size_gb: number | null;
  size_on_disk_gb: number | null;
  error: string | null;
  updated_at: string | null;
};

export type SettingsResponse = {
  settings: AdminSettings;
  options: {
    models: AdminOption[];
    devices: AdminOption[];
    precisions: AdminOption[];
    languages: AdminOption[];
  };
  models: ManagedModel[];
  loaded_model_identifier: string[] | null;
};

export type StatsResponse = {
  summary: {
    total_requests: number;
    avg_total_duration_ms: number | null;
    avg_transcription_duration_ms: number | null;
  };
  history: HistoryEntry[];
};

export type QueueResponse = {
  worker_running: boolean;
  queue_size: number;
  pending_buffer_size: number;
  active_batch_id: string | null;
  active_batch_size: number;
  active_batch_audio_seconds: number;
  active_batch_started_at: string | null;
  last_batch_completed_at: string | null;
  last_batch_duration_ms: number | null;
  last_error: string | null;
  total_batches_processed: number;
  total_segments_processed: number;
  recent_batches: Array<Record<string, unknown>>;
};

export type BenchmarkResponse = {
  ok: boolean;
  file_name: string;
  workflow: string;
  model_id: string;
  transcription_language: string;
  repeat_count: number;
  audio_seconds: number;
  total_audio_seconds: number;
  chunks_per_run: number;
  total_chunks: number;
  batches_used: number;
  total_wall_time_ms: number;
  avg_wall_time_per_run_ms: number;
  rtf: number | null;
  transcripts_match: boolean;
  transcript: string;
  peak_vram_reserved_mb: number | null;
  peak_vram_allocated_mb: number | null;
};

export type AudioProcessMode = "embedding" | "transcript" | "transcript_embedding" | "diarization";

export type SpeakerRefinementMode = "off" | "shadow" | "conservative";

export type KnownSpeakerProfile = {
  id: string;
  embeddings: number[][];
};

export type AudioProcessRequest = {
  schema_version: "2.0";
  mode: AudioProcessMode;
  diarization?: {
    expected_speakers?: number;
    known_speakers: KnownSpeakerProfile[];
    speaker_refinement?: SpeakerRefinementMode;
    unknown_speaker_audio?: boolean;
  };
};

export type UnknownSpeakerAudio = {
  mime_type: string;
  encoding: "base64";
  data: string;
  duration_ms: number;
  snippets: Array<{
    start_ms: number;
    end_ms: number;
    duration_ms: number;
    centrality: number;
  }>;
};

export type AudioProcessUnknownSpeaker = Record<string, unknown> & {
  speaker_id: string;
  diarization_speaker_id: string;
  speaker_kind: "unknown" | "unresolved";
  audio?: UnknownSpeakerAudio;
};

export type AudioProcessTranscriptSegment = {
  start_ms: number;
  end_ms: number;
  speaker_id: string;
  diarization_speaker_id: string;
  refined_diarization_speaker_id?: string;
  speaker_kind: "known" | "unknown" | "unresolved";
  text: string;
  overlap?: boolean;
  [key: string]: unknown;
};

export type AudioProcessResult = {
  transcript?: {
    text: string;
    segments?: AudioProcessTranscriptSegment[];
  };
  embedding?: {
    vector: number[];
  } | null;
  speaker_counts?: {
    expected: number | null;
    detected: number;
    known_provided: number;
    known_assigned: number;
    unknown: number;
    unresolved: number;
  };
  speaker_assignments?: Array<Record<string, unknown>>;
  unknown_speakers?: AudioProcessUnknownSpeaker[];
  unresolved_speakers?: AudioProcessUnknownSpeaker[];
  unresolved_known_speakers?: string[];
  speaker_refinement?: {
    mode: SpeakerRefinementMode;
    status: "disabled" | "not_needed" | "shadow" | "applied" | "rejected";
    eligible_windows: number;
    proposed_turns: number;
    applied_turns: number;
    reassigned_duration_ms: number;
    processing_ms: number;
    rollback_reason: string | null;
    changes_truncated: boolean;
    changes: Array<{
      turn_index: number;
      start_ms: number;
      end_ms: number;
      from_speaker_id: string;
      to_speaker_id: string;
      supporting_windows: number;
      evidence_duration_ms: number;
      evidence_coverage: number;
      weighted_vote_share: number;
      target_cosine: number;
      own_cosine: number;
      similarity_gain: number;
      runner_up_margin: number;
      short_turn_exception: boolean;
      applied: boolean;
    }>;
  };
  [key: string]: unknown;
};

export type AudioProcessResponse = {
  schema_version: "2.0";
  request_id: string;
  status: "completed" | "partial";
  mode: AudioProcessMode;
  audio: {
    duration_ms: number;
  };
  models: Record<string, Record<string, unknown>>;
  timings_ms: Record<string, number>;
  result: AudioProcessResult;
  warnings: Array<Record<string, unknown> | string>;
};

// --- Auth / API keys ---
export type WhoAmI = {
  username: string;
  must_change_password: boolean;
};

export type ApiKeyUsage = {
  total_seconds_processed: number;
  request_count: number;
  last_used_at: string | null;
};

export type ApiKeyInfo = {
  id: string;
  alias: string;
  created_at: string | null;
  usage: ApiKeyUsage;
};

export type CreatedApiKey = {
  id: string;
  alias: string;
  created_at: string;
  token: string;
};

/**
 * All admin requests carry the httpOnly session cookie (same-origin), so no key handling
 * is needed client-side. 401 -> not logged in; 403 password_change_required -> forced
 * password change. Both are surfaced as distinct error messages the UI reacts to.
 */
async function requestJson<T>(input: string, init?: RequestInit): Promise<T> {
  const nextHeaders = new Headers(init?.headers ?? {});
  if (!(init?.body instanceof FormData) && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json");
  }

  const response = await fetch(input, {
    credentials: "include",
    ...init,
    headers: nextHeaders,
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }
  if (response.status === 403) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    if (payload.detail === "password_change_required") {
      throw new Error("password_change_required");
    }
    throw new Error(payload.detail ?? "Forbidden");
  }
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))) as {
      detail?: string;
    };
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

async function readErrorDetail(response: Response): Promise<string> {
  const payload = (await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))) as {
    detail?: string;
  };
  return payload.detail ?? `HTTP ${response.status}`;
}

export async function login(username: string, password: string) {
  return requestJson<WhoAmI>("/api/admin/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function logout() {
  return requestJson<{ ok: boolean }>("/api/admin/auth/logout", { method: "POST" });
}

export async function whoami() {
  return requestJson<WhoAmI>("/api/admin/auth/whoami", { method: "GET" });
}

export async function changePassword(currentPassword: string, newPassword: string) {
  return requestJson<{ ok: boolean; must_change_password: boolean }>("/api/admin/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

export async function listApiKeys() {
  return requestJson<{ keys: ApiKeyInfo[] }>("/api/admin/api-keys", { method: "GET" });
}

export async function createApiKey(alias: string) {
  return requestJson<CreatedApiKey>("/api/admin/api-keys", {
    method: "POST",
    body: JSON.stringify({ alias }),
  });
}

export async function deleteApiKey(id: string) {
  return requestJson<{ ok: boolean }>(`/api/admin/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function getSettings() {
  return requestJson<SettingsResponse>("/api/admin/settings", { method: "GET" });
}

export async function saveSettings(settings: AdminSettings) {
  const {
    dia_api_key_configured: _diaApiKeyConfigured,
    dia_api_key_source: _diaApiKeySource,
    dia_server_base_url_effective: _diaServerBaseUrlEffective,
    dia_server_base_url_source: _diaServerBaseUrlSource,
    ...editableSettings
  } = settings;
  const payload: Record<string, unknown> = { ...editableSettings };
  if (!settings.dia_api_key.trim()) {
    delete payload.dia_api_key;
  }

  return requestJson<SettingsResponse & { ok: boolean; model_reloaded: boolean; model_loaded: boolean | null }>(
    "/api/admin/settings",
    {
      method: "PUT",
      body: JSON.stringify(payload),
    },
  );
}

export async function deleteDiaApiKey() {
  return requestJson<SettingsResponse & { ok: boolean; removed: boolean; environment_fallback_active: boolean }>(
    "/api/admin/settings/dia-api-key",
    { method: "DELETE" },
  );
}

export async function testDiaConnection(diaServerBaseUrl?: string, diaApiKey?: string) {
  return requestJson<DiaConnectionTestResponse>("/api/admin/dia/test", {
    method: "POST",
    body: JSON.stringify({
      dia_server_base_url: diaServerBaseUrl?.trim() || undefined,
      dia_api_key: diaApiKey?.trim() || undefined,
    }),
  });
}

export async function getModels(storagePath?: string) {
  const query = storagePath !== undefined ? `?storage_path=${encodeURIComponent(storagePath)}` : "";
  return requestJson<{ models: ManagedModel[] }>(`/api/admin/models${query}`, { method: "GET" });
}

export async function downloadModel(modelId: string, storagePath: string, huggingfaceToken?: string) {
  return requestJson<{ job: Record<string, unknown>; models: ManagedModel[] }>("/api/admin/models/download", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      storage_path: storagePath,
      huggingface_token: huggingfaceToken,
    }),
  });
}

export async function deleteModel(modelId: string, storagePath: string) {
  return requestJson<{ ok: boolean; removed: boolean; removed_path: string | null; storage_root: string; models: ManagedModel[] }>(
    "/api/admin/models/delete",
    {
      method: "POST",
      body: JSON.stringify({
        model_id: modelId,
        storage_path: storagePath,
      }),
    },
  );
}

export async function getStats() {
  return requestJson<StatsResponse>("/api/admin/stats", { method: "GET" });
}

export async function retryHistoryAudio(historyId: string) {
  return requestJson<HistoryRetryResponse>(
    `/api/admin/history/${encodeURIComponent(historyId)}/retry`,
    { method: "POST" },
  );
}

function filenameFromContentDisposition(headerValue: string | null) {
  if (!headerValue) {
    return null;
  }

  const encodedMatch = headerValue.match(/filename\*=UTF-8''([^;]+)/i);
  if (encodedMatch?.[1]) {
    try {
      return decodeURIComponent(encodedMatch[1].trim().replace(/^"|"$/g, ""));
    } catch {
      // Fall through to the plain filename form.
    }
  }

  const plainMatch = headerValue.match(/filename="([^"]+)"|filename=([^;]+)/i);
  return (plainMatch?.[1] ?? plainMatch?.[2] ?? "").trim() || null;
}

function safeDownloadFilename(value: string | null, fallback: string) {
  const leafName = (value ?? "").split(/[\\/]/).pop()?.trim() ?? "";
  const sanitized = leafName.replace(/[\u0000-\u001f<>:"/\\|?*]/g, "_");
  return sanitized && sanitized !== "." && sanitized !== ".." ? sanitized : fallback;
}

export async function downloadHistoryAudio(historyId: string, fallbackFilename?: string | null) {
  const response = await fetch(`/api/admin/history/${encodeURIComponent(historyId)}/audio`, {
    method: "GET",
    credentials: "include",
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }
  if (response.status === 403) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    if (payload.detail === "password_change_required") {
      throw new Error("password_change_required");
    }
    throw new Error(payload.detail ?? "Forbidden");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }

  const fallback = safeDownloadFilename(fallbackFilename ?? null, `history-audio-${historyId}`);
  return {
    blob: await response.blob(),
    filename: safeDownloadFilename(
      filenameFromContentDisposition(response.headers.get("Content-Disposition")),
      fallback,
    ),
  };
}

export async function getQueue() {
  return requestJson<QueueResponse>("/api/admin/queue", { method: "GET" });
}

export async function runBenchmark(file: File, repeatCount: number) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("repeat_count", String(repeatCount));

  const response = await fetch("/api/admin/benchmark", {
    method: "POST",
    credentials: "include",
    body: formData,
  });

  if (response.status === 401) {
    throw new Error("unauthorized");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }

  return response.json() as Promise<BenchmarkResponse>;
}

export async function processAudioV2(file: File, request: AudioProcessRequest, apiKey?: string) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append(
    "request",
    new Blob([JSON.stringify(request)], { type: "application/json" }),
    "request.json",
  );

  const headers = new Headers();
  if (apiKey?.trim()) {
    headers.set("X-API-Key", apiKey.trim());
  }

  const response = await fetch("/v2/audio/process", {
    method: "POST",
    credentials: "include",
    headers,
    body: formData,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as {
      error?: { code?: string; message?: string };
      detail?: string;
    };
    const message = payload.error?.message ?? payload.detail ?? `HTTP ${response.status}`;
    const code = payload.error?.code;
    throw new Error(code ? `${code}: ${message}` : message);
  }

  return response.json() as Promise<AudioProcessResponse>;
}

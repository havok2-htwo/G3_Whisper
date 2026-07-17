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
  huggingface_token: string;
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
  history: Array<Record<string, unknown>>;
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
  return requestJson<SettingsResponse & { ok: boolean; model_reloaded: boolean; model_loaded: boolean | null }>(
    "/api/admin/settings",
    {
      method: "PUT",
      body: JSON.stringify(settings),
    },
  );
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

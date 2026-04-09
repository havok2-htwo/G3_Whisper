export type AdminOption = {
  label: string;
  value: string;
};

export type AdminSettings = {
  local_model: string;
  local_gpu_device: string;
  local_model_cache_path: string;
  transcription_language: string;
  batch_wait_time_ms: number;
  batch_max_segments: number;
  batch_max_audio_seconds: number;
};

export type SettingsResponse = {
  settings: AdminSettings;
  options: {
    models: AdminOption[];
    devices: AdminOption[];
    languages: AdminOption[];
  };
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

type RequestOptions = RequestInit & {
  allowUnauthorized?: boolean;
};

async function requestJson<T>(input: string, init?: RequestOptions): Promise<T> {
  const response = await fetch(input, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (response.status === 401 && init?.allowUnauthorized) {
    throw new Error("unauthorized");
  }

  if (!response.ok) {
    const payload = await response
      .json()
      .catch(() => ({ detail: `HTTP ${response.status}` })) as { detail?: string };
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function readErrorDetail(response: Response): Promise<string> {
  const payload = await response
    .json()
    .catch(() => ({ detail: `HTTP ${response.status}` })) as { detail?: string };
  return payload.detail ?? `HTTP ${response.status}`;
}

export async function getSession() {
  return requestJson<{ authenticated: true; username: string }>("/api/admin/session", {
    method: "GET",
    allowUnauthorized: true,
  });
}

export async function login(username: string, password: string) {
  return requestJson<{ ok: boolean; username: string }>("/api/admin/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function logout() {
  return requestJson<{ ok: boolean }>("/api/admin/logout", {
    method: "POST",
  });
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
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }

  return response.json() as Promise<BenchmarkResponse>;
}

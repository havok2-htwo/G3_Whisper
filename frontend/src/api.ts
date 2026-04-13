export type AdminOption = {
  label: string;
  value: string;
};

export type AdminKeyMetadata = {
  id: string;
  label: string;
  created_at: string | null;
  last_used_at: string | null;
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
  adminKey?: string;
};

async function requestJson<T>(input: string, init?: RequestOptions): Promise<T> {
  const { adminKey, ...requestInit } = init ?? {};
  const nextHeaders = new Headers(init?.headers ?? {});
  if (!(requestInit.body instanceof FormData) && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json");
  }
  if (adminKey) {
    nextHeaders.set("X-Admin-Key", adminKey);
  }

  const response = await fetch(input, {
    headers: nextHeaders,
    ...requestInit,
  });

  if (response.status === 401) {
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

export async function getKeys(adminKey: string) {
  return requestJson<{ admin_key: AdminKeyMetadata }>("/api/admin/keys", {
    method: "GET",
    adminKey,
  });
}

export async function rotateAdminKey(adminKey: string) {
  return requestJson<{ key: AdminKeyMetadata & { token: string }; keys: { admin_key: AdminKeyMetadata } }>("/api/admin/keys", {
    method: "POST",
    adminKey,
  });
}

export async function getSettings(adminKey: string) {
  return requestJson<SettingsResponse>("/api/admin/settings", {
    method: "GET",
    adminKey,
  });
}

export async function saveSettings(adminKey: string, settings: AdminSettings) {
  return requestJson<SettingsResponse & { ok: boolean; model_reloaded: boolean; model_loaded: boolean | null }>(
    "/api/admin/settings",
    {
      method: "PUT",
      adminKey,
      body: JSON.stringify(settings),
    },
  );
}

export async function getStats(adminKey: string) {
  return requestJson<StatsResponse>("/api/admin/stats", {
    method: "GET",
    adminKey,
  });
}

export async function getQueue(adminKey: string) {
  return requestJson<QueueResponse>("/api/admin/queue", {
    method: "GET",
    adminKey,
  });
}

export async function runBenchmark(adminKey: string, file: File, repeatCount: number) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("repeat_count", String(repeatCount));

  const response = await fetch("/api/admin/benchmark", {
    method: "POST",
    headers: {
      "X-Admin-Key": adminKey,
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }

  return response.json() as Promise<BenchmarkResponse>;
}

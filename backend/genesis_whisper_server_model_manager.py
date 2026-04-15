from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .genesis_whisper_server_globals import LOCAL_ASR_MODEL_SPECS, resolve_local_model_cache_path

BYTES_PER_GB = 1024 ** 3
_model_jobs_lock = threading.Lock()
_model_jobs: dict[tuple[str, str], dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _resolve_storage_root(storage_path: str) -> Path:
    resolved_path = resolve_local_model_cache_path(storage_path)
    if resolved_path:
        return Path(resolved_path)

    env_cache = str(os.getenv("HF_HUB_CACHE") or "").strip()
    if env_cache:
        return Path(env_cache).expanduser().resolve(strict=False)

    return (Path.home() / ".cache" / "huggingface" / "hub").resolve(strict=False)


def _job_key(model_id: str, storage_root: Path) -> tuple[str, str]:
    normalized_root = os.path.normcase(str(storage_root.resolve(strict=False)))
    return model_id, normalized_root


def _repo_root_for_model(model_id: str, storage_root: Path) -> Path:
    return storage_root / _repo_dir_name(model_id)


def _resolve_snapshot_path(repo_root: Path) -> Path | None:
    refs_main_path = repo_root / "refs" / "main"
    snapshots_dir = repo_root / "snapshots"
    snapshot_candidates: list[Path] = []

    if refs_main_path.is_file():
        try:
            revision = refs_main_path.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        if revision:
            snapshot_candidates.append(snapshots_dir / revision)

    if snapshots_dir.is_dir():
        try:
            snapshot_candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
        except OSError:
            pass

    for snapshot_path in snapshot_candidates:
        if (snapshot_path / "preprocessor_config.json").is_file():
            return snapshot_path

    return None


def _directory_size_gb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None

    total_bytes = 0
    for current_root, _, files in os.walk(path):
        for file_name in files:
            file_path = Path(current_root) / file_name
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue

    return round(total_bytes / BYTES_PER_GB, 2)


def _supported_model_specs() -> dict[str, dict[str, Any]]:
    specs = {
        str(spec["value"]): {
            "label": label,
            "backend": spec.get("backend", "whisper"),
            "approx_size_gb": spec.get("approx_size_gb"),
        }
        for label, spec in LOCAL_ASR_MODEL_SPECS.items()
    }
    
    # Add pyannote model explicitly here so it shows up in Cache Manager 
    # but not in the ASR Model dropdown list
    specs["pyannote/embedding"] = {
        "label": "Pyannote Speaker Diarization",
        "backend": "pyannote",
        "approx_size_gb": 0.05
    }
    
    return specs


def list_model_statuses(storage_path: str) -> list[dict[str, Any]]:
    storage_root = _resolve_storage_root(storage_path)
    model_specs = _supported_model_specs()
    with _model_jobs_lock:
        jobs = dict(_model_jobs)

    statuses: list[dict[str, Any]] = []
    for model_id, metadata in model_specs.items():
        repo_root = _repo_root_for_model(model_id, storage_root)
        repo_exists = repo_root.exists()
        snapshot_path = _resolve_snapshot_path(repo_root) if repo_exists else None
        job = jobs.get(_job_key(model_id, storage_root))

        if job and job.get("status") == "downloading":
            status = "downloading"
        elif snapshot_path is not None:
            status = "ready"
        elif job and job.get("status") == "error":
            status = "error"
        elif repo_exists:
            status = "partial"
        else:
            status = "missing"

        size_source = snapshot_path or (repo_root if repo_exists else None)
        statuses.append(
            {
                "id": model_id,
                "label": metadata["label"],
                "backend": metadata["backend"],
                "status": status,
                "local_path": str(snapshot_path.resolve(strict=False)) if snapshot_path is not None else None,
                "cache_path": str(repo_root.resolve(strict=False)) if repo_exists else None,
                "storage_root": str(storage_root),
                "approx_size_gb": metadata["approx_size_gb"],
                "size_on_disk_gb": _directory_size_gb(size_source),
                "error": job.get("error") if job and job.get("status") == "error" else None,
                "updated_at": job.get("updated_at") if job else None,
            }
        )

    return statuses


def queue_model_download(model_id: str, storage_path: str) -> dict[str, Any]:
    model_specs = _supported_model_specs()
    if model_id not in model_specs:
        raise ValueError(f"Unsupported model id '{model_id}'.")

    storage_root = _resolve_storage_root(storage_path)
    storage_root.mkdir(parents=True, exist_ok=True)
    current_job_key = _job_key(model_id, storage_root)

    with _model_jobs_lock:
        existing_job = _model_jobs.get(current_job_key)
        if existing_job and existing_job.get("status") == "downloading":
            return dict(existing_job)
        _model_jobs[current_job_key] = {
            "model_id": model_id,
            "storage_root": str(storage_root),
            "status": "downloading",
            "error": None,
            "updated_at": _now_iso(),
        }

    def worker() -> None:
        try:
            snapshot_download(model_id, cache_dir=str(storage_root), resume_download=True)
        except Exception as exc:
            with _model_jobs_lock:
                _model_jobs[current_job_key] = {
                    "model_id": model_id,
                    "storage_root": str(storage_root),
                    "status": "error",
                    "error": str(exc),
                    "updated_at": _now_iso(),
                }
            return

        with _model_jobs_lock:
            _model_jobs[current_job_key] = {
                "model_id": model_id,
                "storage_root": str(storage_root),
                "status": "ready",
                "error": None,
                "updated_at": _now_iso(),
            }

    threading.Thread(target=worker, daemon=True).start()
    with _model_jobs_lock:
        return dict(_model_jobs[current_job_key])


def delete_model_cache(model_id: str, storage_path: str) -> dict[str, Any]:
    model_specs = _supported_model_specs()
    if model_id not in model_specs:
        raise ValueError(f"Unsupported model id '{model_id}'.")

    storage_root = _resolve_storage_root(storage_path)
    current_job_key = _job_key(model_id, storage_root)
    repo_root = _repo_root_for_model(model_id, storage_root)

    with _model_jobs_lock:
        existing_job = _model_jobs.get(current_job_key)
        if existing_job and existing_job.get("status") == "downloading":
            raise ValueError(f"Model '{model_id}' is still downloading.")

    removed = False
    removed_path: str | None = None
    if repo_root.exists():
        shutil.rmtree(repo_root)
        removed = True
        removed_path = str(repo_root.resolve(strict=False))

    with _model_jobs_lock:
        _model_jobs.pop(current_job_key, None)

    return {
        "ok": True,
        "removed": removed,
        "removed_path": removed_path,
        "storage_root": str(storage_root),
    }

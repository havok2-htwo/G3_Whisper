"""Authenticated streaming client for the configured G3 DIA server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlsplit

import httpx

from .genesis_whisper_server_globals import current_settings, settings_lock
from .genesis_whisper_server_storage import resolve_dia_server_config


@dataclass
class DiaClientError(RuntimeError):
    status_code: int
    code: str
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def _invalid_response() -> DiaClientError:
    return DiaClientError(
        502,
        "DIA_INVALID_RESPONSE",
        "DIA-Server unterstuetzt nicht den erwarteten v2-Vertrag.",
        False,
    )


def _valid_milliseconds(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_dia_payload(payload: Any) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "2.0"
        or payload.get("status") != "completed"
        or not isinstance(payload.get("request_id"), str)
        or not payload.get("request_id")
        or not isinstance(payload.get("model"), dict)
        or not isinstance(payload.get("input"), dict)
        or not isinstance(payload.get("counts"), dict)
        or not _valid_milliseconds(payload.get("total_duration_ms"))
    ):
        raise _invalid_response()

    for key in ("diarization", "exclusive_diarization"):
        items = payload.get(key)
        if not isinstance(items, list):
            raise _invalid_response()
        for item in items:
            if not isinstance(item, dict):
                raise _invalid_response()
            start_ms = item.get("start_ms")
            end_ms = item.get("end_ms")
            speaker_id = item.get("speaker_id")
            if (
                not _valid_milliseconds(start_ms)
                or not _valid_milliseconds(end_ms)
                or end_ms <= start_ms
                or not isinstance(speaker_id, str)
                or not speaker_id.strip()
            ):
                raise _invalid_response()

    overlaps = payload.get("overlaps")
    if not isinstance(overlaps, list):
        raise _invalid_response()
    for item in overlaps:
        if not isinstance(item, dict):
            raise _invalid_response()
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        speaker_ids = item.get("speaker_ids")
        if (
            not _valid_milliseconds(start_ms)
            or not _valid_milliseconds(end_ms)
            or end_ms <= start_ms
            or not isinstance(speaker_ids, list)
            or len(speaker_ids) < 2
            or any(not isinstance(value, str) or not value.strip() for value in speaker_ids)
            or len(set(speaker_ids)) != len(speaker_ids)
        ):
            raise _invalid_response()
    return payload


def _effective_config() -> tuple[str, str]:
    with settings_lock:
        snapshot = current_settings.copy()
    resolved = resolve_dia_server_config(snapshot)
    base_url = str(resolved.get("base_url") or "").strip().rstrip("/")
    api_key = str(resolved.get("api_key") or "").strip()
    if not base_url:
        raise DiaClientError(503, "DIA_NOT_CONFIGURED", "Kein DIA-Server konfiguriert.", False)
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise DiaClientError(503, "DIA_NOT_CONFIGURED", "Die DIA-Server-Konfiguration ist ungueltig.", False)
    return base_url, api_key


async def diarize_v2(
    audio_file: BinaryIO,
    filename: str,
    content_type: str | None,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    base_url, api_key = _effective_config()
    endpoint = urljoin(base_url + "/", "v2/diarize")
    headers = {"X-API-Key": api_key} if api_key else {}
    form_data: dict[str, str] = {}
    if num_speakers is not None:
        form_data["num_speakers"] = str(num_speakers)
    if min_speakers is not None:
        form_data["min_speakers"] = str(min_speakers)
    if max_speakers is not None:
        form_data["max_speakers"] = str(max_speakers)

    try:
        audio_file.seek(0)
        timeout_seconds = max(60.0, float(os.getenv("DIA_REQUEST_TIMEOUT_SECONDS") or "7200"))
        timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=timeout_seconds, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                data=form_data,
                files={"file": (filename, audio_file, content_type or "application/octet-stream")},
            )
    except httpx.TimeoutException as exc:
        raise DiaClientError(504, "DIA_TIMEOUT", "Zeitueberschreitung beim DIA-Server.", True) from exc
    except httpx.HTTPError as exc:
        raise DiaClientError(502, "DIA_UPSTREAM_ERROR", "DIA-Server ist nicht erreichbar.", True) from exc

    if response.status_code in (401, 403):
        raise DiaClientError(
            502,
            "DIA_AUTH_FAILED",
            "Der Whisper-Server konnte sich nicht am DIA-Server authentifizieren.",
            False,
        )
    if response.status_code >= 500:
        raise DiaClientError(
            502,
            "DIA_UPSTREAM_ERROR",
            f"DIA-Server meldete HTTP {response.status_code}.",
            True,
        )
    if response.status_code != 200:
        raise DiaClientError(
            502,
            "DIA_UPSTREAM_ERROR",
            f"DIA-Server lieferte eine unerwartete Antwort (HTTP {response.status_code}).",
            False,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise DiaClientError(502, "DIA_INVALID_RESPONSE", "DIA-Antwort ist kein gueltiges JSON.", False) from exc

    return _validate_dia_payload(payload)


__all__ = ["DiaClientError", "diarize_v2"]

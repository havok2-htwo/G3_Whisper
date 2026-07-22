from __future__ import annotations

import asyncio
import io
import json
import math
import types
import unittest
from unittest import mock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.datastructures import FormData, UploadFile

from backend import genesis_whisper_server_v2 as v2
from backend.genesis_whisper_server_speaker_matching import build_robust_cloud
from backend.genesis_whisper_server_vid import EmbeddedVoiceWindow


def _vector(index: int) -> np.ndarray:
    result = np.zeros(192, dtype=np.float32)
    result[index] = 1.0
    return result


def _ready_cloud(speaker_id: str, vector: np.ndarray):
    base_index = int(np.argmax(np.abs(vector)))
    samples = [
        EmbeddedVoiceWindow(
            vector=(
                lambda candidate: candidate / np.linalg.norm(candidate)
            )(
                vector.copy()
                + np.eye(len(vector), dtype=np.float32)[(base_index + index + 1) % len(vector)] * 0.12
            ),
            start_ms=index * 3000,
            end_ms=(index + 1) * 3000,
            clean_duration_seconds=3.0,
            quality=1.0,
            stitched=False,
        )
        for index in range(3)
    ]
    return build_robust_cloud(speaker_id, samples)


def _refinement_diagnostics(
    mode: str,
    status: str,
    *,
    proposed_turns: int = 0,
    applied_turns: int = 0,
    reassigned_duration_ms: int = 0,
    rollback_reason: str | None = None,
) -> dict:
    return {
        "mode": mode,
        "status": status,
        "eligible_windows": 6,
        "proposed_turns": proposed_turns,
        "applied_turns": applied_turns,
        "reassigned_duration_ms": reassigned_duration_ms,
        "processing_ms": 17,
        "rollback_reason": rollback_reason,
        "changes": [],
        "changes_truncated": False,
    }


class _CapturingApp:
    """Capture FastAPI route functions without depending on local FastAPI's PEP 604 support."""

    def __init__(self) -> None:
        self.routes = {}
        self.state = types.SimpleNamespace(local_gpu_lock=_AsyncLock())

    def post(self, path: str, **_kwargs):
        def decorator(endpoint):
            self.routes[("POST", path)] = endpoint
            return endpoint

        return decorator


class _AsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


def _request(app: _CapturingApp, headers: dict[str, str] | None = None) -> Request:
    encoded_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v2/audio/process",
            "raw_path": b"/v2/audio/process",
            "query_string": b"",
            "headers": encoded_headers,
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
            "app": app,
        }
    )


def _upload() -> UploadFile:
    return UploadFile(filename="fixture.wav", file=io.BytesIO(b"audio fixture"))


def _payload(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


class V2RequestValidationTests(unittest.TestCase):
    def test_all_four_modes_parse(self) -> None:
        for mode in ("embedding", "transcript", "transcript_embedding"):
            with self.subTest(mode=mode):
                parsed = v2._parse_request_json(json.dumps({"schema_version": "2.0", "mode": mode}))
                self.assertEqual(parsed["mode"], mode)
        parsed = v2._parse_request_json(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "mode": "diarization",
                    "diarization": {"expected_speakers": 5, "known_speakers": []},
                }
            )
        )
        self.assertEqual(parsed["diarization"]["expected_speakers"], 5)

    def test_speaker_refinement_defaults_off_and_accepts_supported_modes(self) -> None:
        parsed = v2._parse_request_json(
            json.dumps({"schema_version": "2.0", "mode": "diarization"})
        )
        self.assertEqual(parsed["diarization"]["speaker_refinement"], "off")

        for mode in ("off", "shadow", "conservative"):
            with self.subTest(mode=mode):
                parsed = v2._parse_request_json(
                    json.dumps(
                        {
                            "schema_version": "2.0",
                            "mode": "diarization",
                            "diarization": {"speaker_refinement": mode},
                        }
                    )
                )
                self.assertEqual(parsed["diarization"]["speaker_refinement"], mode)

    def test_invalid_speaker_refinement_values_return_422(self) -> None:
        for value in (None, True, 1, "", "ON", "aggressive"):
            with self.subTest(value=value):
                with self.assertRaises(v2.V2ApiError) as raised:
                    v2._parse_request_json(
                        json.dumps(
                            {
                                "schema_version": "2.0",
                                "mode": "diarization",
                                "diarization": {"speaker_refinement": value},
                            }
                        )
                    )
                self.assertEqual(raised.exception.status_code, 422)
                self.assertEqual(raised.exception.code, "INVALID_REQUEST")

    def test_speaker_refinement_is_rejected_outside_diarization_mode(self) -> None:
        for mode in ("embedding", "transcript", "transcript_embedding"):
            with self.subTest(mode=mode):
                with self.assertRaises(v2.V2ApiError) as raised:
                    v2._parse_request_json(
                        json.dumps(
                            {
                                "schema_version": "2.0",
                                "mode": mode,
                                "diarization": {"speaker_refinement": "shadow"},
                            }
                        )
                    )
                self.assertEqual(raised.exception.status_code, 422)

    def test_unknown_speaker_audio_defaults_false_and_accepts_boolean(self) -> None:
        parsed = v2._parse_request_json(
            json.dumps({"schema_version": "2.0", "mode": "diarization"})
        )
        self.assertIs(parsed["diarization"]["unknown_speaker_audio"], False)

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                parsed = v2._parse_request_json(
                    json.dumps(
                        {
                            "schema_version": "2.0",
                            "mode": "diarization",
                            "diarization": {"unknown_speaker_audio": enabled},
                        }
                    )
                )
                self.assertIs(parsed["diarization"]["unknown_speaker_audio"], enabled)

    def test_unknown_speaker_audio_rejects_non_boolean_values(self) -> None:
        for value in (None, 0, 1, "false", "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaises(v2.V2ApiError) as raised:
                    v2._parse_request_json(
                        json.dumps(
                            {
                                "schema_version": "2.0",
                                "mode": "diarization",
                                "diarization": {"unknown_speaker_audio": value},
                            }
                        )
                    )
                self.assertEqual(raised.exception.status_code, 422)
                self.assertEqual(raised.exception.code, "INVALID_REQUEST")

    def test_512_256_nan_and_zero_profiles_return_422_validation_error(self) -> None:
        invalid_vectors = (
            [1.0] * 512,
            [1.0] * 256,
            [float("nan")] + [0.0] * 191,
            [0.0] * 192,
        )
        for vector in invalid_vectors:
            raw = json.dumps(
                {
                    "schema_version": "2.0",
                    "mode": "diarization",
                    "diarization": {
                        "known_speakers": [{"id": "person", "embeddings": [vector]}]
                    },
                }
            )
            with self.subTest(length=len(vector), first=vector[0]):
                with self.assertRaises(v2.V2ApiError) as raised:
                    v2._parse_request_json(raw)
                self.assertEqual(raised.exception.status_code, 422)


class V2MultipartIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.state.local_gpu_lock = _AsyncLock()
        v2.create_v2_api(self.app)
        self.audio = np.full(32000, 0.1, dtype=np.float32)

    @staticmethod
    def _padded_request(total_bytes: int) -> str:
        prefix = '{"schema_version":"2.0","mode":"embedding"'
        suffix = "}"
        return prefix + (" " * (total_bytes - len(prefix) - len(suffix))) + suffix

    def test_request_part_over_one_mib_is_accepted_and_upload_is_closed(self) -> None:
        raw_request = self._padded_request(2 * 1024 * 1024)
        captured_files = []

        def decode(file_obj, _filename):
            captured_files.append(file_obj)
            return self.audio

        with (
            mock.patch.object(v2, "authorize_api_key", return_value=None),
            mock.patch.object(v2, "load_audio_file", side_effect=decode),
            mock.patch.object(
                v2,
                "_generate_embedding",
                new=mock.AsyncMock(return_value=(_vector(0).tolist(), 3)),
            ),
            mock.patch.object(v2, "_record_log"),
            TestClient(self.app) as client,
        ):
            response = client.post(
                "/v2/audio/process",
                files={
                    "file": ("fixture.wav", b"audio", "audio/wav"),
                    "request": (None, raw_request, "application/json"),
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["result"]["embedding"]["vector"]), 192)
        self.assertEqual(len(captured_files), 1)
        self.assertTrue(captured_files[0].closed)

    def test_named_json_file_part_is_supported(self) -> None:
        raw_request = json.dumps({"schema_version": "2.0", "mode": "embedding"})
        with (
            mock.patch.object(v2, "authorize_api_key", return_value=None),
            mock.patch.object(v2, "load_audio_file", return_value=self.audio),
            mock.patch.object(
                v2,
                "_generate_embedding",
                new=mock.AsyncMock(return_value=(_vector(0).tolist(), 3)),
            ),
            mock.patch.object(v2, "_record_log"),
            TestClient(self.app) as client,
        ):
            response = client.post(
                "/v2/audio/process",
                files={
                    "file": ("fixture.wav", b"audio", "audio/wav"),
                    "request": ("request.json", raw_request, "application/json"),
                },
            )
        self.assertEqual(response.status_code, 200, response.text)

    def test_request_part_over_16_mib_returns_v2_413(self) -> None:
        raw_request = self._padded_request(v2.V2_REQUEST_JSON_MAX_BYTES + 2)
        for request_filename in (None, "request.json"):
            with self.subTest(file_style=request_filename is not None):
                with (
                    mock.patch.object(v2, "authorize_api_key", return_value=None),
                    mock.patch.object(v2, "load_audio_file") as decode,
                    TestClient(self.app) as client,
                ):
                    response = client.post(
                        "/v2/audio/process",
                        files={
                            "file": ("fixture.wav", b"audio", "audio/wav"),
                            "request": (request_filename, raw_request, "application/json"),
                        },
                    )

                self.assertEqual(response.status_code, 413, response.text)
                payload = response.json()
                self.assertEqual(payload["error"]["code"], "REQUEST_METADATA_TOO_LARGE")
                self.assertIsNone(payload["mode"])
                self.assertEqual(payload["models"], {})
                self.assertIn("total", payload["timings_ms"])
                self.assertIsNone(payload["result"])
                self.assertEqual(payload["warnings"], [])
                decode.assert_not_called()

    def test_expected_speakers_is_exact_1_to_64_and_not_below_known_count(self) -> None:
        for invalid in (0, 65, True, "5"):
            with self.subTest(value=invalid):
                with self.assertRaises(v2.V2ApiError) as raised:
                    v2._parse_request_json(
                        json.dumps(
                            {
                                "schema_version": "2.0",
                                "mode": "diarization",
                                "diarization": {"expected_speakers": invalid},
                            }
                        )
                    )
                self.assertEqual(raised.exception.status_code, 422)

        known = [
            {"id": f"p{index}", "embeddings": [_vector(index).tolist()]}
            for index in range(2)
        ]
        with self.assertRaises(v2.V2ApiError):
            v2._parse_request_json(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "mode": "diarization",
                        "diarization": {"expected_speakers": 1, "known_speakers": known},
                    }
                )
            )


class V2EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = _CapturingApp()
        v2.create_v2_api(self.app)
        self.endpoint = self.app.routes[("POST", "/v2/audio/process")]
        self.audio = np.full(16000 * 10, 0.1, dtype=np.float32)

    def _call(self, request_body: dict, *, headers: dict[str, str] | None = None):
        upload = _upload()
        raw_request = json.dumps(request_body)
        form = FormData([("file", upload), ("request", raw_request)])
        with mock.patch.object(
            v2,
            "_parse_multipart_parts",
            new=mock.AsyncMock(return_value=(form, upload, raw_request)),
        ):
            return asyncio.run(
                self.endpoint(
                    _request(self.app, headers),
                )
            )

    def _base_patches(self):
        return (
            mock.patch.object(v2, "authorize_api_key", return_value=None),
            mock.patch.object(v2, "load_audio_file", return_value=self.audio),
            mock.patch.object(v2, "_record_log"),
        )

    def test_embedding_mode_returns_exactly_one_192_vector(self) -> None:
        normalized = [1.0 / math.sqrt(192)] * 192
        patches = self._base_patches()
        with patches[0], patches[1], patches[2], mock.patch.object(
            v2, "_generate_embedding", new=mock.AsyncMock(return_value=(normalized, 11))
        ):
            response = self._call({"schema_version": "2.0", "mode": "embedding"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["mode"], "embedding")
        self.assertEqual(len(payload["result"]["embedding"]["vector"]), 192)
        self.assertEqual(payload["models"]["embedding"]["dimension"], 192)
        self.assertNotIn("embedding_space_id", json.dumps(payload))

    def test_authenticated_admin_session_can_use_v2_tester_without_client_key(self) -> None:
        normalized = _vector(0).tolist()
        with (
            mock.patch.object(
                v2,
                "authorize_api_key",
                side_effect=HTTPException(status_code=401, detail="API key required"),
            ),
            mock.patch.object(v2, "require_admin", return_value={"username": "admin"}) as require_admin,
            mock.patch.object(v2, "load_audio_file", return_value=self.audio),
            mock.patch.object(v2, "_record_log"),
            mock.patch.object(
                v2,
                "_generate_embedding",
                new=mock.AsyncMock(return_value=(normalized, 3)),
            ),
        ):
            response = self._call({"schema_version": "2.0", "mode": "embedding"})

        self.assertEqual(response.status_code, 200, _payload(response))
        require_admin.assert_called_once()

    def test_v2_preserves_api_key_error_without_admin_session(self) -> None:
        with (
            mock.patch.object(
                v2,
                "authorize_api_key",
                side_effect=HTTPException(status_code=401, detail="API key required"),
            ),
            mock.patch.object(
                v2,
                "require_admin",
                side_effect=HTTPException(status_code=401, detail="Authentication required"),
            ),
        ):
            response = self._call({"schema_version": "2.0", "mode": "embedding"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(payload["error"]["code"], "INVALID_API_KEY")

    def test_transcript_mode_has_no_embedding_or_dia_model(self) -> None:
        patches = self._base_patches()
        with patches[0], patches[1], patches[2], mock.patch.object(
            v2,
            "_transcribe_audio",
            new=mock.AsyncMock(return_value=("Ein Transkript", 7, 1, "asr-model")),
        ):
            response = self._call({"schema_version": "2.0", "mode": "transcript"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"], {"transcript": {"text": "Ein Transkript"}})
        self.assertEqual(payload["models"], {"asr": {"id": "asr-model"}})

    def test_explicit_invalid_api_key_is_not_hidden_by_admin_session(self) -> None:
        with (
            mock.patch.object(v2, "authorize_api_key", side_effect=HTTPException(status_code=401, detail="bad key")),
            mock.patch.object(v2, "require_admin", return_value={"username": "admin"}) as require_admin,
        ):
            response = self._call(
                {"schema_version": "2.0", "mode": "embedding"},
                headers={"X-API-Key": "invalid"},
            )

        payload = _payload(response)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(payload["error"]["code"], "INVALID_API_KEY")
        require_admin.assert_not_called()

    def test_transcript_embedding_returns_both_and_only_192_dimensions(self) -> None:
        normalized = _vector(3).tolist()
        patches = self._base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(
                v2,
                "_transcribe_audio",
                new=mock.AsyncMock(return_value=("Ein Transkript", 7, 1, "asr-model")),
            ),
            mock.patch.object(
                v2, "_generate_embedding", new=mock.AsyncMock(return_value=(normalized, 11))
            ),
        ):
            response = self._call(
                {"schema_version": "2.0", "mode": "transcript_embedding"}
            )

        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["transcript"]["text"], "Ein Transkript")
        self.assertEqual(len(payload["result"]["embedding"]["vector"]), 192)
        self.assertEqual(payload["models"]["embedding"]["dimension"], 192)

    def test_transcript_embedding_keeps_transcript_when_voice_window_is_unsuitable(self) -> None:
        patches = self._base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(
                v2,
                "_transcribe_audio",
                new=mock.AsyncMock(return_value=("Trotzdem transkribiert", 7, 1, "asr-model")),
            ),
            mock.patch.object(
                v2,
                "_generate_embedding",
                new=mock.AsyncMock(
                    side_effect=ValueError(
                        "Keine qualitativ geeigneten Sprachfenster fuer ein Stimmembedding gefunden."
                    )
                ),
            ),
        ):
            response = self._call(
                {"schema_version": "2.0", "mode": "transcript_embedding"}
            )

        payload = _payload(response)
        self.assertEqual(response.status_code, 200, payload)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(
            payload["result"],
            {
                "transcript": {"text": "Trotzdem transkribiert"},
                "embedding": None,
            },
        )
        self.assertEqual(payload["warnings"][0]["code"], "VOICE_EMBEDDING_UNAVAILABLE")
        self.assertIn("embedding", payload["timings_ms"])
        self.assertEqual(payload["models"]["embedding"]["dimension"], 192)

    def test_dia_orchestration_runs_only_for_diarization_mode(self) -> None:
        process_dia = mock.AsyncMock(
            return_value=(
                {
                    "transcript": {"text": "Diarisiert", "segments": []},
                    "speaker_counts": {"expected": 5, "detected": 5},
                    "speaker_assignments": [],
                    "unknown_speakers": [],
                    "unresolved_known_speakers": [],
                },
                {"diarization": 4, "transcription": 5, "embedding": 6},
                {
                    "asr": {"id": "asr"},
                    "diarization": {"id": "community-1"},
                    "embedding": {"id": "ReDimNet2-B6", "dimension": 192},
                },
                [],
                "Diarisiert",
            )
        )
        transcribe = mock.AsyncMock(return_value=("Text", 7, 1, "asr"))
        embedding = mock.AsyncMock(return_value=(_vector(0).tolist(), 4))
        patches = self._base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(v2, "_process_diarization", new=process_dia),
            mock.patch.object(v2, "_transcribe_audio", new=transcribe),
            mock.patch.object(v2, "_generate_embedding", new=embedding),
        ):
            for mode in ("embedding", "transcript", "transcript_embedding"):
                response = self._call({"schema_version": "2.0", "mode": mode})
                self.assertEqual(response.status_code, 200)
            self.assertEqual(process_dia.await_count, 0)

            response = self._call(
                {
                    "schema_version": "2.0",
                    "mode": "diarization",
                    "diarization": {"expected_speakers": 5},
                }
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(process_dia.await_count, 1)
        configuration = process_dia.await_args.args[3]
        self.assertEqual(configuration["expected_speakers"], 5)

    def test_default_diarization_passes_refinement_off_without_changing_result_shape(self) -> None:
        legacy_result = {
            "transcript": {"text": "Diarisiert", "segments": []},
            "speaker_counts": {"expected": None, "detected": 1},
            "speaker_assignments": [],
            "unknown_speakers": [],
            "unresolved_speakers": [],
            "unresolved_known_speakers": [],
        }
        process_dia = mock.AsyncMock(
            return_value=(
                legacy_result,
                {"diarization": 4, "transcription": 5, "embedding": 6},
                {
                    "asr": {"id": "asr"},
                    "diarization": {"id": "community-1"},
                    "embedding": {"id": "ReDimNet2-B6", "dimension": 192},
                },
                [],
                "Diarisiert",
            )
        )
        patches = self._base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(v2, "_process_diarization", new=process_dia),
        ):
            response = self._call({"schema_version": "2.0", "mode": "diarization"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 200, payload)
        configuration = process_dia.await_args.args[3]
        self.assertEqual(configuration["speaker_refinement"], "off")
        self.assertEqual(payload["result"], legacy_result)
        self.assertNotIn("speaker_refinement", payload["result"])
        self.assertNotIn("speaker_refinement", payload["timings_ms"])

    def test_enabled_refinement_metadata_and_timing_pass_through_v2_response(self) -> None:
        diagnostics = _refinement_diagnostics(
            "shadow",
            "shadow",
            proposed_turns=2,
        )
        process_dia = mock.AsyncMock(
            return_value=(
                {
                    "transcript": {"text": "Diarisiert", "segments": []},
                    "speaker_counts": {"expected": None, "detected": 2},
                    "speaker_assignments": [],
                    "unknown_speakers": [],
                    "unresolved_speakers": [],
                    "unresolved_known_speakers": [],
                    "speaker_refinement": diagnostics,
                },
                {
                    "diarization": 4,
                    "speaker_refinement": 17,
                    "transcription": 5,
                    "embedding": 6,
                },
                {
                    "asr": {"id": "asr"},
                    "diarization": {"id": "community-1"},
                    "embedding": {"id": "ReDimNet2-B6", "dimension": 192},
                },
                [],
                "Diarisiert",
            )
        )
        patches = self._base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(v2, "_process_diarization", new=process_dia),
        ):
            response = self._call(
                {
                    "schema_version": "2.0",
                    "mode": "diarization",
                    "diarization": {
                        "speaker_refinement": "shadow",
                        "unknown_speaker_audio": True,
                    },
                }
            )

        payload = _payload(response)
        self.assertEqual(response.status_code, 200, payload)
        configuration = process_dia.await_args.args[3]
        self.assertEqual(configuration["speaker_refinement"], "shadow")
        self.assertIs(configuration["unknown_speaker_audio"], True)
        self.assertEqual(payload["result"]["speaker_refinement"], diagnostics)
        self.assertEqual(payload["timings_ms"]["speaker_refinement"], 17)

    def test_invalid_profiles_are_api_422_before_audio_decode(self) -> None:
        invalid_vectors = (
            [1.0] * 512,
            [1.0] * 256,
            [float("nan")] + [0.0] * 191,
            [0.0] * 192,
        )
        for vector in invalid_vectors:
            with self.subTest(length=len(vector), first=vector[0]):
                decode = mock.Mock(return_value=self.audio)
                with (
                    mock.patch.object(v2, "authorize_api_key", return_value=None),
                    mock.patch.object(v2, "load_audio_file", new=decode),
                ):
                    response = self._call(
                        {
                            "schema_version": "2.0",
                            "mode": "diarization",
                            "diarization": {
                                "known_speakers": [
                                    {"id": "old-profile", "embeddings": [vector]}
                                ]
                            },
                        }
                    )

                payload = _payload(response)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")
                self.assertIsNone(payload["mode"])
                self.assertEqual(payload["models"], {})
                self.assertIn("total", payload["timings_ms"])
                self.assertIsNone(payload["result"])
                self.assertEqual(payload["warnings"], [])
                decode.assert_not_called()

    def test_invalid_diarization_options_are_api_422_before_audio_decode(self) -> None:
        invalid_options = (
            {"speaker_refinement": "aggressive"},
            {"speaker_refinement": None},
            {"unknown_speaker_audio": "true"},
            {"unknown_speaker_audio": 1},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                decode = mock.Mock(return_value=self.audio)
                with (
                    mock.patch.object(v2, "authorize_api_key", return_value=None),
                    mock.patch.object(v2, "load_audio_file", new=decode),
                ):
                    response = self._call(
                        {
                            "schema_version": "2.0",
                            "mode": "diarization",
                            "diarization": options,
                        }
                    )

                payload = _payload(response)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")
                decode.assert_not_called()


class V2DiarizationAssemblyTests(unittest.TestCase):
    def _run_refinement_fixture(
        self,
        *,
        mode: str,
        refined_turns: list[dict],
        diagnostics: dict,
    ):
        original_turns = [
            {"start_ms": 0, "end_ms": 6000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 6000, "end_ms": 12000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 12000, "end_ms": 18000, "speaker_id": "SPEAKER_B"},
        ]
        clouds = {
            "SPEAKER_A": _ready_cloud("SPEAKER_A", _vector(0)),
            "SPEAKER_B": _ready_cloud("SPEAKER_B", _vector(1)),
        }
        dia_response = {
            "model": {"id": "community-1"},
            "diarization": original_turns,
            "exclusive_diarization": original_turns,
            "overlaps": [],
        }
        result_object = types.SimpleNamespace(
            turns=refined_turns,
            speaker_clouds=clouds,
            diagnostics=diagnostics,
        )
        events: list[str] = []
        extract_mock = mock.Mock(side_effect=lambda *_args: events.append("embedding") or clouds)
        refine_mock = mock.Mock(side_effect=lambda *_args, **_kwargs: events.append("refinement") or result_object)

        async def transcribe(_request, _audio, turns, _overlaps, _filter):
            events.append("transcription")
            segments = []
            for index, turn in enumerate(turns):
                original_id = str(turn.get("original_speaker_id", turn["speaker_id"]))
                refined_id = str(turn["speaker_id"])
                segment = {
                    "index": index,
                    "start_ms": int(turn["start_ms"]),
                    "end_ms": int(turn["end_ms"]),
                    "diarization_speaker_id": original_id,
                    "refined_diarization_speaker_id": refined_id,
                    "text": f"Turn {index}",
                    "overlap": False,
                }
                segments.append(segment)
            return segments, 20, "asr-model"

        app = _CapturingApp()
        request = _request(app)
        audio = np.full(16000 * 18, 0.1, dtype=np.float32)
        with (
            mock.patch.object(v2, "diarize_v2", new=mock.AsyncMock(return_value=dia_response)),
            mock.patch.object(v2, "extract_speaker_clouds", new=extract_mock),
            mock.patch.object(
                v2,
                "generate_voice_vector",
                side_effect=AssertionError("Refinement must reuse speaker-cloud embeddings"),
            ),
            mock.patch.object(v2, "refine_speaker_turns", new=refine_mock),
            mock.patch.object(v2, "_transcribe_turns", new=mock.AsyncMock(side_effect=transcribe)),
        ):
            processed = asyncio.run(
                v2._process_diarization(
                    request,
                    _upload(),
                    audio,
                    {
                        "expected_speakers": 2,
                        "known_speakers": [],
                        "speaker_refinement": mode,
                        "unknown_speaker_audio": False,
                    },
                    True,
                )
            )
        return processed, events, extract_mock, refine_mock

    def test_conservative_refinement_runs_once_before_asr_and_preserves_provenance(self) -> None:
        refined_turns = [
            {
                "start_ms": 0,
                "end_ms": 6000,
                "speaker_id": "SPEAKER_A",
                "original_speaker_id": "SPEAKER_A",
            },
            {
                "start_ms": 6000,
                "end_ms": 12000,
                "speaker_id": "SPEAKER_B",
                "original_speaker_id": "SPEAKER_A",
            },
            {
                "start_ms": 12000,
                "end_ms": 18000,
                "speaker_id": "SPEAKER_B",
                "original_speaker_id": "SPEAKER_B",
            },
        ]
        diagnostics = _refinement_diagnostics(
            "conservative",
            "applied",
            proposed_turns=1,
            applied_turns=1,
            reassigned_duration_ms=6000,
        )

        (result, timings, _models, _warnings, _text), events, extract, refine = (
            self._run_refinement_fixture(
                mode="conservative",
                refined_turns=refined_turns,
                diagnostics=diagnostics,
            )
        )

        self.assertEqual(events, ["embedding", "refinement", "transcription"])
        extract.assert_called_once()
        refine.assert_called_once()
        refine_mode = refine.call_args.kwargs.get("mode")
        if refine_mode is None:
            refine_mode = refine.call_args.args[0]
        self.assertEqual(refine_mode, "conservative")
        self.assertEqual(result["speaker_refinement"], diagnostics)
        self.assertEqual(timings["speaker_refinement"], 17)
        changed = next(
            segment
            for segment in result["transcript"]["segments"]
            if segment["start_ms"] == 6000
        )
        self.assertEqual(changed["diarization_speaker_id"], "SPEAKER_A")
        self.assertEqual(changed["refined_diarization_speaker_id"], "SPEAKER_B")
        self.assertEqual(changed["speaker_id"], "SPEAKER_B")

    def test_rejected_refinement_rolls_back_all_turn_labels(self) -> None:
        rolled_back_turns = [
            {**turn, "original_speaker_id": turn["speaker_id"]}
            for turn in (
                {"start_ms": 0, "end_ms": 6000, "speaker_id": "SPEAKER_A"},
                {"start_ms": 6000, "end_ms": 12000, "speaker_id": "SPEAKER_A"},
                {"start_ms": 12000, "end_ms": 18000, "speaker_id": "SPEAKER_B"},
            )
        ]
        diagnostics = _refinement_diagnostics(
            "conservative",
            "rejected",
            proposed_turns=1,
            rollback_reason="moved_speech_limit_exceeded",
        )

        (result, timings, _models, _warnings, _text), events, extract, _refine = (
            self._run_refinement_fixture(
                mode="conservative",
                refined_turns=rolled_back_turns,
                diagnostics=diagnostics,
            )
        )

        self.assertEqual(events, ["embedding", "refinement", "transcription"])
        extract.assert_called_once()
        self.assertEqual(result["speaker_refinement"]["status"], "rejected")
        self.assertEqual(
            result["speaker_refinement"]["rollback_reason"],
            "moved_speech_limit_exceeded",
        )
        self.assertEqual(timings["speaker_refinement"], 17)
        for segment in result["transcript"]["segments"]:
            self.assertEqual(
                segment["refined_diarization_speaker_id"],
                segment["diarization_speaker_id"],
            )

    def test_shadow_reports_proposals_without_applying_labels(self) -> None:
        shadow_turns = [
            {**turn, "original_speaker_id": turn["speaker_id"]}
            for turn in (
                {"start_ms": 0, "end_ms": 6000, "speaker_id": "SPEAKER_A"},
                {"start_ms": 6000, "end_ms": 12000, "speaker_id": "SPEAKER_A"},
                {"start_ms": 12000, "end_ms": 18000, "speaker_id": "SPEAKER_B"},
            )
        ]
        diagnostics = _refinement_diagnostics(
            "shadow",
            "shadow",
            proposed_turns=1,
            applied_turns=0,
            reassigned_duration_ms=0,
        )

        (result, timings, _models, _warnings, _text), _events, extract, _refine = (
            self._run_refinement_fixture(
                mode="shadow",
                refined_turns=shadow_turns,
                diagnostics=diagnostics,
            )
        )

        extract.assert_called_once()
        self.assertEqual(result["speaker_refinement"]["status"], "shadow")
        self.assertEqual(result["speaker_refinement"]["proposed_turns"], 1)
        self.assertEqual(result["speaker_refinement"]["applied_turns"], 0)
        self.assertEqual(timings["speaker_refinement"], 17)
        for segment in result["transcript"]["segments"]:
            self.assertEqual(
                segment["refined_diarization_speaker_id"],
                segment["diarization_speaker_id"],
            )

    def test_refined_turns_preserve_original_dia_provenance(self) -> None:
        turns = [
            {
                "start_ms": 0,
                "end_ms": 3000,
                "speaker_id": "SPEAKER_B",
                "original_speaker_id": "SPEAKER_A",
            }
        ]
        app = _CapturingApp()
        app.state.whisper_batch_manager = object()
        request = _request(app)
        audio = np.full(16000 * 3, 0.1, dtype=np.float32)
        batch_result = types.SimpleNamespace(text="Korrigierter Turn")

        with (
            mock.patch.object(
                v2,
                "_processing_key",
                return_value=("asr", "cuda", "cache", "de", "fp16"),
            ),
            mock.patch.object(
                v2,
                "enqueue_audio_segments_bounded",
                new=mock.AsyncMock(return_value=[batch_result]),
            ),
        ):
            segments, _duration_ms, _model_id = asyncio.run(
                v2._transcribe_turns(request, audio, turns, [], True)
            )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["diarization_speaker_id"], "SPEAKER_A")
        self.assertEqual(segments[0]["refined_diarization_speaker_id"], "SPEAKER_B")

    def test_merge_does_not_erase_original_speaker_boundary(self) -> None:
        turns = [
            {
                "start_ms": 0,
                "end_ms": 3000,
                "speaker_id": "SPEAKER_B",
                "original_speaker_id": "SPEAKER_A",
            },
            {
                "start_ms": 3000,
                "end_ms": 6000,
                "speaker_id": "SPEAKER_B",
                "original_speaker_id": "SPEAKER_B",
            },
        ]

        merged = v2._merge_exclusive_turns(turns)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["original_speaker_id"], "SPEAKER_A")
        self.assertEqual(merged[1]["original_speaker_id"], "SPEAKER_B")

    def test_five_expected_four_known_and_one_unknown_are_assembled(self) -> None:
        vectors = [_vector(index) for index in range(5)]
        clouds = {
            f"SPEAKER_{index:02d}": _ready_cloud(f"SPEAKER_{index:02d}", vector)
            for index, vector in enumerate(vectors)
        }
        known = [
            {"id": f"person-{index}", "embeddings": [vectors[index].tolist()]}
            for index in range(4)
        ]
        exclusive = [
            {"start_ms": index * 2000, "end_ms": (index + 1) * 2000, "speaker_id": speaker}
            for index, speaker in enumerate(clouds)
        ]
        segments = [
            {
                "index": index,
                "start_ms": item["start_ms"],
                "end_ms": item["end_ms"],
                "diarization_speaker_id": item["speaker_id"],
                "text": f"Text {index}",
                "overlap": False,
            }
            for index, item in enumerate(exclusive)
        ]
        dia_response = {
            "model": {"id": "community-1"},
            "diarization": exclusive,
            "exclusive_diarization": exclusive,
            "overlaps": [],
        }
        app = _CapturingApp()
        request = _request(app)
        audio = np.full(16000 * 10, 0.1, dtype=np.float32)
        dia_mock = mock.AsyncMock(return_value=dia_response)
        transcribe_mock = mock.AsyncMock(return_value=(segments, 20, "asr-model"))
        build_audio_mock = mock.Mock()

        with (
            mock.patch.object(v2, "diarize_v2", new=dia_mock),
            mock.patch.object(v2, "_transcribe_turns", new=transcribe_mock),
            mock.patch.object(v2, "extract_speaker_clouds", return_value=clouds),
            mock.patch.object(v2, "build_unknown_speaker_audio_assets", new=build_audio_mock),
        ):
            result, timings, models, warnings, transcript = asyncio.run(
                v2._process_diarization(
                    request,
                    _upload(),
                    audio,
                    {"expected_speakers": 5, "known_speakers": known},
                    True,
                )
            )

        dia_mock.assert_awaited_once()
        build_audio_mock.assert_not_called()
        self.assertEqual(dia_mock.await_args.kwargs["num_speakers"], 5)
        self.assertIsNone(dia_mock.await_args.kwargs["min_speakers"])
        self.assertEqual(result["speaker_counts"]["detected"], 5)
        self.assertEqual(result["speaker_counts"]["known_assigned"], 4)
        self.assertEqual(result["speaker_counts"]["unknown"], 1)
        self.assertEqual(result["unresolved_known_speakers"], [])
        self.assertEqual(len(result["unknown_speakers"]), 1)
        self.assertEqual(result["unknown_speakers"][0]["speaker_id"], "SPEAKER_04")
        self.assertTrue(
            all(len(item["vector"]) == 192 for item in result["unknown_speakers"][0]["embeddings"])
        )
        self.assertEqual(models["embedding"]["dimension"], 192)
        self.assertEqual(models["diarization"], {"id": "community-1"})
        self.assertEqual(timings["transcription"], 20)
        self.assertEqual(warnings, [])
        self.assertIn("Text 4", transcript)
        by_dia_id = {
            item["diarization_speaker_id"]: item for item in result["transcript"]["segments"]
        }
        self.assertEqual(by_dia_id["SPEAKER_00"]["speaker_id"], "person-0")
        self.assertEqual(by_dia_id["SPEAKER_00"]["speaker_kind"], "known")
        self.assertEqual(by_dia_id["SPEAKER_04"]["speaker_id"], "SPEAKER_04")
        self.assertEqual(by_dia_id["SPEAKER_04"]["speaker_kind"], "unknown")

    def test_requested_audio_assets_are_added_only_to_unknown_and_unresolved_speakers(self) -> None:
        speaker_ids = ("SPEAKER_KNOWN", "SPEAKER_UNKNOWN", "SPEAKER_UNRESOLVED")
        clouds = {
            speaker_id: _ready_cloud(speaker_id, _vector(index))
            for index, speaker_id in enumerate(speaker_ids)
        }
        turns = [
            {"start_ms": index * 6000, "end_ms": (index + 1) * 6000, "speaker_id": speaker_id}
            for index, speaker_id in enumerate(speaker_ids)
        ]
        transcript_segments = [
            {
                "index": index,
                "start_ms": turn["start_ms"],
                "end_ms": turn["end_ms"],
                "diarization_speaker_id": turn["speaker_id"],
                "text": f"Turn {index}",
                "overlap": False,
            }
            for index, turn in enumerate(turns)
        ]
        dia_response = {
            "model": {"id": "community-1"},
            "diarization": turns,
            "exclusive_diarization": turns,
            "overlaps": [],
        }
        audio_assets = {
            "SPEAKER_UNKNOWN": {
                "mime_type": "audio/mpeg",
                "encoding": "base64",
                "data": "dW5rbm93bg==",
                "duration_ms": 6000,
                "snippets": [
                    {"start_ms": 6000, "end_ms": 12000, "duration_ms": 6000, "centrality": 0.9}
                ],
            },
            "SPEAKER_UNRESOLVED": {
                "mime_type": "audio/mpeg",
                "encoding": "base64",
                "data": "dW5yZXNvbHZlZA==",
                "duration_ms": 6000,
                "snippets": [
                    {"start_ms": 12000, "end_ms": 18000, "duration_ms": 6000, "centrality": 0.8}
                ],
            },
        }
        build_audio = mock.Mock(return_value=audio_assets)
        matches = (
            {"SPEAKER_KNOWN": {"speaker_id": "person-known", "cosine_similarity": 0.9}},
            ["person-unresolved"],
            {"SPEAKER_UNRESOLVED": {"reason": "weak_evidence"}},
        )
        app = _CapturingApp()
        request = _request(app)
        audio = np.full(16000 * 18, 0.1, dtype=np.float32)

        with (
            mock.patch.object(v2, "diarize_v2", new=mock.AsyncMock(return_value=dia_response)),
            mock.patch.object(
                v2,
                "_transcribe_turns",
                new=mock.AsyncMock(return_value=(transcript_segments, 20, "asr-model")),
            ),
            mock.patch.object(v2, "extract_speaker_clouds", return_value=clouds),
            mock.patch.object(v2, "match_known_speakers", return_value=matches),
            mock.patch.object(v2, "build_unknown_speaker_audio_assets", new=build_audio),
        ):
            result, _timings, _models, _warnings, _transcript = asyncio.run(
                v2._process_diarization(
                    request,
                    _upload(),
                    audio,
                    {
                        "expected_speakers": 3,
                        "known_speakers": [
                            {"id": "person-known", "embeddings": [_vector(0).tolist()]},
                            {"id": "person-unresolved", "embeddings": [_vector(2).tolist()]},
                        ],
                        "speaker_refinement": "off",
                        "unknown_speaker_audio": True,
                    },
                    True,
                )
            )

        build_audio.assert_called_once()
        requested_ids = set(build_audio.call_args.args[2])
        self.assertEqual(requested_ids, {"SPEAKER_UNKNOWN", "SPEAKER_UNRESOLVED"})
        unknown = {item["diarization_speaker_id"]: item for item in result["unknown_speakers"]}
        unresolved = {
            item["diarization_speaker_id"]: item for item in result["unresolved_speakers"]
        }
        self.assertEqual(unknown["SPEAKER_UNKNOWN"]["audio"], audio_assets["SPEAKER_UNKNOWN"])
        self.assertEqual(
            unresolved["SPEAKER_UNRESOLVED"]["audio"],
            audio_assets["SPEAKER_UNRESOLVED"],
        )
        known_assignment = next(
            item for item in result["speaker_assignments"] if item["kind"] == "known"
        )
        self.assertNotIn("audio", known_assignment)

    def test_missing_requested_unknown_audio_is_omitted_with_aggregated_warning(self) -> None:
        speaker_id = "SPEAKER_UNKNOWN"
        turns = [{"start_ms": 0, "end_ms": 6000, "speaker_id": speaker_id}]
        cloud = _ready_cloud(speaker_id, _vector(0))
        dia_response = {
            "model": {"id": "community-1"},
            "diarization": turns,
            "exclusive_diarization": turns,
            "overlaps": [],
        }
        segments = [
            {
                "index": 0,
                "start_ms": 0,
                "end_ms": 6000,
                "diarization_speaker_id": speaker_id,
                "text": "Unbekannt",
                "overlap": False,
            }
        ]
        app = _CapturingApp()
        request = _request(app)
        audio = np.full(16000 * 6, 0.1, dtype=np.float32)
        with (
            mock.patch.object(v2, "diarize_v2", new=mock.AsyncMock(return_value=dia_response)),
            mock.patch.object(
                v2,
                "_transcribe_turns",
                new=mock.AsyncMock(return_value=(segments, 20, "asr-model")),
            ),
            mock.patch.object(v2, "extract_speaker_clouds", return_value={speaker_id: cloud}),
            mock.patch.object(v2, "build_unknown_speaker_audio_assets", return_value={}),
        ):
            result, _timings, _models, warnings, _transcript = asyncio.run(
                v2._process_diarization(
                    request,
                    _upload(),
                    audio,
                    {
                        "expected_speakers": 1,
                        "known_speakers": [],
                        "speaker_refinement": "off",
                        "unknown_speaker_audio": True,
                    },
                    True,
                )
            )

        self.assertNotIn("audio", result["unknown_speakers"][0])
        self.assertIn(
            {
                "code": "UNKNOWN_SPEAKER_AUDIO_UNAVAILABLE",
                "speaker_ids": [speaker_id],
            },
            warnings,
        )


if __name__ == "__main__":
    unittest.main()

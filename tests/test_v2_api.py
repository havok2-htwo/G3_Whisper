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
from backend import genesis_whisper_server_lid as lid
from backend.genesis_whisper_server_globals import COHERE_LANGUAGE_CODES
from backend.genesis_whisper_server_lid import LanguageDecision
from backend.genesis_whisper_server_vid import EmbeddedVoiceWindow

# What _transcribe_audio reports for a request that ran with the de setting.
_GERMAN = LanguageDecision(languages=[], language="de", source="setting")


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

    def get(self, path: str, **_kwargs):
        def decorator(endpoint):
            self.routes[("GET", path)] = endpoint
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

    def test_v2_reports_a_refused_gate_as_invalid_api_key(self) -> None:
        # The gate itself (keys, Bearer, admin session, explicit key wins) is tested in test_api_keys.py.
        with mock.patch.object(
            v2,
            "authorize_api_key",
            side_effect=HTTPException(status_code=401, detail="API key required"),
        ):
            response = self._call({"schema_version": "2.0", "mode": "embedding"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(payload["error"]["code"], "INVALID_API_KEY")

    def test_transcript_mode_has_no_embedding_or_diarization_model(self) -> None:
        patches = self._base_patches()
        with patches[0], patches[1], patches[2], mock.patch.object(
            v2,
            "_transcribe_audio",
            new=mock.AsyncMock(return_value=("Ein Transkript", 7, 1, "asr-model", _GERMAN)),
        ):
            response = self._call({"schema_version": "2.0", "mode": "transcript"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"], {"transcript": {"text": "Ein Transkript", "language": "de"}})
        # The language the ASR ran with; no detection block when no detection ran.
        self.assertEqual(payload["models"], {"asr": {"id": "asr-model", "language": "de"}})

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
                new=mock.AsyncMock(return_value=("Ein Transkript", 7, 1, "asr-model", _GERMAN)),
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
                new=mock.AsyncMock(return_value=("Trotzdem transkribiert", 7, 1, "asr-model", _GERMAN)),
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
                "transcript": {"text": "Trotzdem transkribiert", "language": "de"},
                "embedding": None,
            },
        )
        self.assertEqual(payload["warnings"][0]["code"], "VOICE_EMBEDDING_UNAVAILABLE")
        self.assertIn("embedding", payload["timings_ms"])
        self.assertEqual(payload["models"]["embedding"]["dimension"], 192)

    def test_diarization_orchestration_runs_only_for_diarization_mode(self) -> None:
        process_diarization_mock = mock.AsyncMock(
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
        transcribe = mock.AsyncMock(return_value=("Text", 7, 1, "asr", _GERMAN))
        embedding = mock.AsyncMock(return_value=(_vector(0).tolist(), 4))
        patches = self._base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(v2, "_process_diarization", new=process_diarization_mock),
            mock.patch.object(v2, "_transcribe_audio", new=transcribe),
            mock.patch.object(v2, "_generate_embedding", new=embedding),
        ):
            for mode in ("embedding", "transcript", "transcript_embedding"):
                response = self._call({"schema_version": "2.0", "mode": mode})
                self.assertEqual(response.status_code, 200)
            self.assertEqual(process_diarization_mock.await_count, 0)

            response = self._call(
                {
                    "schema_version": "2.0",
                    "mode": "diarization",
                    "diarization": {"expected_speakers": 5},
                }
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(process_diarization_mock.await_count, 1)
        configuration = process_diarization_mock.await_args.args[3]
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
        process_diarization_mock = mock.AsyncMock(
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
            mock.patch.object(v2, "_process_diarization", new=process_diarization_mock),
        ):
            response = self._call({"schema_version": "2.0", "mode": "diarization"})

        payload = _payload(response)
        self.assertEqual(response.status_code, 200, payload)
        configuration = process_diarization_mock.await_args.args[3]
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
        process_diarization_mock = mock.AsyncMock(
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
            mock.patch.object(v2, "_process_diarization", new=process_diarization_mock),
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
        configuration = process_diarization_mock.await_args.args[3]
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


def _silent_frame_probs(audio):
    """Silero stand-in for the diarization fixtures: no speech, so the WXC path falls back to the turn path."""

    return np.zeros(len(audio) // 512, np.float32)


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
        diarization_payload = {
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
        extract_mock = mock.AsyncMock(side_effect=lambda *_args: events.append("embedding") or clouds)
        refine_mock = mock.Mock(side_effect=lambda *_args, **_kwargs: events.append("refinement") or result_object)

        async def transcribe(_request, _audio, turns, _overlaps, _filter, *_gate_and_language):
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
            mock.patch.object(v2, "run_local_diarization", new=mock.AsyncMock(return_value=diarization_payload)),
            mock.patch.object(v2, "silero_frame_probs", side_effect=_silent_frame_probs),
            mock.patch.object(v2, "_extract_speaker_clouds", new=extract_mock),
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

    def test_refined_turns_preserve_original_diarization_provenance(self) -> None:
        turns = [
            {
                "start_ms": 0,
                "end_ms": 3000,
                "speaker_id": "SPEAKER_B",
                "original_speaker_id": "SPEAKER_A",
            }
        ]
        app = _CapturingApp()
        batch_result = types.SimpleNamespace(text="Korrigierter Turn")
        app.state.whisper_batch_manager = types.SimpleNamespace(
            submit_job=mock.AsyncMock(return_value=[batch_result])
        )
        request = _request(app)
        audio = np.full(16000 * 3, 0.1, dtype=np.float32)

        with (
            mock.patch.object(
                v2,
                "_processing_key",
                return_value=("asr", "cuda", "cache", "de", "fp16"),
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
        diarization_payload = {
            "model": {"id": "community-1"},
            "diarization": exclusive,
            "exclusive_diarization": exclusive,
            "overlaps": [],
        }
        app = _CapturingApp()
        request = _request(app)
        audio = np.full(16000 * 10, 0.1, dtype=np.float32)
        diarization_mock = mock.AsyncMock(return_value=diarization_payload)
        transcribe_mock = mock.AsyncMock(return_value=(segments, 20, "asr-model"))
        build_audio_mock = mock.Mock()

        with (
            mock.patch.object(v2, "run_local_diarization", new=diarization_mock),
            mock.patch.object(v2, "silero_frame_probs", side_effect=_silent_frame_probs),
            mock.patch.object(v2, "_transcribe_turns", new=transcribe_mock),
            mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value=clouds)),
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

        diarization_mock.assert_awaited_once()
        build_audio_mock.assert_not_called()
        self.assertEqual(diarization_mock.await_args.kwargs["num_speakers"], 5)
        self.assertIsNone(diarization_mock.await_args.kwargs["min_speakers"])
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
        by_label = {
            item["diarization_speaker_id"]: item for item in result["transcript"]["segments"]
        }
        self.assertEqual(by_label["SPEAKER_00"]["speaker_id"], "person-0")
        self.assertEqual(by_label["SPEAKER_00"]["speaker_kind"], "known")
        self.assertEqual(by_label["SPEAKER_04"]["speaker_id"], "SPEAKER_04")
        self.assertEqual(by_label["SPEAKER_04"]["speaker_kind"], "unknown")

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
        diarization_payload = {
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
            mock.patch.object(v2, "run_local_diarization", new=mock.AsyncMock(return_value=diarization_payload)),
            mock.patch.object(v2, "silero_frame_probs", side_effect=_silent_frame_probs),
            mock.patch.object(
                v2,
                "_transcribe_turns",
                new=mock.AsyncMock(return_value=(transcript_segments, 20, "asr-model")),
            ),
            mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value=clouds)),
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
        diarization_payload = {
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
            mock.patch.object(v2, "run_local_diarization", new=mock.AsyncMock(return_value=diarization_payload)),
            mock.patch.object(v2, "silero_frame_probs", side_effect=_silent_frame_probs),
            mock.patch.object(
                v2,
                "_transcribe_turns",
                new=mock.AsyncMock(return_value=(segments, 20, "asr-model")),
            ),
            mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value={speaker_id: cloud})),
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



COHERE = "CohereLabs/cohere-transcribe-03-2026"
WHISPER_TURBO = "openai/whisper-large-v3-turbo"


def _lid_result(language: str, probability: float = 0.97) -> lid.LidResult:
    rest = (1.0 - probability) / (len(COHERE_LANGUAGE_CODES) - 1)
    return lid.LidResult(
        language=language,
        probability=probability,
        probs={code: (probability if code == language else rest) for code in COHERE_LANGUAGE_CODES},
    )


def _lid_settings(**values) -> dict:
    return {
        "local_model": COHERE,
        "local_gpu_device": "cpu",
        "local_model_cache_path": "",
        "transcription_language": "auto",
        "local_model_precision": "fp16",
        "cohere_language_detection": True,
        "language_detection_model": "openai/whisper-base",
        **values,
    }


def _patch_lid_download(case: unittest.TestCase) -> None:
    patcher = mock.patch.object(lid, "prepare_lid_model", return_value=True)
    patcher.start()
    case.addCleanup(patcher.stop)


class _TextPerChunkManager:
    """submit_job fake: one job per call, the text names the language and the chunk's first sample."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[np.ndarray], tuple]] = []

    async def submit_job(self, segments, request_id, key):
        self.calls.append((list(segments), key))
        return [types.SimpleNamespace(text=f"{key[3]}{int(round(float(segment[0])))}") for segment in segments]


class TranscriptLanguageDetectionTests(unittest.TestCase):
    """Cohere + effective "auto": detect before ASR, submit with the detected language, echo what ran."""

    def setUp(self) -> None:
        _patch_lid_download(self)
        self.manager = _TextPerChunkManager()
        self.app = _CapturingApp()
        self.app.state.whisper_batch_manager = self.manager
        v2.create_v2_api(self.app)
        self.endpoint = self.app.routes[("POST", "/v2/audio/process")]
        self.request = _request(self.app)

    def settings(self, **values):
        return mock.patch.dict(v2.current_settings, _lid_settings(**values), clear=True)

    def transcribe(self, audio, language=None):
        return asyncio.run(v2._transcribe_audio(self.request, audio, language))

    def test_cohere_auto_submits_the_detected_language(self) -> None:
        audio = np.full(16000 * 8, 0.1, dtype=np.float32)
        with self.settings(), mock.patch.object(lid, "detect_languages", return_value=[_lid_result("en")]) as detect:
            text, _, _, _, decision = self.transcribe(audio)

        detect.assert_called_once()
        [(segments, key)] = self.manager.calls
        self.assertEqual(key[3], "en")
        self.assertEqual(text, "en0")
        self.assertEqual((decision.language, decision.source), ("en", "detected"))
        self.assertEqual(
            decision.detection_block(),
            {"model": "openai/whisper-base", "source": "detected", "languages": {"en": 8.0}},
        )

    def test_an_explicit_language_skips_detection(self) -> None:
        audio = np.full(16000 * 8, 0.1, dtype=np.float32)
        for settings, override in (({}, "fr"), ({"transcription_language": "it"}, None)):
            with self.subTest(settings=settings, override=override):
                self.manager.calls.clear()
                with self.settings(**settings), mock.patch.object(lid, "detect_languages") as detect:
                    _, _, _, _, decision = self.transcribe(audio, override)
                detect.assert_not_called()
                self.assertEqual(self.manager.calls[0][1][3], override or "it")
                self.assertIsNone(decision.detection_block())
                self.assertEqual(decision.source, "request" if override else "setting")

    def test_non_cohere_and_detection_off_never_detect(self) -> None:
        audio = np.full(16000 * 8, 0.1, dtype=np.float32)
        for settings, applied in (({"local_model": WHISPER_TURBO}, "auto"), ({"cohere_language_detection": False}, "de")):
            with self.subTest(settings=settings):
                self.manager.calls.clear()
                with (
                    self.settings(**settings),
                    mock.patch.object(lid, "detect_languages") as detect,
                    mock.patch.object(v2, "split_audio_for_whisper", return_value=[audio]),
                ):
                    _, _, _, _, decision = self.transcribe(audio)
                detect.assert_not_called()
                # Whisper detects itself; Cohere without detection keeps the engine's de fallback.
                self.assertEqual(self.manager.calls[0][1][3], "auto")
                self.assertEqual(decision.language, applied)

    def test_endpoint_echoes_the_applied_language_and_the_detection_block(self) -> None:
        audio = np.full(16000 * 6, 0.1, dtype=np.float32)
        upload = _upload()
        raw_request = json.dumps({"schema_version": "2.0", "mode": "transcript"})
        form = FormData([("file", upload), ("request", raw_request)])
        with (
            self.settings(),
            mock.patch.object(
                v2, "_parse_multipart_parts", new=mock.AsyncMock(return_value=(form, upload, raw_request))
            ),
            mock.patch.object(v2, "authorize_api_key", return_value=None),
            mock.patch.object(v2, "load_audio_file", return_value=audio),
            mock.patch.object(v2, "_record_log"),
            mock.patch.object(lid, "detect_languages", return_value=[_lid_result("pl")]),
        ):
            payload = _payload(asyncio.run(self.endpoint(self.request)))

        self.assertEqual(payload["status"], "completed", payload)
        self.assertEqual(payload["models"]["asr"]["language"], "pl")
        self.assertEqual(
            payload["models"]["asr"]["language_detection"],
            {"model": "openai/whisper-base", "source": "detected", "languages": {"pl": 6.0}},
        )
        self.assertEqual(payload["result"]["transcript"]["language"], "pl")
        self.assertIn("language_detection", payload["timings_ms"])

    def test_unavailable_detection_transcribes_german_and_warns(self) -> None:
        audio = np.full(16000 * 6, 0.1, dtype=np.float32)
        failure = lid.LanguageDetectionUnavailable("Language detection model 'openai/whisper-base' is unavailable")
        with self.settings(), mock.patch.object(lid, "detect_languages", side_effect=failure):
            _, _, _, _, decision = self.transcribe(audio)
        self.assertEqual(self.manager.calls[0][1][3], "de")
        self.assertEqual((decision.language, decision.source), ("de", "fallback"))
        self.assertEqual(decision.warnings[0]["code"], "LANGUAGE_DETECTION_UNAVAILABLE")


class DiarizationLanguageDetectionTests(unittest.TestCase):
    """Per-speaker language: decided once per diarization speaker (label) before ASR, carried on segments and assignments."""

    TURNS = [
        {"start_ms": 0, "end_ms": 6000, "speaker_id": "SPEAKER_A"},
        {"start_ms": 6000, "end_ms": 12000, "speaker_id": "SPEAKER_B"},
        {"start_ms": 12000, "end_ms": 18000, "speaker_id": "SPEAKER_A"},
    ]

    def setUp(self) -> None:
        _patch_lid_download(self)

    def run_diarization(self, detected, *, pipeline="turns", settings=None, language=None):
        clouds = {
            "SPEAKER_A": _ready_cloud("SPEAKER_A", _vector(0)),
            "SPEAKER_B": _ready_cloud("SPEAKER_B", _vector(1)),
        }
        diarization_payload = {
            "model": {"id": "community-1"},
            "diarization": self.TURNS,
            "exclusive_diarization": self.TURNS,
            "overlaps": [],
        }
        self.manager = _TextPerChunkManager()
        app = _CapturingApp()
        app.state.whisper_batch_manager = self.manager
        audio = np.full(16000 * 18, 0.1, dtype=np.float32)
        with (
            mock.patch.dict(v2.current_settings, _lid_settings(**(settings or {})), clear=True),
            mock.patch.object(v2, "run_local_diarization", new=mock.AsyncMock(return_value=diarization_payload)),
            mock.patch.object(v2, "silero_frame_probs", side_effect=_silent_frame_probs),
            mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value=clouds)),
            mock.patch.object(lid, "detect_languages", return_value=detected) as detect,
        ):
            processed = asyncio.run(
                v2._process_diarization(
                    _request(app),
                    _upload(),
                    audio,
                    {"expected_speakers": 2, "known_speakers": [], "speaker_refinement": "off"},
                    False,
                    False,
                    pipeline,
                    language,
                )
            )
        return processed, detect

    def test_each_speaker_is_transcribed_and_reported_in_its_language(self) -> None:
        (result, timings, models, warnings, _), detect = self.run_diarization([_lid_result("en"), _lid_result("de")])

        detect.assert_called_once()
        # One pass: a pooled window per speaker, then one per ASR turn (the fake answers only the speakers).
        self.assertEqual(len(detect.call_args.args[0]), 2 + 3)
        self.assertEqual(sorted(key[3] for _, key in self.manager.calls), ["de", "en"])
        self.assertEqual(
            [
                (segment["diarization_speaker_id"], segment["language"], segment["text"])
                for segment in result["transcript"]["segments"]
            ],
            [("SPEAKER_A", "en", "en0"), ("SPEAKER_B", "de", "de0"), ("SPEAKER_A", "en", "en0")],
        )
        assignments = {item["diarization_speaker_id"]: item for item in result["speaker_assignments"]}
        self.assertEqual(
            (assignments["SPEAKER_A"]["language"], assignments["SPEAKER_A"]["language_source"]), ("en", "detected")
        )
        self.assertEqual(
            (assignments["SPEAKER_B"]["language"], assignments["SPEAKER_B"]["language_source"]), ("de", "detected")
        )
        self.assertEqual(models["asr"]["language"], "en")
        self.assertEqual(
            models["asr"]["language_detection"],
            {"model": "openai/whisper-base", "source": "detected", "languages": {"de": 6.0, "en": 12.0}},
        )
        self.assertEqual(result["transcript"]["language"], "en")
        self.assertIn("language_detection", timings)
        self.assertEqual(warnings, [])

    def test_a_weak_speaker_takes_the_majority_of_the_confident_ones(self) -> None:
        (result, _, _, _, _), _ = self.run_diarization([_lid_result("fr"), _lid_result("de", 0.2)])
        assignments = {item["diarization_speaker_id"]: item for item in result["speaker_assignments"]}
        self.assertEqual(
            (assignments["SPEAKER_B"]["language"], assignments["SPEAKER_B"]["language_source"]), ("fr", "majority")
        )
        self.assertEqual([key[3] for _, key in self.manager.calls], ["fr"])

    def test_an_explicit_language_runs_no_detection(self) -> None:
        (result, _, models, _, _), detect = self.run_diarization([], language="it")
        detect.assert_not_called()
        self.assertEqual([key[3] for _, key in self.manager.calls], ["it"])
        self.assertTrue(all(segment["language"] == "it" for segment in result["transcript"]["segments"]))
        self.assertEqual(
            {(item["language"], item["language_source"]) for item in result["speaker_assignments"]},
            {("it", "request")},
        )
        self.assertNotIn("language_detection", models["asr"])

    def _run_wxc(self, regions, speaker_languages, settings=None):
        manager = _TextPerChunkManager()
        app = _CapturingApp()
        app.state.whisper_batch_manager = manager
        align = mock.AsyncMock(return_value=[])
        with (
            mock.patch.dict(v2.current_settings, _lid_settings(**(settings or {})), clear=True),
            mock.patch.object(v2, "silero_frame_probs", return_value=np.ones(4)),
            mock.patch.object(v2, "speech_regions_from_probs", return_value=regions),
            mock.patch.object(v2, "_align_words_sliced", new=align),
            mock.patch.object(v2, "plan_verification_windows", return_value=([], [])),
            mock.patch.object(v2, "finish_verification", return_value={}),
        ):
            _, _, diagnostics, _ = asyncio.run(
                v2._wxc_transcribe_segments(
                    _request(app),
                    np.zeros(16000 * 18, dtype=np.float32),
                    self.TURNS,
                    [],
                    False,
                    None,
                    speaker_languages,
                )
            )
        return manager, align, diagnostics

    def test_wxc_with_one_language_keeps_todays_chunking(self) -> None:
        regions = [(0.2, 5.8), (6.1, 11.7), (12.2, 17.5)]
        single = v2.SpeakerLanguages({"SPEAKER_A": "en", "SPEAKER_B": "en"}, {}, "en")
        before, before_align, _ = self._run_wxc(regions, None, {"transcription_language": "de"})
        after, after_align, _ = self._run_wxc(regions, single)

        self.assertEqual(len(before.calls), 1)
        self.assertEqual(len(after.calls), 1)
        self.assertEqual(before_align.await_args.args[2], after_align.await_args.args[2])
        self.assertEqual(
            [len(segment) for segment in before.calls[0][0]], [len(segment) for segment in after.calls[0][0]]
        )
        self.assertEqual((before.calls[0][1][3], after.calls[0][1][3]), ("de", "en"))

    def test_wxc_with_two_languages_never_mixes_them_in_one_chunk(self) -> None:
        languages = v2.SpeakerLanguages({"SPEAKER_A": "en", "SPEAKER_B": "de"}, {}, "en")
        manager, align, diagnostics = self._run_wxc([(0.2, 17.5)], languages)

        chunks = align.await_args.args[2]
        texts = align.await_args.args[3]
        # Chunks in time order, cut at the turn boundaries, each transcribed in its speaker's language.
        self.assertEqual(chunks, [(0.2, 6.0), (6.0, 12.0), (12.0, 17.5)])
        self.assertEqual([text[:2] for text in texts], ["en", "de", "en"])
        self.assertEqual(sorted(key[3] for _, key in manager.calls), ["de", "en"])
        self.assertEqual(diagnostics["superchunk_languages"], {"de": 1, "en": 2})


def _lid_by_level(levels):
    """detect_languages fake: each window's mean level picks (language, probability) from ``levels``."""

    def detect(windows):
        results = []
        for window in windows:
            level = float(np.mean(window)) if len(window) else 0.0
            for upper, language, probability in levels:
                if level <= upper:
                    results.append(_lid_result(language, probability))
                    break
        return results

    return detect


# 0.1 English, 0.2 German, 0.3 German below LID_TURN_OVERRIDE_CONFIDENCE, 0.4 weak English, 0.5 below
# LID_MIN_CONFIDENCE. A pooled speaker window answers by its mean level.
_LEVELS = [(0.17, "en", 0.97), (0.25, "de", 0.97), (0.35, "de", 0.7), (0.45, "en", 0.7), (0.55, "fr", 0.4)]


def _audio_with_levels(seconds: float, spans) -> np.ndarray:
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    for start_s, end_s, level in spans:
        audio[int(start_s * 16000) : int(end_s * 16000)] = level
    return audio


class TurnLanguageRefinementTests(unittest.TestCase):
    """The speaker's language is the prior; a long enough, confident turn of another language overrides it."""

    def setUp(self) -> None:
        _patch_lid_download(self)

    @staticmethod
    def turns(interjection_end_ms: int = 17000):
        # SPEAKER_A is the "mixed cluster": an English speaker plus a German interjection of someone else.
        return [
            {"start_ms": 0, "end_ms": 8000, "speaker_id": "SPEAKER_A"},
            {"start_ms": 8000, "end_ms": 14000, "speaker_id": "SPEAKER_B"},
            {"start_ms": 14000, "end_ms": interjection_end_ms, "speaker_id": "SPEAKER_A"},
            {"start_ms": 20000, "end_ms": 26000, "speaker_id": "SPEAKER_B"},
        ]

    def run_case(self, turns, audio, *, pipeline="turns", regions=None):
        clouds = {
            "SPEAKER_A": _ready_cloud("SPEAKER_A", _vector(0)),
            "SPEAKER_B": _ready_cloud("SPEAKER_B", _vector(1)),
        }
        diarization_payload = {"model": {"id": "community-1"}, "diarization": turns, "exclusive_diarization": turns, "overlaps": []}
        manager = _TextPerChunkManager()
        app = _CapturingApp()
        app.state.whisper_batch_manager = manager
        original_spans = v2._speaker_lid_spans
        align = mock.AsyncMock(return_value=[])
        with (
            mock.patch.dict(v2.current_settings, _lid_settings(), clear=True),
            mock.patch.object(v2, "run_local_diarization", new=mock.AsyncMock(return_value=diarization_payload)),
            mock.patch.object(v2, "_extract_speaker_clouds", new=mock.AsyncMock(return_value=clouds)),
            # The fixture clouds all sit on 0-9 s; pool each speaker from its own turns instead.
            mock.patch.object(v2, "_speaker_lid_spans", side_effect=lambda cloud, items, *a, **k: original_spans(None, items, *a, **k)),
            mock.patch.object(lid, "detect_languages", side_effect=_lid_by_level(_LEVELS)) as detect,
            mock.patch.object(v2, "silero_frame_probs", return_value=np.ones(4)),
            mock.patch.object(v2, "speech_regions_from_probs", return_value=regions or []),
            mock.patch.object(v2, "_align_words_sliced", new=align),
            mock.patch.object(v2, "plan_verification_windows", return_value=([], [])),
            mock.patch.object(v2, "finish_verification", return_value={}),
        ):
            processed = asyncio.run(
                v2._process_diarization(
                    _request(app),
                    _upload(),
                    audio,
                    {"expected_speakers": 2, "known_speakers": [], "speaker_refinement": "off"},
                    False,
                    False,
                    pipeline,
                    None,
                )
            )
        return processed, manager, detect, align

    def segments_by_start(self, result):
        return {
            segment["start_ms"]: (segment["diarization_speaker_id"], segment["language"], segment["language_source"])
            for segment in result["transcript"]["segments"]
        }

    def test_a_confident_german_turn_in_an_english_cluster_is_decoded_german(self) -> None:
        audio = _audio_with_levels(26, [(0, 8, 0.1), (8, 14, 0.2), (14, 17, 0.2), (20, 26, 0.2)])
        (result, _, models, _, _), manager, detect, _ = self.run_case(self.turns(), audio)

        detect.assert_called_once()  # speakers and turns share one batched pass
        jobs = {key[3]: [round(len(segment) / 16000) for segment in segments] for segments, key in manager.calls}
        # The 3 s interjection runs with the German job although its cluster is English.
        self.assertEqual(jobs, {"en": [8], "de": [6, 3, 6]})
        self.assertEqual(
            self.segments_by_start(result),
            {
                0: ("SPEAKER_A", "en", "speaker"),
                8000: ("SPEAKER_B", "de", "speaker"),
                14000: ("SPEAKER_A", "de", "turn"),
                20000: ("SPEAKER_B", "de", "speaker"),
            },
        )
        assignments = {item["diarization_speaker_id"]: item for item in result["speaker_assignments"]}
        self.assertEqual((assignments["SPEAKER_A"]["language"], assignments["SPEAKER_A"]["language_source"]), ("en", "detected"))
        # The reported seconds follow the turns: the interjection counts as German.
        self.assertEqual(models["asr"]["language_detection"]["languages"], {"de": 15.0, "en": 8.0})

    def test_wxc_cuts_the_chunk_at_the_overriding_turn(self) -> None:
        audio = _audio_with_levels(26, [(0, 8, 0.1), (8, 14, 0.2), (14, 17, 0.2), (20, 26, 0.2)])
        processed, manager, _, align = self.run_case(
            self.turns(), audio, pipeline="wxc", regions=[(0.2, 7.8), (8.1, 16.9), (20.1, 25.8)]
        )
        self.assertEqual(processed[3], [])  # WXC itself ran, no fallback to the turn path

        chunks, texts = align.await_args.args[2], align.await_args.args[3]
        # Without the override the cluster's English would cover 14-17 s (the bug); now German runs on.
        self.assertEqual(chunks, [(0.2, 7.8), (8.1, 25.8)])
        self.assertEqual([text[:2] for text in texts], ["en", "de"])
        self.assertEqual(sorted(key[3] for _, key in manager.calls), ["de", "en"])

    def test_a_turn_below_the_override_confidence_keeps_its_speakers_language(self) -> None:
        audio = _audio_with_levels(26, [(0, 8, 0.1), (8, 14, 0.2), (14, 17, 0.3), (20, 26, 0.2)])
        (result, _, _, _, _), manager, _, _ = self.run_case(self.turns(), audio)

        self.assertEqual(self.segments_by_start(result)[14000], ("SPEAKER_A", "en", "speaker"))
        jobs = {key[3]: [round(len(segment) / 16000) for segment in segments] for segments, key in manager.calls}
        self.assertEqual(jobs, {"en": [8, 3], "de": [6, 6]})

    def test_a_turn_shorter_than_the_minimum_is_not_detected_and_keeps_its_speakers_language(self) -> None:
        turns = self.turns(interjection_end_ms=15200)  # 1.2 s < LID_TURN_MIN_SECONDS
        audio = _audio_with_levels(26, [(0, 8, 0.1), (8, 14, 0.2), (14, 15.2, 0.2), (20, 26, 0.2)])
        (result, _, _, _, _), _, detect, _ = self.run_case(turns, audio)

        windows = [window for call in detect.call_args_list for window in call.args[0]]
        self.assertEqual(len(windows), 2 + 3)  # two speakers, three long turns; the 1.2 s one is skipped
        self.assertEqual(self.segments_by_start(result)[14000], ("SPEAKER_A", "en", "speaker"))

    def test_without_a_language_change_the_chunking_is_unchanged(self) -> None:
        audio = _audio_with_levels(26, [(0, 8, 0.2), (8, 14, 0.2), (14, 17, 0.2), (20, 26, 0.2)])
        regions = [(0.2, 7.8), (8.1, 16.9), (20.1, 25.8)]
        _, manager, _, align = self.run_case(self.turns(), audio, pipeline="wxc", regions=regions)

        self.assertEqual(align.await_args.args[2], v2.build_superchunks(regions))
        self.assertEqual([key[3] for _, key in manager.calls], ["de"])


class TranscriptPieceLanguageTests(unittest.TestCase):
    """Long transcripts: detection per Silero speech piece, superchunks never mix two languages."""

    def setUp(self) -> None:
        _patch_lid_download(self)
        self.manager = _TextPerChunkManager()
        self.app = _CapturingApp()
        self.app.state.whisper_batch_manager = self.manager
        self.request = _request(self.app)

    def transcribe(self, audio, regions):
        with (
            mock.patch.dict(v2.current_settings, _lid_settings(), clear=True),
            mock.patch.object(lid, "superchunk_speech_regions", return_value=regions),
            mock.patch.object(lid, "detect_languages", side_effect=_lid_by_level(_LEVELS)) as detect,
        ):
            return asyncio.run(v2._transcribe_audio(self.request, audio)), detect

    def test_a_german_sentence_then_english_inside_one_old_superchunk_gets_two_languages(self) -> None:
        # 70-82 s German, 82.5-94 s English: build_superchunks made that one chunk, detected as a whole.
        regions = [(1.0, 24.0), (30.0, 52.0), (70.0, 82.0), (82.5, 94.0)]
        audio = _audio_with_levels(100, [(1, 24, 0.2), (30, 52, 0.2), (70, 82, 0.2), (82.5, 94, 0.1)])
        self.assertEqual(v2.build_superchunks(regions)[-1], (70.0, 94.0))

        (text, _, count, _, decision), detect = self.transcribe(audio, regions)

        detect.assert_called_once()
        self.assertEqual(len(detect.call_args.args[0]), 4)  # one window per piece
        self.assertEqual(decision.languages, ["de", "de", "de", "en"])
        self.assertEqual(count, 4)
        self.assertEqual(sorted(key[3] for _, key in self.manager.calls), ["de", "en"])
        en_job = next(segments for segments, key in self.manager.calls if key[3] == "en")
        self.assertEqual([round(len(segment) / 16000, 2) for segment in en_job], [12.0])  # 82.5-94 s plus 2 x 0.25 s pad
        self.assertEqual(decision.seconds, {"de": 57.0, "en": 11.5})

    def test_one_language_keeps_todays_superchunks(self) -> None:
        regions = [(1.0, 24.0), (30.0, 52.0), (70.0, 82.0), (82.5, 94.0)]
        audio = _audio_with_levels(100, [(1, 24, 0.2), (30, 52, 0.2), (70, 82, 0.2), (82.5, 94, 0.2)])
        (_, _, count, _, decision), _ = self.transcribe(audio, regions)

        expected = v2.padded_chunk_audio(audio, v2.build_superchunks(regions))
        [(segments, key)] = self.manager.calls
        self.assertEqual(key[3], "de")
        self.assertEqual(count, len(expected))
        self.assertEqual([len(segment) for segment in segments], [len(segment) for segment in expected])
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(segments, expected)))

    def test_weak_short_and_doubtful_pieces_take_the_nearest_anchor(self) -> None:
        # Mostly German. 24.5-27 s: English at 0.7 (a doubtful deviation from the majority, joins its German
        # neighbour). 70.5-74 s: below LID_MIN_CONFIDENCE, 74.3-75.3 s: too short to detect; both sit
        # between confident English pieces and take English.
        regions = [(1.0, 24.0), (24.5, 27.0), (30.0, 52.0), (60.0, 70.0), (70.5, 74.0), (74.3, 75.3), (75.6, 90.0)]
        audio = _audio_with_levels(
            100,
            [(1, 24, 0.2), (24.5, 27, 0.4), (30, 52, 0.2), (60, 70, 0.1), (70.5, 74, 0.5), (74.3, 75.3, 0.2), (75.6, 90, 0.1)],
        )
        (_, _, _, _, decision), detect = self.transcribe(audio, regions)

        self.assertEqual(len(detect.call_args.args[0]), 6)  # the 1 s piece is not detected
        self.assertEqual(decision.languages, ["de", "de", "de", "en", "en"])
        self.assertEqual({code: round(value, 1) for code, value in decision.seconds.items()}, {"de": 47.5, "en": 29.7})


    def test_choppy_speech_of_only_short_regions_is_still_detected(self) -> None:
        # English back-and-forth: 1.2 s regions with 0.5 s pauses, none of them LID_TURN_MIN_SECONDS long.
        regions = [(1.0 + 1.7 * k, 2.2 + 1.7 * k) for k in range(50)]
        audio = _audio_with_levels(100, [(start, end, 0.1) for start, end in regions])
        (_, _, count, _, decision), detect = self.transcribe(audio, regions)

        detect.assert_called_once()
        windows = detect.call_args.args[0]
        self.assertLess(len(windows), len(regions))  # pooled short pieces, not one window per region
        self.assertTrue(all(len(window) >= lid.LID_TURN_MIN_SECONDS * 16000 for window in windows))
        self.assertEqual((decision.source, set(decision.languages)), ("detected", {"en"}))
        self.assertEqual([key[3] for _, key in self.manager.calls], ["en"])
        self.assertEqual(count, len(v2.build_superchunks(regions)))

    def test_a_language_change_inside_one_long_region_does_not_decode_the_seam_twice(self) -> None:
        # One 60 s Silero region (no pause of 0.3 s): German to 21.6 s, then English. It is detected in three
        # parts; the first cut moves to the quiet moment at 21.6 s.
        regions = [(1.0, 61.0)]
        audio = _audio_with_levels(70, [(1, 21.6, 0.2), (21.64, 61, 0.1)])
        (_, _, count, _, decision), _ = self.transcribe(audio, regions)

        self.assertEqual(decision.languages, ["de", "en"])
        self.assertEqual(count, 2)
        de_job = next(segments for segments, key in self.manager.calls if key[3] == "de")
        en_job = next(segments for segments, key in self.manager.calls if key[3] == "en")
        # Outer edges keep CHUNK_PAD_S; the seam is not padded into the other language's chunk.
        self.assertEqual(len(de_job[0]), int(21.61 * 16000) - int(0.75 * 16000))
        self.assertEqual(len(en_job[0]), int(61.25 * 16000) - int(21.61 * 16000))


if __name__ == "__main__":
    unittest.main()

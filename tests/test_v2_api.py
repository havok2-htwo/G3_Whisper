from __future__ import annotations

import asyncio
import io
import json
import math
import types
import unittest
from unittest import mock

import numpy as np
from fastapi import FastAPI
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


class V2DiarizationAssemblyTests(unittest.TestCase):
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

        with (
            mock.patch.object(v2, "diarize_v2", new=dia_mock),
            mock.patch.object(v2, "_transcribe_turns", new=transcribe_mock),
            mock.patch.object(v2, "extract_speaker_clouds", return_value=clouds),
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


if __name__ == "__main__":
    unittest.main()

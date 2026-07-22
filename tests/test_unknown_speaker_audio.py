from __future__ import annotations

import base64
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from backend import genesis_whisper_server_speaker_audio as speaker_audio
from backend.genesis_whisper_server_speaker_matching import SpeakerCloud
from backend.genesis_whisper_server_vid import EmbeddedVoiceWindow, REDIMNET_EMBEDDING_DIMENSION


def _vector(cosine: float) -> np.ndarray:
    vector = np.zeros(REDIMNET_EMBEDDING_DIMENSION, dtype=np.float32)
    vector[0] = cosine
    vector[1] = float(np.sqrt(max(0.0, 1.0 - cosine**2)))
    return vector


def _sample(
    start_ms: int | None,
    end_ms: int | None,
    cosine: float = 0.95,
    *,
    quality: float = 0.9,
    stitched: bool = False,
) -> EmbeddedVoiceWindow:
    duration_seconds = (
        max(0.0, (end_ms - start_ms) / 1000.0)
        if start_ms is not None and end_ms is not None
        else 3.0
    )
    return EmbeddedVoiceWindow(
        vector=_vector(cosine),
        start_ms=start_ms,
        end_ms=end_ms,
        clean_duration_seconds=duration_seconds,
        quality=quality,
        stitched=stitched,
    )


def _cloud(
    speaker_id: str,
    samples: list[EmbeddedVoiceWindow],
    *,
    inlier_indices: list[int] | None = None,
) -> SpeakerCloud:
    indices = list(range(len(samples))) if inlier_indices is None else inlier_indices
    return SpeakerCloud(
        diarization_speaker_id=speaker_id,
        status="ready",
        prototype=_vector(1.0),
        samples=samples,
        inlier_indices=indices,
        candidate_count=len(samples),
        discarded_outliers=len(samples) - len(indices),
        purity=1.0,
    )


def _completed_mp3(payload: bytes = b"encoded-mp3") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=payload, stderr=b"")


class UnknownSpeakerAudioSelectionTests(unittest.TestCase):
    def test_uses_only_timed_nonstitched_quality_inliers(self) -> None:
        samples = [
            _sample(0, 6000, quality=0.59),
            _sample(6000, 12000, stitched=True),
            _sample(None, None),
            _sample(12000, 18000),
            _sample(18000, 21000, cosine=0.91),
            _sample(21000, 24000, cosine=0.93),
        ]
        cloud = _cloud("SPEAKER_UNKNOWN", samples, inlier_indices=[0, 1, 2, 4, 5])
        audio = np.full(30 * 16_000, 0.25, dtype=np.float32)

        with (
            patch.object(speaker_audio.shutil, "which", return_value="ffmpeg"),
            patch.object(
                speaker_audio.subprocess,
                "run",
                return_value=_completed_mp3(),
            ) as run_mock,
        ):
            assets = speaker_audio.build_unknown_speaker_audio_assets(
                audio,
                {"SPEAKER_UNKNOWN": cloud, "SPEAKER_OTHER": cloud},
                ["SPEAKER_UNKNOWN"],
            )

        self.assertEqual(set(assets), {"SPEAKER_UNKNOWN"})
        asset = assets["SPEAKER_UNKNOWN"]
        self.assertEqual(asset["mime_type"], "audio/mpeg")
        self.assertEqual(asset["encoding"], "base64")
        self.assertEqual(base64.b64decode(asset["data"]), b"encoded-mp3")
        self.assertEqual(asset["duration_ms"], 6000)
        self.assertEqual(
            asset["snippets"],
            [
                {
                    "start_ms": 18000,
                    "end_ms": 24000,
                    "duration_ms": 6000,
                    "centrality": 0.92,
                }
            ],
        )
        pcm_bytes = run_mock.call_args.kwargs["input"]
        self.assertEqual(len(pcm_bytes), 6000 * 16_000 // 1000 * 2)
        pcm = np.frombuffer(pcm_bytes, dtype="<i2")
        self.assertEqual(pcm[0], 0)
        self.assertEqual(pcm[-1], 0)

    def test_selects_central_30_second_slice_from_long_run(self) -> None:
        samples = [
            _sample(index * 3000, (index + 1) * 3000, 0.70 if index < 5 else 0.99)
            for index in range(15)
        ]
        audio = np.full(45 * 16_000, 0.1, dtype=np.float32)

        with (
            patch.object(speaker_audio.shutil, "which", return_value="ffmpeg"),
            patch.object(speaker_audio.subprocess, "run", return_value=_completed_mp3()) as run_mock,
        ):
            asset = speaker_audio.build_unknown_speaker_audio_assets(
                audio,
                {"SPEAKER_00": _cloud("SPEAKER_00", samples)},
                ["SPEAKER_00"],
            )["SPEAKER_00"]

        self.assertEqual(asset["duration_ms"], 30_000)
        self.assertEqual(asset["snippets"][0]["start_ms"], 15_000)
        self.assertEqual(asset["snippets"][0]["end_ms"], 45_000)
        self.assertAlmostEqual(asset["snippets"][0]["centrality"], 0.99, places=5)
        self.assertLessEqual(len(run_mock.call_args.kwargs["input"]), 30 * 16_000 * 2)

    def test_prefers_central_runs_and_returns_chronological_nonoverlapping_snippets(self) -> None:
        samples = []
        samples.extend(_sample(ms, ms + 3000, 0.95) for ms in range(0, 12_000, 3000))
        samples.extend(_sample(ms, ms + 3000, 0.99) for ms in range(15_000, 33_000, 3000))
        samples.extend(_sample(ms, ms + 3000, 0.90) for ms in range(36_000, 45_000, 3000))
        audio = np.full(45 * 16_000, 0.1, dtype=np.float32)

        with (
            patch.object(speaker_audio.shutil, "which", return_value="ffmpeg"),
            patch.object(speaker_audio.subprocess, "run", return_value=_completed_mp3()),
        ):
            asset = speaker_audio.build_unknown_speaker_audio_assets(
                audio,
                {"SPEAKER_00": _cloud("SPEAKER_00", samples)},
                ["SPEAKER_00"],
            )["SPEAKER_00"]

        self.assertEqual(asset["duration_ms"], 30_000)
        self.assertEqual(
            [(item["start_ms"], item["end_ms"]) for item in asset["snippets"]],
            [(0, 12_000), (15_000, 33_000)],
        )
        self.assertTrue(all(item["duration_ms"] >= 5000 for item in asset["snippets"]))
        self.assertTrue(
            all(
                left["end_ms"] <= right["start_ms"]
                for left, right in zip(asset["snippets"], asset["snippets"][1:])
            )
        )

    def test_overlapping_inlier_windows_do_not_duplicate_source_audio(self) -> None:
        cloud = _cloud(
            "SPEAKER_00",
            [_sample(0, 6000, 0.95), _sample(3000, 9000, 0.96)],
        )
        audio = np.full(9 * 16_000, 0.1, dtype=np.float32)

        with (
            patch.object(speaker_audio.shutil, "which", return_value="ffmpeg"),
            patch.object(speaker_audio.subprocess, "run", return_value=_completed_mp3()) as run_mock,
        ):
            asset = speaker_audio.build_unknown_speaker_audio_assets(
                audio, {"SPEAKER_00": cloud}, ["SPEAKER_00"]
            )["SPEAKER_00"]

        self.assertEqual(asset["duration_ms"], 9000)
        self.assertEqual(asset["snippets"][0]["duration_ms"], 9000)
        self.assertEqual(len(run_mock.call_args.kwargs["input"]), 9 * 16_000 * 2)

    def test_omits_speaker_without_five_second_contiguous_source_run(self) -> None:
        cloud = _cloud("SPEAKER_00", [_sample(0, 3000)])
        with patch.object(speaker_audio.subprocess, "run") as run_mock:
            assets = speaker_audio.build_unknown_speaker_audio_assets(
                np.zeros(3 * 16_000, dtype=np.float32),
                {"SPEAKER_00": cloud},
                ["SPEAKER_00", "MISSING"],
            )

        self.assertEqual(assets, {})
        run_mock.assert_not_called()


class UnknownSpeakerAudioEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = np.full(6 * 16_000, 0.1, dtype=np.float32)
        self.cloud = _cloud("SPEAKER_00", [_sample(0, 6000)])

    def test_ffmpeg_receives_mono_16khz_s16le_and_64k_mp3(self) -> None:
        with (
            patch.object(speaker_audio.shutil, "which", return_value="C:/ffmpeg.exe"),
            patch.object(speaker_audio.subprocess, "run", return_value=_completed_mp3()) as run_mock,
        ):
            speaker_audio.build_unknown_speaker_audio_assets(
                self.audio, {"SPEAKER_00": self.cloud}, ["SPEAKER_00"]
            )

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "C:/ffmpeg.exe")
        self.assertIn("s16le", command)
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-b:a") + 1], "64k")
        self.assertEqual(command[-2:], ["mp3", "pipe:1"])

    def test_missing_or_failed_ffmpeg_raises_runtime_error(self) -> None:
        with patch.object(speaker_audio.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "nicht verfuegbar"):
                speaker_audio.build_unknown_speaker_audio_assets(
                    self.audio, {"SPEAKER_00": self.cloud}, ["SPEAKER_00"]
                )

        failed = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"encoder failed")
        with (
            patch.object(speaker_audio.shutil, "which", return_value="ffmpeg"),
            patch.object(speaker_audio.subprocess, "run", return_value=failed),
        ):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                speaker_audio.build_unknown_speaker_audio_assets(
                    self.audio, {"SPEAKER_00": self.cloud}, ["SPEAKER_00"]
                )


if __name__ == "__main__":
    unittest.main()

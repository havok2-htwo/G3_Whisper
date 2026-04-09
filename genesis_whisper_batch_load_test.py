import argparse
import io
import itertools
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import requests
import soundfile as sf
from scipy.signal import resample_poly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Schickt 4 Test-WAVs als 4/8/16/32 gleichzeitige Requests an den laufenden GENESIS Whisper Server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7861", help="Basis-URL des laufenden Servers.")
    parser.add_argument("--audio-dir", default="testaudio", help="Ordner mit den Ausgangs-WAVs.")
    parser.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16, 32], help="Gleichzeitige Request-Mengen.")
    parser.add_argument("--engine", default="local", help="Engine-Formfeld, standardmaessig 'local'.")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP-Timeout pro Request in Sekunden.")
    parser.add_argument("--stagger-ms", type=int, default=0, help="Optionaler Startversatz pro Thread in Millisekunden.")
    parser.add_argument("--admin-username", help="Optionaler Admin-Username fuer Queue-Snapshots.")
    parser.add_argument("--admin-password", help="Optionales Admin-Passwort fuer Queue-Snapshots.")
    parser.add_argument("--json-out", help="Optionaler Pfad fuer einen JSON-Report.")
    return parser.parse_args()


def load_audio_payload(path: Path) -> Dict:
    audio_data, sample_rate = sf.read(str(path), dtype="float32")
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        up = 16000 // divisor
        down = sample_rate // divisor
        audio_data = resample_poly(audio_data, up, down).astype("float32")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 16000, format="WAV", subtype="PCM_16")
    return {
        "bytes": buffer.getvalue(),
        "duration_seconds": round(len(audio_data) / 16000.0, 3),
    }


def build_request_plan(audio_files: List[Path], level: int) -> List[Path]:
    return list(itertools.islice(itertools.cycle(audio_files), level))


def login_admin(base_url: str, username: str, password: str) -> requests.Session:
    session = requests.Session()
    response = session.post(
        f"{base_url.rstrip('/')}/api/admin/login",
        json={"username": username, "password": password},
        timeout=20,
    )
    response.raise_for_status()
    return session


def get_queue_snapshot(session: requests.Session, base_url: str) -> Optional[Dict]:
    try:
        response = session.get(f"{base_url.rstrip('/')}/api/admin/queue", timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def compute_rtf(audio_seconds: float, duration_ms: Optional[float]) -> Optional[float]:
    if not duration_ms or duration_ms <= 0:
        return None
    return round(audio_seconds / (duration_ms / 1000.0), 3)


def select_new_batches(queue_before: Optional[Dict], queue_after: Optional[Dict]) -> List[Dict]:
    if not queue_after:
        return []

    before_ids = set()
    if queue_before:
        before_ids = {entry.get("batch_id") for entry in queue_before.get("recent_batches", []) if entry.get("batch_id")}

    return [
        entry
        for entry in queue_after.get("recent_batches", [])
        if entry.get("batch_id") and entry.get("batch_id") not in before_ids
    ]


def send_parallel_requests(
    base_url: str,
    request_files: List[Path],
    payload_cache: Dict[str, Dict],
    engine: str,
    timeout: float,
    stagger_ms: int,
):
    barrier = threading.Barrier(len(request_files))
    endpoint = f"{base_url.rstrip('/')}/transcribe/"

    def worker(index_and_path):
        index, path = index_and_path
        barrier.wait()
        if stagger_ms > 0:
            time.sleep((stagger_ms * index) / 1000.0)

        started = time.monotonic()
        response = requests.post(
            endpoint,
            files={"file": (path.name, payload_cache[path.name]["bytes"], "audio/wav")},
            data={"engine": engine, "voice_ident": "false"},
            timeout=timeout,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)

        try:
            payload = response.json()
        except Exception:
            payload = {"detail": response.text}

        return {
            "file": path.name,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "transcription_duration_ms": payload.get("transcription_duration_ms"),
            "total_duration_ms": payload.get("total_duration_ms"),
            "transcription_preview": str(payload.get("transcription", ""))[:80],
            "detail": payload.get("detail"),
        }

    started_all = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(request_files)) as executor:
        results = list(executor.map(worker, enumerate(request_files)))
    total_wall_ms = round((time.monotonic() - started_all) * 1000)
    return total_wall_ms, results


def summarize_round(
    level: int,
    submitted_audio_seconds: float,
    total_wall_ms: int,
    results: List[Dict],
    queue_before: Optional[Dict],
    queue_after: Optional[Dict],
) -> Dict:
    ok_results = [result for result in results if result["status_code"] == 200]
    failed_results = [result for result in results if result["status_code"] != 200]
    elapsed_values = [result["elapsed_ms"] for result in results]
    request_total_values = [result["total_duration_ms"] for result in ok_results if result.get("total_duration_ms") is not None]
    request_transcription_values = [
        result["transcription_duration_ms"] for result in ok_results if result.get("transcription_duration_ms") is not None
    ]

    summary = {
        "level": level,
        "submitted_audio_seconds": round(submitted_audio_seconds, 3),
        "total_wall_ms": total_wall_ms,
        "wall_clock_rtf": compute_rtf(submitted_audio_seconds, total_wall_ms),
        "ok_count": len(ok_results),
        "failed_count": len(failed_results),
        "min_elapsed_ms": min(elapsed_values) if elapsed_values else None,
        "avg_elapsed_ms": round(statistics.mean(elapsed_values), 2) if elapsed_values else None,
        "max_elapsed_ms": max(elapsed_values) if elapsed_values else None,
        "avg_request_total_duration_ms": round(statistics.mean(request_total_values), 2) if request_total_values else None,
        "avg_request_transcription_duration_ms": round(statistics.mean(request_transcription_values), 2)
        if request_transcription_values
        else None,
        "results": results,
    }

    if queue_before is not None and queue_after is not None:
        new_batches = select_new_batches(queue_before, queue_after)
        summed_batch_duration_ms = sum((entry.get("duration_ms") or 0) for entry in new_batches)
        summed_batch_audio_seconds = round(sum((entry.get("audio_seconds") or 0) for entry in new_batches), 3)
        summary["queue_before"] = {
            "total_batches_processed": queue_before.get("total_batches_processed"),
            "total_segments_processed": queue_before.get("total_segments_processed"),
        }
        summary["queue_after"] = {
            "total_batches_processed": queue_after.get("total_batches_processed"),
            "total_segments_processed": queue_after.get("total_segments_processed"),
            "recent_batches": queue_after.get("recent_batches", [])[:5],
        }
        summary["batch_delta"] = {
            "batches": (queue_after.get("total_batches_processed") or 0) - (queue_before.get("total_batches_processed") or 0),
            "segments": (queue_after.get("total_segments_processed") or 0) - (queue_before.get("total_segments_processed") or 0),
        }
        summary["new_batches"] = new_batches
        summary["batch_compute_ms_sum"] = summed_batch_duration_ms
        summary["batch_audio_seconds_sum"] = summed_batch_audio_seconds
        summary["batch_rtf"] = compute_rtf(submitted_audio_seconds, summed_batch_duration_ms)

    return summary


def print_round_summary(summary: Dict):
    print("")
    print(f"=== Level {summary['level']} ===")
    print(
        f"Audio: {summary['submitted_audio_seconds']} s | "
        f"Requests OK/Fehler: {summary['ok_count']}/{summary['failed_count']} | "
        f"Wall: {summary['total_wall_ms']} ms | "
        f"RTF(Wall): {summary.get('wall_clock_rtf')} | "
        f"Latenz min/avg/max: {summary['min_elapsed_ms']}/{summary['avg_elapsed_ms']}/{summary['max_elapsed_ms']} ms"
    )
    if summary.get("avg_request_total_duration_ms") is not None:
        print(
            f"Request avg total/transcription: "
            f"{summary['avg_request_total_duration_ms']} / {summary['avg_request_transcription_duration_ms']} ms"
        )

    batch_delta = summary.get("batch_delta")
    if batch_delta is not None:
        print(
            f"Batch-Delta: batches={batch_delta['batches']} segmente={batch_delta['segments']} | "
            f"Batch-Zeit: {summary.get('batch_compute_ms_sum')} ms | "
            f"Batch-Audio: {summary.get('batch_audio_seconds_sum')} s | "
            f"RTF(Batch): {summary.get('batch_rtf')}"
        )
        recent_batches = summary.get("new_batches", [])
        for batch_entry in recent_batches[:5]:
            print(
                f"  recent batch: id={batch_entry.get('batch_id')} size={batch_entry.get('batch_size')} "
                f"audio={batch_entry.get('audio_seconds')}s duration={batch_entry.get('duration_ms')}ms status={batch_entry.get('status')}"
            )

    if summary["failed_count"]:
        for failed in summary["results"]:
            if failed["status_code"] != 200:
                print(
                    f"  FEHLER {failed['file']}: status={failed['status_code']} detail={failed.get('detail')}"
                )


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise SystemExit(f"Keine WAV-Dateien in '{audio_dir}' gefunden.")

    print(f"Gefundene Audios: {[path.name for path in audio_files]}")
    payload_cache = {path.name: load_audio_payload(path) for path in audio_files}
    print("Audios lokal auf 16 kHz vorbereitet.")

    admin_session = None
    if args.admin_username and args.admin_password:
        admin_session = login_admin(args.base_url, args.admin_username, args.admin_password)
        print("Admin-Login erfolgreich. Queue-Snapshots werden mitgezogen.")

    report = {
        "base_url": args.base_url,
        "audio_dir": str(audio_dir),
        "levels": args.levels,
        "engine": args.engine,
        "generated_at_unix": int(time.time()),
        "rounds": [],
    }

    for level in args.levels:
        request_files = build_request_plan(audio_files, level)
        submitted_audio_seconds = round(sum(payload_cache[path.name]["duration_seconds"] for path in request_files), 3)
        queue_before = get_queue_snapshot(admin_session, args.base_url) if admin_session else None
        total_wall_ms, results = send_parallel_requests(
            base_url=args.base_url,
            request_files=request_files,
            payload_cache=payload_cache,
            engine=args.engine,
            timeout=args.timeout,
            stagger_ms=args.stagger_ms,
        )
        queue_after = get_queue_snapshot(admin_session, args.base_url) if admin_session else None
        summary = summarize_round(level, submitted_audio_seconds, total_wall_ms, results, queue_before, queue_after)
        report["rounds"].append(summary)
        print_round_summary(summary)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("")
        print(f"JSON-Report geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()

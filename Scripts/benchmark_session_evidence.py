"""Compare the previous four-clip output with a single video using simulated CCTV."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any

import cv2
import numpy as np

from Action_Detection_SOP.evidence import EvidenceClip, EvidenceClipData
from Action_Detection_SOP.evidence_io import write_evidence_clip
from Action_Detection_SOP.session_evidence import SessionEvidenceWriter, EvidenceEncoder


@dataclass(frozen=True)
class Measurement:
    width: int
    height: int
    approach: str
    files: int
    frames: int
    size_bytes: int
    preparation_ms: float
    max_window_write_ms: float
    finalization_ms: float
    decoding_ms: float
    upload_payload_seconds_10mbps: float
    upload_payload_seconds_100mbps: float


def simulated_windows(width: int, height: int) -> list[EvidenceClipData]:
    """Use identical textured frames in overlaps, small objects, and modest motion."""
    fps = 5.0
    windows = [("roll_entered", 3), ("cleaned_done", 13), ("labeled_done", 15), ("roll_left", 28)]
    rng = np.random.default_rng(20261012)
    texture = rng.integers(-12, 13, (height, width, 1), dtype=np.int16)
    base = np.clip(np.array([195, 205, 215], dtype=np.int16) + texture, 0, 255).astype(np.uint8)
    source_frames: dict[float, np.ndarray] = {}
    result = []
    for name, event in windows:
        times = tuple(round(event - 2 + index / fps, 6) for index in range(21))
        frames = []
        for timestamp in times:
            if timestamp not in source_frames:
                frame = base.copy()
                center = (round(width * (.28 + timestamp * .008)), round(height * .6))
                radius = round(height * .19)
                cv2.circle(frame, center, radius, (145, 170, 195), -1)
                cv2.circle(frame, center, radius, (50, 70, 90), 3)
                if timestamp >= 13:
                    cv2.rectangle(frame, (center[0] - 7, center[1] - 10),
                                  (center[0] + 10, center[1] + 12), (240, 240, 240), -1)
                cv2.putText(frame, f"Synthetic source {timestamp:.1f}s", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (45, 45, 45), 1)
                source_frames[timestamp] = frame
            frames.append(source_frames[timestamp])
        clip = EvidenceClip(name, event, round(event * fps), times[0], times[-1],
                            times[0], times[-1], fps, len(times))
        result.append(EvidenceClipData(clip, frames, times))
    return result


def decode_videos(paths: list[Path]) -> tuple[int, float]:
    started = time.perf_counter()
    frames = 0
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot decode benchmark output: {path}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
        finally:
            cap.release()
    return frames, (time.perf_counter() - started) * 1000


def measure_loopback_upload(*, paths: list[Path], repeats: int = 3) -> float:
    """Exercise the real uploader against a local sink; this is not a LAN benchmark."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from Scripts.sop_uploader import _http_post_file, _parse_server

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            remaining = int(self.headers["Content-Length"])
            while remaining:
                chunk = self.rfile.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise RuntimeError("Incomplete benchmark upload")
                remaining -= len(chunk)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    server = _parse_server(f"http://127.0.0.1:{http.server_port}")
    samples = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            for path in paths:
                _http_post_file(server=server, path="/benchmark", auth_header="benchmark",
                                file_path=path, dry_run=False)
            samples.append((time.perf_counter() - started) * 1000)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
    return statistics.median(samples)


def measure(*, root: Path, windows: list[EvidenceClipData], approach: str) -> Measurement:
    root.mkdir(parents=True, exist_ok=True)
    delays = []
    paths = []
    started = time.perf_counter()
    finalization_ms = 0.0
    if approach in {"single_webm", "single_webm_async"}:
        writer = SessionEvidenceWriter(session_dir=root, fps=windows[0].clip.fps)
        encoder = EvidenceEncoder() if approach == "single_webm_async" else None
        try:
            for window in windows:
                before = time.perf_counter()
                if encoder is not None:
                    encoder.append(writer, window)
                else:
                    writer.append(window)
                delays.append((time.perf_counter() - before) * 1000)
            before = time.perf_counter()
            if encoder is not None:
                encoder.finish(writer, lambda: None)
                encoder.close()
                result = json.loads((root / "evidence.json").read_text())["video"]
            else:
                result = writer.finalize()
            finalization_ms = (time.perf_counter() - before) * 1000
            assert result is not None
            paths = [root / result["file"]]
        finally:
            writer.abort()
    else:
        for index, window in enumerate(windows):
            before = time.perf_counter()
            if approach == "four_mp4":
                metadata = write_evidence_clip(clip_data=window, session_dir=root, index=index)
                assert metadata is not None
                path = root / metadata["file"]
            else:
                path = root / f"{index}.webm"
                h, w = window.frames[0].shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), window.clip.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("VP8 encoder unavailable")
                try:
                    for frame in window.frames:
                        writer.write(frame)
                finally:
                    writer.release()
            paths.append(path)
            delays.append((time.perf_counter() - before) * 1000)
    preparation_ms = (time.perf_counter() - started) * 1000
    count, decoding_ms = decode_videos(paths)
    size = sum(path.stat().st_size for path in paths)
    h, w = windows[0].frames[0].shape[:2]
    return Measurement(w, h, approach, len(paths), count, size, preparation_ms,
                       max(delays), finalization_ms, decoding_ms,
                       size * 8 / 10_000_000, size * 8 / 100_000_000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--loopback-upload", action="store_true",
                        help="Measure the real uploader against a localhost sink; requires socket access.")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    results: list[dict[str, Any]] = []
    for width, height in [(640, 360), (1920, 1080)]:
        windows = simulated_windows(width, height)
        for approach in ["four_mp4", "four_webm", "single_webm", "single_webm_async"]:
            for repeat in range(args.repeats):
                sample = measure(root=args.out_dir / f"{width}x{height}" / approach / str(repeat),
                                 windows=windows, approach=approach)
                results.append({**asdict(sample), "repeat": repeat})
                print(json.dumps(results[-1]), flush=True)
        del windows
    medians = []
    for width, height in [(640, 360), (1920, 1080)]:
        for approach in ["four_mp4", "four_webm", "single_webm", "single_webm_async"]:
            samples = [row for row in results if row["width"] == width and row["approach"] == approach]
            medians.append({"resolution": f"{width}x{height}", "approach": approach,
                **{key: statistics.median(row[key] for row in samples)
                   for key in asdict(Measurement(0, 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0))
                   if key not in {"width", "height", "approach"}}})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.loopback_upload:
        for row in medians:
            root = args.out_dir / row["resolution"] / row["approach"] / "0"
            paths = sorted(root.rglob("*.mp4")) + sorted(root.rglob("*.webm"))
            row["loopback_upload_ms"] = measure_loopback_upload(paths=paths)
    (args.out_dir / "metrics.json").write_text(json.dumps({"data": "simulated", "fps": 5,
        "repeats": args.repeats, "samples": results, "medians": medians,
        "notes": ["Upload seconds are payload-only estimates, not network measurements.",
                  "four_webm controls for codec; four_mp4 is the previous capture writer.",
                  "Single-video preparation includes decode verification; prior writer does not.",
                  "No GPU/model inference or target Jetson measurement is included."]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write event windows directly into one finalized, browser-playable evidence file."""
from __future__ import annotations

from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import threading
from typing import Any, Callable
import uuid

import cv2
import numpy as np

from .evidence import EvidenceClipData

EVENT_LABELS = {
    "roll_entered": "Roll masuk",
    "cleaned_done": "Pembersihan selesai",
    "labeled_done": "Label terpasang",
    "roll_left": "Roll keluar",
    "recording_ended": "Perekaman selesai",
}


class SessionEvidenceWriter:
    """Append completed windows in source order, encoding overlapping frames once."""

    def __init__(self, *, session_dir: Path, fps: float) -> None:
        if not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("Evidence FPS must be finite and between 0 and 240")
        self.session_dir = session_dir
        self.fps = float(fps)
        self._temporary = session_dir / f".evidence-{uuid.uuid4().hex}.webm"
        self._writer: Any = None
        self._size: tuple[int, int] | None = None
        self._times: list[float] = []
        self._seen: set[float] = set()
        self._events: list[dict[str, Any]] = []
        self._closed = False
        self._result: dict[str, Any] | None = None

    def append(self, clip_data: EvidenceClipData) -> None:
        if self._closed:
            raise RuntimeError("Evidence writer is closed")
        times = clip_data.frame_times_s
        if not clip_data.frames:
            return
        if len(times) != len(clip_data.frames) or not math.isclose(clip_data.clip.fps, self.fps, abs_tol=1e-6):
            raise ValueError("Evidence frame timestamps/cadence do not match")
        if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Evidence timestamps must be finite and increasing")
        self._events.append({"name": clip_data.clip.name,
                             "source_time_s": float(clip_data.clip.event_time_s)})
        for timestamp, frame in zip(times, clip_data.frames):
            key = round(timestamp, 9)
            if key in self._seen:
                continue
            if self._times and timestamp <= self._times[-1]:
                raise ValueError("Completed evidence windows arrived out of source order")
            if len(self._times) >= 10_000:
                raise ValueError("Session evidence exceeds the frame limit")
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Evidence frames must be uint8 HWC BGR images")
            h, w = frame.shape[:2]
            if not 0 < w <= 4096 or not 0 < h <= 4096:
                raise ValueError("Evidence frame dimensions must be between 1 and 4096")
            if self._writer is None:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                self._size = ((w + 1) // 2 * 2, (h + 1) // 2 * 2)
                self._writer = cv2.VideoWriter(str(self._temporary), cv2.VideoWriter_fourcc(*"VP80"),
                                               self.fps, self._size)
                if not self._writer.isOpened():
                    self.abort()
                    raise RuntimeError("Session evidence requires OpenCV VP8/WebM encoding on the capture host")
            assert self._size is not None
            target_w, target_h = self._size
            if (w, h) != self._size:
                scale = min(target_w / w, target_h / h)
                scaled_w, scaled_h = max(1, round(w * scale)), max(1, round(h * scale))
                resized = cv2.resize(frame, (scaled_w, scaled_h),
                                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                frame = cv2.copyMakeBorder(resized, 0, target_h - scaled_h, 0, target_w - scaled_w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            self._writer.write(frame)
            self._times.append(float(timestamp))
            self._seen.add(key)

    def finalize(self) -> dict[str, Any] | None:
        """Verify then publish video and manifest before the session is uploadable."""
        if self._closed:
            return self._result
        self._closed = True
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        try:
            cap = cv2.VideoCapture(str(self._temporary))
            try:
                count = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    count += 1
                if count != len(self._times):
                    raise RuntimeError("Session evidence failed decoded-frame verification")
            finally:
                cap.release()
            events = []
            for event in sorted(self._events, key=lambda e: e["source_time_s"]):
                index = bisect_left(self._times, event["source_time_s"])
                candidates = [i for i in (index - 1, index) if 0 <= i < len(self._times)]
                nearest = min(candidates, key=lambda i: abs(self._times[i] - event["source_time_s"]))
                events.append({**event, "label": EVENT_LABELS.get(event["name"], event["name"].replace("_", " ")),
                               "playback_time_s": nearest / self.fps})
            result = {"file": "evidence/session.webm", "fps": self.fps, "frame_count": len(self._times),
                      "duration_s": len(self._times) / self.fps, "frame_times_s": self._times, "events": events}
            final = self.session_dir / result["file"]
            final.parent.mkdir(parents=True, exist_ok=True)
            manifest = self.session_dir / ".evidence.json.tmp"
            manifest.write_text(json.dumps({"version": 2, "video": result, "clips": []}, indent=2), encoding="utf-8")
            self._temporary.replace(final)
            manifest.replace(self.session_dir / "evidence.json")
            self._result = result
            return result
        finally:
            self._temporary.unlink(missing_ok=True)
            (self.session_dir / ".evidence.json.tmp").unlink(missing_ok=True)

    def abort(self) -> None:
        """Release incomplete output; never publish or upload an unfinished file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._temporary.unlink(missing_ok=True)
        self._closed = True


class EvidenceEncoder:
    """One bounded CPU worker; final session publication follows verified encoding."""

    def __init__(self, *, max_pending_frames: int = 128) -> None:
        if not isinstance(max_pending_frames, int) or max_pending_frames <= 0:
            raise ValueError("Evidence frame budget must be a positive integer")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-evidence")
        self._lock = threading.Lock()
        self._pending_frames = 0
        self._limit = max_pending_frames
        self._aborted: set[SessionEvidenceWriter] = set()
        self._errors: list[BaseException] = []

    def _submit(self, action: Callable[[], None], frames: int = 0, *, check_errors: bool = True) -> None:
        if check_errors:
            self.check_errors()
        with self._lock:
            if self._pending_frames + frames > self._limit:
                raise RuntimeError("Evidence encoder backlog exceeded its bounded frame budget")
            self._pending_frames += frames
        future = self._executor.submit(action)

        def finished(completed: Any) -> None:
            with self._lock:
                self._pending_frames -= frames
                error = completed.exception()
                if error is not None:
                    self._errors.append(error)

        future.add_done_callback(finished)

    def check_errors(self) -> None:
        with self._lock:
            if self._errors:
                raise RuntimeError("Evidence preparation failed; inspect capture-host encoder support") from self._errors[0]

    def append(self, writer: SessionEvidenceWriter, clip: EvidenceClipData) -> None:
        def encode() -> None:
            if writer not in self._aborted:
                writer.append(clip)
        self._submit(encode, len(clip.frames))

    def finish(self, writer: SessionEvidenceWriter, publish: Callable[[], None]) -> None:
        def finalize() -> None:
            try:
                if writer in self._aborted:
                    return
                self.check_errors()
                writer.finalize()
                publish()
            finally:
                writer.abort()
        self._submit(finalize, check_errors=False)

    def abort(self, writer: SessionEvidenceWriter, *, discard: bool = False) -> None:
        self._aborted.add(writer)
        def release() -> None:
            writer.abort()
            if discard:
                shutil.rmtree(writer.session_dir, ignore_errors=True)
        # Cleanup must still run when a previous encode failed.
        self._executor.submit(release)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.check_errors()

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TextIO

from Action_Detection_SOP.safety_alerts import (
    HELMET_DIAGNOSTICS_SCHEMA_VERSION,
    HelmetDiagnosticEvent,
    HelmetDiagnosticObservation,
)
from Action_Detection_SOP.source_security import redact_source_credentials

_RUN_END_RESERVE_BYTES = 1024
_FLUSH_EVERY_FRAMES = 25


class HelmetDiagnosticStorageLimit(RuntimeError):
    """Raised when the diagnostics store has reached its configured byte cap."""


class HelmetDiagnosticJsonlWriter:
    """Write bounded, local-only helmet diagnostics as newline-delimited JSON."""

    def __init__(
        self,
        *,
        out_dir: Path,
        date: str,
        source: str,
        camera_id: Optional[str],
        context: Dict[str, Any],
        max_total_bytes: int,
    ) -> None:
        if isinstance(max_total_bytes, bool) or not isinstance(max_total_bytes, int) or max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be a positive integer")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must use YYYY-MM-DD format") from exc

        self.root = out_dir / "diagnostics" / "helmet"
        self.day_dir = self.root / date
        self.root.mkdir(parents=True, exist_ok=True)
        used_bytes = sum(path.stat().st_size for path in self.root.rglob("*.jsonl") if path.is_file())
        self._budget_bytes = max_total_bytes - used_bytes
        if self._budget_bytes <= _RUN_END_RESERVE_BYTES:
            raise HelmetDiagnosticStorageLimit(
                f"helmet diagnostics storage cap reached ({max_total_bytes} bytes)"
            )

        self.run_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = self.day_dir / f"run_{stamp}_{self.run_id}.jsonl"
        self.day_dir.mkdir(parents=True, exist_ok=True)
        self._stream: Optional[TextIO] = self.path.open("x", encoding="utf-8", newline="", buffering=65536)
        self._bytes_written = 0
        self._frames_written = 0
        self._observations_written = 0
        self._events_written = 0
        self._frames_since_flush = 0
        self.truncated = False
        self.failed = False
        self.disabled_reason: Optional[str] = None
        self._accepting = True

        header = {
            "record_type": "run_start",
            "schema_version": HELMET_DIAGNOSTICS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at_utc": self._now_utc(),
            "source": redact_source_credentials(source),
            "camera_id": redact_source_credentials(camera_id) if camera_id else None,
            "context": context,
            "max_total_bytes": max_total_bytes,
        }
        if not self._write_record(header):
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self.path.unlink(missing_ok=True)
            raise HelmetDiagnosticStorageLimit("not enough remaining storage for a diagnostic run header")

    @property
    def accepting(self) -> bool:
        return self._accepting and self._stream is not None

    def write_frame(
        self,
        *,
        frame_idx: int,
        time_s: float,
        segment_id: int,
        observations: Sequence[HelmetDiagnosticObservation],
        events: Sequence[HelmetDiagnosticEvent],
        alert_uids: Sequence[str] = (),
        final_drain: bool = False,
    ) -> bool:
        if not self.accepting:
            return False
        try:
            time_s = float(time_s)
            if not math.isfinite(time_s) or time_s < 0.0:
                return self._fail()
            payload = {
                "record_type": "final_drain" if final_drain else "frame",
                "logged_at_utc": self._now_utc(),
                "segment_id": int(segment_id),
                "frame_idx": int(frame_idx),
                "time_s": round(time_s, 3),
                "alert_uids": list(alert_uids),
                "observations": [item.as_payload() for item in observations],
                "events": [item.as_payload() for item in events],
            }
        except (TypeError, ValueError, OverflowError):
            return self._fail()
        if not self._write_record(payload):
            return False
        if not final_drain:
            self._frames_written += 1
        self._observations_written += len(observations)
        self._events_written += len(events)
        self._frames_since_flush += 1
        if self._frames_since_flush >= _FLUSH_EVERY_FRAMES:
            return self.flush()
        return True

    def write_segment(
        self,
        *,
        segment_id: int,
        reason: str,
        frame_idx: int,
        time_s: float,
    ) -> bool:
        if not self.accepting:
            return False
        return self._write_record(
            {
                "record_type": "capture_segment",
                "logged_at_utc": self._now_utc(),
                "segment_id": int(segment_id),
                "reason": str(reason),
                "frame_idx": int(frame_idx),
                "time_s": round(float(time_s), 3),
            }
        )

    def flush(self) -> bool:
        if self._stream is None:
            return False
        try:
            self._stream.flush()
        except OSError:
            return self._fail()
        self._frames_since_flush = 0
        return self._accepting

    def close(self) -> None:
        stream = self._stream
        if stream is None:
            return
        summary = {
            "record_type": "run_end",
            "ended_at_utc": self._now_utc(),
            "frames_written": self._frames_written,
            "observations_written": self._observations_written,
            "events_written": self._events_written,
            "truncated": self.truncated,
            "failed": self.failed,
            "disabled_reason": self.disabled_reason,
        }
        try:
            data = self._encode(summary)
            if self._bytes_written + len(data) <= self._budget_bytes:
                stream.write(data.decode("utf-8"))
                self._bytes_written += len(data)
            stream.flush()
        except OSError:
            self.failed = True
            self.disabled_reason = "write_error"
        finally:
            try:
                stream.close()
            except OSError:
                self.failed = True
                self.disabled_reason = "write_error"
            self._stream = None
            self._accepting = False

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "path": str(self.path),
            "run_id": self.run_id,
            "bytes_written": self._bytes_written,
            "frames_written": self._frames_written,
            "observations_written": self._observations_written,
            "events_written": self._events_written,
            "truncated": self.truncated,
            "failed": self.failed,
            "disabled_reason": self.disabled_reason,
        }

    def _write_record(self, payload: Dict[str, Any]) -> bool:
        if self._stream is None or not self._accepting:
            return False
        try:
            data = self._encode(payload)
        except (TypeError, ValueError):
            return self._fail()
        if self._bytes_written + len(data) + _RUN_END_RESERVE_BYTES > self._budget_bytes:
            self.truncated = True
            self.disabled_reason = "storage_limit"
            self._accepting = False
            return False
        try:
            self._stream.write(data.decode("utf-8"))
        except OSError:
            return self._fail()
        self._bytes_written += len(data)
        return True

    @staticmethod
    def _encode(payload: Dict[str, Any]) -> bytes:
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _fail(self) -> bool:
        self.failed = True
        self.disabled_reason = "write_error"
        self._accepting = False
        try:
            if self._stream is not None:
                self._stream.close()
        except OSError:
            pass
        self._stream = None
        return False

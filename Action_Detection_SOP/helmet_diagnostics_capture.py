from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from Action_Detection_SOP.helmet_diagnostics_io import HelmetDiagnosticJsonlWriter
from Action_Detection_SOP.safety_alerts import HelmetAlertEngine


class HelmetDiagnosticCapture:
    """
    Coordinate diagnostic draining across frames and capture segments.
    """

    def __init__(
        self,
        *,
        engine: HelmetAlertEngine,
        out_dir: Path,
        date: str,
        source: str,
        camera_id: Optional[str],
        context: Dict[str, Any],
        max_total_bytes: int,
    ) -> None:
        self._engine = engine
        self._out_dir = out_dir
        self._date = date
        self._source = source
        self._camera_id = camera_id
        self._context = context
        self._max_total_bytes = max_total_bytes
        self._writer: Optional[HelmetDiagnosticJsonlWriter] = None
        self._segment_id = 0
        self._warning_emitted = False
        self._disabled_reason: Optional[str] = None

    def start(self) -> bool:
        try:
            self._writer = HelmetDiagnosticJsonlWriter(
                out_dir=self._out_dir,
                date=self._date,
                source=self._source,
                camera_id=self._camera_id,
                context=self._context,
                max_total_bytes=self._max_total_bytes,
            )
        except Exception as exc:
            self._disabled_reason = type(exc).__name__
            self._engine.disable_diagnostics()
            print(
                "WARNING: Helmet diagnostics unavailable; alert processing continues "
                f"({type(exc).__name__})."
            )
            self._warning_emitted = True
            return False
        return True

    @property
    def path(self) -> Optional[Path]:
        return self._writer.path if self._writer is not None else None

    @property
    def run_id(self) -> Optional[str]:
        return self._writer.run_id if self._writer is not None else None

    def persist_frame(
        self,
        *,
        frame_idx: int,
        time_s: float,
        alert_uids: Sequence[str] = (),
        final_drain: bool = False,
    ) -> None:
        writer = self._writer
        if writer is None or not writer.accepting:
            return
        observations = self._engine.pop_diagnostic_observations()
        events = self._engine.pop_diagnostic_events()
        if not final_drain and not (observations or events or alert_uids):
            return
        accepted = writer.write_frame(
            frame_idx=frame_idx,
            time_s=time_s,
            segment_id=self._segment_id,
            observations=observations,
            events=events,
            alert_uids=alert_uids,
            final_drain=final_drain,
        )
        if not accepted:
            self._disable()

    def start_segment(self, *, reason: str, frame_idx: int, time_s: float) -> None:
        self._segment_id += 1
        self._engine.reset_diagnostic_tracks()
        writer = self._writer
        if writer is not None and writer.accepting:
            if not writer.write_segment(
                segment_id=self._segment_id,
                reason=reason,
                frame_idx=frame_idx,
                time_s=time_s,
            ):
                self._disable()

    def close(self, *, frame_idx: int, time_s: float) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            if writer.accepting:
                self.persist_frame(
                    frame_idx=frame_idx,
                    time_s=time_s,
                    final_drain=True,
                )
        finally:
            writer.close()

    def summary(self) -> Dict[str, Any]:
        details: Dict[str, Any]
        if self._writer is not None:
            details = self._writer.summary()
        else:
            details = {
                "enabled": False,
                "failed": self._disabled_reason is not None,
                "disabled_reason": self._disabled_reason,
            }
        details.update(
            {
                "max_total_bytes": self._max_total_bytes,
                "max_total_mb": self._max_total_bytes / (1024 * 1024),
                "requested": True,
                "engine_errors": self._engine.diagnostic_error_count,
                "dropped_observations": self._engine.diagnostic_dropped_observation_count,
                "dropped_events": self._engine.diagnostic_dropped_event_count,
            }
        )
        return details

    def _disable(self) -> None:
        self._engine.disable_diagnostics()
        writer = self._writer
        self._disabled_reason = writer.disabled_reason if writer is not None else "unavailable"
        if not self._warning_emitted:
            print(f"WARNING: Helmet diagnostics stopped ({self._disabled_reason}); alert processing continues.")
            self._warning_emitted = True

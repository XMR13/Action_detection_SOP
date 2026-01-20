"""
Logika deteksi aksi yang dibuat diatas 'yolo_kit'
SOP/action logic layer built on top of `yolo_kit`.

Package ini secara sengaja agar runtime deteksi tetap berada didalam 'yolo_kit'/
dan lebih berfokus kepada 
- ROI/Session
- keadaan SOP + Rules (DONE/ NOT DONE/ UNKOWN)
- Reporting
- Runner (logika running script mainnya)
- tracking evidence untuk menentukan apakah terdapat hal ini atau tidak
"""

from __future__ import annotations

from typing import Any

from .config import SopProfile, load_sop_profile

# Optional dependency boundary:
# `ingest` depends on OpenCV (`cv2`) and should not break imports of the SOP logic
# modules (tests can run without cv2 installed).
try:
    from .ingest import CaptureInfo, get_capture_info, open_capture
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) != "cv2":
        raise

    CaptureInfo = Any  # type: ignore[misc,assignment]

    def open_capture(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python to use capture ingestion.")

    def get_capture_info(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python to use capture ingestion.")
from .reporting import (
    DailyReport,
    today_date_str,
    write_daily_csv,
    write_daily_report,
    write_run_config,
    write_session_artifacts,
    write_session_run_config,
)
from .roi import RoiPolygon, draw_roi, load_roi_json, resolve_roi_for_frame, save_roi_json
from .sop_engine import HelmetRuleConfig, SessionResult, SessionizationConfig, SopEngine, SopEngineConfig, StepStatus

__all__ = [
    "CaptureInfo",
    "get_capture_info",
    "open_capture",
    "SopProfile",
    "load_sop_profile",
    "DailyReport",
    "today_date_str",
    "write_daily_csv",
    "write_daily_report",
    "write_run_config",
    "write_session_artifacts",
    "write_session_run_config",
    "RoiPolygon",
    "draw_roi",
    "load_roi_json",
    "resolve_roi_for_frame",
    "save_roi_json",
    "HelmetRuleConfig",
    "SessionResult",
    "SessionizationConfig",
    "SopEngine",
    "SopEngineConfig",
    "StepStatus",
]

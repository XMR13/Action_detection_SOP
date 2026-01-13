"""
Logika deteksi aksi yang dibuat diatas 'yolo_kit'
SOP/action logic layer built on top of `yolo_kit`.

Package ini secara sengaja agar runtime deteksi tetap berada didalam 'yolo_kit'/
dan lebih berfokus kepada 
- ROI/Session
- keadaan SOP + Rules (DONE/ NOT DONE/ UNKOWN)
- Reporting
"""

from .ingest import CaptureInfo, get_capture_info, open_capture
from .reporting import DailyReport, today_date_str, write_daily_csv, write_daily_report, write_session_artifacts
from .roi import RoiPolygon, draw_roi, load_roi_json, resolve_roi_for_frame, save_roi_json
from .sop_engine import HelmetRuleConfig, SessionResult, SessionizationConfig, SopEngine, SopEngineConfig, StepStatus

__all__ = [
    "CaptureInfo",
    "get_capture_info",
    "open_capture",
    "DailyReport",
    "today_date_str",
    "write_daily_csv",
    "write_daily_report",
    "write_session_artifacts",
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

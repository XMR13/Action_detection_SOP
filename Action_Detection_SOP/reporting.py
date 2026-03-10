"""
Docstring for Action_Detection_SOP.reporting
Script yang digunakan untuk memprogram reporting untuk SOP nya

"""


from __future__ import annotations


import csv
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .sop_engine import SessionResult, StepStatus, iter_roi_status_counts, iter_status_counts
from .shifts import assign_shift_for_interval, parse_iso_datetime


@dataclass(frozen=True)
class ShiftSummary:
    shift_id: str
    shift_name: str
    shift_date: str
    total_sessions: int
    roi_done: int
    roi_not_done: int
    roi_unknown: int
    helmet_done: int
    helmet_not_done: int
    helmet_unknown: int


@dataclass(frozen=True)
class DailyReport:
    date: str
    total_sessions: int
    roi_done: int
    roi_not_done: int
    roi_unknown: int
    helmet_done: int
    helmet_not_done: int
    helmet_unknown: int
    shift_summaries: List[ShiftSummary]


def session_result_to_dict(r: SessionResult) -> Dict[str, Any]:
    payload = asdict(r)
    # Enums to strings
    payload["operator_present"] = str(r.operator_present.value)
    payload["roi_dwell"] = str(r.roi_dwell.value)
    payload["helmet"] = str(r.helmet.value)
    # Stable primary key used by the website/uploader for idempotency across retries and file moves.
    payload.setdefault("session_uid", uuid.uuid4().hex)

    # Shift enrichment (best-effort). Uses ISO timestamps if present.
    start_dt = parse_iso_datetime(payload.get("start_time_iso"))
    end_dt = parse_iso_datetime(payload.get("end_time_iso"))
    if start_dt is not None:
        if end_dt is None:
            end_dt = start_dt
        assignment = assign_shift_for_interval(start_dt=start_dt, end_dt=end_dt)
        if assignment is not None:
            payload.update(assignment.to_iso_fields())

    return payload


def write_session_artifacts(
    *,
    out_dir: Path,
    date: str,
    session: SessionResult,
) -> Path:
    session_dir = out_dir / "sessions" / date / f"session_{session.session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "checklist.json").write_text(
        json.dumps(session_result_to_dict(session), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return session_dir


def write_run_config(*, out_dir: Path, date: str, run_config: Dict[str, Any]) -> Path:
    report_dir = out_dir / "reports" / date
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "run_config.json"
    path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_session_run_config(*, session_dir: Path, run_config: Dict[str, Any]) -> Path:
    path = session_dir / "run_config.json"
    path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_daily_report(
    *,
    out_dir: Path,
    date: str,
    sessions: Iterable[SessionResult],
) -> Path:
    """
    Fungsi yang berguna untuk membuat daily repor, akan ditampilkan menerima input sebagai berikut:
    Args:
        out_dir : path output dari daily report, inside system
        date    : date dalam string
        session : sesi untuk setiap hari yang telah ditetnukan
    
    """
    sessions_list = list(sessions)
    roi_done, roi_not_done, roi_unknown = iter_roi_status_counts(sessions_list)
    done, not_done, unknown = iter_status_counts(sessions_list)

    by_shift: Dict[Tuple[str, str, str], List[SessionResult]] = {}
    for s in sessions_list:
        start_dt = parse_iso_datetime(s.start_time_iso)
        end_dt = parse_iso_datetime(s.end_time_iso)
        if start_dt is None:
            continue
        if end_dt is None:
            end_dt = start_dt
        assignment = assign_shift_for_interval(start_dt=start_dt, end_dt=end_dt)
        if assignment is None:
            continue
        key = (assignment.shift_date, assignment.shift_id, assignment.shift_name)
        by_shift.setdefault(key, []).append(s)

    shift_summaries: List[ShiftSummary] = []
    for (shift_date, shift_id, shift_name), bucket in sorted(by_shift.items()):
        s_roi_done, s_roi_not_done, s_roi_unknown = iter_roi_status_counts(bucket)
        s_done, s_not_done, s_unknown = iter_status_counts(bucket)
        shift_summaries.append(
            ShiftSummary(
                shift_id=shift_id,
                shift_name=shift_name,
                shift_date=shift_date,
                total_sessions=len(bucket),
                roi_done=s_roi_done,
                roi_not_done=s_roi_not_done,
                roi_unknown=s_roi_unknown,
                helmet_done=s_done,
                helmet_not_done=s_not_done,
                helmet_unknown=s_unknown,
            )
        )

    report = DailyReport(
        date=date,
        total_sessions=len(sessions_list),
        roi_done=roi_done,
        roi_not_done=roi_not_done,
        roi_unknown=roi_unknown,
        helmet_done=done,
        helmet_not_done=not_done,
        helmet_unknown=unknown,
        shift_summaries=shift_summaries,
    )
    report_dir = out_dir / "reports" / date
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "daily_report.json"
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_daily_csv(
    *,
    out_dir: Path,
    date: str,
    sessions: Iterable[SessionResult],
) -> Path:
    """
    Fungsi untuk menyimpan daily csv (report yang diperlukan)
    Args:
        out_dir : direktori output
        date    : date dalam string
        session : Sesi untuk setiap hari yang telah ditentukan 
    """
    rows: List[Dict[str, Any]] = [session_result_to_dict(s) for s in sessions]
    report_dir = out_dir / "reports" / date
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "sessions.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def today_date_str(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now()
    return dt.strftime("%Y-%m-%d")

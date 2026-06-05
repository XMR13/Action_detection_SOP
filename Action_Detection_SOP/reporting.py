"""
Docstring for Action_Detection_SOP.reporting
Script yang digunakan untuk memprogram reporting untuk SOP nya

"""
from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .roll_sop_engine import RollComplianceStatus, RollSessionResult
from .sop_engine import SessionResult, StepStatus, iter_roi_status_counts, iter_status_counts
from .shifts import assign_shift_for_interval, parse_iso_datetime


SessionReportResult = Union[SessionResult, RollSessionResult]

def _status_count(rows: Iterable[StepStatus], status: StepStatus) -> int:
    return sum(1 for row in rows if row == status)

def _roll_status_count(rows: Iterable[RollComplianceStatus], status: RollComplianceStatus) -> int:
    return sum(1 for row in rows if row == status)

def _is_roll_session(r: SessionReportResult) -> bool:
    return isinstance(r, RollSessionResult)

def session_result_to_dict(r: SessionReportResult) -> Dict[str, Any]:
    payload = asdict(r)
    if _is_roll_session(r):
        assert isinstance(r, RollSessionResult)
        payload["cleaned"] = str(r.cleaned.value)
        payload["labeled"] = str(r.labeled.value)
        payload["overall_status"] = str(r.overall_status.value)
        payload["duration_s"] = max(0.0, float(r.end_time_s) - float(r.start_time_s))
    else:
        assert isinstance(r, SessionResult)
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
    session: SessionReportResult,
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
    sessions: Iterable[SessionReportResult],
) -> Path:
    """
    Fungsi yang berguna untuk membuat daily repor, akan ditampilkan menerima input sebagai berikut:
    Args:
        out_dir : path output dari daily report, inside system
        date    : date dalam string
        session : sesi untuk setiap hari yang telah ditetnukan
    
    """
    sessions_list = list(sessions)
    if any(_is_roll_session(s) for s in sessions_list):
        roll_sessions = [s for s in sessions_list if isinstance(s, RollSessionResult)]
        report_dir = out_dir / "reports" / date
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "daily_report.json"
        payload = {
            "date": date,
            "sop_profile": "roll_sop_v1",
            "total_sessions": len(roll_sessions),
            "cleaned_done": _status_count((s.cleaned for s in roll_sessions), StepStatus.DONE),
            "cleaned_not_done": _status_count((s.cleaned for s in roll_sessions), StepStatus.NOT_DONE),
            "cleaned_unknown": _status_count((s.cleaned for s in roll_sessions), StepStatus.UNKNOWN),
            "labeled_done": _status_count((s.labeled for s in roll_sessions), StepStatus.DONE),
            "labeled_not_done": _status_count((s.labeled for s in roll_sessions), StepStatus.NOT_DONE),
            "labeled_unknown": _status_count((s.labeled for s in roll_sessions), StepStatus.UNKNOWN),
            "overall_compliant": _roll_status_count(
                (s.overall_status for s in roll_sessions), RollComplianceStatus.COMPLIANT
            ),
            "overall_non_compliant": _roll_status_count(
                (s.overall_status for s in roll_sessions), RollComplianceStatus.NON_COMPLIANT
            ),
            "overall_unknown": _roll_status_count(
                (s.overall_status for s in roll_sessions), RollComplianceStatus.UNKNOWN
            ),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

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

    shift_summaries: List[Dict[str, Any]] = []
    for (shift_date, shift_id, shift_name), bucket in sorted(by_shift.items()):
        s_roi_done, s_roi_not_done, s_roi_unknown = iter_roi_status_counts(bucket)
        s_done, s_not_done, s_unknown = iter_status_counts(bucket)
        shift_summaries.append(
            {
                "shift_id": shift_id,
                "shift_name": shift_name,
                "shift_date": shift_date,
                "total_sessions": len(bucket),
                "roi_done": s_roi_done,
                "roi_not_done": s_roi_not_done,
                "roi_unknown": s_roi_unknown,
                "helmet_done": s_done,
                "helmet_not_done": s_not_done,
                "helmet_unknown": s_unknown,
            }
        )

    payload = {
        "date": date,
        "total_sessions": len(sessions_list),
        "roi_done": roi_done,
        "roi_not_done": roi_not_done,
        "roi_unknown": roi_unknown,
        "helmet_done": done,
        "helmet_not_done": not_done,
        "helmet_unknown": unknown,
        "shift_summaries": shift_summaries,
    }
    report_dir = out_dir / "reports" / date
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "daily_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_daily_csv(
    *,
    out_dir: Path,
    date: str,
    sessions: Iterable[SessionReportResult],
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

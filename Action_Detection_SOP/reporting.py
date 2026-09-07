"""
Docstring for Action_Detection_SOP.reporting
Script yang digunakan untuk memprogram reporting untuk SOP nya

"""
from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .roll_sop_engine import RollComplianceStatus, RollSessionResult
from .sop_engine import SessionResult, StepStatus, iter_roi_status_counts, iter_status_counts
from .shifts import assign_shift_for_interval, parse_iso_datetime


SessionReportResult = Union[SessionResult, RollSessionResult]

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
    session_dir: Optional[Path] = None,
) -> Path:
    if session_dir is None:
        session_dir = out_dir / "sessions" / date / f"session_{session.session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
    elif session_dir.name != f"session_{session.session_id}":
        raise ValueError("session_dir must match the session_id")
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
    append: bool = False,
    sop_profile: Optional[str] = None,
) -> Path:
    """
    Fungsi yang berguna untuk membuat daily repor, akan ditampilkan menerima input sebagai berikut:
    Args:
        out_dir : path output dari daily report, inside system
        date    : date dalam string
        session : sesi untuk setiap hari yang telah ditetnukan
    
    """
    sessions_list = list(sessions)
    if sop_profile == "roll_sop_v1" or any(_is_roll_session(s) for s in sessions_list):
        roll_sessions = [s for s in sessions_list if isinstance(s, RollSessionResult)]
        report_dir = out_dir / "reports" / date
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "daily_report.json"
        payload = {
            "date": date,
            "sop_profile": "roll_sop_v1",
            "total_sessions": 0,
            "cleaned_done": 0,
            "cleaned_not_done": 0,
            "cleaned_unknown": 0,
            "labeled_done": 0,
            "labeled_not_done": 0,
            "labeled_unknown": 0,
            "overall_compliant": 0,
            "overall_non_compliant": 0,
            "overall_unknown": 0,
        }
        if append and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if (
                isinstance(existing, dict)
                and existing.get("date") == date
                and existing.get("sop_profile") == "roll_sop_v1"
            ):
                for key in payload:
                    if key in {"date", "sop_profile"}:
                        continue
                    payload[key] = int(existing.get(key, 0))

        for session in roll_sessions:
            payload["total_sessions"] += 1
            payload["cleaned_done"] += int(session.cleaned == StepStatus.DONE)
            payload["cleaned_not_done"] += int(session.cleaned == StepStatus.NOT_DONE)
            payload["cleaned_unknown"] += int(session.cleaned == StepStatus.UNKNOWN)
            payload["labeled_done"] += int(session.labeled == StepStatus.DONE)
            payload["labeled_not_done"] += int(session.labeled == StepStatus.NOT_DONE)
            payload["labeled_unknown"] += int(session.labeled == StepStatus.UNKNOWN)
            payload["overall_compliant"] += int(session.overall_status == RollComplianceStatus.COMPLIANT)
            payload["overall_non_compliant"] += int(session.overall_status == RollComplianceStatus.NON_COMPLIANT)
            payload["overall_unknown"] += int(session.overall_status == RollComplianceStatus.UNKNOWN)
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
    append: bool = False,
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
        if append and path.exists():
            return path
        path.write_text("", encoding="utf-8")
        return path

    existing_fieldnames: List[str] = []
    existing_rows: List[Dict[str, Any]] = []
    if append and path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_fieldnames and all(set(row).issubset(existing_fieldnames) for row in rows):
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=existing_fieldnames)
                writer.writerows(rows)
            return path

    fieldnames = list(existing_fieldnames)
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)
    return path


def today_date_str(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now()
    return dt.strftime("%Y-%m-%d")


def highest_existing_session_number(*, out_dir: Path, date: str) -> int:
    """Return the highest numeric session directory already present for a date."""
    date_dir = out_dir / "sessions" / date

    # Check whether this is a data directory.
    if not date_dir.is_dir():
        return 0

    highest = 0
    for candidate in date_dir.glob("session_*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("session_")
        if suffix.isdigit():
            highest = max(highest, int(suffix))

    return highest

def date_for_elapsed_time(
    *,
    run_start_dt: Optional[datetime],
    elapsed_s: float,
    fallback_date: str,
) -> str:
    """Return the local calendar date for a timestamp within the current run.

    Long-running RTSP workers can cross midnight without restarting.  Session
    folders should follow the session start date rather than the date captured
    when the worker process was launched.
    """
    if run_start_dt is None:
        return fallback_date
    elapsed = max(0.0, float(elapsed_s))
    return today_date_str(run_start_dt + timedelta(seconds=elapsed))


def session_start_datetime(
    *,
    run_start_dt: Optional[datetime],
    elapsed_s: float,
    live_source: bool,
    wall_clock_dt: Optional[datetime] = None,
) -> datetime:
    """Resolve a session start using wall time for live sources."""
    if live_source:
        return wall_clock_dt or datetime.now()
    if run_start_dt is not None:
        return run_start_dt + timedelta(seconds=max(0.0, float(elapsed_s)))
    return wall_clock_dt or datetime.now()

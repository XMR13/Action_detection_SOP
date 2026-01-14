"""
Docstring for Action_Detection_SOP.reporting
Script yang digunakan untuk memprogram reporting untuk SOP nya

"""


from __future__ import annotations


import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .sop_engine import SessionResult, StepStatus, iter_status_counts


@dataclass(frozen=True)
class DailyReport:
    date: str
    total_sessions: int
    helmet_done: int
    helmet_not_done: int
    helmet_unknown: int


def session_result_to_dict(r: SessionResult) -> Dict[str, Any]:
    payload = asdict(r)
    # Enums to strings
    payload["operator_present"] = str(r.operator_present.value)
    payload["helmet"] = str(r.helmet.value)
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
    done, not_done, unknown = iter_status_counts(sessions_list)
    report = DailyReport(
        date=date,
        total_sessions=len(sessions_list),
        helmet_done=done,
        helmet_not_done=not_done,
        helmet_unknown=unknown,
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

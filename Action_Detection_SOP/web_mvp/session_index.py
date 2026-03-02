from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    checklist_json: Path
    run_config_json: Path
    thumbnail_jpg: Path
    evidence_json: Path


@dataclass(frozen=True)
class SessionArtifact:
    session_uid: str
    date: str
    session_id: str
    checklist: Dict[str, Any]
    evidence: Dict[str, Any]
    paths: SessionPaths


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _iter_session_dirs(data_dir: Path) -> Iterable[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []

    # Primary layout from run_sop_mvp:
    # <out_dir>/sessions/<date>/session_<id>/
    direct_sessions_root = data_dir / "sessions"
    direct_sessions_root_resolved: Optional[Path] = None
    if direct_sessions_root.exists() and direct_sessions_root.is_dir():
        direct_sessions_root_resolved = direct_sessions_root.resolve()
        for date_dir in sorted(direct_sessions_root.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            for session_dir in sorted(date_dir.glob("session_*")):
                if session_dir.is_dir():
                    out.append((date, session_dir, data_dir))

    # Also support nested copied run-output folders:
    # <data_dir>/<any_parent>/sessions/<date>/session_<id>/
    for sessions_root in sorted(data_dir.glob("**/sessions")):
        if not sessions_root.is_dir():
            continue
        # Skip the already-scanned direct root to avoid duplicates.
        if direct_sessions_root_resolved is not None and sessions_root.resolve() == direct_sessions_root_resolved:
            continue
        run_root = sessions_root.parent
        for date_dir in sorted(sessions_root.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            for session_dir in sorted(date_dir.glob("session_*")):
                if session_dir.is_dir():
                    out.append((date, session_dir, run_root))

    # Deduplicate by absolute session path in case two globs resolve to same dir.
    unique: Dict[str, Tuple[str, Path, Path]] = {}
    for date, session_dir, run_root in out:
        unique[str(session_dir.resolve())] = (date, session_dir, run_root)
    return list(unique.values())


def build_session_uid(*, data_dir: Path, run_root: Path, date: str, session_id: str) -> str:
    # Use a path-derived uid to avoid collisions when multiple output folders
    # have the same date/session_id (common in repeated MVP runs).
    if run_root.resolve() == data_dir.resolve():
        prefix = "root"
    else:
        try:
            rel = run_root.resolve().relative_to(data_dir.resolve())
            prefix = "__".join(rel.parts) if rel.parts else "root"
        except ValueError:
            prefix = "__".join(run_root.resolve().parts[-3:])
    safe_prefix = prefix.replace("/", "__").replace("\\", "__")
    return f"{safe_prefix}__{date}__{session_id}"


def parse_session_uid(session_uid: str) -> Tuple[str, str]:
    parts = session_uid.split("__")
    if len(parts) < 3:
        raise ValueError("Invalid session_uid")
    date = parts[-2]
    session_id = parts[-1]
    if not date or not session_id:
        raise ValueError("Invalid session_uid (empty parts)")
    return date, session_id


class SessionIndex:
    def __init__(self, *, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._lock = threading.Lock()
        self._by_uid: Dict[str, SessionArtifact] = {}
        self._last_scan_utc: Optional[str] = None

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def last_scan_utc(self) -> Optional[str]:
        with self._lock:
            return self._last_scan_utc

    def refresh(self) -> None:
        by_uid: Dict[str, SessionArtifact] = {}
        for date, session_dir, run_root in _iter_session_dirs(self._data_dir):
            checklist_path = session_dir / "checklist.json"
            checklist = _safe_read_json(checklist_path)
            session_id = str(checklist.get("session_id") or session_dir.name.replace("session_", ""))
            session_uid = build_session_uid(
                data_dir=self._data_dir,
                run_root=run_root,
                date=date,
                session_id=session_id,
            )
            paths = SessionPaths(
                session_dir=session_dir,
                checklist_json=checklist_path,
                run_config_json=session_dir / "run_config.json",
                thumbnail_jpg=session_dir / "thumbnail.jpg",
                evidence_json=session_dir / "evidence.json",
            )
            evidence = _safe_read_json(paths.evidence_json)
            by_uid[session_uid] = SessionArtifact(
                session_uid=session_uid,
                date=date,
                session_id=session_id,
                checklist=checklist,
                evidence=evidence,
                paths=paths,
            )

        with self._lock:
            self._by_uid = by_uid
            self._last_scan_utc = _utc_now_iso()

    def get(self, session_uid: str) -> Optional[SessionArtifact]:
        with self._lock:
            return self._by_uid.get(session_uid)

    def list(self) -> List[SessionArtifact]:
        with self._lock:
            sessions = list(self._by_uid.values())
        # newest date first; fallback to uid ordering
        sessions.sort(key=lambda s: (s.date, s.session_id), reverse=True)
        return sessions

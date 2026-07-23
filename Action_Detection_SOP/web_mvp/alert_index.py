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
class AlertPaths:
    alert_dir: Path
    alert_json: Path
    thumbnail_jpg: Path


@dataclass(frozen=True)
class AlertArtifact:
    alert_uid: str
    date: str
    alert_type: str
    payload: Dict[str, Any]
    paths: AlertPaths


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_alert_dirs(data_dir: Path) -> Iterable[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []

    direct_alerts_root = data_dir / "alerts"
    direct_alerts_root_resolved: Optional[Path] = None
    if direct_alerts_root.exists() and direct_alerts_root.is_dir():
        direct_alerts_root_resolved = direct_alerts_root.resolve()
        for date_dir in sorted(direct_alerts_root.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            for child in sorted(date_dir.iterdir()):
                if child.is_dir() and (child / "alert.json").exists():
                    out.append((date, child, data_dir))

    for alerts_root in sorted(data_dir.glob("**/alerts")):
        if not alerts_root.is_dir():
            continue
        if direct_alerts_root_resolved is not None and alerts_root.resolve() == direct_alerts_root_resolved:
            continue
        run_root = alerts_root.parent
        for date_dir in sorted(alerts_root.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            for child in sorted(date_dir.iterdir()):
                if child.is_dir() and (child / "alert.json").exists():
                    out.append((date, child, run_root))

    unique: Dict[str, Tuple[str, Path, Path]] = {}
    for date, alert_dir, run_root in out:
        unique[str(alert_dir.resolve())] = (date, alert_dir, run_root)
    return list(unique.values())


class AlertIndex:
    def __init__(self, *, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._lock = threading.Lock()
        self._by_uid: Dict[str, AlertArtifact] = {}
        self._last_scan_utc: Optional[str] = None

    @property
    def last_scan_utc(self) -> Optional[str]:
        with self._lock:
            return self._last_scan_utc

    def refresh(self) -> None:
        by_uid: Dict[str, AlertArtifact] = {}
        for date, alert_dir, _run_root in _iter_alert_dirs(self._data_dir):
            alert_path = alert_dir / "alert.json"
            payload = _safe_read_json(alert_path)
            alert_uid = str(payload.get("alert_uid") or alert_dir.name).strip()
            if not alert_uid:
                continue
            alert_type = str(payload.get("alert_type") or "").strip().upper()
            paths = AlertPaths(
                alert_dir=alert_dir,
                alert_json=alert_path,
                thumbnail_jpg=alert_dir / "thumbnail.jpg",
            )
            by_uid[alert_uid] = AlertArtifact(
                alert_uid=alert_uid,
                date=date,
                alert_type=alert_type,
                payload=payload,
                paths=paths,
            )

        with self._lock:
            self._by_uid = by_uid
            self._last_scan_utc = _utc_now_iso()

    def get(self, alert_uid: str) -> Optional[AlertArtifact]:
        with self._lock:
            return self._by_uid.get(alert_uid)

    def list(self) -> List[AlertArtifact]:
        with self._lock:
            alerts = list(self._by_uid.values())
        alerts.sort(key=lambda a: (a.date, str(a.payload.get("end_time_s") or ""), a.alert_uid), reverse=True)
        return alerts

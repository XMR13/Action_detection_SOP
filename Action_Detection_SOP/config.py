from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SopProfile:
    schema_version: int
    session_start_seconds: float
    session_end_seconds: float
    min_session_seconds: float = 0.0
    roi_dwell_seconds: float
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("sop_profile schema_version must be 1")
        if self.session_start_seconds <= 0:
            raise ValueError("session_start_seconds must be > 0")
        if self.session_end_seconds <= 0:
            raise ValueError("session_end_seconds must be > 0")
        if self.min_session_seconds < 0:
            raise ValueError("min_session_seconds must be >= 0")
        if self.roi_dwell_seconds <= 0:
            raise ValueError("roi_dwell_seconds must be > 0")


def _require_number(payload: Dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing required key: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _require_int(payload: Dict[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"Missing required key: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def load_sop_profile(path: Path) -> SopProfile:
    if not path.exists():
        raise FileNotFoundError(f"SOP profile not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SOP profile JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("SOP profile must be a JSON object")

    allowed = {
        "schema_version",
        "session_start_seconds",
        "session_end_seconds",
        "min_session_seconds",
        "roi_dwell_seconds",
        "notes",
    }
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown SOP profile keys: {unknown}")

    schema_version = _require_int(payload, "schema_version")
    session_start_seconds = _require_number(payload, "session_start_seconds")
    session_end_seconds = _require_number(payload, "session_end_seconds")
    min_session_seconds = float(payload.get("min_session_seconds", 0.0))
    if min_session_seconds < 0:
        raise ValueError("min_session_seconds must be >= 0")
    roi_dwell_seconds = _require_number(payload, "roi_dwell_seconds")
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("notes must be a string if provided")

    return SopProfile(
        schema_version=schema_version,
        session_start_seconds=session_start_seconds,
        session_end_seconds=session_end_seconds,
        min_session_seconds=min_session_seconds,
        roi_dwell_seconds=roi_dwell_seconds,
        notes=notes,
    )

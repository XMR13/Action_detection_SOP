"""Shared, dependency-free rules for helmet diagnostic records."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


HELMET_DIAGNOSTICS_SCHEMA_VERSION = 2
HELMET_DIAGNOSTIC_MAX_TRACKS = 32
HELMET_DIAGNOSTIC_MAX_RECENT_FRAMES = 150


def is_diagnostic_int(value: Any, *, minimum: int = 0) -> bool:
    """Accept JSON integers, but never booleans or floating-point values."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def require_diagnostic_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return int(value)


def validate_shadow_payload(value: Any) -> None:
    """Validate the fields and count relationships shared by writer and reader."""
    if not isinstance(value, Mapping):
        raise TypeError("shadow must be an object")

    required = require_diagnostic_int(value.get("required_frames"), "required_frames", minimum=1)
    window = require_diagnostic_int(value.get("window_frames"), "window_frames", minimum=1)
    observed = require_diagnostic_int(value.get("observed_frames"), "observed_frames")
    verified = require_diagnostic_int(value.get("verified_frames"), "verified_frames")
    streak = require_diagnostic_int(value.get("unverified_streak_frames"), "unverified_streak_frames")
    sustained = value.get("sustained_unverified")

    if window > HELMET_DIAGNOSTIC_MAX_RECENT_FRAMES:
        raise ValueError("shadow window_frames exceeds diagnostic bound")
    if observed > window or verified > observed:
        raise ValueError("shadow frame counts are inconsistent")
    if not isinstance(sustained, bool):
        raise TypeError("sustained_unverified must be a bool")
    if sustained != (streak >= required):
        raise ValueError("shadow sustained_unverified does not match streak")


def validate_candidate_track_ids(value: Iterable[int]) -> tuple[int, ...]:
    """Return unique, sorted person IDs within the diagnostic track bound."""
    try:
        track_ids = tuple(
            require_diagnostic_int(track_id, "candidate_track_id", minimum=1)
            for track_id in value
        )
    except TypeError as exc:
        raise TypeError("candidate_track_ids must contain integers") from exc
    if len(track_ids) > HELMET_DIAGNOSTIC_MAX_TRACKS or len(set(track_ids)) != len(track_ids):
        raise ValueError("candidate_track_ids must be unique and bounded")
    return tuple(sorted(track_ids))

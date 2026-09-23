from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Action_Detection_SOP.roi import RoiPolygon, draw_roi
from Action_Detection_SOP.source_security import redact_source_credentials
from yolo_kit.types import Detection


ALERT_TYPE_NO_HELMET = "NO_HELMET"
SAFETY_PROFILE_HELMET_ALERT_V1 = "helmet_alert_v1"
ALERT_STATUS_PENDING = "PENDING"
MACHINE_STATUS_NO_HELMET = "NO_HELMET"
DEFAULT_HELMET_REQUIRED_SECONDS = 10
DEFAULT_HELMET_ALERT_CONFIDENCE = 0.15
HELMET_DIAGNOSTICS_SCHEMA_VERSION = 1
HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY = 32
HELMET_DIAGNOSTIC_MAX_ASSOCIATIONS = 8
HELMET_DIAGNOSTIC_MAX_OBSERVATIONS = 128
HELMET_DIAGNOSTIC_IOU_THRESHOLD = 0.25
HELMET_DIAGNOSTIC_MAX_MISSED_FRAMES = 2
HELMET_DIAGNOSTIC_MAX_TRACKS = 32
HELMET_DIAGNOSTIC_MAX_PENDING_OBSERVATIONS = 256

"""
---------------------------
DATA SCHEMA
---------------------------
"""


def _diagnostic_float(
    value: float,
    field_name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return number


def _diagnostic_int(value: int, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return int(value)


def _diagnostic_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _diagnostic_box(value: Sequence[float], field_name: str) -> Tuple[float, float, float, float]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must contain four numbers") from exc
    if len(values) != 4:
        raise ValueError(f"{field_name} must contain exactly four values")
    box = tuple(_diagnostic_float(item, f"{field_name}[{idx}]") for idx, item in enumerate(values))
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError(f"{field_name} must have x2 >= x1 and y2 >= y1")
    return box  # type: ignore[return-value]


def _diagnostic_point(value: Sequence[float], field_name: str) -> Tuple[float, float]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must contain two numbers") from exc
    if len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    point = tuple(
        _diagnostic_float(item, f"{field_name}[{idx}]", minimum=0.0, maximum=1.0)
        for idx, item in enumerate(values)
    )
    return point  # type: ignore[return-value]


def _diagnostic_text(value: str, field_name: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters")
    if "\n" in text or "\r" in text:
        raise ValueError(f"{field_name} must not contain newlines")
    return text


def _diagnostic_optional_source(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    text = _diagnostic_text(value, field_name)
    return redact_source_credentials(text)


def _diagnostic_round(value: float) -> float:
    return round(float(value), 3)


@dataclass(frozen=True)
class HelmetDiagnosticAssociation:
    """
    Immutable geometry/evidence for one observed helmet candidate.
    """

    helmet_box: Tuple[float, float, float, float]
    helmet_score: float
    center_inside_person: bool
    center_inside_head: bool
    center_distance_to_person_px: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "helmet_box", _diagnostic_box(self.helmet_box, "helmet_box"))
        object.__setattr__(
            self,
            "helmet_score",
            _diagnostic_float(self.helmet_score, "helmet_score", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "center_inside_person",
            _diagnostic_bool(self.center_inside_person, "center_inside_person"),
        )
        object.__setattr__(
            self,
            "center_inside_head",
            _diagnostic_bool(self.center_inside_head, "center_inside_head"),
        )
        if self.center_distance_to_person_px is not None:
            object.__setattr__(
                self,
                "center_distance_to_person_px",
                _diagnostic_float(
                    self.center_distance_to_person_px,
                    "center_distance_to_person_px",
                    minimum=0.0,
                ),
            )

    def as_payload(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.helmet_box
        return {
            "helmet_box": [_diagnostic_round(value) for value in self.helmet_box],
            "helmet_center": [_diagnostic_round((x1 + x2) * 0.5), _diagnostic_round((y1 + y2) * 0.5)],
            "helmet_score": _diagnostic_round(self.helmet_score),
            "center_inside_person": self.center_inside_person,
            "center_inside_head": self.center_inside_head,
            "center_distance_to_person_px": (
                None
                if self.center_distance_to_person_px is None
                else _diagnostic_round(self.center_distance_to_person_px)
            ),
        }


@dataclass(frozen=True)
class HelmetDiagnosticObservation:
    """
    Immutable, bounded per-frame observation for a diagnostic-only track.
    """

    diagnostic_track_id: int
    frame_idx: int
    time_s: float
    person_box: Tuple[float, float, float, float]
    person_height_px: float
    normalized_position: Tuple[float, float]
    at_frame_edge: bool
    head_visible: bool
    helmet_score_history: Tuple[float, ...] = ()
    helmet_hit_count: int = 0
    helmet_observation_count: int = 0
    helmet_hit_rate: Optional[float] = 0.0
    best_helmet_score: Optional[float] = None
    associations: Tuple[HelmetDiagnosticAssociation, ...] = ()
    track_history_length: int = 1
    track_history_limit: int = HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY
    track_history_truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostic_track_id",
            _diagnostic_int(self.diagnostic_track_id, "diagnostic_track_id", minimum=1),
        )
        object.__setattr__(self, "frame_idx", _diagnostic_int(self.frame_idx, "frame_idx"))
        object.__setattr__(
            self,
            "time_s",
            _diagnostic_float(self.time_s, "time_s", minimum=0.0),
        )
        object.__setattr__(self, "person_box", _diagnostic_box(self.person_box, "person_box"))
        object.__setattr__(
            self,
            "person_height_px",
            _diagnostic_float(self.person_height_px, "person_height_px", minimum=0.0),
        )
        object.__setattr__(
            self,
            "normalized_position",
            _diagnostic_point(self.normalized_position, "normalized_position"),
        )
        object.__setattr__(self, "at_frame_edge", _diagnostic_bool(self.at_frame_edge, "at_frame_edge"))
        object.__setattr__(self, "head_visible", _diagnostic_bool(self.head_visible, "head_visible"))

        score_history = tuple(
            _diagnostic_float(score, f"helmet_score_history[{idx}]", minimum=0.0, maximum=1.0)
            for idx, score in enumerate(tuple(self.helmet_score_history))
        )
        if len(score_history) > HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY:
            raise ValueError(
                "helmet_score_history exceeds "
                f"{HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY} retained values"
            )
        object.__setattr__(self, "helmet_score_history", score_history)

        hit_count = _diagnostic_int(self.helmet_hit_count, "helmet_hit_count")
        observation_count = _diagnostic_int(self.helmet_observation_count, "helmet_observation_count")
        if hit_count > observation_count:
            raise ValueError("helmet_hit_count must be <= helmet_observation_count")
        if len(score_history) > observation_count:
            raise ValueError("helmet_score_history cannot exceed helmet_observation_count")
        object.__setattr__(self, "helmet_hit_count", hit_count)
        object.__setattr__(self, "helmet_observation_count", observation_count)
        if self.helmet_hit_rate is not None:
            hit_rate = _diagnostic_float(
                self.helmet_hit_rate,
                "helmet_hit_rate",
                minimum=0.0,
                maximum=1.0,
            )
            if observation_count == 0 and hit_rate != 0.0:
                raise ValueError("helmet_hit_rate must be 0 when helmet_observation_count is 0")
            object.__setattr__(self, "helmet_hit_rate", hit_rate)
        if self.best_helmet_score is not None:
            object.__setattr__(
                self,
                "best_helmet_score",
                _diagnostic_float(
                    self.best_helmet_score,
                    "best_helmet_score",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

        associations = tuple(self.associations)
        if len(associations) > HELMET_DIAGNOSTIC_MAX_ASSOCIATIONS:
            raise ValueError(
                "associations exceeds "
                f"{HELMET_DIAGNOSTIC_MAX_ASSOCIATIONS} retained values"
            )
        if not all(isinstance(item, HelmetDiagnosticAssociation) for item in associations):
            raise TypeError("associations must contain HelmetDiagnosticAssociation values")
        associations = tuple(
            sorted(
                associations,
                key=lambda item: (
                    item.helmet_box,
                    item.helmet_score,
                    item.center_inside_person,
                    item.center_inside_head,
                    item.center_distance_to_person_px
                    if item.center_distance_to_person_px is not None
                    else -1.0,
                ),
            )
        )
        object.__setattr__(self, "associations", associations)

        object.__setattr__(
            self,
            "track_history_length",
            _diagnostic_int(self.track_history_length, "track_history_length", minimum=0),
        )
        object.__setattr__(
            self,
            "track_history_limit",
            _diagnostic_int(self.track_history_limit, "track_history_limit", minimum=1),
        )
        object.__setattr__(
            self,
            "track_history_truncated",
            _diagnostic_bool(self.track_history_truncated, "track_history_truncated"),
        )

    def as_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": HELMET_DIAGNOSTICS_SCHEMA_VERSION,
            "diagnostic_track_id": self.diagnostic_track_id,
            "frame_idx": self.frame_idx,
            "time_s": _diagnostic_round(self.time_s),
            "person_box": [_diagnostic_round(value) for value in self.person_box],
            "person_height_px": _diagnostic_round(self.person_height_px),
            "normalized_position": [_diagnostic_round(value) for value in self.normalized_position],
            "at_frame_edge": self.at_frame_edge,
            "head_visible": self.head_visible,
            "helmet_score_history": [_diagnostic_round(value) for value in self.helmet_score_history],
            "helmet_hit_count": self.helmet_hit_count,
            "helmet_observation_count": self.helmet_observation_count,
            "helmet_hit_rate": (
                None if self.helmet_hit_rate is None else _diagnostic_round(self.helmet_hit_rate)
            ),
            "best_helmet_score": (
                None if self.best_helmet_score is None else _diagnostic_round(self.best_helmet_score)
            ),
            "associations": [item.as_payload() for item in self.associations],
            "history": {
                "length": self.track_history_length,
                "limit": self.track_history_limit,
                "truncated": self.track_history_truncated,
            },
        }


@dataclass(frozen=True)
class HelmetDiagnosticEvent:
    """
    Immutable bounded episode/event summary for future diagnostic logging.
    """

    event: str
    reason: str
    frame_idx: int
    time_s: float
    episode_start_frame_idx: int
    episode_start_time_s: float
    source: Optional[str] = None
    camera_id: Optional[str] = None
    alert_uid: Optional[str] = None
    observations: Tuple[HelmetDiagnosticObservation, ...] = ()
    observation_limit: int = HELMET_DIAGNOSTIC_MAX_OBSERVATIONS
    observations_truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", _diagnostic_text(self.event, "event", max_length=64))
        object.__setattr__(self, "reason", _diagnostic_text(self.reason, "reason", max_length=128))
        object.__setattr__(self, "frame_idx", _diagnostic_int(self.frame_idx, "frame_idx"))
        object.__setattr__(
            self,
            "time_s",
            _diagnostic_float(self.time_s, "time_s", minimum=0.0),
        )
        object.__setattr__(
            self,
            "episode_start_frame_idx",
            _diagnostic_int(self.episode_start_frame_idx, "episode_start_frame_idx"),
        )
        object.__setattr__(
            self,
            "episode_start_time_s",
            _diagnostic_float(self.episode_start_time_s, "episode_start_time_s", minimum=0.0),
        )
        object.__setattr__(self, "source", _diagnostic_optional_source(self.source, "source"))
        object.__setattr__(self, "camera_id", _diagnostic_optional_source(self.camera_id, "camera_id"))
        if self.alert_uid is not None:
            object.__setattr__(self, "alert_uid", _diagnostic_text(self.alert_uid, "alert_uid"))

        observations = tuple(self.observations)
        if len(observations) > HELMET_DIAGNOSTIC_MAX_OBSERVATIONS:
            raise ValueError(
                "observations exceeds "
                f"{HELMET_DIAGNOSTIC_MAX_OBSERVATIONS} retained values"
            )
        if not all(isinstance(item, HelmetDiagnosticObservation) for item in observations):
            raise TypeError("observations must contain HelmetDiagnosticObservation values")
        observations = tuple(sorted(observations, key=lambda item: (item.frame_idx, item.time_s, item.diagnostic_track_id)))
        object.__setattr__(self, "observations", observations)
        object.__setattr__(
            self,
            "observation_limit",
            _diagnostic_int(self.observation_limit, "observation_limit", minimum=1),
        )
        if len(observations) > self.observation_limit:
            raise ValueError("observations cannot exceed observation_limit")
        object.__setattr__(
            self,
            "observations_truncated",
            _diagnostic_bool(self.observations_truncated, "observations_truncated"),
        )

    def as_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": HELMET_DIAGNOSTICS_SCHEMA_VERSION,
            "event": self.event,
            "reason": self.reason,
            "frame_idx": self.frame_idx,
            "time_s": _diagnostic_round(self.time_s),
            "episode_start_frame_idx": self.episode_start_frame_idx,
            "episode_start_time_s": _diagnostic_round(self.episode_start_time_s),
            "source": self.source,
            "camera_id": self.camera_id,
            "alert_uid": self.alert_uid,
            "observations": [item.as_payload() for item in self.observations],
            "history": {
                "retained_observation_count": len(self.observations),
                "observation_limit": self.observation_limit,
                "truncated": self.observations_truncated,
            },
        }


@dataclass
class _HelmetDiagnosticTrack:
    """Mutable state carried between frames for one temporary person track."""

    track_id: int
    bbox: Detection
    missed_frames: int = 0
    observation_count: int = 0
    hit_count: int = 0
    best_helmet_score: Optional[float] = None
    score_history: List[float] = field(default_factory=list)
    history_truncated: bool = False


class HelmetDiagnosticTracker:
    """
    Deterministic, bounded person tracking used only for diagnostics.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = HELMET_DIAGNOSTIC_IOU_THRESHOLD,
        max_missed_frames: int = HELMET_DIAGNOSTIC_MAX_MISSED_FRAMES,
        max_tracks: int = HELMET_DIAGNOSTIC_MAX_TRACKS,
        score_history_limit: int = HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY,
    ) -> None:
        self.iou_threshold = _diagnostic_float(
            iou_threshold,
            "diagnostic iou_threshold",
            minimum=0.05,
            maximum=0.95,
        )
        self.max_missed_frames = _diagnostic_int(
            max_missed_frames,
            "diagnostic max_missed_frames",
        )
        self.max_tracks = _diagnostic_int(max_tracks, "diagnostic max_tracks", minimum=1)
        self.score_history_limit = _diagnostic_int(
            score_history_limit,
            "diagnostic score_history_limit",
            minimum=1,
        )
        if self.score_history_limit > HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY:
            raise ValueError(
                "diagnostic score_history_limit must be <= "
                f"{HELMET_DIAGNOSTIC_MAX_SCORE_HISTORY}"
            )
        self._tracks: List[_HelmetDiagnosticTrack] = []
        self._next_track_id = 1

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Reset temporary IDs and state at a source/engine boundary."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        *,
        persons: Sequence[Detection],
        helmets: Sequence[Detection],
        frame_idx: int,
        time_s: float,
        frame_size: Tuple[int, int],
        head_top_fraction: float,
        verification_confidence: float,
    ) -> Tuple[HelmetDiagnosticObservation, ...]:
        """Match current people to temporary tracks and return observations.

        This method only consumes detections and produces diagnostics. It does
        not make or change any helmet-alert decision.
        """

        frame_idx = _diagnostic_int(frame_idx, "diagnostic frame_idx")
        time_s = _diagnostic_float(time_s, "diagnostic time_s", minimum=0.0)
        frame_width, frame_height = frame_size
        if isinstance(frame_width, bool) or isinstance(frame_height, bool):
            raise TypeError("diagnostic frame_size must contain integers")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("diagnostic frame_size must be positive")
        head_top_fraction = _diagnostic_float(
            head_top_fraction,
            "diagnostic head_top_fraction",
            minimum=0.05,
            maximum=0.8,
        )
        verification_confidence = _diagnostic_float(
            verification_confidence,
            "diagnostic verification_confidence",
            minimum=0.0,
            maximum=1.0,
        )

        ordered_persons = _diagnostic_ordered_persons(persons)
        if len(ordered_persons) > self.max_tracks:
            ordered_persons = ordered_persons[: self.max_tracks]

        assignments: Dict[int, _HelmetDiagnosticTrack] = {}
        matched_track_indices: set[int] = set()
        matched_person_indices: set[int] = set()
        matches: List[Tuple[float, int, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for person_index, person in enumerate(ordered_persons):
                iou = _detection_iou(track.bbox, person)
                if iou >= self.iou_threshold:
                    matches.append((iou, track.track_id, person_index, track_index))
        matches.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, _, person_index, track_index in matches:
            if track_index in matched_track_indices or person_index in matched_person_indices:
                continue
            track = self._tracks[track_index]
            track.bbox = ordered_persons[person_index]
            track.missed_frames = 0
            assignments[person_index] = track
            matched_track_indices.add(track_index)
            matched_person_indices.add(person_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_track_indices:
                track.missed_frames += 1
        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        for person_index, person in enumerate(ordered_persons):
            if person_index in matched_person_indices or len(self._tracks) >= self.max_tracks:
                continue
            track = _HelmetDiagnosticTrack(track_id=self._next_track_id, bbox=person)
            self._next_track_id += 1
            self._tracks.append(track)
            assignments[person_index] = track

        observations: List[HelmetDiagnosticObservation] = []
        for person_index, person in enumerate(ordered_persons):
            track = assignments.get(person_index)
            if track is None:
                continue
            all_associations = _helmet_diagnostic_associations(
                person,
                helmets,
                head_top_fraction=head_top_fraction,
            )
            associated_scores = tuple(
                association.helmet_score
                for association in all_associations
                if association.center_inside_person and association.center_inside_head
            )
            associations = _prioritize_diagnostic_associations(all_associations)
            frame_best_score = max(associated_scores) if associated_scores else None
            track.observation_count += 1
            if any(score >= verification_confidence for score in associated_scores):
                track.hit_count += 1
            if frame_best_score is not None:
                track.best_helmet_score = (
                    frame_best_score
                    if track.best_helmet_score is None
                    else max(track.best_helmet_score, frame_best_score)
                )
                if len(track.score_history) >= self.score_history_limit:
                    track.score_history.pop(0)
                    track.history_truncated = True
                track.score_history.append(frame_best_score)

            x1, y1, x2, y2 = (
                float(person.x1),
                float(person.y1),
                float(person.x2),
                float(person.y2),
            )
            touches_edge = x1 <= 0.0 or y1 <= 0.0 or x2 >= frame_width or y2 >= frame_height
            center_x = min(1.0, max(0.0, ((x1 + x2) * 0.5) / float(frame_width)))
            center_y = min(1.0, max(0.0, ((y1 + y2) * 0.5) / float(frame_height)))
            observations.append(
                HelmetDiagnosticObservation(
                    diagnostic_track_id=track.track_id,
                    frame_idx=frame_idx,
                    time_s=time_s,
                    person_box=(x1, y1, x2, y2),
                    person_height_px=max(0.0, y2 - y1),
                    normalized_position=(center_x, center_y),
                    at_frame_edge=touches_edge,
                    head_visible=not (y1 <= 0.0),
                    helmet_score_history=tuple(track.score_history),
                    helmet_hit_count=track.hit_count,
                    helmet_observation_count=track.observation_count,
                    helmet_hit_rate=track.hit_count / float(track.observation_count),
                    best_helmet_score=track.best_helmet_score,
                    associations=associations,
                    track_history_length=len(track.score_history),
                    track_history_limit=self.score_history_limit,
                    track_history_truncated=track.history_truncated,
                )
            )
        return tuple(sorted(observations, key=lambda item: item.diagnostic_track_id))


@dataclass(frozen=True)
class HelmetAlertConfig:
    required_seconds: float = DEFAULT_HELMET_REQUIRED_SECONDS
    strong_helmet_confidence: float = 0.35
    verification_confidence: float = DEFAULT_HELMET_ALERT_CONFIDENCE
    analysis_fps: float = 5.0
    recovery_seconds: float = 2.0
    absence_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    min_person_height_px: int = 120
    head_top_fraction: float = 0.35
    max_gap_frames: int = 1
    safety_area_id: str = "helmet_area_main"

    def __post_init__(self) -> None:
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be > 0")
        if self.required_seconds <= 0:
            raise ValueError("required_seconds must be > 0")
        if not (0.0 <= self.verification_confidence <= 1.0):
            raise ValueError("verification_confidence must be within [0, 1]")
        if not (0.0 <= self.strong_helmet_confidence <= 1.0):
            raise ValueError("strong_helmet_confidence must be within [0, 1]")
        if self.verification_confidence > self.strong_helmet_confidence:
            raise ValueError("verification_confidence must be <= strong_helmet_confidence")
        if self.recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        if self.absence_seconds <= 0:
            raise ValueError("absence_seconds must be > 0")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if self.min_person_height_px < 0:
            raise ValueError("min_person_height_px must be >= 0")
        if not (0.05 <= self.head_top_fraction <= 0.8):
            raise ValueError("head_top_fraction must be within [0.05, 0.8]")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be >= 0")

    @property
    def required_frames(self) -> int:
        return max(1, int(round(self.required_seconds * self.analysis_fps)))

    @property
    def recovery_frames(self) -> int:
        return max(1, int(round(self.recovery_seconds * self.analysis_fps)))

    @property
    def absence_frames(self) -> int:
        return max(1, int(round(self.absence_seconds * self.analysis_fps)))


@dataclass(frozen=True)
class HelmetAlertCandidate:
    box: Tuple[float, float, float, float]
    height_px: float
    score: float

    @classmethod
    def from_detection(cls, det: Detection) -> "HelmetAlertCandidate":
        return cls(
            box=(float(det.x1), float(det.y1), float(det.x2), float(det.y2)),
            height_px=max(0.0, float(det.y2) - float(det.y1)),
            score=float(det.score),
        )

    def as_payload(self) -> Dict[str, Any]:
        """return the detected helmet alert candiate"""
        return {
            "box": [round(v, 3) for v in self.box],
            "height_px" : round(float(self.height_px), 3),
            "score" : round(float(self.score), 3)
        }


@dataclass(frozen=True)
class HelmetAlert:

    """
    Appropriate data helmet class output
    from the candidates
    """

    alert_uid: str
    alert_type: str
    safety_profile: str
    start_time_s: float
    end_time_s: float
    trigger_frame_idx: int
    source: str
    camera_id: Optional[str]
    safety_area_id: str
    primary: HelmetAlertCandidate
    candidates: Tuple[HelmetAlertCandidate, ...]
    best_helmet_score: Optional[float] = None
    helmet_confidence_floor: float = DEFAULT_HELMET_ALERT_CONFIDENCE
    helmet_strong_confidence: float = 0.35
    related_session_uid: Optional[str] = None
    notes: Tuple[str, ...] = ()
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

    def to_payload(self, *, run_start_dt: Optional[datetime], fallback_date: str) -> Dict[str, Any]:
        start_iso = (
            self.start_datetime.isoformat(timespec="seconds")
            if self.start_datetime is not None
            else _iso_at(run_start_dt, self.start_time_s)
        )
        end_iso = (
            self.end_datetime.isoformat(timespec="seconds")
            if self.end_datetime is not None
            else _iso_at(run_start_dt, self.end_time_s)
        )
        start_date = start_iso[:10] if start_iso else fallback_date
        end_date = end_iso[:10] if end_iso else start_date
        return {
            "alert_uid": self.alert_uid,
            "alert_type": self.alert_type,
            "safety_profile": self.safety_profile,
            "start_time_s": float(self.start_time_s),
            "end_time_s": float(self.end_time_s),
            "start_time_iso": start_iso,
            "end_time_iso": end_iso,
            "start_date": start_date,
            "end_date": end_date,
            "camera_id": self.camera_id,
            "source": self.source,
            "status": ALERT_STATUS_PENDING,
            "machine_status": MACHINE_STATUS_NO_HELMET,
            "safety_area_id": self.safety_area_id,
            "person_box": [round(v, 3) for v in self.primary.box],
            "person_height_px": round(float(self.primary.height_px), 3),
            "person_count": int(len(self.candidates)),
            "candidates": [c.as_payload() for c in self.candidates],
            "best_helmet_score": (
                None if self.best_helmet_score is None else round(float(self.best_helmet_score), 3)
            ),
            "helmet_confidence_floor": round(float(self.helmet_confidence_floor), 3),
            "helmet_strong_confidence": round(float(self.helmet_strong_confidence), 3),
            "related_session_uid": self.related_session_uid,
            "thumbnail": "thumbnail.jpg",
            "artifacts": {
                "thumbnail": "thumbnail.jpg",
                "clip": None,
            },
            "trigger_frame_idx": int(self.trigger_frame_idx),
            "notes": list(self.notes),
        }


class HelmetAlertEngine:
    def __init__(
        self,
        cfg: HelmetAlertConfig,
        *,
        source: str,
        camera_id: Optional[str] = None,
        diagnostics_enabled: bool = False,
        diagnostic_tracker: Optional[HelmetDiagnosticTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.source = redact_source_credentials(str(source)) or "source"
        self.camera_id = redact_source_credentials(str(camera_id)) if camera_id else None
        if not isinstance(diagnostics_enabled, bool):
            raise TypeError("diagnostics_enabled must be a bool")
        if diagnostic_tracker is not None and not diagnostics_enabled:
            raise ValueError("diagnostic_tracker requires diagnostics_enabled=True")
        if diagnostics_enabled and diagnostic_tracker is None:
            diagnostic_tracker = HelmetDiagnosticTracker()
        self._diagnostic_tracker = diagnostic_tracker
        self._diagnostic_observations: List[HelmetDiagnosticObservation] = []
        self._diagnostic_error_count = 0
        self._active = False
        self._alert_emitted = False
        self._episode_start_time_s = 0.0
        self._episode_start_frame_idx = 0
        self._episode_start_datetime: Optional[datetime] = None
        self._no_helmet_frames = 0
        self._no_helmet_gap_frames = 0
        self._recovery_frames = 0
        self._absence_frames = 0
        self._cooldown_until_s = 0.0
        self._episode_best_helmet_score: Optional[float] = None

    @property
    def diagnostics_enabled(self) -> bool:
        return self._diagnostic_tracker is not None

    @property
    def diagnostic_error_count(self) -> int:
        return self._diagnostic_error_count

    def pop_diagnostic_observations(self) -> Tuple[HelmetDiagnosticObservation, ...]:
        observations = tuple(self._diagnostic_observations)
        self._diagnostic_observations.clear()
        return observations

    def update(
        self,
        *,
        time_s: float,
        frame_idx: int,
        persons: Sequence[Detection],
        helmets: Sequence[Detection],
        safety_roi: RoiPolygon,
        related_session_uid: Optional[str] = None,
        wall_dt: Optional[datetime] = None,
    ) -> Tuple[HelmetAlert, ...]:
        qualifying = _qualifying_persons(
            persons,
            safety_roi=safety_roi,
            min_person_height_px=self.cfg.min_person_height_px,
        )
        self._collect_diagnostic_observations(
            qualifying=qualifying,
            helmets=helmets,
            frame_idx=frame_idx,
            time_s=time_s,
            safety_roi=safety_roi,
        )
        associated_scores = {
            id(person): _associated_helmet_scores(
                person,
                helmets,
                head_top_fraction=self.cfg.head_top_fraction,
            )
            for person in qualifying
        }
        strong_helmeted_present = any(
            any(score >= self.cfg.strong_helmet_confidence for score in associated_scores[id(person)])
            for person in qualifying
        )
        candidate_persons = tuple(
            person
            for person in qualifying
            if not any(score >= self.cfg.strong_helmet_confidence for score in associated_scores[id(person)])
        )
        frame_helmet_scores = tuple(
            score for person in candidate_persons for score in associated_scores[id(person)]
        )
        frame_best_helmet_score = max(frame_helmet_scores) if frame_helmet_scores else None
        candidates = tuple(HelmetAlertCandidate.from_detection(person) for person in candidate_persons)
        # With whole-frame coverage, do not let a weak helmet on one person
        # suppress an alert for another person in the same frame.
        verification_helmet_present = bool(candidate_persons) and all(
            any(score >= self.cfg.verification_confidence for score in associated_scores[id(person)])
            for person in candidate_persons
        )

        if candidates:
            if not self._active:
                self._start_episode(time_s=float(time_s), frame_idx=int(frame_idx), wall_dt=wall_dt)
            if frame_best_helmet_score is not None:
                if self._episode_best_helmet_score is None:
                    self._episode_best_helmet_score = frame_best_helmet_score
                else:
                    self._episode_best_helmet_score = max(self._episode_best_helmet_score, frame_best_helmet_score)
            self._no_helmet_frames += 1
            self._no_helmet_gap_frames = 0
            self._recovery_frames = 0
            self._absence_frames = 0
            if (
                not self._alert_emitted
                and float(time_s) >= self._cooldown_until_s
                and self._no_helmet_frames >= self.cfg.required_frames
            ):
                # Weak helmet detections are deliberately ignored while the
                # episode is accumulating. At the alert boundary they get a
                # final verification chance before NO_HELMET is emitted.
                if verification_helmet_present:
                    self._close_episode(time_s=float(time_s))
                    return ()
                alert = self._build_alert(
                    time_s=float(time_s),
                    frame_idx=int(frame_idx),
                    candidates=candidates,
                    related_session_uid=related_session_uid,
                    wall_dt=wall_dt,
                )
                self._alert_emitted = True
                return (alert,)
            return ()

        if not self._active:
            return ()

        if strong_helmeted_present:
            self._recovery_frames += 1
            self._absence_frames = 0
            self._no_helmet_gap_frames = 0
        else:
            self._absence_frames += 1
            self._recovery_frames = 0
            self._no_helmet_gap_frames += 1
            if not self._alert_emitted and self._no_helmet_gap_frames > self.cfg.max_gap_frames:
                self._no_helmet_frames = 0

        if self._recovery_frames >= self.cfg.recovery_frames or self._absence_frames >= self.cfg.absence_frames:
            self._close_episode(time_s=float(time_s))

        return ()

    def _collect_diagnostic_observations(
        self,
        *,
        qualifying: Sequence[Detection],
        helmets: Sequence[Detection],
        frame_idx: int,
        time_s: float,
        safety_roi: RoiPolygon,
    ) -> None:
        tracker = self._diagnostic_tracker
        frame_size = safety_roi.frame_size
        if tracker is None or frame_size is None:
            return
        try:
            observations = tracker.update(
                persons=qualifying,
                helmets=helmets,
                frame_idx=int(frame_idx),
                time_s=float(time_s),
                frame_size=frame_size,
                head_top_fraction=self.cfg.head_top_fraction,
                verification_confidence=self.cfg.verification_confidence,
            )
        except Exception:
            # Diagnostic collection is a side channel. A malformed diagnostic
            # input must never change the alert decision path.
            self._diagnostic_error_count += 1
            return
        self._diagnostic_observations.extend(observations)
        overflow = len(self._diagnostic_observations) - HELMET_DIAGNOSTIC_MAX_PENDING_OBSERVATIONS
        if overflow > 0:
            del self._diagnostic_observations[:overflow]

    def flush(self, *, time_s: float) -> None:
        if self._active:
            self._close_episode(time_s=float(time_s))

    def _start_episode(self, *, time_s: float, frame_idx: int, wall_dt: Optional[datetime]) -> None:
        self._active = True
        self._alert_emitted = False
        self._episode_start_time_s = float(time_s)
        self._episode_start_frame_idx = int(frame_idx)
        self._episode_start_datetime = wall_dt
        self._no_helmet_frames = 0
        self._no_helmet_gap_frames = 0
        self._recovery_frames = 0
        self._absence_frames = 0
        self._episode_best_helmet_score = None

    def _close_episode(self, *, time_s: float) -> None:
        if self._alert_emitted:
            self._cooldown_until_s = float(time_s) + float(self.cfg.cooldown_seconds)
        self._active = False
        self._alert_emitted = False
        self._no_helmet_frames = 0
        self._no_helmet_gap_frames = 0
        self._recovery_frames = 0
        self._absence_frames = 0
        self._episode_start_datetime = None
        self._episode_best_helmet_score = None

    def _build_alert(
        self,
        *,
        time_s: float,
        frame_idx: int,
        candidates: Sequence[HelmetAlertCandidate],
        related_session_uid: Optional[str],
        wall_dt: Optional[datetime],
    ) -> HelmetAlert:
        sorted_candidates = tuple(sorted(candidates, key=lambda c: (c.height_px, c.score), reverse=True))
        primary = sorted_candidates[0]
        uid = make_alert_uid(
            alert_type=ALERT_TYPE_NO_HELMET,
            source=self.source,
            start_time_s=self._episode_start_time_s,
            start_frame_idx=self._episode_start_frame_idx,
        )
        return HelmetAlert(
            alert_uid=uid,
            alert_type=ALERT_TYPE_NO_HELMET,
            safety_profile=SAFETY_PROFILE_HELMET_ALERT_V1,
            start_time_s=self._episode_start_time_s,
            end_time_s=float(time_s),
            trigger_frame_idx=int(frame_idx),
            source=self.source,
            camera_id=self.camera_id,
            safety_area_id=self.cfg.safety_area_id,
            primary=primary,
            candidates=sorted_candidates,
            best_helmet_score=self._episode_best_helmet_score,
            helmet_confidence_floor=self.cfg.verification_confidence,
            helmet_strong_confidence=self.cfg.strong_helmet_confidence,
            related_session_uid=related_session_uid,
            notes=("sustained_no_helmet",),
            start_datetime=self._episode_start_datetime,
            end_datetime=wall_dt,
        )


def make_alert_uid(*, alert_type: str, source: str, start_time_s: float, start_frame_idx: int) -> str:
    safe_source = redact_source_credentials(str(source)) or "source"
    raw = f"{alert_type}|{safe_source}|{float(start_time_s):.3f}|{int(start_frame_idx)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"alert_{str(alert_type).lower()}_{int(start_frame_idx):06d}_{digest}"


def draw_helmet_alert_thumbnail(
    image_bgr: np.ndarray,
    *,
    safety_roi: RoiPolygon,
    candidates: Sequence[HelmetAlertCandidate],
    label: str = "NO HELMET",
) -> np.ndarray:
    if image_bgr is None or not hasattr(image_bgr, "shape"):
        raise TypeError("image_bgr must be a NumPy array.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {getattr(image_bgr, 'shape', None)}")

    out = draw_roi(image_bgr, safety_roi, color=(0, 255, 255))
    h, w = out.shape[:2]
    for idx, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.box
        x1i = int(np.clip(round(x1), 0, w - 1))
        y1i = int(np.clip(round(y1), 0, h - 1))
        x2i = int(np.clip(round(x2), 0, w - 1))
        y2i = int(np.clip(round(y2), 0, h - 1))
        color = (0, 0, 255) if idx == 0 else (0, 128, 255)
        thickness = 3 if idx == 0 else 2
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, thickness=thickness)
        text = label if idx == 0 else f"{label} #{idx + 1}"
        cv2.putText(
            out,
            text,
            (x1i, max(18, y1i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def write_helmet_alert_artifacts(
    *,
    out_dir: Path,
    date: str,
    alert: HelmetAlert,
    frame_bgr: np.ndarray,
    safety_roi: RoiPolygon,
    run_start_dt: Optional[datetime],
) -> Path:
    alert_dir = out_dir / "alerts" / date / alert.alert_uid
    alert_dir.mkdir(parents=True, exist_ok=True)
    thumb = draw_helmet_alert_thumbnail(frame_bgr, safety_roi=safety_roi, candidates=alert.candidates)
    thumb_path = alert_dir / "thumbnail.jpg"
    ok = cv2.imwrite(str(thumb_path), thumb)
    if not ok:
        raise RuntimeError(f"Failed to write helmet alert thumbnail: {thumb_path}")
    payload = alert.to_payload(run_start_dt=run_start_dt, fallback_date=date)
    (alert_dir / "alert.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return alert_dir


def _qualifying_persons(
    persons: Sequence[Detection],
    *,
    safety_roi: RoiPolygon,
    min_person_height_px: int,
) -> Tuple[Detection, ...]:
    out: List[Detection] = []
    """
    if the box of the person is biggger
    """
    for person in persons:
        height = max(0.0, float(person.y2) - float(person.y1))
        if height < float(min_person_height_px):
            continue
        if not _box_overlaps_roi(person, safety_roi):
            continue
        out.append(person)
    return tuple(out)


def _box_overlaps_roi(det: Detection, roi: RoiPolygon) -> bool:
    x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
    if x2 <= x1 or y2 <= y1:
        return False

    sample_points = (
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    )
    if any(roi.contains_point(px, py) for px, py in sample_points):
        return True

    for px, py in roi.points:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def _diagnostic_ordered_persons(persons: Sequence[Detection]) -> List[Detection]:
    indexed = list(enumerate(persons))
    indexed.sort(
        key=lambda item: (
            -max(0.0, float(item[1].y2) - float(item[1].y1)),
            -float(item[1].score),
            float(item[1].x1),
            float(item[1].y1),
            float(item[1].x2),
            float(item[1].y2),
            item[0],
        )
    )
    return [person for _, person in indexed]


def _detection_iou(a: Detection, b: Detection) -> float:
    x1 = max(float(a.x1), float(b.x1))
    y1 = max(float(a.y1), float(b.y1))
    x2 = min(float(a.x2), float(b.x2))
    y2 = min(float(a.y2), float(b.y2))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, float(a.x2) - float(a.x1)) * max(0.0, float(a.y2) - float(a.y1))
    area_b = max(0.0, float(b.x2) - float(b.x1)) * max(0.0, float(b.y2) - float(b.y1))

    denominator = area_a + area_b - intersection
    return 0.0 if denominator <= 0.0 else intersection / denominator


def _helmet_center_geometry(
    person: Detection,
    helmet: Detection,
    *,
    head_top_fraction: float,
) -> Tuple[float, float, bool, bool]:
    """Return the helmet center and whether it falls in the person's body/head regions."""

    head_y2 = float(person.y1) + (float(person.y2) - float(person.y1)) * float(head_top_fraction)
    # The point under test is the helmet box center; the person box supplies
    # only the enclosing body and head-region boundaries.
    center_x = (float(helmet.x1) + float(helmet.x2)) * 0.5
    center_y = (float(helmet.y1) + float(helmet.y2)) * 0.5
    inside_person = (
        float(person.x1) <= center_x <= float(person.x2)
        and float(person.y1) <= center_y <= float(person.y2)
    )
    inside_head = (
        float(person.x1) <= center_x <= float(person.x2)
        and float(person.y1) <= center_y <= head_y2
    )
    return center_x, center_y, inside_person, inside_head


def _helmet_diagnostic_associations(
    person: Detection,
    helmets: Sequence[Detection],
    *,
    head_top_fraction: float,
) -> Tuple[HelmetDiagnosticAssociation, ...]:
    associations: List[HelmetDiagnosticAssociation] = []
    person_center_x = (float(person.x1) + float(person.x2)) * 0.5
    person_center_y = (float(person.y1) + float(person.y2)) * 0.5
    for helmet in helmets:
        center_x, center_y, inside_person, inside_head = _helmet_center_geometry(
            person,
            helmet,
            head_top_fraction=head_top_fraction,
        )
        distance = math.hypot(center_x - person_center_x, center_y - person_center_y)
        associations.append(
            HelmetDiagnosticAssociation(
                helmet_box=(
                    float(helmet.x1),
                    float(helmet.y1),
                    float(helmet.x2),
                    float(helmet.y2),
                ),
                helmet_score=float(helmet.score),
                center_inside_person=inside_person,
                center_inside_head=inside_head,
                center_distance_to_person_px=distance,
            )
        )
    associations.sort(
        key=lambda item: (
            -item.helmet_score,
            item.helmet_box,
            item.center_inside_person,
            item.center_inside_head,
        )
    )
    return tuple(associations)


def _prioritize_diagnostic_associations(
    associations: Sequence[HelmetDiagnosticAssociation],
) -> Tuple[HelmetDiagnosticAssociation, ...]:
    """Retain head-associated evidence before unrelated high-score boxes."""
    associated = [
        item
        for item in associations
        if item.center_inside_person and item.center_inside_head
    ]
    unrelated = [item for item in associations if item not in associated]
    return tuple((associated + unrelated)[:HELMET_DIAGNOSTIC_MAX_ASSOCIATIONS])


def _helmet_associated_with_person(
    person: Detection,
    helmets: Sequence[Detection],
    *,
    head_top_fraction: float,
) -> bool:
    return bool(_associated_helmet_scores(person, helmets, head_top_fraction=head_top_fraction))


def _associated_helmet_scores(
    person: Detection,
    helmets: Sequence[Detection],
    *,
    head_top_fraction: float,
) -> Tuple[float, ...]:
    scores: List[float] = []
    for helmet in helmets:
        _, _, _, inside_head = _helmet_center_geometry(
            person,
            helmet,
            head_top_fraction=head_top_fraction,
        )
        if inside_head:
            scores.append(float(helmet.score))

    return tuple(scores)


def _iso_at(run_start_dt: Optional[datetime], offset_s: float) -> Optional[str]:
    if run_start_dt is None:
        return None
    return (run_start_dt + timedelta(seconds=float(offset_s))).isoformat(timespec="seconds")

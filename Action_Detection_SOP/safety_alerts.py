from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Action_Detection_SOP.roi import RoiPolygon, draw_roi
from yolo_kit.types import Detection


ALERT_TYPE_NO_HELMET = "NO_HELMET"
SAFETY_PROFILE_HELMET_ALERT_V1 = "helmet_alert_v1"
ALERT_STATUS_PENDING = "PENDING"
MACHINE_STATUS_NO_HELMET = "NO_HELMET"

"""
---------------------------
DATA SCHEMA
---------------------------
"""

@dataclass(frozen=True)
class HelmetAlertConfig:
    required_seconds: float = 5.0
    analysis_fps: float = 5.0
    recovery_seconds: float = 2.0
    absence_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    min_person_height_px: int = 120
    head_top_fraction: float = 0.35
    max_gap_frames: int = 1
    safety_area_id: str = "helmet_area_main"

    def __post_init__(self) -> None:
        #just basics check to make sure the values that is supplid is in the appropriate range
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be > 0")
        if self.required_seconds <= 0:
            raise ValueError("required_seconds must be > 0")
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
    related_session_uid: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def to_payload(self, *, run_start_dt: Optional[datetime], fallback_date: str) -> Dict[str, Any]:
        start_iso = _iso_at(run_start_dt, self.start_time_s)
        end_iso = _iso_at(run_start_dt, self.end_time_s)
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
    def __init__(self, cfg: HelmetAlertConfig, *, source: str, camera_id: Optional[str] = None) -> None:
        self.cfg = cfg
        self.source = str(source)
        self.camera_id = str(camera_id) if camera_id else None
        self._active = False
        self._alert_emitted = False
        self._episode_start_time_s = 0.0
        self._episode_start_frame_idx = 0
        self._no_helmet_frames = 0
        self._no_helmet_gap_frames = 0
        self._recovery_frames = 0
        self._absence_frames = 0
        self._cooldown_until_s = 0.0

    def update(
        self,
        *,
        time_s: float,
        frame_idx: int,
        persons: Sequence[Detection],
        helmets: Sequence[Detection],
        safety_roi: RoiPolygon,
        related_session_uid: Optional[str] = None,
    ) -> Tuple[HelmetAlert, ...]:
        qualifying = _qualifying_persons(
            persons,
            safety_roi=safety_roi,
            min_person_height_px=self.cfg.min_person_height_px,
        )
        candidates = tuple(
            HelmetAlertCandidate.from_detection(p)
            for p in qualifying
            if not _helmet_associated_with_person(p, helmets, head_top_fraction=self.cfg.head_top_fraction)
        )
        helmeted_present = any(
            _helmet_associated_with_person(p, helmets, head_top_fraction=self.cfg.head_top_fraction)
            for p in qualifying
        )

        if candidates:
            if not self._active:
                self._start_episode(time_s=float(time_s), frame_idx=int(frame_idx))
            self._no_helmet_frames += 1
            self._no_helmet_gap_frames = 0
            self._recovery_frames = 0
            self._absence_frames = 0
            if (
                not self._alert_emitted
                and float(time_s) >= self._cooldown_until_s
                and self._no_helmet_frames >= self.cfg.required_frames
            ):
                alert = self._build_alert(
                    time_s=float(time_s),
                    frame_idx=int(frame_idx),
                    candidates=candidates,
                    related_session_uid=related_session_uid,
                )
                self._alert_emitted = True
                return (alert,)
            return ()

        if not self._active:
            return ()

        if helmeted_present:
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

    def flush(self, *, time_s: float) -> None:
        if self._active:
            self._close_episode(time_s=float(time_s))

    def _start_episode(self, *, time_s: float, frame_idx: int) -> None:
        self._active = True
        self._alert_emitted = False
        self._episode_start_time_s = float(time_s)
        self._episode_start_frame_idx = int(frame_idx)
        self._no_helmet_frames = 0
        self._no_helmet_gap_frames = 0
        self._recovery_frames = 0
        self._absence_frames = 0

    def _close_episode(self, *, time_s: float) -> None:
        if self._alert_emitted:
            self._cooldown_until_s = float(time_s) + float(self.cfg.cooldown_seconds)
        self._active = False
        self._alert_emitted = False
        self._no_helmet_frames = 0
        self._no_helmet_gap_frames = 0
        self._recovery_frames = 0
        self._absence_frames = 0

    def _build_alert(
        self,
        *,
        time_s: float,
        frame_idx: int,
        candidates: Sequence[HelmetAlertCandidate],
        related_session_uid: Optional[str],
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
            related_session_uid=related_session_uid,
            notes=("sustained_no_helmet",),
        )


def make_alert_uid(*, alert_type: str, source: str, start_time_s: float, start_frame_idx: int) -> str:
    source_slug = _slugify(source)[:32] or "source"
    raw = f"{alert_type}|{source}|{float(start_time_s):.3f}|{int(start_frame_idx)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"alert_{str(alert_type).lower()}_{source_slug}_{int(start_frame_idx):06d}_{digest}"


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


def _helmet_associated_with_person(
    person: Detection,
    helmets: Sequence[Detection],
    *,
    head_top_fraction: float,
) -> bool:
    head_y2 = float(person.y1) + (float(person.y2) - float(person.y1)) * float(head_top_fraction)
    for helmet in helmets:
        cx = (float(helmet.x1) + float(helmet.x2)) * 0.5
        cy = (float(helmet.y1) + float(helmet.y2)) * 0.5
        if float(person.x1) <= cx <= float(person.x2) and float(person.y1) <= cy <= head_y2:
            return True
    return False


def _iso_at(run_start_dt: Optional[datetime], offset_s: float) -> Optional[str]:
    if run_start_dt is None:
        return None
    return (run_start_dt + timedelta(seconds=float(offset_s))).isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return out or "source"

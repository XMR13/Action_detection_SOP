"""
Preliminary skeleton repo untuk engine inti SOP nya, 
berfungsi sebagai engine untuk proses SOP secara lengkapnya

Note: Masih dalam tahap alpha (0.1), subject to changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from yolo_kit.types import Detection


class StepStatus(str, Enum):
    DONE = "DONE"
    NOT_DONE = "NOT_DONE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EvidenceEvent:
    name: str
    time_s: float
    frame_idx: int
    session_id: str


@dataclass(frozen=True)
class SessionizationConfig:
    start_seconds: float = 2.0
    end_seconds: float = 3.0
    analysis_fps: float = 5.0

    def __post_init__(self) -> None:
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be > 0")
        if self.start_seconds <= 0:
            raise ValueError("start_seconds must be > 0")
        if self.end_seconds <= 0:
            raise ValueError("end_seconds must be > 0")

    @property
    def start_frames(self) -> int:
        return max(1, int(round(self.start_seconds * self.analysis_fps)))

    @property
    def end_frames(self) -> int:
        return max(1, int(round(self.end_seconds * self.analysis_fps)))


@dataclass(frozen=True)
class HelmetRuleConfig:
    required_seconds: float = 2.0
    analysis_fps: float = 5.0
    head_top_fraction: float = 0.35
    min_person_height_px: int = 0
    # jika session length lebih pendek daripada evidence, default kembali ke UNKNOWN
    short_session_is_unknown: bool = True
    max_gap_frames: int = 1

    def __post_init__(self) -> None:
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be > 0")
        if self.required_seconds <= 0:
            raise ValueError("required_seconds must be > 0")
        if not (0.05 <= self.head_top_fraction <= 0.8):
            raise ValueError("head_top_fraction must be within [0.05, 0.8]")
        if self.min_person_height_px < 0:
            raise ValueError("min_person_height_px must be >= 0")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be >= 0")

    @property
    def required_frames(self) -> int:
        return max(1, int(round(self.required_seconds * self.analysis_fps)))


@dataclass(frozen=True)
class RoiDwellRuleConfig:

    """
    Important dataclas variable explanation
        required_seconds : the time needed to start the session
        iou_match_threshold : the IOU for the ROI and the person box
    """

    required_seconds: float = 5.0
    analysis_fps: float = 5.0
    max_gap_frames: int = 2
    max_track_missed: int = 5
    iou_match_threshold: float = 0.25
    min_person_height_px: int = 0
    # jika session length lebih pendek daripada evidence, default kembali ke UNKNOWN
    short_session_is_unknown: bool = True

    def __post_init__(self) -> None:
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be > 0")
        if self.required_seconds <= 0:
            raise ValueError("required_seconds must be > 0")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be >= 0")
        if self.max_track_missed < 0:
            raise ValueError("max_track_missed must be >= 0")
        if not (0.05 <= self.iou_match_threshold <= 0.95):
            raise ValueError("iou_match_threshold must be within [0.05, 0.95]")
        if self.min_person_height_px < 0:
            raise ValueError("min_person_height_px must be >= 0")
        if self.max_track_missed < self.max_gap_frames:
            raise ValueError("max_track_missed must be >= max_gap_frames")

    @property
    def required_frames(self) -> int:
        return max(1, int(round(self.required_seconds * self.analysis_fps)))


@dataclass(frozen=True)
class SopEngineConfig:
    session: SessionizationConfig = SessionizationConfig()
    helmet: Optional[HelmetRuleConfig] = HelmetRuleConfig()
    roi_dwell: Optional[RoiDwellRuleConfig] = RoiDwellRuleConfig()
    helmet_disabled_note: str = "helmet_check_disabled"
    roi_dwell_disabled_note: str = "roi_dwell_disabled"

    def __post_init__(self) -> None:
        # Jika kedua konfig sesuai dengan fps analysis nya.
        if self.helmet is None:
            pass
        elif abs(self.session.analysis_fps - self.helmet.analysis_fps) > 1e-6:
            raise ValueError("SessionizationConfig.analysis_fps must match HelmetRuleConfig.analysis_fps")
        if self.roi_dwell is None:
            return
        if abs(self.session.analysis_fps - self.roi_dwell.analysis_fps) > 1e-6:
            raise ValueError("SessionizationConfig.analysis_fps must match RoiDwellRuleConfig.analysis_fps")


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    start_time_s: float
    end_time_s: float
    operator_present: StepStatus
    roi_dwell: StepStatus
    helmet: StepStatus
    total_frames: int
    roi_dwell_max_frames: int
    helmet_positive_frames: int
    start_time_iso: Optional[str] = None
    end_time_iso: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    notes: Tuple[str, ...] = ()


@dataclass
class _EvidenceCounter:
    required_frames: int
    max_gap_frames: int = 0
    _positive_streak: int = 0
    _gap_streak: int = 0
    achieved: bool = False

    def update(self, is_positive: bool) -> bool:
        if self.achieved:
            return True
        if is_positive:
            self._positive_streak += 1
            self._gap_streak = 0
        else:
            if self._positive_streak == 0:
                return False
            self._gap_streak += 1
            if self._gap_streak > self.max_gap_frames:
                self._positive_streak = 0
                self._gap_streak = 0

        if self._positive_streak >= self.required_frames:
            self.achieved = True
        return self.achieved


@dataclass
class _PresenceSessionizer:
    start_frames: int
    end_frames: int
    active: bool = False
    _present_streak: int = 0
    _absent_streak: int = 0

    def update(self, present_now: bool) -> Optional[str]:
        if present_now:
            self._present_streak += 1
            self._absent_streak = 0
        else:
            self._absent_streak += 1
            self._present_streak = 0

        if not self.active and self._present_streak >= self.start_frames:
            self.active = True
            self._absent_streak = 0
            return "start"

        if self.active and self._absent_streak >= self.end_frames:
            self.active = False
            self._present_streak = 0
            return "end"

        return None


def _box_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a.x2 - a.x1)) * max(0.0, (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1)) * max(0.0, (b.y2 - b.y1))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def helmet_associated_with_person(
    persons: Sequence[Detection],
    helmets: Sequence[Detection],
    *,
    head_top_fraction: float,
) -> bool:
    for p in persons:
        head_y2 = p.y1 + (p.y2 - p.y1) * head_top_fraction
        for h in helmets:
            cx = (h.x1 + h.x2) * 0.5
            cy = (h.y1 + h.y2) * 0.5
            if p.x1 <= cx <= p.x2 and p.y1 <= cy <= head_y2:
                return True
    return False


def _max_person_height_px(persons: Sequence[Detection]) -> float:
    if not persons:
        return 0.0
    return max((p.y2 - p.y1) for p in persons)


@dataclass
class _Tracklet:
    track_id: int
    bbox: Detection
    dwell_frames: int = 0
    gap_frames: int = 0
    missed_frames: int = 0


@dataclass
class _RoiDwellTracker:
    max_gap_frames: int
    max_track_missed: int
    iou_match_threshold: float
    min_person_height_px: int = 0
    _tracks: List[_Tracklet] = field(default_factory=list)
    _next_id: int = 1

    def update(self, persons_in_roi: Sequence[Detection]) -> int:
        dets = [
            d for d in persons_in_roi if (d.y2 - d.y1) >= float(self.min_person_height_px)
        ]
        if not dets and not self._tracks:
            return 0

        matches: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(dets):
                iou = _box_iou(t.bbox, d)
                if iou >= self.iou_match_threshold:
                    matches.append((iou, ti, di))
        matches.sort(reverse=True, key=lambda x: x[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, ti, di in matches:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            track = self._tracks[ti]
            track.bbox = dets[di]
            track.dwell_frames += 1
            track.gap_frames = 0
            track.missed_frames = 0

        for ti, track in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            track.missed_frames += 1
            track.gap_frames += 1
            if track.gap_frames > self.max_gap_frames:
                track.dwell_frames = 0

        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            self._tracks.append(
                _Tracklet(track_id=self._next_id, bbox=det, dwell_frames=1, gap_frames=0, missed_frames=0)
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.missed_frames <= self.max_track_missed]
        if not self._tracks:
            return 0
        return max(t.dwell_frames for t in self._tracks)

@dataclass
class _ActiveSession:
    session_id: str
    start_time_s: float
    start_frame_idx: int
    total_frames: int = 0
    roi_dwell_max_frames: int = 0
    helmet_positive_frames: int = 0
    roi_dwell_tracker: Optional[_RoiDwellTracker] = None
    helmet_evidence: Optional[_EvidenceCounter] = None
    max_person_height_px: float = 0.0
    max_roi_person_height_px: float = 0.0
    roi_dwell_done: bool = False
    helmet_done: bool = False
    notes: List[str] = field(default_factory=list)


class SopEngine:
    """
    MVP (Minimum Viable Product untuk) untuk engine SOP
    - Sessionization: operator per sesi didalam ROI  (based on person presence)
    - Steps/langkah:
      - operator_present: Selesai untuk setiap sesi yang terlewat
      - roi_dwell: DONE/NOT_DONE/UNKNOWN Apakah operator cukup lama di ROI
      - helmet: DONE/NOT_DONE/UNKNOWN Apakah helm tersebut ada (global, tidak dibatasi ROI)
    """

    def __init__(self, cfg: SopEngineConfig):
        self.cfg = cfg
        self._sessionizer = _PresenceSessionizer(
            start_frames=cfg.session.start_frames,
            end_frames=cfg.session.end_frames,
        )
        self._active: Optional[_ActiveSession] = None
        self._session_counter = 0
        self._events: List[EvidenceEvent] = []

    @property
    def active_session_id(self) -> Optional[str]:
        return None if self._active is None else self._active.session_id

    @property
    def active_roi_dwell_frames(self) -> int:
        if self._active is None:
            return 0
        return int(self._active.roi_dwell_max_frames)

    def pop_events(self) -> Tuple[EvidenceEvent, ...]:
        if not self._events:
            return ()
        events = tuple(self._events)
        self._events.clear()
        return events

    def update(
        self,
        *,
        time_s: float,
        frame_idx: int,
        persons_in_roi: Sequence[Detection],
        persons_all: Sequence[Detection],
        helmets_all: Sequence[Detection],
    ) -> Optional[SessionResult]:
        if time_s < 0:
            raise ValueError("time_s must be >= 0")
        if frame_idx < 0:
            raise ValueError("frame_idx must be >= 0")

        person_present = len(persons_in_roi) > 0
        event = self._sessionizer.update(person_present)

        if event == "start":
            self._session_counter += 1
            session_id = f"{self._session_counter:06d}"
            helmet_counter: Optional[_EvidenceCounter] = None
            if self.cfg.helmet is not None:
                helmet_counter = _EvidenceCounter(
                    required_frames=self.cfg.helmet.required_frames,
                    max_gap_frames=self.cfg.helmet.max_gap_frames,
                )
            roi_tracker: Optional[_RoiDwellTracker] = None
            if self.cfg.roi_dwell is not None:
                roi_tracker = _RoiDwellTracker(
                    max_gap_frames=self.cfg.roi_dwell.max_gap_frames,
                    max_track_missed=self.cfg.roi_dwell.max_track_missed,
                    iou_match_threshold=self.cfg.roi_dwell.iou_match_threshold,
                    min_person_height_px=self.cfg.roi_dwell.min_person_height_px,
                )
            self._active = _ActiveSession(
                session_id=session_id,
                start_time_s=time_s,
                start_frame_idx=frame_idx,
                helmet_evidence=helmet_counter,
                roi_dwell_tracker=roi_tracker,
            )
            if self.cfg.helmet is None:
                self._active.notes.append(self.cfg.helmet_disabled_note)
            if self.cfg.roi_dwell is None:
                self._active.notes.append(self.cfg.roi_dwell_disabled_note)

        if self._active is not None and self._sessionizer.active:
            self._active.total_frames += 1
            self._active.max_person_height_px = max(
                self._active.max_person_height_px, _max_person_height_px(persons_all)
            )
            self._active.max_roi_person_height_px = max(
                self._active.max_roi_person_height_px, _max_person_height_px(persons_in_roi)
            )

            if self.cfg.roi_dwell is not None and self._active.roi_dwell_tracker is not None:
                current_max = self._active.roi_dwell_tracker.update(persons_in_roi)
                if current_max > self._active.roi_dwell_max_frames:
                    self._active.roi_dwell_max_frames = current_max
                if not self._active.roi_dwell_done:
                    required = self.cfg.roi_dwell.required_frames
                    if required > 0 and self._active.roi_dwell_max_frames >= required:
                        self._active.roi_dwell_done = True
                        self._events.append(
                            EvidenceEvent(
                                name="roi_dwell_done",
                                time_s=float(time_s),
                                frame_idx=int(frame_idx),
                                session_id=self._active.session_id,
                            )
                        )

            if self.cfg.helmet is not None and self._active.helmet_evidence is not None:
                helmet_ok = helmet_associated_with_person(
                    persons_all,
                    helmets_all,
                    head_top_fraction=self.cfg.helmet.head_top_fraction,
                )
                if helmet_ok:
                    self._active.helmet_positive_frames += 1
                self._active.helmet_evidence.update(helmet_ok)
                if not self._active.helmet_done and self._active.helmet_evidence.achieved:
                    self._active.helmet_done = True
                    self._events.append(
                        EvidenceEvent(
                            name="helmet_done",
                            time_s=float(time_s),
                            frame_idx=int(frame_idx),
                            session_id=self._active.session_id,
                        )
                    )

        if event == "end":
            if self._active is None:
                return None
            res = self._finalize_session(end_time_s=time_s)
            self._active = None
            return res

        return None

    def flush(self, *, time_s: float) -> Optional[SessionResult]:
        """
        Force-close an active session (e.g., end-of-stream).
        """
        if self._active is None:
            return None
        res = self._finalize_session(end_time_s=time_s)
        self._active = None
        self._sessionizer.active = False
        return res

    def _finalize_session(self, *, end_time_s: float) -> SessionResult:
        assert self._active is not None

        notes: list[str] = list(self._active.notes)

        operator_status = StepStatus.DONE
        roi_status = StepStatus.UNKNOWN

        total = self._active.total_frames
        if self.cfg.roi_dwell is None:
            roi_status = StepStatus.UNKNOWN
        else:
            required = self.cfg.roi_dwell.required_frames
            achieved = self._active.roi_dwell_max_frames >= required
            if achieved:
                roi_status = StepStatus.DONE
            else:
                if self.cfg.roi_dwell.short_session_is_unknown and total < required:
                    roi_status = StepStatus.UNKNOWN
                    notes.append("session_too_short_for_roi_dwell_decision")
                elif (
                    self.cfg.roi_dwell.min_person_height_px
                    and self._active.max_roi_person_height_px < self.cfg.roi_dwell.min_person_height_px
                ):
                    roi_status = StepStatus.UNKNOWN
                    notes.append("roi_person_too_small_for_reliable_dwell")
                else:
                    roi_status = StepStatus.NOT_DONE

        if self.cfg.helmet is None:
            helmet_status = StepStatus.UNKNOWN
        else:
            required = self.cfg.helmet.required_frames
            achieved = bool(self._active.helmet_evidence is not None and self._active.helmet_evidence.achieved)
            if achieved:
                helmet_status = StepStatus.DONE
            else:
                if self.cfg.helmet.short_session_is_unknown and total < required:
                    helmet_status = StepStatus.UNKNOWN
                    notes.append("session_too_short_for_helmet_decision")
                elif self.cfg.helmet.min_person_height_px and self._active.max_person_height_px < self.cfg.helmet.min_person_height_px:
                    helmet_status = StepStatus.UNKNOWN
                    notes.append("person_too_small_for_reliable_helmet")
                else:
                    helmet_status = StepStatus.NOT_DONE

        return SessionResult(
            session_id=self._active.session_id,
            start_time_s=self._active.start_time_s,
            end_time_s=end_time_s,
            operator_present=operator_status,
            roi_dwell=roi_status,
            helmet=helmet_status,
            total_frames=total,
            roi_dwell_max_frames=self._active.roi_dwell_max_frames,
            helmet_positive_frames=self._active.helmet_positive_frames,
            notes=tuple(notes),
        )


def iter_status_counts(results: Iterable[SessionResult]) -> Tuple[int, int, int]:
    done = 0
    not_done = 0
    unknown = 0
    for r in results:
        if r.helmet == StepStatus.DONE:
            done += 1
        elif r.helmet == StepStatus.NOT_DONE:
            not_done += 1
        else:
            unknown += 1
    return done, not_done, unknown


def iter_roi_status_counts(results: Iterable[SessionResult]) -> Tuple[int, int, int]:
    done = 0
    not_done = 0
    unknown = 0
    for r in results:
        if r.roi_dwell == StepStatus.DONE:
            done += 1
        elif r.roi_dwell == StepStatus.NOT_DONE:
            not_done += 1
        else:
            unknown += 1
    return done, not_done, unknown

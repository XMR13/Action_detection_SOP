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
class SopEngineConfig:
    session: SessionizationConfig = SessionizationConfig()
    helmet: Optional[HelmetRuleConfig] = HelmetRuleConfig()
    helmet_disabled_note: str = "helmet_check_disabled"

    def __post_init__(self) -> None:
        # Jika kedua konfig sesuai dengan fps analysis nya.
        if self.helmet is None:
            return
        if abs(self.session.analysis_fps - self.helmet.analysis_fps) > 1e-6:
            raise ValueError("SessionizationConfig.analysis_fps must match HelmetRuleConfig.analysis_fps")


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    start_time_s: float
    end_time_s: float
    operator_present: StepStatus
    helmet: StepStatus
    total_frames: int
    helmet_positive_frames: int
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
class _ActiveSession:
    session_id: str
    start_time_s: float
    start_frame_idx: int
    total_frames: int = 0
    helmet_positive_frames: int = 0
    helmet_evidence: Optional[_EvidenceCounter] = None
    max_person_height_px: float = 0.0
    notes: List[str] = field(default_factory=list)


class SopEngine:
    """
    MVP (Minimum Viable Product untuk) untuk engine SOP
    - Sessionization: operator per sesi didalam ROI  (based on person presence)
    - Steps/langkah:
      - operator_present: Selesai untuk setiap sesi yang terlewat
      - helmet: DONE/NOT_DONE/UNKNOWN Apakah helm tersebut ada
    """

    def __init__(self, cfg: SopEngineConfig):
        self.cfg = cfg
        self._sessionizer = _PresenceSessionizer(
            start_frames=cfg.session.start_frames,
            end_frames=cfg.session.end_frames,
        )
        self._active: Optional[_ActiveSession] = None
        self._session_counter = 0

    @property
    def active_session_id(self) -> Optional[str]:
        return None if self._active is None else self._active.session_id

    def update(
        self,
        *,
        time_s: float,
        frame_idx: int,
        persons: Sequence[Detection],
        helmets: Sequence[Detection],
    ) -> Optional[SessionResult]:
        if time_s < 0:
            raise ValueError("time_s must be >= 0")
        if frame_idx < 0:
            raise ValueError("frame_idx must be >= 0")

        person_present = len(persons) > 0
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
            self._active = _ActiveSession(
                session_id=session_id,
                start_time_s=time_s,
                start_frame_idx=frame_idx,
                helmet_evidence=helmet_counter,
            )
            if self.cfg.helmet is None:
                self._active.notes.append(self.cfg.helmet_disabled_note)

        if self._active is not None and self._sessionizer.active:
            self._active.total_frames += 1
            self._active.max_person_height_px = max(self._active.max_person_height_px, _max_person_height_px(persons))

            if self.cfg.helmet is not None and self._active.helmet_evidence is not None:
                helmet_ok = helmet_associated_with_person(
                    persons,
                    helmets,
                    head_top_fraction=self.cfg.helmet.head_top_fraction,
                )
                if helmet_ok:
                    self._active.helmet_positive_frames += 1
                self._active.helmet_evidence.update(helmet_ok)

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

        total = self._active.total_frames
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
            helmet=helmet_status,
            total_frames=total,
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

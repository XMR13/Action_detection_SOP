from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from yolo_kit.types import Detection

from .session import RollSessionConfig, RollSessionizer
from .sop_engine import EvidenceEvent, StepStatus


class RollComplianceStatus(str, Enum):
    COMPLIANT = "SESUAI SOP"
    NON_COMPLIANT = "TIDAK SESUAI SOP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RollEvidenceRuleConfig:
    "This is the config needed for both the cleaning and the labeling session"
    required_seconds: float = 1.0
    analysis_fps: float = 5.0
    max_gap_frames: int = 1
    min_tool_coverage: float = 0.35
    min_iou: float = 0.05
    short_session_is_unknown: bool = True

    def __post_init__(self) -> None:
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must have the value of  >-0")
        if self.required_seconds <= 0:
            raise ValueError("required_seconds must be > 0")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be >= 0")
        if not (0.0 <= self.min_tool_coverage <= 1.0):
            raise ValueError("min_tool_coverage must be within [0, 1]")
        if not (0.0 <= self.min_iou <= 1.0):
            raise ValueError("min_iou must be within [0, 1]")

    @property
    def required_frames(self) -> int:
        return max(1, int(round(self.required_seconds * self.analysis_fps)))


@dataclass(frozen=True)
class RollSopEngineConfig:
    session: RollSessionConfig = RollSessionConfig()
    cleaning: RollEvidenceRuleConfig = RollEvidenceRuleConfig()
    labeling: RollEvidenceRuleConfig = RollEvidenceRuleConfig()
    roll_class_id: int = 2
    cleaning_cloth_class_id: int = 3
    label_class_id: int = 4
    sop_profile: str = "roll_sop_v1"

    def __post_init__(self) -> None:
        fps = self.session.analysis_fps
        if abs(fps - self.cleaning.analysis_fps) > 1e-6:
            raise ValueError("RollSessionConfig.analysis_fps must match cleaning.analysis_fps")
        if abs(fps - self.labeling.analysis_fps) > 1e-6:
            raise ValueError("RollSessionConfig.analysis_fps must match labeling.analysis_fps")


@dataclass(frozen=True)
class RollSessionResult:
    """output data dari session rol setelah proses"""
    session_id: str
    sop_profile: str
    start_time: float
    cleaned: StepStatus
    labeled: StepStatus
    overall_status: RollComplianceStatus
    total_frames: int
    roll_present_frames: int
    cleaning_positive_frames: int
    labeling_positive_frames: int
    notes: Tuple[str, ...] = ()



@dataclass
class _TemporalEvidence:
    required_frames: int
    max_gap_frames: int
    positive_frames: int = 0
    achieved: bool = False
    _positive_streak: int = 0
    _gap_streak: int = 0

    def update(self, positive: bool) -> bool:
        if positive:
            self.positive_frames += 1

        if self.achieved:
            return True

        if positive:
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
class _ActiveRollSession:
    session_id: str
    start_time_s: float
    start_frame_idx: int
    total_frames: int = 0
    roll_present_frames: int = 0
    cleaning: _TemporalEvidence = field(default_factory=lambda: _TemporalEvidence(1, 0))
    labeling: _TemporalEvidence = field(default_factory=lambda: _TemporalEvidence(1, 0))
    cleaning_done: bool = False
    labeling_done: bool = False
    notes: List[str] = field(default_factory=list)


def _area(d: Detection) -> float:
    return max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)


def _intersection_area(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: Detection, b: Detection) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    denom = _area(a) + _area(b) - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _center_inside(inner: Detection, outer: Detection) -> bool:
    cx = (inner.x1 + inner.x2) * 0.5
    cy = (inner.y1 + inner.y2) * 0.5
    return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2


def detection_overlaps_roll(
    detection: Detection,
    roll: Detection,
    *,
    min_tool_coverage: float,
    min_iou: float,
) -> bool:
    tool_area = _area(detection)
    if tool_area <= 0:
        return False
    inter = _intersection_area(detection, roll)
    coverage = inter / tool_area
    return _center_inside(detection, roll) or coverage >= min_tool_coverage or _box_iou(detection, roll) >= min_iou


def _best_roll(rolls: Sequence[Detection]) -> Optional[Detection]:
    if not rolls:
        return None
    return max(rolls, key=lambda d: (_area(d), d.score))


def _any_evidence_on_roll(
    detections: Sequence[Detection],
    roll: Optional[Detection],
    *,
    cfg: RollEvidenceRuleConfig,
) -> bool:
    if roll is None:
        return False
    return any(
        detection_overlaps_roll(
            d,
            roll,
            min_tool_coverage=cfg.min_tool_coverage,
            min_iou=cfg.min_iou,
        )
        for d in detections
    )


class RollSopEngine:
    """
    Prototype roll-centric SOP engine for `roll_sop_v1`.

    This is intentionally model-agnostic: callers pass already ROI-gated roll,
    cleaning-cloth, and label detections. The runner can wire class filtering and
    ROI handling later without changing these business rules.
    """

    def __init__(self, cfg: RollSopEngineConfig):
        self.cfg = cfg
        self._sessionizer = RollSessionizer(cfg.session)
        self._active: Optional[_ActiveRollSession] = None
        self._session_counter = 0
        self._events: List[EvidenceEvent] = []

    @property
    def active_session_id(self) -> Optional[str]:
        return None if self._active is None else self._active.session_id

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
        rolls: Sequence[Detection],
        cleaning_cloths: Sequence[Detection],
        labels: Sequence[Detection],
    ) -> Optional[RollSessionResult]:
        if time_s < 0:
            raise ValueError("time_s must be >= 0")
        if frame_idx < 0:
            raise ValueError("frame_idx must be >= 0")

        roll = _best_roll(rolls)
        event = self._sessionizer.update(roll is not None)

        if event == "start":
            self._session_counter += 1
            session_id = f"{self._session_counter:06d}"
            self._active = _ActiveRollSession(
                session_id=session_id,
                start_time_s=time_s,
                start_frame_idx=frame_idx,
                cleaning=_TemporalEvidence(
                    required_frames=self.cfg.cleaning.required_frames,
                    max_gap_frames=self.cfg.cleaning.max_gap_frames,
                ),
                labeling=_TemporalEvidence(
                    required_frames=self.cfg.labeling.required_frames,
                    max_gap_frames=self.cfg.labeling.max_gap_frames,
                ),
            )
            self._events.append(
                EvidenceEvent(
                    name="roll_entered",
                    time_s=float(time_s),
                    frame_idx=int(frame_idx),
                    session_id=session_id,
                )
            )

        if self._active is not None and self._sessionizer.active:
            self._active.total_frames += 1
            if roll is not None:
                self._active.roll_present_frames += 1

            cleaning_positive = _any_evidence_on_roll(cleaning_cloths, roll, cfg=self.cfg.cleaning)
            labeling_positive = _any_evidence_on_roll(labels, roll, cfg=self.cfg.labeling)

            self._active.cleaning.update(cleaning_positive)
            self._active.labeling.update(labeling_positive)

            if not self._active.cleaning_done and self._active.cleaning.achieved:
                self._active.cleaning_done = True
                self._events.append(
                    EvidenceEvent(
                        name="cleaned_done",
                        time_s=float(time_s),
                        frame_idx=int(frame_idx),
                        session_id=self._active.session_id,
                    )
                )

            if not self._active.labeling_done and self._active.labeling.achieved:
                self._active.labeling_done = True
                self._events.append(
                    EvidenceEvent(
                        name="labeled_done",
                        time_s=float(time_s),
                        frame_idx=int(frame_idx),
                        session_id=self._active.session_id,
                    )
                )

        if event == "end":
            if self._active is None:
                return None
            result = self._finalize_session(end_time_s=time_s, frame_idx=frame_idx)
            self._active = None
            return result

        return None

    def flush(self, *, time_s: float, frame_idx: int) -> Optional[RollSessionResult]:
        if self._active is None:
            return None
        result = self._finalize_session(end_time_s=time_s, frame_idx=frame_idx)
        self._active = None
        self._sessionizer.reset()
        return result

    def _finalize_session(self, *, end_time_s: float, frame_idx: int) -> RollSessionResult:
        assert self._active is not None
        notes = list(self._active.notes)

        cleaned = self._step_status(
            evidence=self._active.cleaning,
            rule=self.cfg.cleaning,
            total_frames=self._active.total_frames,
            short_note="session_too_short_for_cleaning_decision",
            partial_note="insufficient_cleaning_evidence",
        )
        labeled = self._step_status(
            evidence=self._active.labeling,
            rule=self.cfg.labeling,
            total_frames=self._active.total_frames,
            short_note="session_too_short_for_labeling_decision",
            partial_note="insufficient_labeling_evidence",
        )

        if cleaned == StepStatus.DONE and labeled == StepStatus.DONE:
            overall = RollComplianceStatus.COMPLIANT
        elif cleaned == StepStatus.NOT_DONE or labeled == StepStatus.NOT_DONE:
            overall = RollComplianceStatus.NON_COMPLIANT
        else:
            overall = RollComplianceStatus.UNKNOWN

        if cleaned == StepStatus.UNKNOWN and self._active.cleaning.positive_frames > 0:
            notes.append("insufficient_cleaning_evidence")
        if labeled == StepStatus.UNKNOWN and self._active.labeling.positive_frames > 0:
            notes.append("insufficient_labeling_evidence")

        self._events.append(
            EvidenceEvent(
                name="roll_left",
                time_s=float(end_time_s),
                frame_idx=int(frame_idx),
                session_id=self._active.session_id,
            )
        )

        return RollSessionResult(
            session_id=self._active.session_id,
            sop_profile=self.cfg.sop_profile,
            start_time_s=self._active.start_time_s,
            end_time_s=end_time_s,
            cleaned=cleaned,
            labeled=labeled,
            overall_status=overall,
            total_frames=self._active.total_frames,
            roll_present_frames=self._active.roll_present_frames,
            cleaning_positive_frames=self._active.cleaning.positive_frames,
            labeling_positive_frames=self._active.labeling.positive_frames,
            notes=tuple(dict.fromkeys(notes)),
        )

    @staticmethod
    def _step_status(
        *,
        evidence: _TemporalEvidence,
        rule: RollEvidenceRuleConfig,
        total_frames: int,
        short_note: str,
        partial_note: str,
    ) -> StepStatus:
        del short_note, partial_note
        if evidence.achieved:
            return StepStatus.DONE
        if rule.short_session_is_unknown and total_frames < rule.required_frames:
            return StepStatus.UNKNOWN
        if evidence.positive_frames > 0:
            return StepStatus.UNKNOWN
        return StepStatus.NOT_DONE

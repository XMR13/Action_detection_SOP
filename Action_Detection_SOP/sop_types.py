"""Shared SOP result and evidence types, independent of inference backends."""

from dataclasses import dataclass
from enum import Enum


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

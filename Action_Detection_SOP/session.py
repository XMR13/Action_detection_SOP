from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_ROLL_SESSION_END_S = 5.0
DEFAULT_ROLL_SESSION_START_S = 3.0

#konfigurasi roll yang akan digunakan untuk hal - hal ini
@dataclass(frozen=True)
class RollSessionConfig:
    start_seconds: float = DEFAULT_ROLL_SESSION_START_S
    end_seconds: float = DEFAULT_ROLL_SESSION_END_S
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
        return max(1, int(round(self.start_seconds * self.analysis_fps))) #number of frames so that the coniditoned is to start

    @property
    def end_frames(self) -> int:
        return max(1, int(round(self.end_seconds * self.analysis_fps))) #number of frames so that the session is ending


class RollSessionizer:
    """
    Start after sustained roll presence and end after consecutive absence.

    Used by roll_sop_v1. A detected roll resets the absence streak so temporary
    occlusion shorter than end_seconds does not close the active session.
    """
    def __init__(self, cfg: RollSessionConfig) -> None:
        self.cfg = cfg
        self.active = False
        self._present_streak = 0
        self._absent_streak = 0

    def update(self, roll_present: bool) -> Optional[str]:
        if roll_present:
            self._present_streak +=1
            self._absent_streak = 0
        else:
            self._present_streak = 0
            self._absent_streak +=1

        #if the self ais active and the roll is present
        if not self.active and self._present_streak >= self.cfg.start_frames:
            self.active = True
            self._absent_streak = 0
            return "start"

        #if the roll is not there and the frame of absend is bigger than the threshold frames
        if self.active and self._absent_streak >= self.cfg.end_frames:
            self.active = False
            self._present_streak = 0
            return "end"

        return None

    def reset(self) -> None:
        self.active = False
        self._present_streak = 0
        self._absent_streak = 0

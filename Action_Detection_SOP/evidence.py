from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, List, Tuple


@dataclass(frozen=True)
class EvidenceClipConfig:
    pre_seconds: float = 2.0
    post_seconds: float = 2.0
    max_seconds: float = 6.0
    analysis_fps: float = 5.0

    def __post_init__(self) -> None:
        if self.pre_seconds < 0:
            raise ValueError("pre_seconds must be >= 0")
        if self.post_seconds < 0:
            raise ValueError("post_seconds must be >= 0")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be > 0")

    def resolved_window(self) -> Tuple[float, float]:
        pre = float(self.pre_seconds)
        post = float(self.post_seconds)
        if pre + post > self.max_seconds:
            if pre >= self.max_seconds:
                pre = float(self.max_seconds)
                post = 0.0
            else:
                post = max(0.0, float(self.max_seconds) - pre)
        return pre, post

    @property
    def pre_frames(self) -> int:
        pre, _ = self.resolved_window()
        return max(0, int(math.ceil(pre * self.analysis_fps)))


@dataclass(frozen=True)
class EvidenceClip:
    name: str
    event_time_s: float
    event_frame_idx: int
    start_time_s: float
    end_time_s: float
    actual_start_time_s: float
    actual_end_time_s: float
    fps: float
    frame_count: int


@dataclass
class EvidenceClipData:
    clip: EvidenceClip
    frames: List[Any]


@dataclass
class _PendingClip:
    name: str
    event_time_s: float
    event_frame_idx: int
    start_time_s: float
    end_time_s: float
    frames: List[Any] = field(default_factory=list)
    times: List[float] = field(default_factory=list)

    def append(self, time_s: float, frame: Any) -> None:
        self.times.append(time_s)
        self.frames.append(frame)


class EvidenceClipper:
    def __init__(self, cfg: EvidenceClipConfig) -> None:
        self.cfg = cfg
        pre_s, post_s = cfg.resolved_window()
        self._pre_seconds = pre_s
        self._post_seconds = post_s
        buffer_len = max(1, cfg.pre_frames + 1)
        self._buffer: Deque[Tuple[float, Any]] = deque(maxlen=buffer_len)
        self._pending: List[_PendingClip] = []

    def reset(self) -> None:
        self._buffer.clear()
        self._pending.clear()

    def has_pending(self) -> bool:
        return bool(self._pending)

    def add_frame(self, *, time_s: float, frame: Any) -> List[EvidenceClipData]:
        self._buffer.append((time_s, frame))
        if not self._pending:
            return []

        completed: List[EvidenceClipData] = []
        still_pending: List[_PendingClip] = []
        for clip in self._pending:
            if clip.start_time_s - 1e-9 <= time_s <= clip.end_time_s + 1e-9:
                if not clip.times or time_s > clip.times[-1] + 1e-9:
                    clip.append(time_s, frame)

            if time_s >= clip.end_time_s - 1e-9:
                completed.append(self._finalize_clip(clip))
            else:
                still_pending.append(clip)

        self._pending = still_pending
        return completed

    def trigger(self, *, name: str, time_s: float, frame_idx: int) -> None:
        start = time_s - self._pre_seconds
        end = time_s + self._post_seconds
        clip = _PendingClip(
            name=name,
            event_time_s=time_s,
            event_frame_idx=frame_idx,
            start_time_s=start,
            end_time_s=end,
        )
        for t, f in self._buffer:
            if t + 1e-9 < start:
                continue
            if t - 1e-9 > end:
                continue
            clip.append(t, f)
        self._pending.append(clip)

    def flush(self) -> List[EvidenceClipData]:
        completed = [self._finalize_clip(c) for c in self._pending]
        self._pending = []
        return completed

    def _finalize_clip(self, clip: _PendingClip) -> EvidenceClipData:
        if clip.times:
            actual_start = clip.times[0]
            actual_end = clip.times[-1]
        else:
            actual_start = clip.start_time_s
            actual_end = clip.start_time_s
        payload = EvidenceClip(
            name=clip.name,
            event_time_s=clip.event_time_s,
            event_frame_idx=clip.event_frame_idx,
            start_time_s=clip.start_time_s,
            end_time_s=clip.end_time_s,
            actual_start_time_s=actual_start,
            actual_end_time_s=actual_end,
            fps=float(self.cfg.analysis_fps),
            frame_count=len(clip.frames),
        )
        return EvidenceClipData(clip=payload, frames=list(clip.frames))

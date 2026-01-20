from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from Action_Detection_SOP.evidence import EvidenceClipData


def write_evidence_clip(
    *,
    clip_data: EvidenceClipData,
    session_dir: Path,
    index: int,
) -> Optional[Dict[str, object]]:
    if not clip_data.frames:
        return None
    evidence_dir = session_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{clip_data.clip.name}_{index:02d}.mp4"
    path = evidence_dir / filename
    frame0 = clip_data.frames[0]
    h, w = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_out = float(clip_data.clip.fps)
    writer = cv2.VideoWriter(str(path), fourcc, fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open evidence video writer: {path}")
    for frame in clip_data.frames:
        writer.write(frame)
    writer.release()

    return {
        "name": clip_data.clip.name,
        "file": str(path.relative_to(session_dir)),
        "event_time_s": float(clip_data.clip.event_time_s),
        "event_frame_idx": int(clip_data.clip.event_frame_idx),
        "start_time_s": float(clip_data.clip.start_time_s),
        "end_time_s": float(clip_data.clip.end_time_s),
        "actual_start_time_s": float(clip_data.clip.actual_start_time_s),
        "actual_end_time_s": float(clip_data.clip.actual_end_time_s),
        "frame_count": int(clip_data.clip.frame_count),
        "fps": float(clip_data.clip.fps),
    }


def write_evidence_manifest(*, session_dir: Path, clips: List[Dict[str, object]]) -> Optional[Path]:
    if not clips:
        return None
    path = session_dir / "evidence.json"
    path.write_text(json.dumps({"clips": clips}, indent=2, sort_keys=True), encoding="utf-8")
    return path

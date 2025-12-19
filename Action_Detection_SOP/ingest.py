from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2


@dataclass(frozen=True)
class CaptureInfo:
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]


def open_capture(*, video: Optional[str] = None, webcam: Optional[int] = None, rtsp: Optional[str] = None) -> cv2.VideoCapture:
    sources = [video is not None, webcam is not None, rtsp is not None]
    if sum(bool(s) for s in sources) != 1:
        raise ValueError("Exactly one of video/webcam/rtsp must be provided.")

    if video is not None:
        cap = cv2.VideoCapture(video)
    elif rtsp is not None:
        cap = cv2.VideoCapture(rtsp)
    else:
        cap = cv2.VideoCapture(int(webcam))

    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")
    return cap


def get_capture_info(cap: cv2.VideoCapture) -> CaptureInfo:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps_val = None
    else:
        fps_val = float(fps)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w_val = int(w) if w and w > 0 else None
    h_val = int(h) if h and h > 0 else None

    return CaptureInfo(fps=fps_val, width=w_val, height=h_val)


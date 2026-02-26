from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2


@dataclass(frozen=True)
class CaptureInfo:
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    frame_count: Optional[int]


def _set_capture_property(cap: cv2.VideoCapture, prop_name: str, value: Optional[Union[int, float]]) -> None:
    if value is None:
        return
    prop_id = getattr(cv2, prop_name, None)
    if prop_id is None:
        return
    try:
        cap.set(int(prop_id), float(value))
    except Exception:
        # Best-effort only: support depends on OpenCV backend build.
        return


def _open_with_optional_api(source: Union[str, int], api_preference: Optional[int]) -> cv2.VideoCapture:
    if api_preference is None:
        return cv2.VideoCapture(source)
    try:
        return cv2.VideoCapture(source, int(api_preference))
    except Exception:
        return cv2.VideoCapture(source)


def open_capture(
    *,
    video: Optional[str] = None,
    webcam: Optional[int] = None,
    rtsp: Optional[str] = None,
    rtsp_prefer_ffmpeg: bool = True,
    rtsp_open_timeout_ms: Optional[int] = None,
    rtsp_read_timeout_ms: Optional[int] = None,
    rtsp_buffer_size: Optional[int] = None,
) -> cv2.VideoCapture:
    sources = [video is not None, webcam is not None, rtsp is not None]
    if sum(bool(s) for s in sources) != 1:
        raise ValueError("Exactly one of video/webcam/rtsp must be provided.")

    if video is not None:
        cap = cv2.VideoCapture(video)
    elif rtsp is not None:
        ffmpeg_api = getattr(cv2, "CAP_FFMPEG", None) if rtsp_prefer_ffmpeg else None
        cap = _open_with_optional_api(rtsp, ffmpeg_api)
        if ffmpeg_api is not None and not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(rtsp)
    else:
        cap = cv2.VideoCapture(int(webcam))

    # RTSP capture knobs are best-effort and backend-dependent.
    if rtsp is not None:
        _set_capture_property(cap, "CAP_PROP_BUFFERSIZE", rtsp_buffer_size)
        _set_capture_property(cap, "CAP_PROP_OPEN_TIMEOUT_MSEC", rtsp_open_timeout_ms)
        _set_capture_property(cap, "CAP_PROP_READ_TIMEOUT_MSEC", rtsp_read_timeout_ms)

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

    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if count is None or count <= 0:
        count_val = None
    else:
        count_val = int(count)

    return CaptureInfo(fps=fps_val, width=w_val, height=h_val, frame_count=count_val)

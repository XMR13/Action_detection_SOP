"""Session thumbnail encoding.

Thumbnails are shown in the review queue and as the still-image fallback on the
session detail page once retention has removed the evidence video, so they are
downscaled (not cropped) and JPEG-compressed rather than stored as full frames.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ThumbnailSpec:
    max_width: int = 960
    jpeg_quality: int = 80

    def __post_init__(self) -> None:
        if self.max_width < 64:
            raise ValueError(f"max_width must be >= 64, got {self.max_width}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")


DEFAULT_THUMBNAIL_SPEC = ThumbnailSpec()


def resize_for_thumbnail(frame_bgr: np.ndarray, spec: ThumbnailSpec = DEFAULT_THUMBNAIL_SPEC) -> np.ndarray:
    """Downscale to ``spec.max_width`` keeping aspect ratio; never upscales."""
    if frame_bgr.ndim not in (2, 3) or frame_bgr.size == 0:
        raise ValueError(f"expected a non-empty HxW or HxWxC image, got shape {frame_bgr.shape}")
    h, w = frame_bgr.shape[:2]
    if w <= spec.max_width:
        return frame_bgr
    new_h = max(1, round(h * spec.max_width / w))
    return cv2.resize(frame_bgr, (spec.max_width, new_h), interpolation=cv2.INTER_AREA)


def encode_thumbnail(frame_bgr: np.ndarray, spec: ThumbnailSpec = DEFAULT_THUMBNAIL_SPEC) -> bytes:
    ok, buf = cv2.imencode(".jpg", resize_for_thumbnail(frame_bgr, spec), [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode thumbnail JPEG")
    return buf.tobytes()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, then rename over ``path``."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_thumbnail(path: Path, frame_bgr: np.ndarray, spec: ThumbnailSpec = DEFAULT_THUMBNAIL_SPEC) -> int:
    """Encode and atomically write a thumbnail. Returns bytes written."""
    data = encode_thumbnail(frame_bgr, spec)
    atomic_write_bytes(path, data)
    return len(data)

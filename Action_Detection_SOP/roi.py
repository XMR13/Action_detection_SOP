from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass(frozen=True)
class RoiPolygon:
    """
    Menentukan ROI dalam gambar koordinate (pixel)

    Titik (x,y) adalah tuple dari koordinate framenya
    """

    points: Tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("ROI polygon must have at least 3 points.")

    def as_contour(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def bounding_rect(self, *, expand_px: int = 0) -> Tuple[int, int, int, int]:
        if expand_px < 0:
            raise ValueError("expand_px must be >= 0")
        x, y, w, h = cv2.boundingRect(self.as_contour())
        x0 = x - expand_px
        y0 = y - expand_px
        x1 = x + w + expand_px
        y1 = y + h + expand_px
        return x0, y0, x1, y1

    def contains_point(self, x: float, y: float) -> bool:
        res = cv2.pointPolygonTest(self.as_contour(), (float(x), float(y)), False)
        return res >= 0


def clamp_rect_to_frame(
    rect_xyxy: Tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect_xyxy
    x0 = max(0, min(int(x0), frame_width))
    y0 = max(0, min(int(y0), frame_height))
    x1 = max(0, min(int(x1), frame_width))
    y1 = max(0, min(int(y1), frame_height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid rect after clamping: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def load_roi_json(path: Path) -> RoiPolygon:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ROI JSON must be an object.")
    poly = raw.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError("ROI JSON must include 'polygon' as a list of >= 3 [x, y] points.")

    points: list[Point] = []
    for idx, p in enumerate(poly):
        if (
            not isinstance(p, (list, tuple))
            or len(p) != 2
            or not isinstance(p[0], (int, float))
            or not isinstance(p[1], (int, float))
        ):
            raise ValueError(f"Invalid ROI point at index {idx}: {p!r}")
        points.append((int(p[0]), int(p[1])))

    return RoiPolygon(points=tuple(points))


def save_roi_json(path: Path, polygon: RoiPolygon) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"polygon": [[int(x), int(y)] for (x, y) in polygon.points]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def draw_roi(image_bgr: np.ndarray, polygon: RoiPolygon, *, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    if image_bgr is None or not hasattr(image_bgr, "shape"):
        raise TypeError("image_bgr must be a NumPy array.")
    if image_bgr.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {getattr(image_bgr, 'shape', None)}")

    out = image_bgr.copy()
    pts = np.asarray(polygon.points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    for (x, y) in polygon.points:
        cv2.circle(out, (int(x), int(y)), radius=4, color=color, thickness=-1)
    return out


def iter_points(points: Sequence[Point]) -> Iterable[Point]:
    for p in points:
        yield int(p[0]), int(p[1])


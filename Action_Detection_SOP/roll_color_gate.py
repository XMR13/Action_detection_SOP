"""
Keep blue wrapped, already-completed rolls out of first-floor SOP sessions.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from yolo_kit.types import Detection

"""
-----------------------
ONLY DETECT BLUE ROLL
-----------------------
"""

BLUE_HSV_LOWER = (95, 80, 35)
BLUE_HSV_UPPER = (130, 255, 255)
DEFAULT_MIN_BLUE_FRACTION = 0.30
MIN_ROLL_BOX_PIXEL = 20


def blue_fraction_on_roll(frame_bgr: np.ndarray, roll: Detection) -> float:
    """
    Measure blue only inside the roll box, away from its border and overlays.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image")
    coordinates = (roll.x1, roll.y1, roll.x2, roll.y2)
    if not all(math.isfinite(value) for value in coordinates):
        return 0.0

    height, width = frame_bgr.shape[:2]

    #we define the coordinates so that only inside roll
    x1 = max(0, min(width, math.ceil(roll.x1))) 
    y1 = max(0, min(height, math.ceil(roll.y1)))
    x2 = max(0, min(width, math.floor(roll.x2)))
    y2 = max(0, min(height, math.floor(roll.y2)))
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < MIN_ROLL_BOX_PIXEL or box_height < MIN_ROLL_BOX_PIXEL:
        return 0.0

    #add some inside into the roll itself to avoid bounding box colour, jitter, etc
    inset_x = max(1, round(box_width * 0.08))
    inset_y = max(1, round(box_height * 0.08))
    crop = frame_bgr[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    #check if the pixel inside that bouding box berada pada rentang ini 
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    return float(cv2.countNonZero(blue_mask)) / float(blue_mask.size)


def exclude_blue_rolls(
    frame_bgr: np.ndarray,
    rolls: Sequence[Detection],
    *,
    min_blue_fraction: float = DEFAULT_MIN_BLUE_FRACTION,
) -> Tuple[List[Detection], List[Detection]]:
    """
    Return eligible and excluded rolls; uncertain rolls remain eligible.
    """
    if not (0.0 < min_blue_fraction <= 1.0):
        raise ValueError("min_blue_fraction must be within (0, 1]")
    eligible: List[Detection] = []
    excluded: List[Detection] = []
    for roll in rolls:
        #jika persentase pixel yang bieru lebih besar daripada persentase yang telah ditentukan 
        #maka exclude roll tersebut
        if blue_fraction_on_roll(frame_bgr, roll) >= min_blue_fraction:
            excluded.append(roll)
        else:
            eligible.append(roll)
    return eligible, excluded

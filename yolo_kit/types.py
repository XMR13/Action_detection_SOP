from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """
    Generic detection representation used across backends.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: Optional[int] = None

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

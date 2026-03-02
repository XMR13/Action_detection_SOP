from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WebMvpSettings:
    data_dir: Path
    db_path: Path
    ui_dir: Path
    admin_password: Optional[str] = None
    auto_rescan_seconds: float = 0.0
    auto_approve_done_enabled: bool = True
    auto_approve_min_duration_s: float = 8.0

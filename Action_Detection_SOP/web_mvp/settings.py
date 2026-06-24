from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


#sturktur data untuk setting data web_mvp
@dataclass(frozen=True)
class WebMvpSettings:
    data_dir: Path
    db_path: Path
    ui_dir: Path
    admin_username: str = "admin"
    admin_password: Optional[str] = None
    auto_rescan_seconds: float = 0.0
    auto_approve_done_enabled: bool = True
    auto_approve_min_duration_s: float = 8.0

    #settings for disk
    #1. warning, jika lebih banyak dari 80%
    #2. Critical, jika lebih banyak dari 90%
    disk_warning_used_pct: float = 80.0
    disk_critical_used_pct: float = 90.0

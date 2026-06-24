from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import uvicorn

from Action_Detection_SOP.web_mvp.app import create_app
from Action_Detection_SOP.web_mvp.settings import WebMvpSettings


def _default_data_dir() -> Path:
    # Default to repo-local `data/` (matches SOP runner output), but allow override via env.
    env = os.environ.get("SOP_DATA_DIR", "").strip()
    return Path(env) if env else (Path.cwd() / "data")


def _default_db_path(data_dir: Path) -> Path:
    return data_dir / "web_mvp.sqlite3"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SOP Review Website MVP (FastAPI + SQLite).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--ui-dir", type=Path, default=(Path.cwd() / "mockups"))
    parser.add_argument("--admin-username", default=os.environ.get("SOP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--admin-password", default=os.environ.get("SOP_ADMIN_PASSWORD"))
    parser.add_argument(
        "--auto-rescan-seconds",
        type=float,
        default=float(os.environ.get("SOP_AUTO_RESCAN_S", "0")),
        help="Auto-rescan sessions under <data_dir>/.../sessions when they change (0 disables).",
    )
    parser.add_argument(
        "--disable-auto-approve-done",
        action="store_true",
        help="Disable auto-qualifying low-risk DONE sessions; require manual review for all sessions.",
    )
    parser.add_argument(
        "--auto-approve-min-duration-s",
        type=float,
        default=float(os.environ.get("SOP_AUTO_APPROVE_MIN_DURATION_S", "8.0")),
        help="Minimum session duration (seconds) before DONE can be auto-qualified.",
    )
    parser.add_argument(
        "--disk-warning-used-pct",
        type=float,
        default=float(os.environ.get("SOP_DISK_WARNING_USED_PCT", "75.0")),
        help="Mark disk health warning when used percentage is at or above this value.",
    )
    parser.add_argument(
        "--disk-critical-used-pct",
        type=float,
        default=float(os.environ.get("SOP_DISK_CRITICAL_USED_PCT", "85.0")),
        help="Mark disk health critical when used percentage is at or above this value.",
    )
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only).")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    db_path: Path = args.db_path if args.db_path is not None else _default_db_path(data_dir)
    ui_dir: Path = args.ui_dir

    settings = WebMvpSettings(
        data_dir=data_dir,
        db_path=db_path,
        ui_dir=ui_dir,
        admin_username=str(args.admin_username),
        admin_password=args.admin_password,
        auto_rescan_seconds=float(args.auto_rescan_seconds),
        auto_approve_done_enabled=not bool(args.disable_auto_approve_done),
        auto_approve_min_duration_s=float(args.auto_approve_min_duration_s),
        disk_warning_used_pct=float(args.disk_warning_used_pct),
        disk_critical_used_pct=float(args.disk_critical_used_pct),
    )
    app = create_app(settings)
    uvicorn.run(app, host=str(args.host), port=int(args.port), reload=bool(args.reload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

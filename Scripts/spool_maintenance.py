from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from Scripts import sop_uploader as uploader


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and maintain SOP uploader spool state.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data", help="Root containing uploader_spool.")
    parser.add_argument(
        "--spool-dir",
        type=Path,
        default=None,
        help="Optional explicit spool directory (default: <data-dir>/uploader_spool).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="Show current spool snapshot and state file payload.")

    requeue = sub.add_parser("requeue-dead", help="Move dead tasks back to pending for retry.")
    requeue.add_argument("--limit", type=int, default=0, help="Max dead tasks to inspect (0 = no limit).")
    requeue.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")

    prune = sub.add_parser("prune-done", help="Delete old done task records by age.")
    prune.add_argument("--older-than-days", type=float, default=7.0, help="Delete done records older than this many days.")
    prune.add_argument("--apply", action="store_true", help="Apply deletions. Default is dry-run.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    spool = uploader._resolve_spool_paths(data_dir=Path(args.data_dir), spool_dir=args.spool_dir)
    snapshot_before = uploader._collect_spool_snapshot(spool)
    state_file = uploader._read_spool_state(spool)

    if args.command == "inspect":
        _print_json(
            {
                "command": "inspect",
                "spool_root": str(spool.root),
                "snapshot": snapshot_before,
                "state_file": state_file,
            }
        )
        return 0

    if args.command == "requeue-dead":
        if int(args.limit) < 0:
            raise SystemExit("--limit must be >= 0")
        dry_run = not bool(args.apply)
        if not dry_run:
            uploader._ensure_spool_dirs(spool)
        stats = uploader.requeue_dead_tasks(spool=spool, dry_run=dry_run, limit=int(args.limit))
        snapshot_after = uploader._collect_spool_snapshot(spool)
        _print_json(
            {
                "command": "requeue-dead",
                "dry_run": dry_run,
                "spool_root": str(spool.root),
                "stats": asdict(stats),
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "state_file": state_file,
            }
        )
        return 0

    if args.command == "prune-done":
        older_than_days = float(args.older_than_days)
        if older_than_days < 0:
            raise SystemExit("--older-than-days must be >= 0")
        dry_run = not bool(args.apply)
        if not dry_run:
            uploader._ensure_spool_dirs(spool)
        stats = uploader.prune_done_tasks(
            spool=spool,
            older_than_days=older_than_days,
            dry_run=dry_run,
        )
        snapshot_after = uploader._collect_spool_snapshot(spool)
        _print_json(
            {
                "command": "prune-done",
                "dry_run": dry_run,
                "older_than_days": older_than_days,
                "spool_root": str(spool.root),
                "stats": asdict(stats),
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "state_file": state_file,
            }
        )
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

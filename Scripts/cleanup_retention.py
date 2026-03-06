from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class RetentionPolicy:
    transcoded_cache_days: float = 7.0
    annotated_video_days: float = 14.0
    evidence_clip_days: float = 30.0
    uploader_done_days: float = 7.0
    uploader_dead_days: float = 14.0


@dataclass(frozen=True)
class RetentionAction:
    category: str
    path: Path
    age_days: float
    size_bytes: int


@dataclass(frozen=True)
class RetentionSummary:
    dry_run: bool
    deleted_files: int
    deleted_bytes: int
    matched_files: int
    matched_bytes: int


def _file_age_days(path: Path, *, now_ts: float) -> float:
    return max(0.0, (now_ts - float(path.stat().st_mtime)) / 86400.0)


def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return ()
    return (path for path in root.rglob(pattern) if path.is_file())


def _eligible_files(*, data_dir: Path, policy: RetentionPolicy, now_ts: Optional[float] = None) -> List[RetentionAction]:
    now = float(now_ts if now_ts is not None else time.time())
    candidates: list[tuple[str, Iterable[Path], float]] = [
        ("transcoded_cache", _iter_files(data_dir / "_web_cache" / "transcoded", "*.mp4"), float(policy.transcoded_cache_days)),
        ("annotated_video", _iter_files(data_dir / "sessions", "annotated.mp4"), float(policy.annotated_video_days)),
        ("evidence_clip", _iter_files(data_dir / "sessions", "*.mp4"), float(policy.evidence_clip_days)),
        ("uploader_done", _iter_files(data_dir / "uploader_spool" / "done", "*.json"), float(policy.uploader_done_days)),
        ("uploader_dead", _iter_files(data_dir / "uploader_spool" / "dead", "*.json"), float(policy.uploader_dead_days)),
    ]
    out: List[RetentionAction] = []
    for category, files, min_days in candidates:
        for path in files:
            if category == "evidence_clip" and "evidence" not in path.parts:
                continue
            try:
                age_days = _file_age_days(path, now_ts=now)
                size_bytes = int(path.stat().st_size)
            except OSError:
                continue
            if age_days < min_days:
                continue
            out.append(
                RetentionAction(
                    category=category,
                    path=path,
                    age_days=age_days,
                    size_bytes=size_bytes,
                )
            )
    out.sort(key=lambda item: (item.category, str(item.path)))
    return out


def apply_retention(*, data_dir: Path, policy: RetentionPolicy, dry_run: bool, now_ts: Optional[float] = None) -> RetentionSummary:
    actions = _eligible_files(data_dir=data_dir, policy=policy, now_ts=now_ts)
    matched_bytes = sum(item.size_bytes for item in actions)
    deleted_files = 0
    deleted_bytes = 0
    for item in actions:
        if dry_run:
            continue
        try:
            item.path.unlink(missing_ok=True)
        except OSError:
            continue
        deleted_files += 1
        deleted_bytes += int(item.size_bytes)
    return RetentionSummary(
        dry_run=dry_run,
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
        matched_files=len(actions),
        matched_bytes=matched_bytes,
    )


def _actions_to_json(actions: List[RetentionAction]) -> List[dict]:
    return [
        {
            "category": item.category,
            "path": str(item.path),
            "age_days": round(float(item.age_days), 3),
            "size_bytes": int(item.size_bytes),
        }
        for item in actions
    ]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cleanup retention-managed generated artifacts under a SOP data directory.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--transcoded-cache-days", type=float, default=7.0)
    parser.add_argument("--annotated-video-days", type=float, default=14.0)
    parser.add_argument("--evidence-clip-days", type=float, default=30.0)
    parser.add_argument("--uploader-done-days", type=float, default=7.0)
    parser.add_argument("--uploader-dead-days", type=float, default=14.0)
    parser.add_argument("--apply", action="store_true", help="Delete matched files. Default is dry-run.")
    args = parser.parse_args(argv)

    policy = RetentionPolicy(
        transcoded_cache_days=float(args.transcoded_cache_days),
        annotated_video_days=float(args.annotated_video_days),
        evidence_clip_days=float(args.evidence_clip_days),
        uploader_done_days=float(args.uploader_done_days),
        uploader_dead_days=float(args.uploader_dead_days),
    )
    actions = _eligible_files(data_dir=args.data_dir, policy=policy)
    summary = apply_retention(data_dir=args.data_dir, policy=policy, dry_run=not bool(args.apply))
    payload = {
        "data_dir": str(args.data_dir),
        "policy": asdict(policy),
        "summary": asdict(summary),
        "actions": _actions_to_json(actions),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

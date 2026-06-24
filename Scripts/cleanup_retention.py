from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

@dataclass(frozen=True)
class RetentionPolicy:
    """Validated runtime representation of the configured retention policy."""
    transcoded_cache_days: float
    annotated_video_days: float
    evidence_clip_days: float
    uploader_done_days: float
    uploader_dead_days: float


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


_POLICY_KEYS = {field.name for field in fields(RetentionPolicy)}

def _parse_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml_mapping(path: Path) -> Dict[str, object]:
    """
    Load the small YAML subset used by configs/retention.yaml without adding
    a PyYAML dependency to the deployment environment.
    """

    root: Dict[str, object] = {}
    current_section: Optional[str] = None
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_no}: expected 'key: value'")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key")

        if indent == 0:
            if value:
                root[key] = _parse_scalar(value)
                current_section = None
            else:
                section: Dict[str, object] = {}
                root[key] = section
                current_section = key
            continue

        if current_section is None or not isinstance(root.get(current_section), dict):
            raise ValueError(f"{path}:{line_no}: nested value without a parent section")
        section = root[current_section]
        assert isinstance(section, dict)
        section[key] = _parse_scalar(value)

    return root

def _load_policy_config(config_path: Path) -> Tuple[Optional[Path], RetentionPolicy]:
    payload = _load_simple_yaml_mapping(config_path)
    raw_data_dir = payload.get("data_dir")
    data_dir = Path(str(raw_data_dir)) if raw_data_dir not in (None, "") else None

    raw_retention = payload.get("retention", {})
    if not isinstance(raw_retention, dict):
        raise ValueError(f"{config_path}: retention must be a mapping")

    unknown_keys = sorted(set(raw_retention) - _POLICY_KEYS)
    if unknown_keys:
        raise ValueError(f"{config_path}: unsupported retention keys: {', '.join(unknown_keys)}")

    missing_keys = sorted(_POLICY_KEYS - set(raw_retention))
    if missing_keys:
        raise ValueError(f"{config_path}: missing retention keys: {', '.join(missing_keys)}")

    policy_values: Dict[str, float] = {}
    for key in _POLICY_KEYS:
        value = raw_retention[key]
        try:
            policy_values[key] = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{config_path}: retention.{key} must be a number") from e

    return data_dir, RetentionPolicy(**policy_values)


def _override_policy(base: RetentionPolicy, overrides: Mapping[str, Optional[float]]) -> RetentionPolicy:
    policy_values = asdict(base)
    for key, value in overrides.items():
        if value is not None:
            policy_values[key] = float(value)
    return RetentionPolicy(**policy_values)


def _file_age_days(path: Path, *, now_ts: float) -> float:
    return max(0.0, (now_ts - float(path.stat().st_mtime)) / 86400.0)


def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return ()
    return (path for path in root.rglob(pattern) if path.is_file())


def _eligible_files(*, data_dir: Path, policy: RetentionPolicy, now_ts: Optional[float] = None) -> List[RetentionAction]:
    now = float(now_ts if now_ts is not None else time.time())
    candidates: List[Tuple[str, Iterable[Path], float]] = [
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


def _actions_to_json(actions: List[RetentionAction]) -> List[Dict[str, object]]:
    return [
        {
            "category": item.category,
            "path": str(item.path),
            "age_days": round(float(item.age_days), 3),
            "size_bytes": int(item.size_bytes),
        }
        for item in actions
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cleanup retention-managed generated artifacts under a SOP data directory.")
    parser.add_argument("--config", type=Path, required=True, help="Retention YAML config containing all policy values.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--transcoded-cache-days", type=float, default=None)
    parser.add_argument("--annotated-video-days", type=float, default=None)
    parser.add_argument("--evidence-clip-days", type=float, default=None)
    parser.add_argument("--uploader-done-days", type=float, default=None)
    parser.add_argument("--uploader-dead-days", type=float, default=None)
    parser.add_argument("--apply", action="store_true", help="Delete matched files. Default is dry-run.")
    args = parser.parse_args(argv)

    config_data_dir, policy = _load_policy_config(args.config)
    data_dir = config_data_dir if config_data_dir is not None else Path.cwd() / "data"
    if args.data_dir is not None:
        data_dir = args.data_dir

    policy = _override_policy(
        policy,
        {
            "transcoded_cache_days": args.transcoded_cache_days,
            "annotated_video_days": args.annotated_video_days,
            "evidence_clip_days": args.evidence_clip_days,
            "uploader_done_days": args.uploader_done_days,
            "uploader_dead_days": args.uploader_dead_days,
        },
    )

    actions = _eligible_files(data_dir=data_dir, policy=policy)
    summary = apply_retention(data_dir=data_dir, policy=policy, dry_run=not bool(args.apply))
    payload = {
        "config": str(args.config),
            "data_dir": str(data_dir),
        "policy": asdict(policy),
        "summary": asdict(summary),
        "actions": _actions_to_json(actions),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

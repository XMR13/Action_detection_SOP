from __future__ import annotations

import os
import time
from pathlib import Path

from Scripts.cleanup_retention import RetentionPolicy, _eligible_files, apply_retention


def _touch_old(path: Path, *, age_days: float, now_ts: float, body: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    old_ts = now_ts - (age_days * 86400.0)
    os.utime(path, (old_ts, old_ts))


def test_retention_targets_only_generated_artifacts(tmp_path: Path) -> None:
    now_ts = time.time()
    data_dir = tmp_path / "data"
    _touch_old(data_dir / "_web_cache" / "transcoded" / "uid1" / "cache.mp4", age_days=10, now_ts=now_ts)
    _touch_old(data_dir / "sessions" / "2026-03-01" / "session_001" / "annotated.mp4", age_days=20, now_ts=now_ts)
    _touch_old(
        data_dir / "sessions" / "2026-03-01" / "session_001" / "evidence" / "clip01.mp4",
        age_days=40,
        now_ts=now_ts,
    )
    _touch_old(data_dir / "uploader_spool" / "done" / "task1.json", age_days=10, now_ts=now_ts)
    _touch_old(data_dir / "uploader_spool" / "dead" / "task2.json", age_days=20, now_ts=now_ts)
    _touch_old(data_dir / "sessions" / "2026-03-01" / "session_001" / "checklist.json", age_days=60, now_ts=now_ts)
    _touch_old(data_dir / "sessions" / "2026-03-01" / "session_001" / "thumbnail.jpg", age_days=60, now_ts=now_ts)

    actions = _eligible_files(data_dir=data_dir, policy=RetentionPolicy(), now_ts=now_ts)
    categories = {(item.category, item.path.name) for item in actions}
    assert ("transcoded_cache", "cache.mp4") in categories
    assert ("annotated_video", "annotated.mp4") in categories
    assert ("evidence_clip", "clip01.mp4") in categories
    assert ("uploader_done", "task1.json") in categories
    assert ("uploader_dead", "task2.json") in categories
    assert ("evidence_clip", "checklist.json") not in categories
    assert ("evidence_clip", "thumbnail.jpg") not in categories


def test_retention_apply_deletes_matched_files_only(tmp_path: Path) -> None:
    now_ts = time.time()
    data_dir = tmp_path / "data"
    old_clip = data_dir / "sessions" / "2026-03-01" / "session_001" / "evidence" / "clip01.mp4"
    old_checklist = data_dir / "sessions" / "2026-03-01" / "session_001" / "checklist.json"
    _touch_old(old_clip, age_days=40, now_ts=now_ts)
    _touch_old(old_checklist, age_days=40, now_ts=now_ts)

    summary = apply_retention(data_dir=data_dir, policy=RetentionPolicy(), dry_run=False, now_ts=now_ts)
    assert summary.deleted_files == 1
    assert not old_clip.exists()
    assert old_checklist.exists()

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from Scripts.cleanup_retention import RetentionPolicy, _eligible_files, apply_retention, main


def _touch_old(path: Path, *, age_days: float, now_ts: float, body: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    old_ts = now_ts - (age_days * 86400.0)
    os.utime(path, (old_ts, old_ts))


def _test_policy() -> RetentionPolicy:
    return RetentionPolicy(
        transcoded_cache_days=7.0,
        annotated_video_days=14.0,
        evidence_clip_days=30.0,
        uploader_done_days=7.0,
        uploader_dead_days=14.0,
    )


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

    actions = _eligible_files(data_dir=data_dir, policy=_test_policy(), now_ts=now_ts)
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

    summary = apply_retention(data_dir=data_dir, policy=_test_policy(), dry_run=False, now_ts=now_ts)
    assert summary.deleted_files == 1
    assert not old_clip.exists()
    assert old_checklist.exists()


def test_retention_config_loads_policy_and_cli_override(tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    config_path = tmp_path / "retention.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {data_dir}",
                "retention:",
                "  transcoded_cache_days: 3",
                "  annotated_video_days: 4",
                "  evidence_clip_days: 5",
                "  uploader_done_days: 6",
                "  uploader_dead_days: 7",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--evidence-clip-days", "9"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["config"] == str(config_path)
    assert payload["data_dir"] == str(data_dir)
    assert payload["policy"] == {
        "transcoded_cache_days": 3.0,
        "annotated_video_days": 4.0,
        "evidence_clip_days": 9.0,
        "uploader_done_days": 6.0,
        "uploader_dead_days": 7.0,
    }
    assert payload["summary"]["dry_run"] is True


def test_retention_config_requires_every_policy_value(tmp_path: Path) -> None:
    config_path = tmp_path / "retention.yaml"
    config_path.write_text(
        "\n".join(
            [
                "retention:",
                "  transcoded_cache_days: 3",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing retention keys"):
        main(["--config", str(config_path)])

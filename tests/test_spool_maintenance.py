from __future__ import annotations

import json
import os
import time
from pathlib import Path

from Scripts import sop_uploader as uploader
from Scripts import spool_maintenance


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_session_dir(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    session_dir = data_dir / "sessions" / "2026-03-06" / "session_001"
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        session_dir / "checklist.json",
        {
            "session_id": "001",
            "operator_present": "DONE",
            "roi_dwell": "DONE",
            "helmet": "DONE",
        },
    )
    return data_dir, session_dir


def _spool(data_dir: Path) -> uploader.SpoolPaths:
    spool = uploader._resolve_spool_paths(data_dir=data_dir, spool_dir=None)
    uploader._ensure_spool_dirs(spool)
    return spool


def test_spool_maintenance_inspect_outputs_snapshot(tmp_path: Path, capsys) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    spool = _spool(data_dir)
    sessions = list(uploader._iter_sessions(data_dir))
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    rc = spool_maintenance.main(["--data-dir", str(data_dir), "inspect"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "inspect"
    assert payload["snapshot"]["pending"]["files"] == 1
    assert payload["snapshot"]["pending_retry"]["ready_now_files"] == 1


def test_spool_maintenance_requeue_dead_apply(tmp_path: Path, capsys) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    spool = _spool(data_dir)
    sessions = list(uploader._iter_sessions(data_dir))
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    pending_path = next(spool.pending.glob("*.json"))
    task = uploader._load_task(pending_path)
    uploader._record_dead_task(spool=spool, pending_path=pending_path, task=task, reason="forced_dead")
    assert len(list(spool.dead.glob("*.json"))) == 1

    rc = spool_maintenance.main(["--data-dir", str(data_dir), "requeue-dead", "--apply"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "requeue-dead"
    assert payload["dry_run"] is False
    assert payload["stats"]["requeued"] == 1
    assert len(list(spool.dead.glob("*.json"))) == 0
    assert len(list(spool.pending.glob("*.json"))) == 1


def test_spool_maintenance_prune_done_apply(tmp_path: Path, capsys) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    spool = _spool(data_dir)
    done_path = spool.done / "old_done.json"
    done_path.write_text("{}", encoding="utf-8")
    old_ts = time.time() - (5 * 86400)
    os.utime(done_path, (old_ts, old_ts))

    rc = spool_maintenance.main(
        ["--data-dir", str(data_dir), "prune-done", "--older-than-days", "2", "--apply"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "prune-done"
    assert payload["dry_run"] is False
    assert payload["stats"]["deleted_files"] == 1
    assert done_path.exists() is False

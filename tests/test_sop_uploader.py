from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

from Scripts import sop_uploader as uploader


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_session_dir(tmp_path: Path) -> Tuple[Path, Path]:
    data_dir = tmp_path / "data"
    session_dir = data_dir / "sessions" / "2026-03-05" / "session_001"
    session_dir.mkdir(parents=True, exist_ok=True)
    checklist_path = session_dir / "checklist.json"
    _write_json(
        checklist_path,
        {
            "session_id": "001",
            "operator_present": "DONE",
            "roi_dwell": "DONE",
            "helmet": "UNKNOWN",
        },
    )
    return data_dir, session_dir


def _spool(data_dir: Path) -> uploader.SpoolPaths:
    spool = uploader._resolve_spool_paths(data_dir=data_dir, spool_dir=None)
    uploader._ensure_spool_dirs(spool)
    return spool


def _server_auth() -> Tuple[uploader.Server, str]:
    server = uploader._parse_server("http://127.0.0.1:8000")
    auth = uploader._basic_auth("admin", "secret")
    return server, auth


def test_enqueue_creates_tasks_and_persists_uid(tmp_path: Path) -> None:
    data_dir, session_dir = _build_session_dir(tmp_path)
    _write_json(session_dir / "run_config.json", {"model": "demo"})
    _write_json(session_dir / "evidence.json", {"clips": []})
    evidence_dir = session_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "clip_01.mp4").write_bytes(b"fake")

    sessions = list(uploader._iter_sessions(data_dir))
    spool = _spool(data_dir)
    stats = uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    assert stats.queued == 4
    assert len(list(spool.pending.glob("*.json"))) == 4

    checklist = uploader._read_json(session_dir / "checklist.json")
    assert isinstance(checklist.get("session_uid"), str)
    assert checklist.get("start_date") == "2026-03-05"
    assert checklist.get("end_date") == "2026-03-05"

    stats_second = uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)
    assert stats_second.skipped_pending == 4
    assert stats_second.queued == 0


def test_retryable_error_schedules_backoff(tmp_path: Path, monkeypatch) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    sessions = list(uploader._iter_sessions(data_dir))
    spool = _spool(data_dir)
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    def _always_retryable(**_: object) -> None:
        raise uploader.RetryableUploadError("temporary network failure")

    monkeypatch.setattr(uploader, "_execute_task", _always_retryable)

    server, auth = _server_auth()
    started_at = time.time()
    stats = uploader._process_pending_tasks(
        spool=spool,
        server=server,
        auth_header=auth,
        dry_run=False,
        max_attempts=3,
        retry_wait_s=1.0,
        retry_backoff=2.0,
        retry_wait_max_s=10.0,
        process_limit=0,
    )

    assert stats.attempted == 1
    assert stats.retry_scheduled == 1
    assert stats.dead == 0

    pending_files = list(spool.pending.glob("*.json"))
    assert len(pending_files) == 1
    task = uploader._load_task(pending_files[0])
    assert task.attempts == 1
    assert task.next_retry_ts >= started_at + 0.8


def test_retry_exhausted_moves_task_to_dead(tmp_path: Path, monkeypatch) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    sessions = list(uploader._iter_sessions(data_dir))
    spool = _spool(data_dir)
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    def _always_retryable(**_: object) -> None:
        raise uploader.RetryableUploadError("temporary network failure")

    monkeypatch.setattr(uploader, "_execute_task", _always_retryable)

    server, auth = _server_auth()
    stats = uploader._process_pending_tasks(
        spool=spool,
        server=server,
        auth_header=auth,
        dry_run=False,
        max_attempts=1,
        retry_wait_s=1.0,
        retry_backoff=2.0,
        retry_wait_max_s=10.0,
        process_limit=0,
    )

    assert stats.dead == 1
    assert len(list(spool.pending.glob("*.json"))) == 0
    assert len(list(spool.dead.glob("*.json"))) == 1


def test_done_task_is_skipped_until_payload_changes(tmp_path: Path, monkeypatch) -> None:
    data_dir, session_dir = _build_session_dir(tmp_path)
    sessions = list(uploader._iter_sessions(data_dir))
    spool = _spool(data_dir)
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    def _always_success(**_: object) -> None:
        return None

    monkeypatch.setattr(uploader, "_execute_task", _always_success)
    server, auth = _server_auth()
    first_stats = uploader._process_pending_tasks(
        spool=spool,
        server=server,
        auth_header=auth,
        dry_run=False,
        max_attempts=0,
        retry_wait_s=1.0,
        retry_backoff=2.0,
        retry_wait_max_s=10.0,
        process_limit=0,
    )
    assert first_stats.succeeded == 1
    assert len(list(spool.done.glob("*.json"))) == 1

    second_enqueue = uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)
    assert second_enqueue.skipped_done == 1
    assert len(list(spool.pending.glob("*.json"))) == 0

    checklist = uploader._read_json(session_dir / "checklist.json")
    checklist["notes"] = ["updated"]
    _write_json(session_dir / "checklist.json", checklist)

    third_enqueue = uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)
    assert third_enqueue.queued == 1
    assert len(list(spool.pending.glob("*.json"))) == 1


def test_collect_spool_snapshot_and_write_state(tmp_path: Path) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    sessions = list(uploader._iter_sessions(data_dir))
    spool = _spool(data_dir)
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    snapshot = uploader._collect_spool_snapshot(spool, now_ts=time.time())
    assert snapshot["pending"]["files"] == 1
    assert snapshot["pending_retry"]["ready_now_files"] == 1
    assert snapshot["health"] == "backlog"

    uploader._write_spool_state(
        spool=spool,
        snapshot=snapshot,
        cycle=3,
        watch_mode=True,
        sessions_scanned=len(sessions),
        enqueue_stats=uploader.EnqueueStats(queued=1),
        process_stats=uploader.ProcessStats(),
        last_success_utc=None,
        last_dead_utc=None,
    )
    state = uploader._read_spool_state(spool)
    assert state["exists"] is True
    payload = state["payload"]
    assert isinstance(payload, dict)
    assert payload["cycle"] == 3
    assert payload["watch_mode"] is True
    assert payload["spool"]["pending"]["files"] == 1
    assert payload["spool"]["pending_retry"]["ready_now_files"] == 1


def test_requeue_dead_and_prune_done_helpers(tmp_path: Path) -> None:
    data_dir, _ = _build_session_dir(tmp_path)
    sessions = list(uploader._iter_sessions(data_dir))
    spool = _spool(data_dir)
    uploader._enqueue_session_tasks(sessions=sessions, spool=spool, dry_run=False)

    pending_path = next(spool.pending.glob("*.json"))
    pending_task = uploader._load_task(pending_path)
    uploader._record_dead_task(spool=spool, pending_path=pending_path, task=pending_task, reason="forced_dead")
    assert len(list(spool.pending.glob("*.json"))) == 0
    assert len(list(spool.dead.glob("*.json"))) == 1

    dry_stats = uploader.requeue_dead_tasks(spool=spool, dry_run=True)
    assert dry_stats.matched == 1
    assert dry_stats.requeued == 1
    assert len(list(spool.pending.glob("*.json"))) == 0
    assert len(list(spool.dead.glob("*.json"))) == 1

    apply_stats = uploader.requeue_dead_tasks(spool=spool, dry_run=False)
    assert apply_stats.requeued == 1
    assert len(list(spool.pending.glob("*.json"))) == 1
    assert len(list(spool.dead.glob("*.json"))) == 0

    pending_path = next(spool.pending.glob("*.json"))
    pending_task = uploader._load_task(pending_path)
    uploader._record_done_task(spool=spool, pending_path=pending_path, task=pending_task, reason="promoted")
    assert len(list(spool.done.glob("*.json"))) == 1

    done_path = next(spool.done.glob("*.json"))
    old_ts = time.time() - (3 * 86400)
    os.utime(done_path, (old_ts, old_ts))

    prune_stats = uploader.prune_done_tasks(spool=spool, older_than_days=2.0, dry_run=False)
    assert prune_stats.matched_files == 1
    assert prune_stats.deleted_files == 1
    assert len(list(spool.done.glob("*.json"))) == 0

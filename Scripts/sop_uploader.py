from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import quote, urlsplit

from Action_Detection_SOP.reconnect_policy import reconnect_wait_seconds


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _sha1_bytes(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


@dataclass(frozen=True)
class Server:
    scheme: str
    host: str
    port: int
    base_path: str

    @property
    def is_https(self) -> bool:
        return self.scheme.lower() == "https"

    def conn(self) -> HTTPConnection:
        if self.is_https:
            return HTTPSConnection(self.host, self.port, timeout=60)
        return HTTPConnection(self.host, self.port, timeout=60)


def _parse_server(url: str) -> Server:
    s = urlsplit(url)
    if s.scheme not in ("http", "https"):
        raise ValueError("server must be http(s) URL")
    host = s.hostname or ""
    if not host:
        raise ValueError("server must include hostname")
    port = int(s.port or (443 if s.scheme == "https" else 80))
    base_path = s.path.rstrip("/")
    return Server(scheme=s.scheme, host=host, port=port, base_path=base_path)


def _basic_auth(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _read_response(resp: HTTPResponse) -> Tuple[int, str]:
    body = resp.read()
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = repr(body)
    return int(resp.status), text


class HTTPStatusError(RuntimeError):
    def __init__(self, *, method: str, path: str, status: int, body: str) -> None:
        self.method = method
        self.path = path
        self.status = int(status)
        self.body = body
        super().__init__(f"{method} {path} failed: HTTP {status}: {body}")


class RetryableUploadError(RuntimeError):
    pass


class PermanentUploadError(RuntimeError):
    pass


def _http_put_json(
    *,
    server: Server,
    path: str,
    auth_header: str,
    payload: Dict[str, Any],
    dry_run: bool,
) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Content-Length": str(len(body)),
    }
    if dry_run:
        return
    conn = server.conn()
    try:
        conn.request("PUT", path, body=body, headers=headers)
        resp = conn.getresponse()
        status, text = _read_response(resp)
        if status >= 400:
            raise HTTPStatusError(method="PUT", path=path, status=status, body=text)
    finally:
        conn.close()


def _iter_file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _http_post_file(
    *,
    server: Server,
    path: str,
    auth_header: str,
    file_path: Path,
    content_type: Optional[str] = None,
    dry_run: bool,
) -> None:
    guessed_type = mimetypes.guess_type(file_path.name)[0]
    content_type = content_type or guessed_type or "application/octet-stream"
    content_length = int(file_path.stat().st_size)
    headers = {
        "Authorization": auth_header,
        "Content-Type": content_type,
        "Accept": "application/json",
        "Content-Length": str(content_length),
    }
    if dry_run:
        return
    conn = server.conn()
    try:
        conn.putrequest("POST", path)
        for k, v in headers.items():
            conn.putheader(k, v)
        conn.endheaders()
        for chunk in _iter_file_chunks(file_path):
            conn.send(chunk)
        resp = conn.getresponse()
        status, text = _read_response(resp)
        if status >= 400:
            raise HTTPStatusError(method="POST", path=path, status=status, body=text)
    finally:
        conn.close()


@dataclass(frozen=True)
class LocalSession:
    date: str
    session_dir: Path
    checklist_path: Path


def _iter_sessions(data_dir: Path) -> Iterable[LocalSession]:
    sessions_root = data_dir / "sessions"
    if not sessions_root.exists():
        return []
    out: list[LocalSession] = []
    for date_dir in sorted(sessions_root.iterdir()):
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        for session_dir in sorted(date_dir.glob("session_*")):
            if not session_dir.is_dir():
                continue
            checklist_path = session_dir / "checklist.json"
            if not checklist_path.exists():
                continue
            out.append(LocalSession(date=date, session_dir=session_dir, checklist_path=checklist_path))
    return out


def _ensure_session_uid(checklist: Dict[str, Any]) -> str:
    uid = checklist.get("session_uid")
    if isinstance(uid, str) and uid.strip():
        return uid.strip()
    uid = uuid.uuid4().hex
    checklist["session_uid"] = uid
    return uid


def _ensure_session_id(checklist: Dict[str, Any], session_dir: Path) -> str:
    sid = checklist.get("session_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    derived = session_dir.name.replace("session_", "")
    checklist["session_id"] = derived
    return derived


def _ensure_dates(checklist: Dict[str, Any], date: str) -> None:
    if not checklist.get("start_date"):
        checklist["start_date"] = date
    if not checklist.get("end_date"):
        checklist["end_date"] = date


def _artifact_paths(session_dir: Path) -> Iterable[Path]:
    for name in ("run_config.json", "thumbnail.jpg", "evidence.json"):
        p = session_dir / name
        if p.exists() and p.is_file():
            yield p
    evidence_dir = session_dir / "evidence"
    if evidence_dir.exists() and evidence_dir.is_dir():
        for p in sorted(evidence_dir.glob("*")):
            if p.is_file():
                yield p


def _checklist_fingerprint(checklist: Dict[str, Any]) -> str:
    payload = json.dumps(checklist, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _sha1_bytes(payload)


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{int(stat.st_size)}:{int(stat.st_mtime_ns)}"


@dataclass
class SpoolTask:
    task_id: str
    kind: str
    session_uid: str
    session_id: str
    date: str
    session_dir: str
    source_path: str
    source_fingerprint: str
    rel_path: Optional[str]
    payload: Optional[Dict[str, Any]]
    attempts: int
    next_retry_ts: float
    created_at_utc: str
    updated_at_utc: str
    last_error: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "session_uid": self.session_uid,
            "session_id": self.session_id,
            "date": self.date,
            "session_dir": self.session_dir,
            "source_path": self.source_path,
            "source_fingerprint": self.source_fingerprint,
            "rel_path": self.rel_path,
            "payload": self.payload,
            "attempts": int(self.attempts),
            "next_retry_ts": float(self.next_retry_ts),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "last_error": self.last_error,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SpoolTask":
        return cls(
            task_id=str(payload["task_id"]),
            kind=str(payload["kind"]),
            session_uid=str(payload["session_uid"]),
            session_id=str(payload.get("session_id", "")),
            date=str(payload.get("date", "")),
            session_dir=str(payload.get("session_dir", "")),
            source_path=str(payload["source_path"]),
            source_fingerprint=str(payload["source_fingerprint"]),
            rel_path=None if payload.get("rel_path") is None else str(payload.get("rel_path")),
            payload=None if payload.get("payload") is None else dict(payload["payload"]),
            attempts=int(payload.get("attempts", 0)),
            next_retry_ts=float(payload.get("next_retry_ts", 0.0)),
            created_at_utc=str(payload.get("created_at_utc", _utc_now_iso())),
            updated_at_utc=str(payload.get("updated_at_utc", _utc_now_iso())),
            last_error=str(payload.get("last_error", "")),
        )


@dataclass(frozen=True)
class SpoolPaths:
    root: Path
    pending: Path
    done: Path
    dead: Path


@dataclass
class EnqueueStats:
    queued: int = 0
    skipped_done: int = 0
    skipped_pending: int = 0
    skipped_dead: int = 0
    updated_pending: int = 0
    requeued_dead: int = 0


@dataclass
class ProcessStats:
    attempted: int = 0
    succeeded: int = 0
    retry_scheduled: int = 0
    dead: int = 0
    deferred: int = 0
    skipped_limit: int = 0


def _resolve_spool_paths(*, data_dir: Path, spool_dir: Optional[Path]) -> SpoolPaths:
    root = Path(spool_dir) if spool_dir is not None else Path(data_dir) / "uploader_spool"
    return SpoolPaths(root=root, pending=root / "pending", done=root / "done", dead=root / "dead")


def _ensure_spool_dirs(spool: SpoolPaths) -> None:
    spool.pending.mkdir(parents=True, exist_ok=True)
    spool.done.mkdir(parents=True, exist_ok=True)
    spool.dead.mkdir(parents=True, exist_ok=True)


def _task_files(spool: SpoolPaths, task_id: str) -> Tuple[Path, Path, Path]:
    return (
        spool.pending / f"{task_id}.json",
        spool.done / f"{task_id}.json",
        spool.dead / f"{task_id}.json",
    )


def _load_task(path: Path) -> SpoolTask:
    payload = _read_json(path)
    return SpoolTask.from_payload(payload)


def _save_task(path: Path, task: SpoolTask) -> None:
    _atomic_write_json(path, task.to_payload())


def _save_task_record(path: Path, task: SpoolTask, *, status: str, reason: str = "") -> None:
    payload = task.to_payload()
    payload["status"] = status
    payload["reason"] = reason
    payload["recorded_at_utc"] = _utc_now_iso()
    _atomic_write_json(path, payload)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _build_upsert_task(*, session: LocalSession, checklist: Dict[str, Any], session_uid: str, session_id: str) -> SpoolTask:
    fingerprint = _checklist_fingerprint(checklist)
    task_key = _sha1_text(f"upsert|{session_uid}")
    now_iso = _utc_now_iso()
    return SpoolTask(
        task_id=task_key,
        kind="upsert",
        session_uid=session_uid,
        session_id=session_id,
        date=session.date,
        session_dir=str(session.session_dir),
        source_path=str(session.checklist_path),
        source_fingerprint=fingerprint,
        rel_path=None,
        payload=checklist,
        attempts=0,
        next_retry_ts=0.0,
        created_at_utc=now_iso,
        updated_at_utc=now_iso,
        last_error="",
    )


def _build_artifact_task(
    *,
    session: LocalSession,
    session_uid: str,
    session_id: str,
    artifact: Path,
) -> SpoolTask:
    rel_path = artifact.relative_to(session.session_dir).as_posix()
    fingerprint = _file_fingerprint(artifact)
    task_key = _sha1_text(f"artifact|{session_uid}|{rel_path}")
    now_iso = _utc_now_iso()
    return SpoolTask(
        task_id=task_key,
        kind="artifact",
        session_uid=session_uid,
        session_id=session_id,
        date=session.date,
        session_dir=str(session.session_dir),
        source_path=str(artifact),
        source_fingerprint=fingerprint,
        rel_path=rel_path,
        payload=None,
        attempts=0,
        next_retry_ts=0.0,
        created_at_utc=now_iso,
        updated_at_utc=now_iso,
        last_error="",
    )


def _queue_task(*, spool: SpoolPaths, task: SpoolTask, dry_run: bool) -> str:
    pending_path, done_path, dead_path = _task_files(spool, task.task_id)
    had_dead = dead_path.exists()
    if done_path.exists():
        done_payload = _read_json(done_path)
        if str(done_payload.get("source_fingerprint", "")) == task.source_fingerprint:
            return "skipped_done"
        if not dry_run:
            _safe_unlink(done_path)
    if pending_path.exists():
        pending_task = _load_task(pending_path)
        if pending_task.source_fingerprint == task.source_fingerprint:
            return "skipped_pending"
        if not dry_run:
            task.attempts = 0
            task.next_retry_ts = 0.0
            task.last_error = ""
            task.updated_at_utc = _utc_now_iso()
            task.created_at_utc = pending_task.created_at_utc
            _save_task(pending_path, task)
        return "updated_pending"
    if dead_path.exists():
        dead_payload = _read_json(dead_path)
        if str(dead_payload.get("source_fingerprint", "")) == task.source_fingerprint:
            return "skipped_dead"
        if not dry_run:
            _safe_unlink(dead_path)
    if not dry_run:
        _save_task(pending_path, task)
    if had_dead:
        return "requeued_dead"
    return "queued"


def _update_enqueue_stats(stats: EnqueueStats, result: str) -> None:
    if result == "queued":
        stats.queued += 1
    elif result == "skipped_done":
        stats.skipped_done += 1
    elif result == "skipped_pending":
        stats.skipped_pending += 1
    elif result == "skipped_dead":
        stats.skipped_dead += 1
    elif result == "updated_pending":
        stats.updated_pending += 1
    elif result == "requeued_dead":
        stats.requeued_dead += 1


def _enqueue_session_tasks(*, sessions: Iterable[LocalSession], spool: SpoolPaths, dry_run: bool) -> EnqueueStats:
    stats = EnqueueStats()
    for session in sessions:
        checklist = _read_json(session.checklist_path)
        session_uid = _ensure_session_uid(checklist)
        session_id = _ensure_session_id(checklist, session.session_dir)
        _ensure_dates(checklist, session.date)
        if not dry_run:
            _write_json(session.checklist_path, checklist)
        upsert_task = _build_upsert_task(
            session=session,
            checklist=checklist,
            session_uid=session_uid,
            session_id=session_id,
        )
        result = _queue_task(spool=spool, task=upsert_task, dry_run=dry_run)
        _update_enqueue_stats(stats, result)
        for artifact in _artifact_paths(session.session_dir):
            artifact_task = _build_artifact_task(
                session=session,
                session_uid=session_uid,
                session_id=session_id,
                artifact=artifact,
            )
            result = _queue_task(spool=spool, task=artifact_task, dry_run=dry_run)
            _update_enqueue_stats(stats, result)
    return stats


def _iter_pending_task_files(spool: SpoolPaths) -> Iterable[Path]:
    if not spool.pending.exists():
        return []
    return sorted(spool.pending.glob("*.json"))


def _is_retryable_http_status(*, status: int, kind: str) -> bool:
    if status in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    if kind == "artifact" and status == 404:
        # Artifact upload may race against server index refresh after session upsert.
        return True
    return False


def _execute_task(*, task: SpoolTask, server: Server, auth_header: str, dry_run: bool) -> None:
    try:
        if task.kind == "upsert":
            payload = dict(task.payload or {})
            if not payload:
                payload = _read_json(Path(task.source_path))
            path = f"{server.base_path}/api/sessions/{task.session_uid}"
            print(f"[{_utc_now_iso()}] upsert {task.session_uid} ({task.date}/{task.session_id})")
            _http_put_json(
                server=server,
                path=path,
                auth_header=auth_header,
                payload=payload,
                dry_run=dry_run,
            )
            return
        if task.kind == "artifact":
            artifact_path = Path(task.source_path)
            if not artifact_path.exists() or not artifact_path.is_file():
                raise PermanentUploadError(f"Missing artifact file: {artifact_path}")
            rel_path = task.rel_path or artifact_path.name
            path = f"{server.base_path}/api/sessions/{task.session_uid}/artifacts?rel_path={quote(rel_path)}"
            print(f"[{_utc_now_iso()}] upload {task.session_uid}:{rel_path}")
            _http_post_file(
                server=server,
                path=path,
                auth_header=auth_header,
                file_path=artifact_path,
                dry_run=dry_run,
            )
            return
        raise PermanentUploadError(f"Unsupported task kind: {task.kind}")
    except HTTPStatusError as exc:
        if _is_retryable_http_status(status=exc.status, kind=task.kind):
            raise RetryableUploadError(str(exc)) from exc
        raise PermanentUploadError(str(exc)) from exc
    except (TimeoutError, OSError, ConnectionError) as exc:
        raise RetryableUploadError(str(exc)) from exc


def _record_done_task(*, spool: SpoolPaths, pending_path: Path, task: SpoolTask, reason: str = "") -> None:
    _, done_path, dead_path = _task_files(spool, task.task_id)
    _safe_unlink(dead_path)
    _save_task_record(done_path, task, status="done", reason=reason)
    _safe_unlink(pending_path)


def _record_dead_task(*, spool: SpoolPaths, pending_path: Path, task: SpoolTask, reason: str) -> None:
    _, _, dead_path = _task_files(spool, task.task_id)
    _save_task_record(dead_path, task, status="dead", reason=reason)
    _safe_unlink(pending_path)


def _process_pending_tasks(
    *,
    spool: SpoolPaths,
    server: Server,
    auth_header: str,
    dry_run: bool,
    max_attempts: int,
    retry_wait_s: float,
    retry_backoff: float,
    retry_wait_max_s: float,
    process_limit: int,
) -> ProcessStats:
    stats = ProcessStats()
    pending_files = list(_iter_pending_task_files(spool))
    pending_with_tasks: list[Tuple[Path, SpoolTask]] = []
    for path in pending_files:
        try:
            task = _load_task(path)
        except Exception as exc:
            # Corrupted spool file should not block other tasks.
            now_iso = _utc_now_iso()
            broken = SpoolTask(
                task_id=path.stem,
                kind="invalid",
                session_uid="",
                session_id="",
                date="",
                session_dir="",
                source_path=str(path),
                source_fingerprint="",
                rel_path=None,
                payload=None,
                attempts=0,
                next_retry_ts=0.0,
                created_at_utc=now_iso,
                updated_at_utc=now_iso,
                last_error=str(exc),
            )
            _record_dead_task(spool=spool, pending_path=path, task=broken, reason="invalid_task_payload")
            stats.dead += 1
            continue
        pending_with_tasks.append((path, task))
    pending_with_tasks.sort(key=lambda item: (float(item[1].next_retry_ts), item[1].created_at_utc, item[1].task_id))
    now_ts = time.time()
    due_processed = 0
    for pending_path, task in pending_with_tasks:
        if task.next_retry_ts > now_ts:
            stats.deferred += 1
            continue
        if process_limit > 0 and due_processed >= process_limit:
            stats.skipped_limit += 1
            continue
        due_processed += 1
        stats.attempted += 1
        try:
            _execute_task(task=task, server=server, auth_header=auth_header, dry_run=dry_run)
        except RetryableUploadError as exc:
            task.attempts += 1
            task.updated_at_utc = _utc_now_iso()
            task.last_error = str(exc)
            if max_attempts > 0 and task.attempts >= max_attempts:
                _record_dead_task(spool=spool, pending_path=pending_path, task=task, reason="retry_exhausted")
                stats.dead += 1
                print(f"[{_utc_now_iso()}] dead {task.task_id}: {task.last_error}")
                continue
            wait_s = reconnect_wait_seconds(
                attempt=max(task.attempts, 1),
                base_wait_s=retry_wait_s,
                backoff=retry_backoff,
                wait_cap_s=retry_wait_max_s,
            )
            task.next_retry_ts = time.time() + wait_s
            _save_task(pending_path, task)
            stats.retry_scheduled += 1
            print(f"[{_utc_now_iso()}] retry {task.task_id} in {wait_s:.1f}s: {task.last_error}")
        except PermanentUploadError as exc:
            task.updated_at_utc = _utc_now_iso()
            task.last_error = str(exc)
            _record_dead_task(spool=spool, pending_path=pending_path, task=task, reason="permanent_error")
            stats.dead += 1
            print(f"[{_utc_now_iso()}] dead {task.task_id}: {task.last_error}")
        else:
            task.updated_at_utc = _utc_now_iso()
            task.last_error = ""
            _record_done_task(spool=spool, pending_path=pending_path, task=task)
            stats.succeeded += 1
    return stats


def _next_pending_retry_ts(spool: SpoolPaths) -> Optional[float]:
    due_values: list[float] = []
    for path in _iter_pending_task_files(spool):
        try:
            task = _load_task(path)
        except Exception:
            continue
        due_values.append(float(task.next_retry_ts))
    if not due_values:
        return None
    return min(due_values)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload completed SOP sessions to the review website (web MVP).")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data", help="Root containing sessions/YYYY-MM-DD.")
    parser.add_argument("--server", required=True, help="Website base URL, e.g. http://10.0.0.10:8000")
    parser.add_argument("--username", default=os.environ.get("SOP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.environ.get("SOP_ADMIN_PASSWORD"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD filter.")
    parser.add_argument(
        "--spool-dir",
        type=Path,
        default=None,
        help="Spool state directory (default: <data-dir>/uploader_spool).",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously enqueue + process with polling.")
    parser.add_argument("--poll-s", type=float, default=5.0, help="Watch mode poll interval in seconds.")
    parser.add_argument("--process-limit", type=int, default=256, help="Max due tasks processed per cycle (0=unlimited).")
    parser.add_argument("--max-attempts", type=int, default=0, help="Max retry attempts per task (0=retry forever).")
    parser.add_argument("--retry-wait-s", type=float, default=2.0, help="Base retry wait in seconds.")
    parser.add_argument(
        "--retry-wait-max-s",
        type=float,
        default=120.0,
        help="Max retry wait after backoff (0 = no cap).",
    )
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Retry backoff multiplier (>=1.0).")
    args = parser.parse_args(argv)

    if args.max_attempts < 0:
        raise SystemExit("--max-attempts must be >= 0")
    if args.retry_wait_s < 0.0:
        raise SystemExit("--retry-wait-s must be >= 0")
    if args.retry_backoff < 1.0:
        raise SystemExit("--retry-backoff must be >= 1.0")
    if args.retry_wait_max_s < 0.0:
        raise SystemExit("--retry-wait-max-s must be >= 0")
    if args.poll_s <= 0.0:
        raise SystemExit("--poll-s must be > 0")
    if args.process_limit < 0:
        raise SystemExit("--process-limit must be >= 0")
    if not args.password and not args.dry_run:
        raise SystemExit("Missing --password (or SOP_ADMIN_PASSWORD).")

    server = _parse_server(str(args.server))
    auth_header = _basic_auth(str(args.username), str(args.password or ""))
    spool = _resolve_spool_paths(data_dir=Path(args.data_dir), spool_dir=args.spool_dir)
    if not args.dry_run:
        _ensure_spool_dirs(spool)

    dead_total = 0
    cycle = 0
    while True:
        cycle += 1
        sessions = list(_iter_sessions(Path(args.data_dir)))
        if args.date:
            sessions = [s for s in sessions if s.date == str(args.date)]
        enqueue_stats = _enqueue_session_tasks(
            sessions=sessions,
            spool=spool,
            dry_run=bool(args.dry_run),
        )
        process_stats = _process_pending_tasks(
            spool=spool,
            server=server,
            auth_header=auth_header,
            dry_run=bool(args.dry_run),
            max_attempts=int(args.max_attempts),
            retry_wait_s=float(args.retry_wait_s),
            retry_backoff=float(args.retry_backoff),
            retry_wait_max_s=float(args.retry_wait_max_s),
            process_limit=int(args.process_limit),
        )
        dead_total += process_stats.dead

        pending_count = len(list(_iter_pending_task_files(spool)))
        print(
            f"[{_utc_now_iso()}] cycle={cycle} sessions={len(sessions)} "
            f"queued={enqueue_stats.queued} updated_pending={enqueue_stats.updated_pending} "
            f"skipped_done={enqueue_stats.skipped_done} skipped_pending={enqueue_stats.skipped_pending} "
            f"skipped_dead={enqueue_stats.skipped_dead} requeued_dead={enqueue_stats.requeued_dead} "
            f"attempted={process_stats.attempted} succeeded={process_stats.succeeded} "
            f"retry_scheduled={process_stats.retry_scheduled} deferred={process_stats.deferred} "
            f"skipped_limit={process_stats.skipped_limit} dead={process_stats.dead} pending={pending_count}"
        )
        if not args.watch:
            break

        next_ts = _next_pending_retry_ts(spool)
        if next_ts is None:
            sleep_s = float(args.poll_s)
        else:
            until_next_due = max(0.0, next_ts - time.time())
            sleep_s = min(float(args.poll_s), max(0.2, until_next_due))
        print(f"[{_utc_now_iso()}] watch sleep {sleep_s:.1f}s")
        time.sleep(sleep_s)

    print(f"[{_utc_now_iso()}] done: dead={dead_total}")
    return 1 if dead_total > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

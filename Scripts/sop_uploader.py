from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import quote, urlsplit


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
            raise RuntimeError(f"PUT {path} failed: HTTP {status}: {text}")
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
            raise RuntimeError(f"POST {path} failed: HTTP {status}: {text}")
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload completed SOP sessions to the review website (web MVP).")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data", help="Root containing sessions/YYYY-MM-DD.")
    parser.add_argument("--server", required=True, help="Website base URL, e.g. http://10.0.0.10:8000")
    parser.add_argument("--username", default=os.environ.get("SOP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.environ.get("SOP_ADMIN_PASSWORD"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD filter.")
    args = parser.parse_args(argv)

    if not args.password:
        raise SystemExit("Missing --password (or SOP_ADMIN_PASSWORD).")

    server = _parse_server(str(args.server))
    auth_header = _basic_auth(str(args.username), str(args.password))

    sessions = list(_iter_sessions(Path(args.data_dir)))
    if args.date:
        sessions = [s for s in sessions if s.date == str(args.date)]

    for s in sessions:
        checklist = _read_json(s.checklist_path)
        session_uid = _ensure_session_uid(checklist)
        _ensure_session_id(checklist, s.session_dir)
        _ensure_dates(checklist, s.date)

        # Persist the UID into the artifact for idempotency across retries and file moves.
        if not args.dry_run:
            _write_json(s.checklist_path, checklist)

        put_path = f"{server.base_path}/api/sessions/{session_uid}"
        print(f"[{_utc_now_iso()}] upsert session {session_uid} ({s.date}/{checklist.get('session_id')})")
        _http_put_json(
            server=server,
            path=put_path,
            auth_header=auth_header,
            payload=checklist,
            dry_run=bool(args.dry_run),
        )

        for artifact in _artifact_paths(s.session_dir):
            rel = artifact.relative_to(s.session_dir).as_posix()
            post_path = f"{server.base_path}/api/sessions/{session_uid}/artifacts?rel_path={quote(rel)}"
            print(f"[{_utc_now_iso()}] upload {session_uid}:{rel}")
            _http_post_file(
                server=server,
                path=post_path,
                auth_header=auth_header,
                file_path=artifact,
                dry_run=bool(args.dry_run),
            )

    print(f"[{_utc_now_iso()}] done: sessions={len(sessions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

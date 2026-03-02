from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class ReviewRecord:
    session_uid: str
    review_status: str
    review_note: str
    overrides: Dict[str, Any]
    created_at_utc: str
    updated_at_utc: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
              session_uid TEXT PRIMARY KEY,
              review_status TEXT NOT NULL,
              review_note TEXT NOT NULL,
              overrides_json TEXT NOT NULL,
              created_at_utc TEXT NOT NULL,
              updated_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_updated ON reviews(updated_at_utc)")


def get_review(db_path: Path, session_uid: str) -> Optional[ReviewRecord]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT session_uid, review_status, review_note, overrides_json, created_at_utc, updated_at_utc FROM reviews WHERE session_uid = ?",
            (session_uid,),
        ).fetchone()
        if row is None:
            return None
        overrides = json.loads(row["overrides_json"]) if row["overrides_json"] else {}
        return ReviewRecord(
            session_uid=str(row["session_uid"]),
            review_status=str(row["review_status"]),
            review_note=str(row["review_note"]),
            overrides=overrides if isinstance(overrides, dict) else {},
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )


def get_reviews_by_uid(db_path: Path, session_uids: Iterable[str]) -> Dict[str, ReviewRecord]:
    uids = list(session_uids)
    if not uids:
        return {}
    placeholders = ",".join("?" for _ in uids)
    query = (
        "SELECT session_uid, review_status, review_note, overrides_json, created_at_utc, updated_at_utc "
        f"FROM reviews WHERE session_uid IN ({placeholders})"
    )
    out: Dict[str, ReviewRecord] = {}
    with _connect(db_path) as conn:
        for row in conn.execute(query, tuple(uids)).fetchall():
            overrides = json.loads(row["overrides_json"]) if row["overrides_json"] else {}
            rec = ReviewRecord(
                session_uid=str(row["session_uid"]),
                review_status=str(row["review_status"]),
                review_note=str(row["review_note"]),
                overrides=overrides if isinstance(overrides, dict) else {},
                created_at_utc=str(row["created_at_utc"]),
                updated_at_utc=str(row["updated_at_utc"]),
            )
            out[rec.session_uid] = rec
    return out


def upsert_review(
    *,
    db_path: Path,
    session_uid: str,
    review_status: str,
    review_note: str,
    overrides: Dict[str, Any],
) -> ReviewRecord:
    now = _utc_now_iso()
    overrides_json = json.dumps(overrides or {}, sort_keys=True)
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT created_at_utc FROM reviews WHERE session_uid = ?",
            (session_uid,),
        ).fetchone()
        created = now if existing is None else str(existing["created_at_utc"])
        conn.execute(
            """
            INSERT INTO reviews (session_uid, review_status, review_note, overrides_json, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_uid) DO UPDATE SET
              review_status=excluded.review_status,
              review_note=excluded.review_note,
              overrides_json=excluded.overrides_json,
              updated_at_utc=excluded.updated_at_utc
            """,
            (session_uid, review_status, review_note, overrides_json, created, now),
        )
    return ReviewRecord(
        session_uid=session_uid,
        review_status=review_status,
        review_note=review_note,
        overrides=overrides or {},
        created_at_utc=created,
        updated_at_utc=now,
    )


from __future__ import annotations

import sqlite3
from pathlib import Path

from Action_Detection_SOP.web_mvp.review_store import get_review, get_reviews_by_uid, init_db, upsert_review


def test_scope_migration_preserves_existing_review_and_is_repeatable(tmp_path: Path) -> None:
    db = tmp_path / "reviews.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE reviews (session_uid TEXT PRIMARY KEY, review_status TEXT NOT NULL, "
                     "review_note TEXT NOT NULL, overrides_json TEXT NOT NULL, created_at_utc TEXT NOT NULL, "
                     "updated_at_utc TEXT NOT NULL)")
        conn.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)",
                     ("old", "QUALIFIED", "evidence checked", '{"labeled": "DONE"}', "created", "updated"))
    init_db(db)
    init_db(db)
    old = get_review(db, "old")
    assert old is not None
    assert (old.review_status, old.review_note, old.overrides) == (
        "QUALIFIED", "evidence checked", {"labeled": "DONE"})
    assert (old.created_at_utc, old.updated_at_utc, old.scope_reason) == ("created", "updated", None)


def test_scope_reason_round_trip_and_return_to_scored_review(tmp_path: Path) -> None:
    db = tmp_path / "reviews.sqlite3"
    init_db(db)
    excluded = upsert_review(db_path=db, session_uid="roll", review_status="OUT_OF_SCOPE",
                             review_note="passing", overrides={}, scope_reason="PASSING_THROUGH")
    assert get_review(db, "roll") == excluded
    assert get_reviews_by_uid(db, ["roll"])["roll"] == excluded
    approved = upsert_review(db_path=db, session_uid="roll", review_status="QUALIFIED",
                             review_note="corrected", overrides={"cleaned": "DONE"})
    assert approved.scope_reason is None
    assert approved.created_at_utc == excluded.created_at_utc
    assert get_review(db, "roll") == approved

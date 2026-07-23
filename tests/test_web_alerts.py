from __future__ import annotations

import json
from pathlib import Path

from Action_Detection_SOP.web_mvp.alert_index import AlertIndex
from Action_Detection_SOP.web_mvp.app import create_app
from Action_Detection_SOP.web_mvp.review_store import (
    get_alert_review,
    get_alert_reviews_by_uid,
    init_db,
    upsert_alert_review,
)
from Action_Detection_SOP.web_mvp.settings import WebMvpSettings


def _write_alert(root: Path, *, date: str, alert_uid: str, status: str = "PENDING") -> Path:
    alert_dir = root / "alerts" / date / alert_uid
    alert_dir.mkdir(parents=True, exist_ok=True)
    (alert_dir / "alert.json").write_text(
        json.dumps(
            {
                "alert_uid": alert_uid,
                "alert_type": "NO_HELMET",
                "safety_profile": "helmet_alert_v1",
                "status": status,
                "machine_status": "NO_HELMET",
                "start_date": date,
                "end_date": date,
                "start_time_s": 1.0,
                "end_time_s": 5.0,
                "person_count": 1,
                "thumbnail": "thumbnail.jpg",
            }
        ),
        encoding="utf-8",
    )
    (alert_dir / "thumbnail.jpg").write_bytes(b"fakejpg")
    return alert_dir


def test_alert_index_reads_direct_and_nested_alert_roots(tmp_path: Path) -> None:
    _write_alert(tmp_path, date="2026-06-29", alert_uid="alert_direct")
    _write_alert(tmp_path / "run_001", date="2026-06-30", alert_uid="alert_nested")

    index = AlertIndex(data_dir=tmp_path)
    index.refresh()

    direct = index.get("alert_direct")
    nested = index.get("alert_nested")
    assert direct is not None
    assert nested is not None
    assert direct.date == "2026-06-29"
    assert nested.date == "2026-06-30"
    assert direct.alert_type == "NO_HELMET"
    assert direct.paths.thumbnail_jpg.exists()
    assert {a.alert_uid for a in index.list()} == {"alert_direct", "alert_nested"}


def test_alert_review_store_is_independent_from_session_reviews(tmp_path: Path) -> None:
    db_path = tmp_path / "web.sqlite3"
    init_db(db_path)

    created = upsert_alert_review(
        db_path=db_path,
        alert_uid="alert_001",
        status="CONFIRMED",
        review_note="verified",
    )
    assert created.alert_uid == "alert_001"
    assert created.status == "CONFIRMED"

    updated = upsert_alert_review(
        db_path=db_path,
        alert_uid="alert_001",
        status="DISMISSED",
        review_note="false positive",
    )
    assert updated.created_at_utc == created.created_at_utc
    assert updated.updated_at_utc >= created.updated_at_utc

    fetched = get_alert_review(db_path, "alert_001")
    assert fetched is not None
    assert fetched.status == "DISMISSED"
    assert fetched.review_note == "false positive"

    by_uid = get_alert_reviews_by_uid(db_path, ["alert_001", "missing"])
    assert set(by_uid) == {"alert_001"}


def test_login_contract_accepts_json_request_body(tmp_path: Path) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    app = create_app(
        WebMvpSettings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "web.sqlite3",
            ui_dir=ui_dir,
            admin_username="admin",
            admin_password="secret",
        )
    )

    login_operation = app.openapi()["paths"]["/api/auth/login"]["post"]

    assert "requestBody" in login_operation
    assert not login_operation.get("parameters")

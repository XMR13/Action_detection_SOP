from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from Action_Detection_SOP.web_mvp.app import API_CONTRACT_VERSION, create_app
from Action_Detection_SOP.web_mvp.settings import WebMvpSettings


def _auth_headers(username: str = "admin", password: str = "secret") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _build_client(tmp_path: Path) -> TestClient:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    settings = WebMvpSettings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "reviews.sqlite3",
        ui_dir=ui_dir,
        admin_username="admin",
        admin_password="secret",
    )
    return TestClient(create_app(settings))


def _put_min_session(client: TestClient, *, session_uid: str, start_date: str = "2026-03-03") -> None:
    payload = {
        "session_uid": session_uid,
        "session_id": "s001",
        "start_date": start_date,
        "operator_present": "DONE",
        "roi_dwell": "DONE",
        "helmet": "UNKNOWN",
    }
    res = client.put(f"/api/sessions/{session_uid}", headers=_auth_headers(), json=payload)
    assert res.status_code == 200


def _post_artifact(client: TestClient, *, session_uid: str, rel_path: str, body: bytes) -> None:
    res = client.post(
        f"/api/sessions/{session_uid}/artifacts",
        headers=_auth_headers(),
        params={"rel_path": rel_path},
        content=body,
    )
    assert res.status_code == 200


def test_health_and_config_expose_contract_version(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["api_contract_version"] == API_CONTRACT_VERSION

        cfg = client.get("/api/config", headers=_auth_headers())
        assert cfg.status_code == 200
        assert cfg.json()["api_contract_version"] == API_CONTRACT_VERSION


def test_contract_endpoints_require_auth(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        res = client.get("/api/sessions")
        assert res.status_code == 401


def test_put_session_rejects_invalid_dates(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        invalid_calendar = {
            "session_uid": "uid_invalid_day",
            "session_id": "s001",
            "start_date": "2026-02-31",
        }
        res_invalid = client.put("/api/sessions/uid_invalid_day", headers=_auth_headers(), json=invalid_calendar)
        assert res_invalid.status_code == 400

        missing_dates = {
            "session_uid": "uid_missing_dates",
            "session_id": "s001",
        }
        res_missing = client.put("/api/sessions/uid_missing_dates", headers=_auth_headers(), json=missing_dates)
        assert res_missing.status_code == 400


def test_put_review_enforces_override_keys_and_values(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_min_session(client, session_uid="uid_review")

        bad_key = {
            "review_status": "QUALIFIED",
            "review_note": "",
            "overrides": {"unknown_step": "DONE"},
        }
        res_bad_key = client.put("/api/sessions/uid_review/review", headers=_auth_headers(), json=bad_key)
        assert res_bad_key.status_code == 400

        bad_value = {
            "review_status": "QUALIFIED",
            "review_note": "",
            "overrides": {"helmet": "INVALID"},
        }
        res_bad_value = client.put("/api/sessions/uid_review/review", headers=_auth_headers(), json=bad_value)
        assert res_bad_value.status_code == 400

        good_overrides = {
            "review_status": "QUALIFIED",
            "review_note": "checked",
            "overrides": {"helmet": "done", "roi_dwell": "unknown"},
        }
        res_good = client.put("/api/sessions/uid_review/review", headers=_auth_headers(), json=good_overrides)
        assert res_good.status_code == 200
        review = res_good.json()["review"]
        assert review["overrides"]["helmet"] == "DONE"
        assert review["overrides"]["roi_dwell"] == "UNKNOWN"


def test_artifact_upload_rejects_path_traversal(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_min_session(client, session_uid="uid_artifact")

        res = client.post(
            "/api/sessions/uid_artifact/artifacts",
            headers=_auth_headers(),
            params={"rel_path": "../escape.txt"},
            content=b"bad",
        )
        assert res.status_code == 400


def test_sessions_pagination_metadata_and_slicing(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        for idx in range(30):
            _put_min_session(client, session_uid=f"uid_page_{idx:03d}")

        page_1 = client.get("/api/sessions", headers=_auth_headers(), params={"page": 1, "page_size": 25})
        assert page_1.status_code == 200
        body_1 = page_1.json()
        assert body_1["total"] == 30
        assert body_1["page"] == 1
        assert body_1["page_size"] == 25
        assert body_1["total_pages"] == 2
        assert body_1["has_prev"] is False
        assert body_1["has_next"] is True
        assert len(body_1["sessions"]) == 25

        page_2 = client.get("/api/sessions", headers=_auth_headers(), params={"page": 2, "page_size": 25})
        assert page_2.status_code == 200
        body_2 = page_2.json()
        assert body_2["total"] == 30
        assert body_2["page"] == 2
        assert body_2["total_pages"] == 2
        assert body_2["has_prev"] is True
        assert body_2["has_next"] is False
        assert len(body_2["sessions"]) == 5


def test_sessions_evidence_filter_modes(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_min_session(client, session_uid="uid_thumb_only")
        _post_artifact(client, session_uid="uid_thumb_only", rel_path="thumbnail.jpg", body=b"fakejpg")

        _put_min_session(client, session_uid="uid_clip_only")
        _post_artifact(
            client,
            session_uid="uid_clip_only",
            rel_path="evidence.json",
            body=b'{"clips":[{"file":"evidence/c01.mp4"}]}',
        )

        _put_min_session(client, session_uid="uid_clip_thumb")
        _post_artifact(client, session_uid="uid_clip_thumb", rel_path="thumbnail.jpg", body=b"fakejpg")
        _post_artifact(
            client,
            session_uid="uid_clip_thumb",
            rel_path="evidence.json",
            body=b'{"clips":[{"file":"evidence/c01.mp4"}]}',
        )

        clip_thumb = client.get(
            "/api/sessions",
            headers=_auth_headers(),
            params={"evidence": "CLIP_THUMB", "page": 1, "page_size": 25},
        )
        assert clip_thumb.status_code == 200
        clip_thumb_uids = {str(s["session_uid"]) for s in clip_thumb.json()["sessions"]}
        assert clip_thumb_uids == {"uid_clip_thumb"}

        clip_only = client.get(
            "/api/sessions",
            headers=_auth_headers(),
            params={"evidence": "CLIP_ONLY", "page": 1, "page_size": 25},
        )
        assert clip_only.status_code == 200
        clip_only_uids = {str(s["session_uid"]) for s in clip_only.json()["sessions"]}
        assert clip_only_uids == {"uid_clip_only"}

        thumb_only = client.get(
            "/api/sessions",
            headers=_auth_headers(),
            params={"evidence": "THUMB_ONLY", "page": 1, "page_size": 25},
        )
        assert thumb_only.status_code == 200
        thumb_only_uids = {str(s["session_uid"]) for s in thumb_only.json()["sessions"]}
        assert thumb_only_uids == {"uid_thumb_only"}

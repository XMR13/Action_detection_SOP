from __future__ import annotations

import base64
import csv
import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional

import pytest

_HAS_HTTPX = importlib.util.find_spec("httpx") is not None
pytestmark = pytest.mark.skipif(not _HAS_HTTPX, reason="fastapi/starlette TestClient requires httpx")

if _HAS_HTTPX:
    from fastapi.testclient import TestClient
else:
    TestClient = object  # type: ignore[misc,assignment]

from Action_Detection_SOP.web_mvp.app import API_CONTRACT_VERSION, create_app
from Action_Detection_SOP.web_mvp.settings import WebMvpSettings


def _auth_headers(username: str = "admin", password: str = "secret") -> Dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _build_client(
    tmp_path: Path,
    *,
    disk_warning_used_pct: float = 75.0,
    disk_critical_used_pct: float = 85.0,
) -> TestClient:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    settings = WebMvpSettings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "reviews.sqlite3",
        ui_dir=ui_dir,
        admin_username="admin",
        admin_password="secret",
        disk_warning_used_pct=disk_warning_used_pct,
        disk_critical_used_pct=disk_critical_used_pct,
    )
    return TestClient(create_app(settings))


def _put_min_session(
    client: TestClient,
    *,
    session_uid: str,
    start_date: str = "2026-03-03",
    extra: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "session_uid": session_uid,
        "session_id": "s001",
        "start_date": start_date,
        "operator_present": "DONE",
        "roi_dwell": "DONE",
        "helmet": "UNKNOWN",
    }
    if extra:
        payload.update(extra)
    res = client.put(f"/api/sessions/{session_uid}", headers=_auth_headers(), json=payload)
    assert res.status_code == 200


def _put_roll_session(
    client: TestClient,
    *,
    session_uid: str,
    cleaned: str = "DONE",
    labeled: str = "DONE",
    overall_status: str = "SESUAI SOP",
) -> None:
    payload = {
        "session_uid": session_uid,
        "session_id": "roll001",
        "start_date": "2026-06-10",
        "sop_profile": "roll_sop_v1",
        "start_time_s": 1.0,
        "end_time_s": 6.0,
        "cleaned": cleaned,
        "labeled": labeled,
        "overall_status": overall_status,
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


def test_admin_ops_reports_spool_and_storage_state(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_min_session(client, session_uid="uid_ops")
        _post_artifact(client, session_uid="uid_ops", rel_path="thumbnail.jpg", body=b"fakejpg")
        _post_artifact(client, session_uid="uid_ops", rel_path="run_config.json", body=b'{"cfg":1}')
        _post_artifact(client, session_uid="uid_ops", rel_path="evidence.json", body=b'{"clips":[{"file":"evidence/c01.mp4"}]}')
        _post_artifact(client, session_uid="uid_ops", rel_path="evidence/c01.mp4", body=b"clipdata")

        data_dir = tmp_path / "data"
        spool_pending = data_dir / "uploader_spool" / "pending"
        spool_dead = data_dir / "uploader_spool" / "dead"
        spool_root = data_dir / "uploader_spool"
        spool_pending.mkdir(parents=True, exist_ok=True)
        spool_dead.mkdir(parents=True, exist_ok=True)
        (spool_pending / "task_a.json").write_text("{}", encoding="utf-8")
        (spool_dead / "task_b.json").write_text("{}", encoding="utf-8")
        (spool_root / "state.json").write_text(
            json.dumps(
                {
                    "generated_at_utc": "2026-03-06T09:00:00+00:00",
                    "watch_mode": True,
                    "cycle": 7,
                    "last_success_utc": "2026-03-06T09:00:00+00:00",
                    "last_dead_utc": None,
                }
            ),
            encoding="utf-8",
        )

        cache_dir = data_dir / "_web_cache" / "transcoded" / "uid_ops"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "clip.mp4").write_bytes(b"fakeclip")

        ops = client.get("/api/admin/ops", headers=_auth_headers())
        assert ops.status_code == 200
        payload = ops.json()
        assert payload["status"] == "ok"
        assert payload["session_count"] == 1
        assert payload["database"]["exists"] is True
        assert payload["uploader_spool"]["pending"]["files"] == 1
        assert payload["uploader_spool"]["dead"]["files"] == 1
        assert payload["uploader_spool"]["pending_retry"]["ready_now_files"] == 1
        assert payload["uploader_spool"]["state_file"]["exists"] is True
        assert payload["uploader_spool"]["state_file"]["watch_mode"] is True
        assert payload["uploader_spool"]["health"]["status"] == "error"
        assert "dead_tasks_present" in payload["uploader_spool"]["health"]["issues"]
        assert payload["disk"]["health"]["status"] in {"ok", "warning", "critical"}
        assert "used_pct" in payload["disk"]["health"]
        assert "free_pct" in payload["disk"]["health"]
        assert payload["settings"]["disk_warning_used_pct"] == 75.0
        assert payload["settings"]["disk_critical_used_pct"] == 85.0
        assert payload["cache"]["files"] == 1
        assert payload["reports"]["path"].endswith("/data/reports")
        assert payload["managed_storage"]["total_bytes"] > 0
        assert payload["managed_storage"]["percent_of_disk"] >= 0.0
        assert payload["managed_storage"]["total_files"] >= 6
        assert payload["managed_storage"]["sessions"]["categories"]["thumbnails"]["files"] == 1
        assert payload["managed_storage"]["sessions"]["categories"]["run_configs"]["files"] == 1
        assert payload["managed_storage"]["sessions"]["categories"]["evidence_manifests"]["files"] == 1
        assert payload["managed_storage"]["sessions"]["categories"]["evidence_clips"]["files"] == 1


def test_admin_ops_reports_critical_disk_health_when_threshold_is_crossed(tmp_path: Path) -> None:
    with _build_client(tmp_path, disk_warning_used_pct=0.0, disk_critical_used_pct=0.0) as client:
        ops = client.get("/api/admin/ops", headers=_auth_headers())
        assert ops.status_code == 200
        payload = ops.json()
        assert payload["disk"]["health"]["status"] == "critical"
        assert "disk_used_pct_critical" in payload["disk"]["health"]["issues"]

        storage = client.get("/api/admin/storage", headers=_auth_headers())
        assert storage.status_code == 200
        assert storage.json()["disk_health"]["status"] == "critical"


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


def test_roll_session_api_exposes_structured_sop_and_auto_approves_with_evidence(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_roll_session(client, session_uid="uid_roll_done")
        _post_artifact(client, session_uid="uid_roll_done", rel_path="thumbnail.jpg", body=b"fakejpg")

        rows = client.get("/api/sessions", headers=_auth_headers(), params={"page": 1, "page_size": 25})
        assert rows.status_code == 200
        row = rows.json()["sessions"][0]
        assert row["session_uid"] == "uid_roll_done"
        assert row["machine_sop"] == "DONE"
        assert row["final_sop"] == "DONE"
        assert row["review_status"] == "QUALIFIED"
        assert row["review_source"] == "AUTO"
        assert row["machine_helmet"] == "UNKNOWN"
        assert row["sop"]["profile"] == "roll_sop_v1"
        assert row["sop"]["machine"]["cleaned"] == "DONE"
        assert row["sop"]["machine"]["labeled"] == "DONE"
        assert row["sop"]["machine"]["overall_status"] == "SESUAI SOP"
        assert row["sop"]["machine"]["labels"]["cleaned"] == "Sudah dibersihkan"
        assert row["sop"]["inconsistent"] is False

        detail = client.get("/api/sessions/uid_roll_done", headers=_auth_headers())
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["sop"]["final"]["overall_status"] == "SESUAI SOP"
        assert payload["auto_review_reason"] == "roll_policy_pass"


def test_roll_sessions_do_not_count_as_unknown_helmet_stats(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_roll_session(client, session_uid="uid_roll_stats")
        _post_artifact(client, session_uid="uid_roll_stats", rel_path="thumbnail.jpg", body=b"fakejpg")

        stats = client.get("/api/stats", headers=_auth_headers())
        assert stats.status_code == 200
        payload = stats.json()
        assert payload["total_sessions"] == 1
        assert payload["helmet_session_count"] == 0
        assert payload["machine_helmet_unknown"] == 0
        assert payload["final_helmet_unknown"] == 0
        assert payload["unknown"] == 0
        assert payload["final_sop_done"] == 1

def test_roll_review_allows_step_and_overall_overrides(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_roll_session(
            client,
            session_uid="uid_roll_review",
            cleaned="DONE",
            labeled="NOT_DONE",
            overall_status="TIDAK SESUAI SOP",
        )

        step_override = {
            "review_status": "QUALIFIED",
            "review_note": "label was visible in evidence",
            "overrides": {"labeled": "DONE"},
        }
        res_step = client.put("/api/sessions/uid_roll_review/review", headers=_auth_headers(), json=step_override)
        assert res_step.status_code == 200
        detail_step = client.get("/api/sessions/uid_roll_review", headers=_auth_headers())
        assert detail_step.status_code == 200
        sop_step = detail_step.json()["sop"]
        assert sop_step["final"]["labeled"] == "DONE"
        assert sop_step["final"]["overall_status"] == "SESUAI SOP"
        assert sop_step["final"]["status"] == "DONE"

        overall_override = {
            "review_status": "NOT_QUALIFIED",
            "review_note": "operator says result differs from evidence",
            "overrides": {"labeled": "DONE", "overall_status": "TIDAK SESUAI SOP"},
        }
        res_overall = client.put("/api/sessions/uid_roll_review/review", headers=_auth_headers(), json=overall_override)
        assert res_overall.status_code == 200
        detail_overall = client.get("/api/sessions/uid_roll_review", headers=_auth_headers())
        assert detail_overall.status_code == 200
        sop_overall = detail_overall.json()["sop"]
        assert sop_overall["final"]["labeled"] == "DONE"
        assert sop_overall["final"]["overall_status"] == "TIDAK SESUAI SOP"
        assert sop_overall["final"]["status"] == "NOT_DONE"


def test_roll_review_rejects_legacy_keys_and_noncanonical_overall_status(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_roll_session(client, session_uid="uid_roll_strict")

        bad_key = {
            "review_status": "QUALIFIED",
            "review_note": "",
            "overrides": {"helmet": "DONE"},
        }
        res_bad_key = client.put("/api/sessions/uid_roll_strict/review", headers=_auth_headers(), json=bad_key)
        assert res_bad_key.status_code == 400

        bad_overall = {
            "review_status": "QUALIFIED",
            "review_note": "",
            "overrides": {"overall_status": "DONE"},
        }
        res_bad_overall = client.put("/api/sessions/uid_roll_strict/review", headers=_auth_headers(), json=bad_overall)
        assert res_bad_overall.status_code == 400


def test_put_roll_session_rejects_noncanonical_overall_status(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        payload = {
            "session_uid": "uid_roll_bad_overall",
            "session_id": "roll001",
            "start_date": "2026-06-10",
            "sop_profile": "roll_sop_v1",
            "cleaned": "DONE",
            "labeled": "DONE",
            "overall_status": "DONE",
        }
        res = client.put("/api/sessions/uid_roll_bad_overall", headers=_auth_headers(), json=payload)
        assert res.status_code == 400


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


def test_sessions_csv_export_uses_filters_but_not_pagination(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        for idx in range(3):
            _put_min_session(
                client,
                session_uid=f"uid_csv_match_{idx}",
                start_date="2026-06-19",
                extra={
                    "start_time_iso": f"2026-06-19T08:0{idx}:00+00:00",
                    "end_time_iso": f"2026-06-19T08:0{idx}:30+00:00",
                    "start_time_s": 10.0,
                    "end_time_s": 40.0,
                    "shift_id": "S1",
                    "shift_name": "Shift 1",
                },
            )
        _put_min_session(client, session_uid="uid_csv_other_date", start_date="2026-06-20")

        res = client.get(
            "/api/sessions/export.csv",
            headers=_auth_headers(),
            params={
                "date": "2026-06-19",
                "review_status": "PENDING",
                "shift": "S1",
                "page": 1,
                "page_size": 1,
            },
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/csv")
        assert "sop_review_queue.csv" in res.headers["content-disposition"]

        rows = list(csv.DictReader(res.text.splitlines()))
        assert {row["session_uid"] for row in rows} == {
            "uid_csv_match_0",
            "uid_csv_match_1",
            "uid_csv_match_2",
        }
        assert {row["duration_s"] for row in rows} == {"30.0"}
        assert all(row["shift"] == "Shift 1" for row in rows)
        assert all(row["review_status"] == "PENDING" for row in rows)


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


def test_sessions_shift_filter_modes(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        _put_min_session(client, session_uid="uid_shift_1", extra={"shift_id": "S1", "shift_name": "Shift 1"})
        _put_min_session(client, session_uid="uid_shift_2", extra={"shift_id": "S2", "shift_name": "Shift 2"})
        _put_min_session(client, session_uid="uid_shift_3", extra={"shift_id": "S3", "shift_name": "Shift 3"})

        all_rows = client.get("/api/sessions", headers=_auth_headers(), params={"shift": "ALL", "page": 1, "page_size": 25})
        assert all_rows.status_code == 200
        all_payload = all_rows.json()
        assert all_payload["shift"] == "ALL"
        all_uids = {str(s["session_uid"]) for s in all_payload["sessions"]}
        assert all_uids == {"uid_shift_1", "uid_shift_2", "uid_shift_3"}

        s1_rows = client.get("/api/sessions", headers=_auth_headers(), params={"shift": "S1", "page": 1, "page_size": 25})
        assert s1_rows.status_code == 200
        s1_payload = s1_rows.json()
        assert s1_payload["shift"] == "S1"
        s1_uids = {str(s["session_uid"]) for s in s1_payload["sessions"]}
        assert s1_uids == {"uid_shift_1"}

        s2_rows = client.get("/api/sessions", headers=_auth_headers(), params={"shift": "2", "page": 1, "page_size": 25})
        assert s2_rows.status_code == 200
        s2_payload = s2_rows.json()
        assert s2_payload["shift"] == "S2"
        s2_uids = {str(s["session_uid"]) for s in s2_payload["sessions"]}
        assert s2_uids == {"uid_shift_2"}

"""Exercise scope reviews over real HTTP using the existing server dependencies."""
from __future__ import annotations

import base64
import csv
import io
import json
import socket
import threading
import time
from datetime import datetime, timedelta
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from Action_Detection_SOP.web_mvp.app import create_app
from Action_Detection_SOP.web_mvp.settings import WebMvpSettings
from Action_Detection_SOP.web_mvp.review_store import upsert_review

HttpRequest = Callable[..., tuple[int, bytes]]


@pytest.fixture
def request_api(tmp_path: Path) -> Iterator[HttpRequest]:
    settings = WebMvpSettings(data_dir=tmp_path / "data", db_path=tmp_path / "reviews.sqlite3",
                              ui_dir=tmp_path / "ui", admin_password="secret")
    settings.ui_dir.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="error"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started, "Local API did not start"
            root = f"http://127.0.0.1:{sock.getsockname()[1]}"
            auth = base64.b64encode(b"admin:secret").decode("ascii")

            def request(method: str, path: str, payload: dict | None = None) -> tuple[int, bytes]:
                body = None if payload is None else json.dumps(payload).encode()
                req = Request(root + path, data=body, method=method,
                              headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"})
                try:
                    with urlopen(req, timeout=5) as response:
                        return response.status, response.read()
                except HTTPError as response:
                    return response.code, response.read()

            yield request
        finally:
            server.should_exit = True
            worker.join(timeout=5)
            assert not worker.is_alive(), "Local API did not stop"


def _put_roll(request: HttpRequest, uid: str, cleaned: str = "UNKNOWN", labeled: str = "UNKNOWN") -> None:
    code, body = request("PUT", f"/api/sessions/{uid}", {
        "session_uid": uid, "session_id": uid, "start_date": "2026-09-30",
        "sop_profile": "roll_sop_v1", "start_time_s": 100, "end_time_s": 145,
        "cleaned": cleaned, "labeled": labeled,
        "overall_status": "SESUAI SOP" if cleaned == labeled == "DONE" else "TIDAK SESUAI SOP" if "NOT_DONE" in (cleaned, labeled) else "UNKNOWN",
    })
    assert code == 200, body.decode()


def _json(request: HttpRequest, path: str) -> dict:
    code, body = request("GET", path)
    assert code == 200
    return json.loads(body)


def _put_roll_at(request: HttpRequest, uid: str, timestamp: str, **extra: object) -> None:
    start = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    code, body = request("PUT", f"/api/sessions/{uid}", {
        "session_uid": uid, "session_id": uid, "start_date": start.date().isoformat(),
        "sop_profile": "roll_sop_v1", "start_time_s": 0, "end_time_s": 60,
        "start_time_iso": timestamp, "end_time_iso": (start + timedelta(minutes=1)).isoformat(),
        "cleaned": "UNKNOWN", "labeled": "UNKNOWN", "overall_status": "UNKNOWN", **extra,
    })
    assert code == 200, body.decode()


def test_shift_date_consistent_across_queue_detail_stats_and_csv(request_api: HttpRequest) -> None:
    # Simulate an old record whose saved shift date incorrectly used its calendar date.
    _put_roll_at(request_api, "overnight", "2026-10-04T03:00:00",
                 shift_id="S1", shift_date="2026-10-04", shift_name="Shift 1")
    _put_roll_at(request_api, "daytime", "2026-10-04T08:00:00+07:00")
    _put_roll_at(request_api, "late", "2026-10-03T23:30:00+07:00")
    query = "/api/sessions?date=2026-10-03&shift=S3&sort=OLDEST&page_size=1"
    first = _json(request_api, query)
    assert first["total"] == 2
    assert first["sessions"][0]["session_uid"] == "late"
    row = _json(request_api, query + "&page=2")["sessions"][0]
    assert (row["session_uid"], row["date"], row["shift_id"]) == ("overnight", "2026-10-03", "S3")
    assert row["shift_date"] == "2026-10-03"
    assert row["storage_date"] == "2026-10-04"
    assert _json(request_api, "/api/stats?date=2026-10-03")["total_sessions"] == 2
    assert _json(request_api, "/api/stats?date=2026-10-04")["total_sessions"] == 1
    assert _json(request_api, "/api/sessions?date=2026-10-04&shift=S3")["total"] == 0
    for query in ("date_from=2026-10-03&date_to=2026-10-03", "date_to=2026-10-03"):
        assert _json(request_api, "/api/sessions?" + query)["total"] == 2
    assert _json(request_api, "/api/sessions?date_from=2026-10-04")["total"] == 1
    detail = _json(request_api, "/api/sessions/overnight")
    assert (detail["date"], detail["shift_date"], detail["shift_id"]) == ("2026-10-03", "2026-10-03", "S3")
    assert detail["checklist"]["start_time_iso"] == "2026-10-04T03:00:00"
    assert detail["storage_date"] == "2026-10-04"
    code, body = request_api("GET", "/api/sessions/export.csv?date=2026-10-03&shift=S3")
    assert code == 200
    rows = list(csv.DictReader(io.StringIO(body.decode())))
    assert {row["session_uid"] for row in rows} == {"late", "overnight"}
    row = next(row for row in rows if row["session_uid"] == "overnight")
    assert (row["date"], row["shift"], row["start_time"]) == ("2026-10-03", "Shift 3", "2026-10-04T03:00:00")


def test_website_shift_boundaries_and_utc_records(request_api: HttpRequest) -> None:
    cases = [
        ("morning_before", "2026-10-04T07:29:59+07:00", "S3", "2026-10-03"),
        ("morning_at", "2026-10-04T07:30:00+07:00", "S1", "2026-10-04"),
        ("afternoon_before", "2026-10-04T15:29:59+07:00", "S1", "2026-10-04"),
        ("afternoon_at", "2026-10-04T15:30:00+07:00", "S2", "2026-10-04"),
        ("night_before", "2026-10-04T23:29:59+07:00", "S2", "2026-10-04"),
        ("night_at", "2026-10-04T23:30:00+07:00", "S3", "2026-10-04"),
        ("utc", "2026-10-03T20:00:00Z", "S3", "2026-10-03"),
    ]
    for uid, stamp, shift_id, shift_date in cases:
        # Point records verify the exact boundary without interval-overlap effects.
        _put_roll_at(request_api, uid, stamp, end_time_iso=stamp)
        rows = _json(request_api, f"/api/sessions?date={shift_date}&shift={shift_id}")["sessions"]
        assert uid in {row["session_uid"] for row in rows}
        detail = _json(request_api, f"/api/sessions/{uid}")
        assert (detail["shift_id"], detail["date"]) == (shift_id, shift_date)


def test_timestamp_free_records_keep_saved_shift_or_storage_date(request_api: HttpRequest) -> None:
    _put_roll(request_api, "storage_fallback")
    _put_roll_at(request_api, "saved_shift", "2026-10-04T03:00:00",
                 start_time_iso=None, end_time_iso=None,
                 shift_id="S3", shift_date="2026-10-03", shift_name="Shift 3")
    _put_roll_at(request_api, "invalid_shift", "2026-10-04T03:00:00",
                 start_time_iso=None, end_time_iso=None, shift_date="2026-02-30")
    assert _json(request_api, "/api/sessions?date=2026-09-30")["total"] == 1
    assert _json(request_api, "/api/sessions?date=2026-10-03&shift=S3")["total"] == 1
    assert _json(request_api, "/api/sessions/invalid_shift")["date"] == "2026-10-04"


def test_mixed_naive_and_utc_timestamps_sort_in_actual_wib_order(request_api: HttpRequest) -> None:
    _put_roll_at(request_api, "earlier", "2026-10-04T03:00:00")
    _put_roll_at(request_api, "later", "2026-10-03T20:05:00Z")  # 4 Oct 03:05 WIB
    rows = _json(request_api, "/api/sessions?date=2026-10-03&sort=OLDEST")["sessions"]
    assert [row["session_uid"] for row in rows] == ["earlier", "later"]


def test_helmet_alerts_follow_same_shift_date_and_keep_media_paths(request_api: HttpRequest) -> None:
    code, body = request_api("PUT", "/api/alerts/overnight_alert", {
        "alert_uid": "overnight_alert", "alert_type": "NO_HELMET",
        "start_date": "2026-10-04", "start_time_iso": "2026-10-04T03:00:00+07:00",
        "end_time_iso": "2026-10-04T03:00:10+07:00",
    })
    assert code == 200, body.decode()
    assert request_api("POST", "/api/alerts/overnight_alert/artifacts?rel_path=thumbnail.jpg", {})[0] == 200
    rows = _json(request_api, "/api/alerts?date_from=2026-10-03&date_to=2026-10-03")["alerts"]
    assert len(rows) == 1
    assert (rows[0]["date"], rows[0]["shift_id"], rows[0]["storage_date"]) == ("2026-10-03", "S3", "2026-10-04")
    assert _json(request_api, "/api/alerts?date=2026-10-04")["total"] == 0
    detail = _json(request_api, "/api/alerts/overnight_alert")
    assert detail["date"] == detail["shift_date"] == "2026-10-03"
    assert detail["alert"]["start_time_iso"] == "2026-10-04T03:00:00+07:00"
    assert request_api("GET", detail["thumbnail_url"])[0] == 200
    code, body = request_api("GET", "/api/alerts/export.csv?date=2026-10-03")
    assert code == 200
    row = next(csv.DictReader(io.StringIO(body.decode())))
    assert (row["date"], row["shift_date"], row["storage_date"]) == ("2026-10-03", "2026-10-03", "2026-10-04")


def test_next_review_excludes_archives_before_pagination(
    request_api: HttpRequest, tmp_path: Path,
) -> None:
    archive = tmp_path / "data" / "sessions" / "2026-09-30" / "archive"
    archive.mkdir(parents=True)
    (archive / "checklist.json").write_text(json.dumps({
        "session_uid": "archive", "session_id": "old", "sop_profile": "operator_mvp_a",
        "operator_present": "DONE", "roi_dwell": "DONE", "helmet": "DONE",
        "start_time_iso": "2026-09-30T12:00:00",
    }))
    _put_roll(request_api, "roll")

    query = "/api/sessions?operator_verdict=NEEDS_REVIEW&sort=NEWEST&limit=1"
    # The ordinary list keeps archives readable, and the archive sorts first.
    assert _json(request_api, query)["sessions"][0]["session_uid"] == "archive"
    pending = _json(request_api, query + "&reviewable_only=true")
    assert pending["total"] == 1
    assert [row["session_uid"] for row in pending["sessions"]] == ["roll"]

    assert request_api("PUT", "/api/sessions/roll/review", {
        "review_status": "QUALIFIED", "review_note": "checked",
        "overrides": {"cleaned": "DONE", "labeled": "DONE"},
    })[0] == 200
    assert _json(request_api, query + "&reviewable_only=true")["sessions"] == []
    assert _json(request_api, "/api/sessions/archive")["sop"]["read_only"] is True


def test_archived_operator_evidence_is_readable_but_reviews_and_ingest_are_blocked(
    request_api: HttpRequest, tmp_path: Path,
) -> None:
    session_dir = tmp_path / "data" / "sessions" / "2026-09-30" / "archived"
    session_dir.mkdir(parents=True)
    checklist = {
        "session_uid": "archived", "session_id": "old1", "start_date": "2026-09-30",
        "sop_profile": "operator_mvp_a", "start_time_s": 0, "end_time_s": 100,
        "operator_present": "DONE", "roi_dwell": "DONE", "helmet": "UNKNOWN",
    }
    checklist_path = session_dir / "checklist.json"
    checklist_path.write_text(json.dumps(checklist))
    thumbnail = session_dir / "thumbnail.jpg"
    thumbnail.write_bytes(b"archived-picture")
    upsert_review(db_path=tmp_path / "reviews.sqlite3", session_uid="archived",
                  review_status="QUALIFIED", review_note="old review", overrides={"helmet": "DONE"})
    assert request_api("POST", "/api/admin/rescan")[0] == 200

    detail = _json(request_api, "/api/sessions/archived")
    assert detail["sop"]["read_only"] is True
    assert detail["sop"]["final"]["helmet"] == "DONE"
    assert detail["review_status"] == "QUALIFIED"
    assert request_api("GET", detail["thumbnail_url"])[1] == b"archived-picture"
    assert request_api("PUT", "/api/sessions/archived/review", {
        "review_status": "NOT_QUALIFIED", "review_note": "new review",
    })[0] == 400
    assert _json(request_api, "/api/sessions/archived")["review"]["review_note"] == "old review"

    original = checklist_path.read_bytes()
    assert request_api("PUT", "/api/sessions/archived", checklist)[0] == 400
    roll_payload = {**checklist, "sop_profile": "roll_sop_v1", "cleaned": "DONE",
                    "labeled": "DONE", "overall_status": "SESUAI SOP"}
    assert request_api("PUT", "/api/sessions/archived", roll_payload)[0] == 400
    for rel_path in ("thumbnail.jpg", "checklist.json"):
        assert request_api("POST", f"/api/sessions/archived/artifacts?rel_path={rel_path}", roll_payload)[0] == 400
    assert checklist_path.read_bytes() == original
    assert thumbnail.read_bytes() == b"archived-picture"

    # An unreviewed archive must stay pending even with DONE steps and evidence.
    unchecked = session_dir.with_name("unchecked")
    unchecked.mkdir()
    (unchecked / "checklist.json").write_text(json.dumps({
        **checklist, "session_uid": "unchecked", "helmet": "DONE",
    }))
    (unchecked / "thumbnail.jpg").write_bytes(b"picture")
    assert request_api("POST", "/api/admin/rescan")[0] == 200
    assert _json(request_api, "/api/sessions/unchecked")["review_source"] == "PENDING"


def test_new_operator_upload_is_rejected_before_storage_changes(
    request_api: HttpRequest, tmp_path: Path,
) -> None:
    code, body = request_api("PUT", "/api/sessions/removed", {
        "session_uid": "removed", "session_id": "old", "start_date": "2026-09-30",
        "operator_present": "DONE", "roi_dwell": "DONE", "helmet": "DONE",
    })
    assert code == 400
    assert "archived and read-only" in body.decode()
    assert not (tmp_path / "data" / "sessions" / "2026-09-30" / "removed").exists()


def test_roll_upload_cannot_replace_archive_before_rescan(
    request_api: HttpRequest, tmp_path: Path,
) -> None:
    path = tmp_path / "data" / "sessions" / "2026-09-30" / "unscanned" / "checklist.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "session_uid": "unscanned", "session_id": "old", "operator_present": "DONE",
    }))
    original = path.read_bytes()
    assert request_api("PUT", "/api/sessions/unscanned", {
        "session_uid": "unscanned", "session_id": "new", "start_date": "2026-09-30",
        "sop_profile": "roll_sop_v1", "cleaned": "DONE", "labeled": "DONE",
        "overall_status": "SESUAI SOP",
    })[0] == 400
    assert path.read_bytes() == original


def test_artifact_upload_cannot_bypass_roll_metadata_validation(request_api: HttpRequest) -> None:
    _put_roll(request_api, "roll")
    assert request_api("POST", "/api/sessions/roll/artifacts?rel_path=checklist.json", {
        "session_uid": "roll", "operator_present": "DONE",
    })[0] == 400
    assert _json(request_api, "/api/sessions/roll")["sop"]["profile"] == "roll_sop_v1"
    assert request_api("POST", "/api/sessions/roll/artifacts?rel_path=thumbnail.jpg", {})[0] == 200


def test_passing_roll_resolves_review_and_stays_out_of_compliance(request_api: HttpRequest) -> None:
    request = request_api
    _put_roll(request, "passing")
    payload = {"review_status": "OUT_OF_SCOPE", "scope_reason": "PASSING_THROUGH", "review_note": "=passing"}
    for _ in range(2):
        assert request("PUT", "/api/sessions/passing/review", payload)[0] == 200
    detail = _json(request, "/api/sessions/passing")
    assert detail["operator_verdict"] == "OUT_OF_SCOPE"
    assert detail["machine_sop"] == detail["final_sop"] == "UNKNOWN"
    assert detail["scope_reason"] == "PASSING_THROUGH"
    assert detail["checklist"]["end_time_s"] == 145
    assert _json(request, "/api/sessions?operator_verdict=NEEDS_REVIEW")["sessions"] == []
    excluded = _json(request, "/api/sessions?operator_verdict=OUT_OF_SCOPE")["sessions"]
    assert [row["session_uid"] for row in excluded] == ["passing"]
    assert _json(request, "/api/sessions?review_status=OUT_OF_SCOPE")["total"] == 1
    stats = _json(request, "/api/stats")
    assert (stats["out_of_scope"], stats["pending"], stats["verdict_needs_review"]) == (1, 0, 0)
    assert stats["review_completion_pct"] == 100
    assert stats["compliance_pct"] is None
    assert stats["final_sop_unknown"] == 0
    assert stats["machine_sop_unknown"] == 1
    code, body = request("GET", "/api/sessions/export.csv?operator_verdict=OUT_OF_SCOPE")
    assert code == 200
    row = list(csv.DictReader(io.StringIO(body.decode())))[0]
    assert row["scope_reason"] == "PASSING_THROUGH"
    assert row["operator_verdict"] == "OUT_OF_SCOPE"
    assert row["review_note"] == "'=passing"
    # An uploader metadata retry cannot overwrite the human exclusion.
    _put_roll(request, "passing", "DONE", "DONE")
    assert _json(request, "/api/sessions/passing")["operator_verdict"] == "OUT_OF_SCOPE"
    # Returning to a scored review still checks the corrected final SOP.
    assert request("PUT", "/api/sessions/passing/review", {"review_status": "NOT_QUALIFIED"})[0] == 400
    assert request("PUT", "/api/sessions/passing/review", {"review_status": "QUALIFIED"})[0] == 200
    assert _json(request, "/api/sessions/passing")["scope_reason"] is None
    assert _json(request, "/api/stats")["compliance_pct"] == 100


def test_compliance_uses_scored_decisions_only(request_api: HttpRequest) -> None:
    request = request_api
    for uid, step in [("approved", "DONE"), ("rejected", "NOT_DONE"), ("pending", "UNKNOWN"), ("excluded", "DONE")]:
        _put_roll(request, uid, step, step)
    for uid, payload in [
        ("approved", {"review_status": "QUALIFIED"}),
        ("rejected", {"review_status": "NOT_QUALIFIED"}),
        ("excluded", {"review_status": "OUT_OF_SCOPE", "scope_reason": "ALREADY_WRAPPED"}),
    ]:
        assert request("PUT", f"/api/sessions/{uid}/review", payload)[0] == 200
    stats = _json(request, "/api/stats")
    assert (stats["verdict_done"], stats["verdict_not_done"], stats["verdict_needs_review"], stats["out_of_scope"]) == (1, 1, 1, 1)
    assert stats["compliance_pct"] == stats["reviewed_final_sop_done_pct"] == 50
    assert stats["review_completion_pct"] == 75
    assert stats["final_sop_unknown_pct"] == pytest.approx(100 / 3)
    assert stats["final_sop_done"] == 1
    assert _json(request, "/api/stats?date=2026-09-29")["compliance_pct"] is None


@pytest.mark.parametrize("payload,expected", [
    ({"review_status": "OUT_OF_SCOPE"}, 400),
    ({"review_status": "OUT_OF_SCOPE", "scope_reason": "invalid"}, 422),
    ({"review_status": "OUT_OF_SCOPE", "scope_reason": "OTHER", "review_note": "  "}, 400),
    ({"review_status": "QUALIFIED", "scope_reason": "PASSING_THROUGH"}, 400),
])
def test_invalid_scope_decision_does_not_write_review(request_api: HttpRequest, payload: dict, expected: int) -> None:
    _put_roll(request_api, "invalid")
    assert request_api("PUT", "/api/sessions/invalid/review", payload)[0] == expected
    assert _json(request_api, "/api/sessions/invalid")["review"] is None


def test_other_reason_requires_explanation(request_api: HttpRequest) -> None:
    _put_roll(request_api, "other")
    assert request_api("PUT", "/api/sessions/other/review", {
        "review_status": "OUT_OF_SCOPE", "scope_reason": "OTHER", "review_note": "Oversize bypass roll",
    })[0] == 200
    assert _json(request_api, "/api/sessions/other")["review"]["review_note"] == "Oversize bypass roll"


def test_old_conflicting_review_stays_out_of_compliance(request_api: HttpRequest, tmp_path: Path) -> None:
    _put_roll(request_api, "conflicting", "DONE", "DONE")
    upsert_review(db_path=tmp_path / "reviews.sqlite3", session_uid="conflicting",
                  review_status="NOT_QUALIFIED", review_note="old decision", overrides={})
    stats = _json(request_api, "/api/stats")
    assert stats["verdict_needs_review"] == 1
    assert stats["review_completion_pct"] == 0
    assert stats["compliance_pct"] is None


def test_exclusion_preserves_prior_corrections_when_omitted(request_api: HttpRequest) -> None:
    _put_roll(request_api, "corrected", "DONE", "UNKNOWN")
    assert request_api("PUT", "/api/sessions/corrected/review", {
        "review_status": "QUALIFIED", "overrides": {"labeled": "DONE"},
    })[0] == 200
    assert request_api("PUT", "/api/sessions/corrected/review", {
        "review_status": "OUT_OF_SCOPE", "scope_reason": "ALREADY_WRAPPED",
    })[0] == 200
    detail = _json(request_api, "/api/sessions/corrected")
    assert detail["review"]["overrides"] == {"labeled": "DONE"}
    assert detail["machine_sop"] == "UNKNOWN"
    assert detail["final_sop"] == "DONE"
    assert detail["operator_verdict"] == "OUT_OF_SCOPE"

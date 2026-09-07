from pathlib import Path

from datetime import datetime

from Action_Detection_SOP.reporting import (
    date_for_elapsed_time,
    highest_existing_session_number,
    session_result_to_dict,
    session_start_datetime,
    write_daily_csv,
    write_daily_report,
    write_session_artifacts,
)
from Action_Detection_SOP.roll_sop_engine import RollComplianceStatus, RollSessionResult
from Action_Detection_SOP.sop_engine import StepStatus


def _roll_session(
    *,
    session_id: str,
    cleaned: StepStatus,
    labeled: StepStatus,
    overall_status: RollComplianceStatus,
) -> RollSessionResult:
    return RollSessionResult(
        session_id=session_id,
        sop_profile="roll_sop_v1",
        start_time_s=1.0,
        end_time_s=4.5,
        cleaned=cleaned,
        labeled=labeled,
        overall_status=overall_status,
        total_frames=10,
        roll_present_frames=8,
        cleaning_positive_frames=3,
        labeling_positive_frames=5,
    )


def test_roll_session_dict_contains_roll_fields_and_duration() -> None:
    payload = session_result_to_dict(
        _roll_session(
            session_id="000001",
            cleaned=StepStatus.DONE,
            labeled=StepStatus.DONE,
            overall_status=RollComplianceStatus.COMPLIANT,
        )
    )

    assert payload["sop_profile"] == "roll_sop_v1"
    assert payload["cleaned"] == "DONE"
    assert payload["labeled"] == "DONE"
    assert payload["overall_status"] == "SESUAI SOP"
    assert payload["duration_s"] == 3.5
    assert isinstance(payload["session_uid"], str)


def test_date_for_elapsed_time_rolls_over_at_midnight() -> None:
    run_start = datetime(2026, 9, 7, 23, 59, 50)

    assert date_for_elapsed_time(run_start_dt=run_start, elapsed_s=5, fallback_date="2026-09-07") == "2026-09-07"
    assert date_for_elapsed_time(run_start_dt=run_start, elapsed_s=15, fallback_date="2026-09-07") == "2026-09-08"


def test_session_start_datetime_uses_wall_clock_for_live_source() -> None:
    wall_clock = datetime(2026, 9, 8, 0, 0, 5)

    assert session_start_datetime(
        run_start_dt=datetime(2026, 9, 7, 23, 59, 50),
        elapsed_s=15,
        live_source=True,
        wall_clock_dt=wall_clock,
    ) == wall_clock


def test_session_start_datetime_uses_video_timeline_for_recorded_source() -> None:
    assert session_start_datetime(
        run_start_dt=datetime(2026, 9, 7, 23, 59, 50),
        elapsed_s=15,
        live_source=False,
    ) == datetime(2026, 9, 8, 0, 0, 5)


def test_highest_existing_session_number_ignores_non_numeric_directories(tmp_path: Path) -> None:
    date_dir = tmp_path / "sessions" / "2026-09-07"
    date_dir.mkdir(parents=True)
    (date_dir / "session_000012").mkdir()
    (date_dir / "session_000895").mkdir()
    (date_dir / "session_backup").mkdir()
    (date_dir / "session_000999.tmp").mkdir()
    (date_dir / "session_001000").write_text("not a directory", encoding="utf-8")

    assert highest_existing_session_number(out_dir=tmp_path, date="2026-09-07") == 895


def test_write_session_artifacts_reuses_reserved_directory(tmp_path: Path) -> None:
    session = _roll_session(
        session_id="000001",
        cleaned=StepStatus.DONE,
        labeled=StepStatus.DONE,
        overall_status=RollComplianceStatus.COMPLIANT,
    )
    session_dir = tmp_path / "sessions" / "2026-09-07" / "session_000001"
    session_dir.mkdir(parents=True)
    sentinel = session_dir / "evidence" / "roll_entered_001.mp4"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"existing evidence")

    returned = write_session_artifacts(
        out_dir=tmp_path,
        date="2026-09-07",
        session=session,
        session_dir=session_dir,
    )

    assert returned == session_dir
    assert sentinel.read_bytes() == b"existing evidence"
    assert (session_dir / "checklist.json").exists()


def test_write_session_artifacts_rejects_mismatched_reserved_directory(tmp_path: Path) -> None:
    session = _roll_session(
        session_id="000001",
        cleaned=StepStatus.DONE,
        labeled=StepStatus.DONE,
        overall_status=RollComplianceStatus.COMPLIANT,
    )

    try:
        write_session_artifacts(
            out_dir=tmp_path,
            date="2026-09-07",
            session=session,
            session_dir=tmp_path / "sessions" / "2026-09-07" / "session_000002",
        )
    except ValueError as exc:
        assert str(exc) == "session_dir must match the session_id"
    else:
        raise AssertionError("expected mismatched session_dir to be rejected")


def test_roll_daily_outputs_use_roll_summary(tmp_path: Path) -> None:
    sessions = [
        _roll_session(
            session_id="000001",
            cleaned=StepStatus.DONE,
            labeled=StepStatus.DONE,
            overall_status=RollComplianceStatus.COMPLIANT,
        ),
        _roll_session(
            session_id="000002",
            cleaned=StepStatus.UNKNOWN,
            labeled=StepStatus.DONE,
            overall_status=RollComplianceStatus.UNKNOWN,
        ),
    ]

    report_path = write_daily_report(out_dir=tmp_path, date="2026-06-03", sessions=sessions)
    csv_path = write_daily_csv(out_dir=tmp_path, date="2026-06-03", sessions=sessions)

    report = report_path.read_text(encoding="utf-8")
    assert '"sop_profile": "roll_sop_v1"' in report
    assert '"overall_compliant": 1' in report
    assert '"overall_unknown": 1' in report
    assert csv_path.exists()


def test_roll_daily_outputs_append_without_losing_previous_sessions(tmp_path: Path) -> None:
    first = _roll_session(
        session_id="000001",
        cleaned=StepStatus.DONE,
        labeled=StepStatus.DONE,
        overall_status=RollComplianceStatus.COMPLIANT,
    )
    second = _roll_session(
        session_id="000002",
        cleaned=StepStatus.UNKNOWN,
        labeled=StepStatus.NOT_DONE,
        overall_status=RollComplianceStatus.UNKNOWN,
    )

    write_daily_report(out_dir=tmp_path, date="2026-06-03", sessions=[first], append=True)
    write_daily_csv(out_dir=tmp_path, date="2026-06-03", sessions=[first], append=True)
    write_daily_report(out_dir=tmp_path, date="2026-06-03", sessions=[second], append=True)
    write_daily_csv(out_dir=tmp_path, date="2026-06-03", sessions=[second], append=True)

    report = (tmp_path / "reports" / "2026-06-03" / "daily_report.json").read_text(encoding="utf-8")
    assert '"total_sessions": 2' in report
    assert '"overall_compliant": 1' in report
    assert '"overall_unknown": 1' in report
    assert (tmp_path / "reports" / "2026-06-03" / "sessions.csv").read_text(encoding="utf-8").count("\n") == 3


def test_empty_roll_daily_report_uses_roll_schema(tmp_path: Path) -> None:
    path = write_daily_report(
        out_dir=tmp_path,
        date="2026-06-03",
        sessions=[],
        sop_profile="roll_sop_v1",
    )

    report = path.read_text(encoding="utf-8")
    assert '"sop_profile": "roll_sop_v1"' in report
    assert '"total_sessions": 0' in report

from pathlib import Path

from Action_Detection_SOP.reporting import session_result_to_dict, write_daily_csv, write_daily_report
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

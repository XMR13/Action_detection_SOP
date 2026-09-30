from __future__ import annotations

from types import SimpleNamespace

import pytest

from Action_Detection_SOP.web_mvp.review_store import ReviewRecord
from Action_Detection_SOP.web_mvp.sop_status import (
    ReviewOverrideError,
    effective_review_for_session,
    evaluate_sop_status,
    normalize_session_checklist_payload,
    operator_verdict,
    validate_review_overrides,
    validate_roll_review_decision,
)


def _session(checklist: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(checklist=checklist)


def _review(overrides: dict[str, object]) -> ReviewRecord:
    return ReviewRecord(
        session_uid="uid",
        review_status="PENDING",
        review_note="",
        overrides=overrides,
        created_at_utc="2026-06-10T00:00:00+00:00",
        updated_at_utc="2026-06-10T00:00:00+00:00",
    )


def test_operator_verdict_requires_matching_review_and_final_sop() -> None:
    assert operator_verdict(review_status="QUALIFIED", final_sop="DONE") == "DONE"
    assert operator_verdict(review_status="NOT_QUALIFIED", final_sop="NOT_DONE") == "NOT_DONE"
    assert operator_verdict(review_status="PENDING", final_sop="DONE") == "NEEDS_REVIEW"
    assert operator_verdict(review_status="QUALIFIED", final_sop="NOT_DONE") == "NEEDS_REVIEW"
    assert operator_verdict(review_status="NOT_QUALIFIED", final_sop="DONE") == "NEEDS_REVIEW"


def test_roll_review_decision_requires_matching_final_result() -> None:
    session = _session(
        {"sop_profile": "roll_sop_v1", "cleaned": "DONE", "labeled": "DONE", "overall_status": "SESUAI SOP"}
    )
    with pytest.raises(ReviewOverrideError, match="Tidak sesuai SOP"):
        validate_roll_review_decision(session=session, review_status="NOT_QUALIFIED", overrides={})
    validate_roll_review_decision(
        session=session,
        review_status="NOT_QUALIFIED",
        overrides={"overall_status": "TIDAK SESUAI SOP"},
    )


def test_out_of_scope_is_manual_and_preserves_machine_results() -> None:
    session = _session({"sop_profile": "roll_sop_v1", "cleaned": "UNKNOWN", "labeled": "NOT_DONE"})
    review = ReviewRecord("roll", "OUT_OF_SCOPE", "", {}, "created", "updated", "PASSING_THROUGH")
    validate_roll_review_decision(session=session, review_status="OUT_OF_SCOPE", overrides={},
                                  scope_reason="PASSING_THROUGH")
    status = evaluate_sop_status(session=session, review=review)
    assert status.machine_sop == status.final_sop == "NOT_DONE"
    assert operator_verdict(review_status=review.review_status, final_sop=status.final_sop) == "OUT_OF_SCOPE"
    effective = effective_review_for_session(session=session, review=review, auto_approve_done_enabled=True,
                                             auto_approve_min_duration_s=0, has_evidence=True)
    assert (effective.status, effective.source) == ("OUT_OF_SCOPE", "MANUAL")


@pytest.mark.parametrize("reason,note", [(None, ""), ("INVALID", ""), ("OTHER", "  ")])
def test_out_of_scope_requires_valid_reason_and_other_explanation(reason: str | None, note: str) -> None:
    session = _session({"sop_profile": "roll_sop_v1", "cleaned": "UNKNOWN", "labeled": "UNKNOWN"})
    with pytest.raises(ReviewOverrideError):
        validate_roll_review_decision(session=session, review_status="OUT_OF_SCOPE", overrides={},
                                      scope_reason=reason, review_note=note)


def test_scope_reason_cannot_be_used_for_approval_or_legacy_profile() -> None:
    roll = _session({"sop_profile": "roll_sop_v1", "cleaned": "DONE", "labeled": "DONE"})
    with pytest.raises(ReviewOverrideError):
        validate_roll_review_decision(session=roll, review_status="QUALIFIED", overrides={},
                                      scope_reason="PASSING_THROUGH")
    with pytest.raises(ReviewOverrideError):
        validate_roll_review_decision(session=_session({}), review_status="OUT_OF_SCOPE", overrides={},
                                      scope_reason="PASSING_THROUGH")


def test_legacy_sop_summary_keeps_old_operator_fields() -> None:
    summary = evaluate_sop_status(
        session=_session({"operator_present": "DONE", "roi_dwell": "DONE", "helmet": "UNKNOWN"}),
        review=_review({"helmet": "DONE"}),
    ).summary

    assert summary["profile"] == "operator_mvp_a"
    assert summary["machine"]["status"] == "UNKNOWN"
    assert summary["final"]["status"] == "DONE"
    assert summary["final"]["helmet"] == "DONE"


def test_roll_sop_summary_uses_explicit_roll_fields() -> None:
    status = evaluate_sop_status(
        session=_session(
            {
                "sop_profile": "roll_sop_v1",
                "cleaned": "DONE",
                "labeled": "DONE",
                "overall_status": "SESUAI SOP",
            }
        ),
        review=None,
    )
    summary = status.summary

    assert status.profile == "roll_sop_v1"
    assert summary["profile"] == "roll_sop_v1"
    assert summary["machine"]["cleaned"] == "DONE"
    assert summary["machine"]["labeled"] == "DONE"
    assert summary["machine"]["overall_status"] == "SESUAI SOP"
    assert summary["machine"]["status"] == "DONE"
    assert summary["machine"]["labels"]["cleaned"] == "Sudah dibersihkan"
    assert summary["inconsistent"] is False


def test_roll_step_override_recomputes_final_overall_status() -> None:
    summary = evaluate_sop_status(
        session=_session(
            {
                "sop_profile": "roll_sop_v1",
                "cleaned": "DONE",
                "labeled": "NOT_DONE",
                "overall_status": "TIDAK SESUAI SOP",
            }
        ),
        review=_review({"labeled": "DONE"}),
    ).summary

    assert summary["machine"]["status"] == "NOT_DONE"
    assert summary["final"]["labeled"] == "DONE"
    assert summary["final"]["overall_status"] == "SESUAI SOP"
    assert summary["final"]["status"] == "DONE"


def test_roll_overall_override_wins_over_step_derivation() -> None:
    summary = evaluate_sop_status(
        session=_session(
            {
                "sop_profile": "roll_sop_v1",
                "cleaned": "DONE",
                "labeled": "DONE",
                "overall_status": "SESUAI SOP",
            }
        ),
        review=_review({"overall_status": "TIDAK SESUAI SOP"}),
    ).summary

    assert summary["final"]["cleaned"] == "DONE"
    assert summary["final"]["labeled"] == "DONE"
    assert summary["final"]["overall_status"] == "TIDAK SESUAI SOP"
    assert summary["final"]["status"] == "NOT_DONE"


def test_roll_summary_flags_inconsistent_machine_artifact() -> None:
    summary = evaluate_sop_status(
        session=_session(
            {
                "sop_profile": "roll_sop_v1",
                "cleaned": "DONE",
                "labeled": "NOT_DONE",
                "overall_status": "SESUAI SOP",
            }
        ),
        review=None,
    ).summary

    assert summary["machine"]["status"] == "NOT_DONE"
    assert summary["machine"]["overall_status"] == "SESUAI SOP"
    assert summary["inconsistent"] is True


def test_roll_override_validation_is_profile_aware_and_strict() -> None:
    checklist = {"sop_profile": "roll_sop_v1", "cleaned": "DONE", "labeled": "DONE"}

    assert validate_review_overrides(
        checklist=checklist,
        raw={"cleaned": "UNKNOWN", "overall_status": "TIDAK SESUAI SOP"},
    ) == {"cleaned": "UNKNOWN", "overall_status": "TIDAK SESUAI SOP"}

    with pytest.raises(ReviewOverrideError):
        validate_review_overrides(checklist=checklist, raw={"helmet": "DONE"})

    with pytest.raises(ReviewOverrideError):
        validate_review_overrides(checklist=checklist, raw={"overall_status": "DONE"})


def test_roll_session_payload_validation_requires_canonical_overall_status() -> None:
    payload = normalize_session_checklist_payload(
        {
            "session_uid": "uid_roll",
            "session_id": "roll001",
            "start_date": "2026-06-10",
            "sop_profile": "roll_sop_v1",
            "cleaned": "done",
            "labeled": "NOT DONE",
            "overall_status": "TIDAK SESUAI SOP",
        }
    )

    assert payload["sop_profile"] == "roll_sop_v1"
    assert payload["cleaned"] == "DONE"
    assert payload["labeled"] == "NOT_DONE"
    assert payload["overall_status"] == "TIDAK SESUAI SOP"

    with pytest.raises(ReviewOverrideError):
        normalize_session_checklist_payload(
            {
                "sop_profile": "roll_sop_v1",
                "cleaned": "DONE",
                "labeled": "DONE",
                "overall_status": "DONE",
            }
        )


def test_roll_auto_approve_requires_only_compliant_machine_result() -> None:
    session = _session(
        {
            "sop_profile": "roll_sop_v1",
            "cleaned": "DONE",
            "labeled": "DONE",
            "overall_status": "SESUAI SOP",
        }
    )

    approved_without_evidence = effective_review_for_session(
        session=session,
        review=None,
        auto_approve_done_enabled=True,
        auto_approve_min_duration_s=8.0,
        has_evidence=False,
    )
    approved = effective_review_for_session(
        session=session,
        review=None,
        auto_approve_done_enabled=True,
        auto_approve_min_duration_s=8.0,
        has_evidence=True,
    )
    approved_with_pending_review = effective_review_for_session(
        session=session,
        review=_review({}),
        auto_approve_done_enabled=True,
        auto_approve_min_duration_s=8.0,
        has_evidence=False,
    )

    assert approved_without_evidence.status == "QUALIFIED"
    assert approved_without_evidence.source == "AUTO"
    assert approved_without_evidence.auto_reason == "roll_policy_pass"
    assert approved.status == "QUALIFIED"
    assert approved.source == "AUTO"
    assert approved.auto_reason == "roll_policy_pass"
    assert approved_with_pending_review.status == "QUALIFIED"
    assert approved_with_pending_review.source == "AUTO"
    assert approved_with_pending_review.auto_reason == "roll_policy_pass"

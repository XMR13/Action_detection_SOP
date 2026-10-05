from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple

from .review_store import ReviewRecord

ARCHIVED_OPERATOR_PROFILE = "operator_mvp_a"
ARCHIVED_SESSION_ERROR = (
    "Operator sessions are archived and read-only. Upload and review roll_sop_v1 sessions instead."
)
ROLL_PROFILE = "roll_sop_v1"

STEP_STATUS_VALUES = {"DONE", "NOT_DONE", "UNKNOWN"}
ROLL_OVERALL_COMPLIANT = "SESUAI SOP"
ROLL_OVERALL_NON_COMPLIANT = "TIDAK SESUAI SOP"
ROLL_OVERALL_UNKNOWN = "UNKNOWN"
ROLL_OVERALL_STATUS_VALUES = {
    ROLL_OVERALL_COMPLIANT,
    ROLL_OVERALL_NON_COMPLIANT,
    ROLL_OVERALL_UNKNOWN,
}


@dataclass(frozen=True)
class WebSopStatus:
    """Web-facing roll status and read-only compatibility for archived sessions."""

    profile: str
    summary: Dict[str, Any]
    machine_sop: str
    final_sop: str
    machine_helmet: str = "UNKNOWN"
    final_helmet: str = "UNKNOWN"
    machine_roi_dwell: str = "UNKNOWN"


@dataclass(frozen=True)
class EffectiveReview:
    status: Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING", "OUT_OF_SCOPE"]
    source: Literal["MANUAL", "AUTO", "PENDING"]
    auto_reason: Optional[str] = None


class ReviewOverrideError(ValueError):
    pass


class _SopPolicy(Protocol):
    profile: str

    def build_status(self, *, session: Any, review: Optional[ReviewRecord]) -> WebSopStatus:
        ...

    def should_auto_approve(
        self,
        *,
        status: WebSopStatus,
        session: Any,
        auto_approve_min_duration_s: float,
        has_evidence: bool,
    ) -> Tuple[bool, Optional[str]]:
        ...


def evaluate_sop_status(*, session: Any, review: Optional[ReviewRecord]) -> WebSopStatus:
    """Turn a stored checklist and optional review into the web SOP contract."""
    return _policy_for_checklist(session.checklist).build_status(session=session, review=review)


def validate_review_overrides(*, checklist: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, str]:
    _require_roll_checklist(checklist)
    return _ROLL_POLICY.validate_overrides(raw)


def operator_verdict(*, review_status: str, final_sop: str) -> str:
    """Return one operator-facing outcome; conflicting stored values need review."""
    if review_status == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    if review_status == "QUALIFIED" and final_sop == "DONE":
        return "DONE"
    if review_status == "NOT_QUALIFIED" and final_sop == "NOT_DONE":
        return "NOT_DONE"
    return "NEEDS_REVIEW"


def validate_roll_review_decision(
    *, session: Any, review_status: str, overrides: Dict[str, str],
    scope_reason: Optional[str] = None, review_note: str = "",
) -> None:
    """Keep a new roll review decision aligned with its final SOP result."""
    _require_roll_checklist(session.checklist)
    if review_status == "OUT_OF_SCOPE":
        if scope_reason not in {"PASSING_THROUGH", "ALREADY_WRAPPED", "OTHER"}:
            raise ReviewOverrideError("Pilih alasan di luar cakupan SOP")
        if scope_reason == "OTHER" and not review_note.strip():
            raise ReviewOverrideError("Tuliskan penjelasan untuk alasan Lainnya")
        return
    if scope_reason is not None:
        raise ReviewOverrideError("Alasan di luar cakupan hanya berlaku untuk keputusan Di luar cakupan SOP")
    if review_status == "PENDING":
        return
    candidate = ReviewRecord(
        session_uid="",
        review_status=review_status,
        review_note="",
        overrides=overrides,
        created_at_utc="",
        updated_at_utc="",
    )
    final_sop = evaluate_sop_status(session=session, review=candidate).final_sop
    expected = "DONE" if review_status == "QUALIFIED" else "NOT_DONE"
    if final_sop != expected:
        label = "Sesuai SOP" if expected == "DONE" else "Tidak sesuai SOP"
        raise ReviewOverrideError(
            f"Review decision requires final SOP result {label}; correct the step or final SOP override before saving"
        )


def normalize_session_checklist_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_roll_checklist(payload)
    return _ROLL_POLICY.normalize_payload(payload)


def effective_review_for_session(
    *,
    session: Any,
    review: Optional[ReviewRecord],
    auto_approve_done_enabled: bool,
    auto_approve_min_duration_s: float,
    has_evidence: bool,
) -> EffectiveReview:
    if review is not None:
        manual_status = str(review.review_status).upper()
        if manual_status == "OUT_OF_SCOPE":
            return EffectiveReview(status="OUT_OF_SCOPE", source="MANUAL")
        if manual_status == "QUALIFIED":
            return EffectiveReview(status="QUALIFIED", source="MANUAL")
        if manual_status == "NOT_QUALIFIED":
            return EffectiveReview(status="NOT_QUALIFIED", source="MANUAL")
        # PENDING is not a final human decision. Let the automatic policy
        # evaluate the machine result instead of allowing an old placeholder
        # review row to keep an otherwise complete session pending forever.

    if not auto_approve_done_enabled:
        return EffectiveReview(status="PENDING", source="PENDING", auto_reason="auto_approve_disabled")

    status = evaluate_sop_status(session=session, review=None)
    allow_auto, reason = _policy_for_checklist(session.checklist).should_auto_approve(
        status=status,
        session=session,
        auto_approve_min_duration_s=auto_approve_min_duration_s,
        has_evidence=has_evidence,
    )
    if allow_auto:
        return EffectiveReview(status="QUALIFIED", source="AUTO", auto_reason=reason)
    return EffectiveReview(status="PENDING", source="PENDING", auto_reason=reason)


class _RollSopPolicy:
    profile = ROLL_PROFILE
    override_keys = {"cleaned", "labeled", "overall_status"}

    def build_status(self, *, session: Any, review: Optional[ReviewRecord]) -> WebSopStatus:
        machine_cleaned = _normalize_step_status(session.checklist.get("cleaned"))
        machine_labeled = _normalize_step_status(session.checklist.get("labeled"))
        machine_overall = _normalize_roll_overall_status(session.checklist.get("overall_status"))
        derived_machine_overall = _roll_overall_from_steps(cleaned=machine_cleaned, labeled=machine_labeled)
        machine_sop = _normalized_status_from_roll_overall(derived_machine_overall)

        final_cleaned = _review_step_status(
            machine_status=machine_cleaned,
            review=review,
            step_key="cleaned",
        )
        final_labeled = _review_step_status(
            machine_status=machine_labeled,
            review=review,
            step_key="labeled",
        )
        final_overall = self._final_overall_status(
            review=review,
            final_cleaned=final_cleaned,
            final_labeled=final_labeled,
        )
        final_sop = _normalized_status_from_roll_overall(final_overall)

        summary = {
            "profile": ROLL_PROFILE,
            "machine": {
                "cleaned": machine_cleaned,
                "labeled": machine_labeled,
                "overall_status": machine_overall,
                "status": machine_sop,
                "labels": _roll_labels(
                    cleaned=machine_cleaned,
                    labeled=machine_labeled,
                    overall_status=machine_overall,
                    status=machine_sop,
                ),
            },
            "final": {
                "cleaned": final_cleaned,
                "labeled": final_labeled,
                "overall_status": final_overall,
                "status": final_sop,
                "labels": _roll_labels(
                    cleaned=final_cleaned,
                    labeled=final_labeled,
                    overall_status=final_overall,
                    status=final_sop,
                ),
            },
            "inconsistent": machine_overall != derived_machine_overall,
        }
        return WebSopStatus(
            profile=ROLL_PROFILE,
            summary=summary,
            machine_sop=machine_sop,
            final_sop=final_sop,
        )

    def validate_overrides(self, raw: Dict[str, Any]) -> Dict[str, str]:
        return _validate_overrides(
            raw=raw,
            allowed_keys=self.override_keys,
            overall_key="overall_status",
        )

    def normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)
        missing = [key for key in ("cleaned", "labeled", "overall_status") if key not in normalized]
        if missing:
            raise ReviewOverrideError(f"Missing roll SOP field(s): {', '.join(missing)}")

        normalized["sop_profile"] = ROLL_PROFILE
        normalized["cleaned"] = _validate_step_value(key="cleaned", value=normalized.get("cleaned"))
        normalized["labeled"] = _validate_step_value(key="labeled", value=normalized.get("labeled"))
        normalized["overall_status"] = _validate_roll_overall_value(
            key="overall_status",
            value=normalized.get("overall_status"),
        )
        return normalized

    def should_auto_approve(
        self,
        *,
        status: WebSopStatus,
        session: Any,
        auto_approve_min_duration_s: float,
        has_evidence: bool,
    ) -> Tuple[bool, Optional[str]]:
        machine = status.summary["machine"]
        if (
            machine.get("overall_status") != ROLL_OVERALL_COMPLIANT
            or machine.get("cleaned") != "DONE"
            or machine.get("labeled") != "DONE"
        ):
            return False, "roll_sop_not_done"
        return True, "roll_policy_pass"

    def _final_overall_status(
        self,
        *,
        review: Optional[ReviewRecord],
        final_cleaned: str,
        final_labeled: str,
    ) -> str:
        if review is not None and isinstance(review.overrides, dict):
            raw_overall = review.overrides.get("overall_status")
            if isinstance(raw_overall, str) and raw_overall:
                return _normalize_roll_overall_status(raw_overall)
        return _roll_overall_from_steps(cleaned=final_cleaned, labeled=final_labeled)


class _ArchivedOperatorReader:
    """Read historical checklists and stored overrides without an active SOP engine."""

    profile = ARCHIVED_OPERATOR_PROFILE

    def build_status(self, *, session: Any, review: Optional[ReviewRecord]) -> WebSopStatus:
        keys = ("operator_present", "roi_dwell", "helmet")
        machine = {key: _normalize_step_status(session.checklist.get(key)) for key in keys}
        final = {
            key: _review_step_status(machine_status=machine[key], review=review, step_key=key)
            for key in keys
        }
        machine_sop = _sop_status_from_steps(*machine.values())
        final_sop = _sop_status_from_steps(*final.values())
        return WebSopStatus(
            profile=self.profile,
            summary={
                "profile": self.profile,
                "machine": {**machine, "status": machine_sop},
                "final": {**final, "status": final_sop},
                "inconsistent": False,
                "read_only": True,
            },
            machine_sop=machine_sop,
            final_sop=final_sop,
            machine_helmet=machine["helmet"],
            final_helmet=final["helmet"],
            machine_roi_dwell=machine["roi_dwell"],
        )

    def should_auto_approve(
        self,
        *,
        status: WebSopStatus,
        session: Any,
        auto_approve_min_duration_s: float,
        has_evidence: bool,
    ) -> Tuple[bool, Optional[str]]:
        return False, "archived_operator_session"


_ROLL_POLICY = _RollSopPolicy()
_ARCHIVED_OPERATOR_READER = _ArchivedOperatorReader()


def _policy_for_checklist(checklist: Dict[str, Any]) -> _SopPolicy:
    if _is_roll_profile(checklist):
        return _ROLL_POLICY
    return _ARCHIVED_OPERATOR_READER


def _is_roll_profile(checklist: Dict[str, Any]) -> bool:
    profile = checklist.get("sop_profile")
    if profile is not None:
        return isinstance(profile, str) and profile.strip() == ROLL_PROFILE
    return any(key in checklist for key in ("cleaned", "labeled", "overall_status"))


def _require_roll_checklist(checklist: Dict[str, Any]) -> None:
    if not _is_roll_profile(checklist):
        raise ReviewOverrideError(ARCHIVED_SESSION_ERROR)


def _normalize_step_status(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    normalized = value.strip().upper().replace(" ", "_")
    if normalized in STEP_STATUS_VALUES:
        return normalized
    return "UNKNOWN"


def _normalize_roll_overall_status(value: Any) -> str:
    if not isinstance(value, str):
        return ROLL_OVERALL_UNKNOWN
    normalized = value.strip().upper()
    if normalized in ROLL_OVERALL_STATUS_VALUES:
        return normalized
    return ROLL_OVERALL_UNKNOWN


def _review_step_status(*, machine_status: str, review: Optional[ReviewRecord], step_key: str) -> str:
    if review is None or not isinstance(review.overrides, dict):
        return machine_status
    override = review.overrides.get(step_key)
    if isinstance(override, str) and override:
        return _normalize_step_status(override)
    return machine_status


def _sop_status_from_steps(*steps: str) -> str:
    normalized = [_normalize_step_status(step) for step in steps]
    if any(step == "NOT_DONE" for step in normalized):
        return "NOT_DONE"
    if normalized and all(step == "DONE" for step in normalized):
        return "DONE"
    return "UNKNOWN"


def _roll_overall_from_steps(*, cleaned: str, labeled: str) -> str:
    cleaned_status = _normalize_step_status(cleaned)
    labeled_status = _normalize_step_status(labeled)
    if cleaned_status == "DONE" and labeled_status == "DONE":
        return ROLL_OVERALL_COMPLIANT
    if cleaned_status == "NOT_DONE" or labeled_status == "NOT_DONE":
        return ROLL_OVERALL_NON_COMPLIANT
    return ROLL_OVERALL_UNKNOWN


def _normalized_status_from_roll_overall(overall_status: str) -> str:
    normalized = _normalize_roll_overall_status(overall_status)
    if normalized == ROLL_OVERALL_COMPLIANT:
        return "DONE"
    if normalized == ROLL_OVERALL_NON_COMPLIANT:
        return "NOT_DONE"
    return "UNKNOWN"


def _validate_overrides(
    *,
    raw: Dict[str, Any],
    allowed_keys: set[str],
    overall_key: Optional[str] = None,
) -> Dict[str, str]:
    validated: Dict[str, str] = {}
    for key, value in raw.items():
        if key not in allowed_keys:
            allowed = ", ".join(sorted(allowed_keys))
            raise ReviewOverrideError(f"Invalid override key `{key}` (allowed: {allowed})")
        if key == overall_key:
            validated[key] = _validate_roll_overall_value(key=key, value=value)
        else:
            validated[key] = _validate_step_value(key=key, value=value)
    return validated


def _validate_step_value(*, key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ReviewOverrideError(f"Invalid override value type for `{key}`")
    normalized = _normalize_step_status(value)
    if normalized not in STEP_STATUS_VALUES:
        raise ReviewOverrideError(f"Invalid override value for `{key}`")
    if normalized == "UNKNOWN" and value.strip().upper().replace(" ", "_") != "UNKNOWN":
        raise ReviewOverrideError(f"Invalid override value for `{key}`")
    return normalized


def _validate_roll_overall_value(*, key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ReviewOverrideError(f"Invalid override value type for `{key}`")
    normalized = value.strip().upper()
    if normalized not in ROLL_OVERALL_STATUS_VALUES:
        allowed = ", ".join(sorted(ROLL_OVERALL_STATUS_VALUES))
        raise ReviewOverrideError(f"Invalid override value for `{key}` (allowed: {allowed})")
    return normalized


def _roll_labels(*, cleaned: str, labeled: str, overall_status: str, status: str) -> Dict[str, str]:
    return {
        "cleaned": _step_status_label(field="cleaned", status=cleaned),
        "labeled": _step_status_label(field="labeled", status=labeled),
        "overall_status": _overall_status_label(overall_status),
        "status": _normalized_status_label(status),
    }


def _step_status_label(*, field: str, status: str) -> str:
    labels = {
        ("cleaned", "DONE"): "Sudah dibersihkan",
        ("cleaned", "NOT_DONE"): "Belum dibersihkan",
        ("cleaned", "UNKNOWN"): "Status pembersihan belum jelas",
        ("labeled", "DONE"): "Sudah diberi label",
        ("labeled", "NOT_DONE"): "Belum diberi label",
        ("labeled", "UNKNOWN"): "Status label belum jelas",
    }
    normalized = _normalize_step_status(status)
    return labels.get((field, normalized), normalized.replace("_", " "))


def _overall_status_label(status: str) -> str:
    normalized = _normalize_roll_overall_status(status)
    labels = {
        ROLL_OVERALL_COMPLIANT: "Sesuai SOP",
        ROLL_OVERALL_NON_COMPLIANT: "Tidak sesuai SOP",
        ROLL_OVERALL_UNKNOWN: "Status SOP belum jelas",
    }
    return labels[normalized]


def _normalized_status_label(status: str) -> str:
    normalized = _normalize_step_status(status)
    labels = {
        "DONE": "Selesai",
        "NOT_DONE": "Tidak selesai",
        "UNKNOWN": "Belum jelas",
    }
    return labels[normalized]

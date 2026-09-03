from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple

from .review_store import ReviewRecord

LEGACY_PROFILE = "operator_mvp_a"
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
    """Web-facing SOP status plus temporary legacy flat-field compatibility."""

    profile: str
    summary: Dict[str, Any]
    machine_sop: str
    final_sop: str
    machine_helmet: str = "UNKNOWN"
    final_helmet: str = "UNKNOWN"
    machine_roi_dwell: str = "UNKNOWN"


@dataclass(frozen=True)
class EffectiveReview:
    status: Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"]
    source: Literal["MANUAL", "AUTO", "PENDING"]
    auto_reason: Optional[str] = None


class ReviewOverrideError(ValueError):
    pass


class _SopPolicy(Protocol):
    profile: str

    def build_status(self, *, session: Any, review: Optional[ReviewRecord]) -> WebSopStatus:
        ...

    def validate_overrides(self, raw: Dict[str, Any]) -> Dict[str, str]:
        ...

    def normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
    return _policy_for_checklist(checklist).validate_overrides(raw)


def normalize_session_checklist_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _policy_for_checklist(payload).normalize_payload(payload)


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


class _LegacySopPolicy:
    profile = LEGACY_PROFILE
    override_keys = {"operator_present", "roi_dwell", "helmet"}

    def build_status(self, *, session: Any, review: Optional[ReviewRecord]) -> WebSopStatus:
        machine_operator = _normalize_step_status(session.checklist.get("operator_present"))
        machine_roi = _normalize_step_status(session.checklist.get("roi_dwell"))
        machine_helmet = _normalize_step_status(session.checklist.get("helmet"))
        machine_sop = _sop_status_from_steps(machine_operator, machine_roi, machine_helmet)

        final_operator = _review_step_status(
            machine_status=machine_operator,
            review=review,
            step_key="operator_present",
        )
        final_roi = _review_step_status(
            machine_status=machine_roi,
            review=review,
            step_key="roi_dwell",
        )
        final_helmet = _review_step_status(
            machine_status=machine_helmet,
            review=review,
            step_key="helmet",
        )
        final_sop = _sop_status_from_steps(final_operator, final_roi, final_helmet)

        summary = {
            "profile": LEGACY_PROFILE,
            "machine": {
                "operator_present": machine_operator,
                "roi_dwell": machine_roi,
                "helmet": machine_helmet,
                "status": machine_sop,
            },
            "final": {
                "operator_present": final_operator,
                "roi_dwell": final_roi,
                "helmet": final_helmet,
                "status": final_sop,
            },
            "inconsistent": False,
        }
        return WebSopStatus(
            profile=LEGACY_PROFILE,
            summary=summary,
            machine_sop=machine_sop,
            final_sop=final_sop,
            machine_helmet=machine_helmet,
            final_helmet=final_helmet,
            machine_roi_dwell=machine_roi,
        )

    def validate_overrides(self, raw: Dict[str, Any]) -> Dict[str, str]:
        return _validate_overrides(raw=raw, allowed_keys=self.override_keys)

    def normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return dict(payload)

    def should_auto_approve(
        self,
        *,
        status: WebSopStatus,
        session: Any,
        auto_approve_min_duration_s: float,
        has_evidence: bool,
    ) -> Tuple[bool, Optional[str]]:
        if status.machine_helmet != "DONE":
            return False, "helmet_not_done"
        if status.machine_roi_dwell != "DONE":
            return False, "roi_not_done"
        if _session_duration_s(session) < float(auto_approve_min_duration_s):
            return False, "duration_too_short"

        blocker = _auto_approve_blocker(session.checklist.get("notes"))
        if blocker:
            return False, f"blocked_by_note:{blocker}"
        if not has_evidence:
            return False, "no_evidence"
        return True, "policy_pass"


_ROLL_POLICY = _RollSopPolicy()
_LEGACY_POLICY = _LegacySopPolicy()


def _policy_for_checklist(checklist: Dict[str, Any]) -> _SopPolicy:
    if _is_roll_profile(checklist):
        return _ROLL_POLICY
    return _LEGACY_POLICY


def _is_roll_profile(checklist: Dict[str, Any]) -> bool:
    profile = checklist.get("sop_profile")
    if isinstance(profile, str) and profile.strip() == ROLL_PROFILE:
        return True
    return any(key in checklist for key in ("cleaned", "labeled", "overall_status"))


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


def _session_duration_s(session: Any) -> float:
    start_s = float(session.checklist.get("start_time_s") or 0.0)
    end_s = float(session.checklist.get("end_time_s") or 0.0)
    return max(0.0, end_s - start_s)


def _auto_approve_blocker(notes: Any) -> Optional[str]:
    if not isinstance(notes, list):
        return None
    for raw in notes:
        if not isinstance(raw, str):
            continue
        tag = raw.strip().lower()
        if not tag:
            continue
        if ("too_short" in tag) or ("too_small" in tag) or ("disabled" in tag):
            return tag
    return None

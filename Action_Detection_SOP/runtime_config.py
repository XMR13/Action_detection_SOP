from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Action_Detection_SOP.config import SopProfile, load_sop_profile
from yolo_kit import load_class_names

#THE CONSTANT CONFIG VALUES
DEFAULT_SESSION_START_S = 2.0
DEFAULT_SESSION_END_S = 3.0
DEFAULT_ROLL_SESSION_START_S = 1.0
DEFAULT_ROLL_SESSION_END_S = 2.0
DEFAULT_ROI_DWELL_S = 8.0
PROFILE_OPERATOR_MVP_A = "operator_mvp_a"
PROFILE_ROLL_SOP_V1 = "roll_sop_v1"
KNOWN_SOP_PROFILES = {PROFILE_OPERATOR_MVP_A, PROFILE_ROLL_SOP_V1}

"""
----------------------------
CONFIG
----------------------------
"""
@dataclass(frozen=True)
class ResolvedSessionTimingConfig:
    start_s: float
    end_s: float
    min_session_s: float


@dataclass(frozen=True)
class ResolvedOperatorRulesConfig:
    roi_dwell_s: float

@dataclass(frozen=True)
class ResolvedSopProfileConfig:
    name: str
    path: Optional[Path]
    profile: Optional[SopProfile]

@dataclass(frozen=True)
class ResolvedClassConfig:
    #all configuration for the class side
    #or detection of thins
    class_names: Dict[int, str]
    class_conf_thresholds: Dict[int, float]
    active_class_ids: Tuple[int, ...]
    person_ids: Tuple[int, ...]
    helmet_ids: Tuple[int, ...]
    helmet_disabled: bool
    roll_ids: Tuple[int, ...]
    cleaning_cloth_ids: Tuple[int, ...]
    paper_label_ids: Tuple[int, ...]
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedRunConfig:
    sop_profile: ResolvedSopProfileConfig
    session_timing: ResolvedSessionTimingConfig
    operator_rules: Optional[ResolvedOperatorRulesConfig]
    classes: ResolvedClassConfig


def _resolve_seconds(cli_value: Optional[float], profile_value: Optional[float], default_value: float) -> float:
    """This is for choosing the values in the order of cli_value --> profile_value --> default value"""
    if cli_value is not None:
        return float(cli_value)

    if profile_value is not None:
        return float(profile_value)

    return float(default_value)

def _name_to_ids(class_names: Dict[int, str], labels: Sequence[str]) -> List[int]:
    wanted = {s.strip().lower() for s in labels if s.strip()}
    ids: List[int] = []
    for cid, name in class_names.items():
        if str(name).strip().lower() in wanted:
            ids.append(int(cid))
    return ids


def _parse_label_conf(raw_values: Sequence[str], *, class_names: Dict[int, str]) -> Dict[int, float]:
    if not raw_values:
        return {}
    if not class_names:
        raise ValueError("--label-conf requires --metadata so label names can be resolved.")

    by_name = {str(name).strip().lower(): int(class_id) for class_id, name in class_names.items()}
    out: Dict[int, float] = {}
    for raw in raw_values:
        if "=" not in str(raw):
            raise ValueError(f"--label-conf must use label=value format, got: {raw!r}")
        raw_label, raw_conf = str(raw).split("=", 1)
        label = raw_label.strip().lower()
        if not label:
            raise ValueError(f"--label-conf has an empty label: {raw!r}")
        if label not in by_name:
            raise ValueError(f"--label-conf label {label!r} is not in metadata names: {sorted(by_name)}")
        try:
            conf = float(raw_conf)
        except ValueError as exc:
            raise ValueError(f"--label-conf value must be a number within [0, 1], got: {raw!r}") from exc
        if conf < 0.0 or conf > 1.0:
            raise ValueError(f"--label-conf value must be within [0, 1], got: {raw!r}")
        out[int(by_name[label])] = conf
    return out

def _resolve_sop_profile(args: argparse.Namespace) -> ResolvedSopProfileConfig:
    name = PROFILE_OPERATOR_MVP_A
    path = None
    profile = None
    raw = str(args.sop_profile).strip() if args.sop_profile else ""
    if raw:
        if raw in KNOWN_SOP_PROFILES:
            name = raw
        else:
            path = Path(raw)
            profile = load_sop_profile(path)

    return ResolvedSopProfileConfig(name=name, path=path, profile=profile)

def _resolve_session_timing(args: argparse.Namespace, sop_profile: ResolvedSopProfileConfig) -> ResolvedSessionTimingConfig:
    profile = sop_profile.profile
    #the default value
    default_start_s = DEFAULT_ROLL_SESSION_START_S if sop_profile.name == PROFILE_ROLL_SOP_V1 else DEFAULT_SESSION_START_S
    default_end_s = DEFAULT_ROLL_SESSION_END_S if sop_profile.name == PROFILE_ROLL_SOP_V1 else DEFAULT_SESSION_END_S
    return ResolvedSessionTimingConfig(
        start_s=_resolve_seconds(
            args.start_s,
            profile.session_start_seconds if profile else None,
            default_start_s,
        ),
        end_s=_resolve_seconds(
            args.end_s,
            profile.session_end_seconds if profile else None,
            default_end_s,
        ),
        min_session_s=_resolve_seconds(
            args.min_session_s,
            profile.min_session_seconds if profile else None,
            0.0,
        ),
    )


def _resolve_operator_rules(
    args: argparse.Namespace,
    sop_profile: ResolvedSopProfileConfig,
) -> Optional[ResolvedOperatorRulesConfig]:
    if sop_profile.name != PROFILE_OPERATOR_MVP_A:
        return None

    profile = sop_profile.profile
    return ResolvedOperatorRulesConfig(
        roi_dwell_s=_resolve_seconds(
            args.roi_dwell_s,
            profile.roi_dwell_seconds if profile else None,
            DEFAULT_ROI_DWELL_S,
        ),
    )


def _resolve_classes(args: argparse.Namespace, sop_profile_name: str) -> ResolvedClassConfig:
    class_names = load_class_names(args.metadata) if args.metadata else {}
    class_conf_thresholds = _parse_label_conf(args.label_conf, class_names=class_names)
    person_ids = tuple(_name_to_ids(class_names, args.person_label))
    helmet_disabled = bool(args.skip_helmet)
    helmet_alerts_enabled = bool(getattr(args, "enable_helmet_alerts", False))
    helmet_label_ids = tuple(_name_to_ids(class_names, args.helmet_label))
    helmet_ids = helmet_label_ids if helmet_alerts_enabled or not helmet_disabled else ()
    roll_ids = tuple(_name_to_ids(class_names, args.roll_label))
    cleaning_cloth_ids = tuple(_name_to_ids(class_names, args.cleaning_cloth_label))
    paper_label_ids = tuple(_name_to_ids(class_names, args.paper_label))
    warnings: List[str] = []

    if helmet_alerts_enabled:
        "jika kelas helmet dan person tidak tersedia di config maka trhow an errors"
        if not person_ids:
            raise ValueError(f"Could not resolve person class ids from labels: {args.person_label!r}")
        if not helmet_label_ids:
            raise ValueError(
                f"Could not resolve helmet class ids from labels: {args.helmet_label!r}. "
                "Helmet alerts require a helmet-capable metadata/model pair."
            )

    if sop_profile_name == PROFILE_OPERATOR_MVP_A and not person_ids:
        raise ValueError(f"Could not resolve person class ids from labels: {args.person_label!r}")
    if sop_profile_name == PROFILE_ROLL_SOP_V1:
        if not roll_ids:
            raise ValueError(f"Could not resolve roll class ids from labels: {args.roll_label!r}")
        if not cleaning_cloth_ids:
            raise ValueError(
                f"Could not resolve cleaning cloth class ids from labels: {args.cleaning_cloth_label!r}"
            )
        if not paper_label_ids:
            raise ValueError(f"Could not resolve paper label class ids from labels: {args.paper_label!r}")
    if sop_profile_name == PROFILE_OPERATOR_MVP_A and not helmet_ids and not helmet_disabled:
        if args.require_helmet_class:
            raise ValueError(
                f"Could not resolve helmet class ids from labels: {args.helmet_label!r}. "
                "Provide a metadata.yaml that includes a helmet class (or pass --skip-helmet)."
            )
        warnings.append(
            f"WARNING: Could not resolve helmet class ids from labels: {args.helmet_label!r}. "
            "Helmet check will be disabled (helmet=UNKNOWN)."
        )
        helmet_disabled = True
        helmet_ids = ()

    if sop_profile_name == PROFILE_ROLL_SOP_V1:
        active = set(roll_ids + cleaning_cloth_ids + paper_label_ids)
        if helmet_alerts_enabled:
            active.update(person_ids)
            active.update(helmet_label_ids)
        active_class_ids = tuple(sorted(active))
    else:
        active_class_ids = tuple(sorted(set(person_ids + helmet_ids)))

    if class_conf_thresholds:
        selected_ids = set(active_class_ids)
        unused_overrides = sorted(set(class_conf_thresholds) - selected_ids)
        if unused_overrides:
            names = [class_names.get(class_id, str(class_id)) for class_id in unused_overrides]
            raise ValueError(
                "--label-conf includes classes excluded by the active SOP profile: "
                f"{names}. Add the class to the profile labels or remove the override."
            )

    return ResolvedClassConfig(
        class_names=class_names,
        class_conf_thresholds=class_conf_thresholds,
        active_class_ids=active_class_ids,
        person_ids=person_ids,
        helmet_ids=helmet_ids,
        helmet_disabled=helmet_disabled,
        roll_ids=roll_ids,
        cleaning_cloth_ids=cleaning_cloth_ids,
        paper_label_ids=paper_label_ids,
        warnings=tuple(warnings),
    )


def resolve_run_config(args: argparse.Namespace) -> ResolvedRunConfig:
    source_count = int(args.video is not None) + int(args.webcam is not None) + int(args.rtsp is not None)
    if source_count != 1:
        raise ValueError("Exactly one source must be set: --video or --webcam or --rtsp (or via --config).")

    sop_profile = _resolve_sop_profile(args)
    session_timing = _resolve_session_timing(args, sop_profile)
    operator_rules = _resolve_operator_rules(args, sop_profile)
    classes = _resolve_classes(args, sop_profile.name)
    return ResolvedRunConfig(
        sop_profile=sop_profile,
        session_timing=session_timing,
        operator_rules=operator_rules,
        classes=classes,
    )

from pathlib import Path

import pytest

from Action_Detection_SOP.runtime_config import (
    PROFILE_ROLL_SOP_V1,
    resolve_run_config,
)
from Action_Detection_SOP.safety_alerts import (
    DEFAULT_HELMET_ALERT_CONFIDENCE,
    DEFAULT_HELMET_REQUIRED_SECONDS,
    HelmetAlertConfig,
)
from Scripts.run_sop_mvp import build_parser


def _metadata(path: Path, names: dict[int, str]) -> Path:
    lines = ["names:"]
    for class_id, name in names.items():
        lines.append(f"  {class_id}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_helmet_alert_default_is_shared_by_engine_and_cli() -> None:
    args = build_parser().parse_args(["--video", "sample.mp4"])

    assert DEFAULT_HELMET_REQUIRED_SECONDS == 10
    assert HelmetAlertConfig().required_seconds == DEFAULT_HELMET_REQUIRED_SECONDS
    assert args.helmet_alert_s == DEFAULT_HELMET_REQUIRED_SECONDS
    assert args.helmet_alert_confidence == DEFAULT_HELMET_ALERT_CONFIDENCE


def test_helmet_alert_diagnostics_are_opt_in_and_have_a_storage_cap() -> None:
    default_args = build_parser().parse_args(["--video", "sample.mp4"])
    diagnostic_args = build_parser().parse_args(
        [
            "--video",
            "sample.mp4",
            "--enable-helmet-alerts",
            "--helmet-alert-diagnostics",
            "--helmet-diagnostics-max-mb",
            "64",
        ]
    )

    assert default_args.helmet_alert_diagnostics is False
    assert default_args.helmet_diagnostics_max_mb == 128
    assert diagnostic_args.helmet_alert_diagnostics is True
    assert diagnostic_args.helmet_diagnostics_max_mb == 64


def test_helmet_alert_camera_id_defaults_from_deployment_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOP_HELMET_ALERT_CAMERA_ID", "Camera 16 RW3")

    args = build_parser().parse_args(["--video", "sample.mp4"])

    assert args.helmet_alert_camera_id == "Camera 16 RW3"


def test_resolves_roll_profile_classes_and_timing_defaults(tmp_path: Path) -> None:
    metadata = _metadata(
        tmp_path / "metadata.yaml",
        {
            0: "person",
            1: "helmet",
            2: "roll",
            3: "cleaning_cloth",
            4: "label",
        },
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--video",
            "sample.mp4",
            "--metadata",
            str(metadata),
            "--sop-profile",
            PROFILE_ROLL_SOP_V1,
            "--label-conf",
            "cleaning_cloth=0.08",
        ]
    )

    resolved = resolve_run_config(args)

    assert resolved.sop_profile.name == PROFILE_ROLL_SOP_V1
    assert resolved.session_timing.start_s == 3.0
    assert resolved.session_timing.end_s == 5.0
    assert resolved.classes.active_class_ids == (2, 3, 4)
    assert resolved.classes.roll_ids == (2,)
    assert resolved.classes.cleaning_cloth_ids == (3,)
    assert resolved.classes.paper_label_ids == (4,)
    assert resolved.classes.class_conf_thresholds == {3: 0.08}


def test_roll_profile_includes_person_and_helmet_when_alerts_are_enabled(tmp_path: Path) -> None:
    metadata = _metadata(
        tmp_path / "metadata.yaml",
        {
            0: "person",
            1: "helmet",
            2: "roll",
            3: "cleaning_cloth",
            4: "label",
        },
    )
    roi = tmp_path / "helmet_alert_roi.json"
    roi.write_text('{"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}', encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--video",
            "sample.mp4",
            "--metadata",
            str(metadata),
            "--sop-profile",
            PROFILE_ROLL_SOP_V1,
            "--enable-helmet-alerts",
            "--helmet-alert-roi",
            str(roi),
        ]
    )

    resolved = resolve_run_config(args)

    assert resolved.classes.active_class_ids == (0, 1, 2, 3, 4)
    assert resolved.classes.person_ids == (0,)
    assert resolved.classes.helmet_ids == (1,)
    assert resolved.classes.class_conf_thresholds == {1: DEFAULT_HELMET_ALERT_CONFIDENCE}


def test_explicit_helmet_label_confidence_overrides_alert_floor(tmp_path: Path) -> None:
    metadata = _metadata(
        tmp_path / "metadata.yaml",
        {0: "person", 1: "helmet", 2: "roll", 3: "cleaning_cloth", 4: "label"},
    )
    args = build_parser().parse_args(
        [
            "--video",
            "sample.mp4",
            "--metadata",
            str(metadata),
            "--sop-profile",
            PROFILE_ROLL_SOP_V1,
            "--enable-helmet-alerts",
            "--label-conf",
            "helmet=0.22",
        ]
    )

    resolved = resolve_run_config(args)

    assert resolved.classes.class_conf_thresholds == {1: 0.22}


def test_helmet_alerts_fail_fast_without_person_class(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.yaml", {1: "helmet", 2: "roll", 3: "cleaning_cloth", 4: "label"})
    parser = build_parser()
    args = parser.parse_args(
        [
            "--video",
            "sample.mp4",
            "--metadata",
            str(metadata),
            "--sop-profile",
            PROFILE_ROLL_SOP_V1,
            "--enable-helmet-alerts",
            "--helmet-alert-roi",
            "helmet_alert_roi.json",
        ]
    )

    with pytest.raises(ValueError, match="person class ids"):
        resolve_run_config(args)


def test_helmet_alerts_fail_fast_without_helmet_class(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.yaml", {0: "person", 2: "roll", 3: "cleaning_cloth", 4: "label"})
    parser = build_parser()
    args = parser.parse_args(
        [
            "--video",
            "sample.mp4",
            "--metadata",
            str(metadata),
            "--sop-profile",
            PROFILE_ROLL_SOP_V1,
            "--enable-helmet-alerts",
            "--helmet-alert-roi",
            "helmet_alert_roi.json",
        ]
    )

    with pytest.raises(ValueError, match="helmet class ids"):
        resolve_run_config(args)


def test_resolves_roll_profile_path_timing(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.yaml", {0: "roll", 1: "cleaning_cloth", 2: "label"})
    profile = tmp_path / "profile.json"
    profile.write_text(
        """
{
  "schema_version": 1,
  "session_start_seconds": 3.0,
  "session_end_seconds": 4.0,
  "min_session_seconds": 1.5
}
""".strip(),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["--video", "sample.mp4", "--metadata", str(metadata), "--sop-profile", str(profile)])

    resolved = resolve_run_config(args)

    assert resolved.sop_profile.name == PROFILE_ROLL_SOP_V1
    assert resolved.sop_profile.path == profile
    assert resolved.session_timing.start_s == 3.0
    assert resolved.session_timing.end_s == 4.0
    assert resolved.session_timing.min_session_s == 1.5
    assert resolved.classes.active_class_ids == (0, 1, 2)


def test_default_profile_is_roll_and_cli_timing_overrides_profile(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.yaml", {0: "roll", 1: "cleaning_cloth", 2: "label"})
    args = build_parser().parse_args(["--video", "sample.mp4", "--metadata", str(metadata)])
    resolved = resolve_run_config(args)
    assert resolved.sop_profile.name == PROFILE_ROLL_SOP_V1
    assert resolved.session_timing.start_s == 3.0
    assert resolved.session_timing.end_s == 5.0
    assert resolved.classes.active_class_ids == (0, 1, 2)
    assert resolved.classes.helmet_ids == ()

    profile = tmp_path / "profile.json"
    profile.write_text('{"schema_version": 1, "session_start_seconds": 4, "session_end_seconds": 8}')
    args = build_parser().parse_args([
        "--video", "sample.mp4", "--metadata", str(metadata), "--sop-profile", str(profile),
        "--start-s", "6", "--end-s", "10",
    ])
    resolved = resolve_run_config(args)
    assert resolved.session_timing.start_s == 6.0
    assert resolved.session_timing.end_s == 10.0


def test_removed_operator_profile_has_actionable_error() -> None:
    args = build_parser().parse_args(["--video", "sample.mp4", "--sop-profile", "operator_mvp_a"])
    with pytest.raises(ValueError, match="operator_mvp_a has been removed"):
        resolve_run_config(args)


@pytest.mark.parametrize("flag", ["--roi-dwell-s", "--helmet-s", "--skip-helmet"])
def test_removed_operator_cli_flags_are_rejected(flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--video", "sample.mp4", flag])

from pathlib import Path

from Action_Detection_SOP.runtime_config import (
    PROFILE_OPERATOR_MVP_A,
    PROFILE_ROLL_SOP_V1,
    resolve_run_config,
)
from Scripts.run_sop_mvp import build_parser


def _metadata(path: Path, names: dict[int, str]) -> Path:
    lines = ["names:"]
    for class_id, name in names.items():
        lines.append(f"  {class_id}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    assert resolved.timing.start_s == 1.0
    assert resolved.timing.end_s == 2.0
    assert resolved.classes.active_class_ids == (2, 3, 4)
    assert resolved.classes.roll_ids == (2,)
    assert resolved.classes.cleaning_cloth_ids == (3,)
    assert resolved.classes.paper_label_ids == (4,)
    assert resolved.classes.class_conf_thresholds == {3: 0.08}


def test_resolves_legacy_profile_path_timing(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.yaml", {0: "person", 1: "helmet"})
    profile = tmp_path / "profile.json"
    profile.write_text(
        """
{
  "schema_version": 1,
  "session_start_seconds": 3.0,
  "session_end_seconds": 4.0,
  "min_session_seconds": 1.5,
  "roi_dwell_seconds": 9.0
}
""".strip(),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["--video", "sample.mp4", "--metadata", str(metadata), "--sop-profile", str(profile)])

    resolved = resolve_run_config(args)

    assert resolved.sop_profile.name == PROFILE_OPERATOR_MVP_A
    assert resolved.sop_profile.path == profile
    assert resolved.timing.start_s == 3.0
    assert resolved.timing.end_s == 4.0
    assert resolved.timing.min_session_s == 1.5
    assert resolved.timing.roi_dwell_s == 9.0
    assert resolved.classes.active_class_ids == (0, 1)


def test_operator_profile_missing_helmet_warns_and_disables_helmet(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.yaml", {0: "person"})
    parser = build_parser()
    args = parser.parse_args(["--video", "sample.mp4", "--metadata", str(metadata)])

    resolved = resolve_run_config(args)

    assert resolved.sop_profile.name == PROFILE_OPERATOR_MVP_A
    assert resolved.classes.helmet_disabled is True
    assert resolved.classes.active_class_ids == (0,)
    assert resolved.classes.warnings

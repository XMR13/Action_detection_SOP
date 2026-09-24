import numpy as np

from Action_Detection_SOP.run_config_loader import apply_run_config
from Action_Detection_SOP.roll_color_gate import blue_fraction_on_roll, exclude_blue_rolls
from Action_Detection_SOP.session import RollSessionConfig, RollSessionizer
from Scripts.run_sop_mvp import build_parser
from yolo_kit.types import Detection


BLUE = (145, 65, 35)
GRAY = (105, 105, 105)


def _roll(x1=20, y1=20, x2=140, y2=140):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=0.9, class_id=2)


def test_blue_outer_cover_excludes_completed_roll_but_not_plain_roll():
    frame = np.full((160, 320, 3), GRAY, dtype=np.uint8)
    frame[20:140, 20:108] = BLUE  # outer cover beside the gray end face
    blue_roll = _roll()
    plain_roll = _roll(180, 20, 300, 140)

    eligible, excluded = exclude_blue_rolls(frame, [blue_roll, plain_roll])

    assert eligible == [plain_roll]
    assert excluded == [blue_roll]
    assert blue_fraction_on_roll(frame, blue_roll) > 0.30


def test_blue_scene_and_small_blue_label_do_not_exclude_plain_roll():
    frame = np.full((180, 180, 3), BLUE, dtype=np.uint8)
    frame[20:160, 20:160] = GRAY
    frame[65:95, 65:95] = BLUE  # an attached label is smaller than the cover
    roll = _roll(20, 20, 160, 160)

    eligible, excluded = exclude_blue_rolls(frame, [roll])

    assert eligible == [roll]
    assert excluded == []


def test_uncertain_small_roll_keeps_sop_session_eligible():
    frame = np.full((160, 160, 3), GRAY, dtype=np.uint8)
    tiny_roll = _roll(30, 30, 45, 45)
    sessionizer = RollSessionizer(RollSessionConfig(start_seconds=0.4, end_seconds=0.4, analysis_fps=5))

    for _ in range(2):
        eligible, excluded = exclude_blue_rolls(frame, [tiny_roll])
        event = sessionizer.update(bool(eligible))

    assert excluded == []
    assert event == "start"


def test_roll_exclusion_can_be_enabled_through_run_config():
    parser = build_parser()
    args = parser.parse_args([])

    apply_run_config(
        args=args,
        payload={"exclude_blue_rolls": True, "blue_roll_min_fraction": 0.35},
        cli_dests=set(),
        parser=parser,
    )

    assert args.exclude_blue_rolls is True
    assert args.blue_roll_min_fraction == 0.35

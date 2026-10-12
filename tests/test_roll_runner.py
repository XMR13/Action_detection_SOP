"""Exercise the roll runner's real session/report flow without model inference."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Action_Detection_SOP import runner_mvp
from Action_Detection_SOP.ingest import CaptureInfo
from Scripts.run_sop_mvp import build_parser
from yolo_kit.types import Detection


class _Capture:
    def __init__(
        self, frame_count: int, *, frame_sizes: list[tuple[int, int]] | None = None,
        reconnect_after: int | None = None,
    ) -> None:
        self.frame_count = frame_count
        self.frame_sizes = frame_sizes or [(160, 160)] * frame_count
        self.reconnect_after = reconnect_after
        self.disconnected = False
        self.frames_read = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames_read == self.reconnect_after and not self.disconnected:
            self.disconnected = True
            return False, None
        if self.frames_read == self.frame_count:
            return False, None
        width, height = self.frame_sizes[self.frames_read]
        self.frames_read += 1
        return True, np.zeros((height, width, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


@dataclass(frozen=True)
class _Run:
    out_dir: Path
    args: argparse.Namespace
    capture: _Capture
    class_ids: list[int]
    # Counts observed before each frame's inference: reserved sessions, checklists.
    artifacts_before_frame: dict[int, tuple[int, int]]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    detections: list[list[Detection]],
    *extra_args: str,
    frame_sizes: list[tuple[int, int]] | None = None,
    reconnect_after: int | None = None,
    roi_points: list[list[int]] | None = None,
) -> _Run:
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        "names:\n  0: person\n  1: helmet\n  2: roll\n  3: cleaning_cloth\n  4: label\n",
        encoding="utf-8",
    )
    roi = tmp_path / "roi.json"
    roi.write_text(
        json.dumps({
            "polygon": roi_points or [[0, 0], [159, 0], [159, 159], [0, 159]],
            "frame_size": {"width": 160, "height": 160},
        }),
        encoding="utf-8",
    )
    out_dir = tmp_path / "output"
    args = build_parser().parse_args([
        "--video", "stub.mp4", "--model", str(tmp_path / "stub.onnx"),
        "--metadata", str(metadata), "--roi", str(roi), "--out-dir", str(out_dir),
        "--analysis-fps", "1", "--no-evidence", "--no-thumb", "--no-progress",
        *extra_args,
    ])
    if reconnect_after is not None:
        # Stop on an extra terminal frame so --reconnect does not loop at EOF.
        args.reconnect = True
        args.max_frames = len(detections) + 1
        if frame_sizes is not None:
            frame_sizes = [*frame_sizes, frame_sizes[-1]]
    capture = _Capture(
        len(detections) + int(reconnect_after is not None),
        frame_sizes=frame_sizes, reconnect_after=reconnect_after,
    )
    monkeypatch.setattr(runner_mvp, "open_capture", lambda **kwargs: capture)
    monkeypatch.setattr(
        runner_mvp, "get_capture_info",
        lambda cap: CaptureInfo(fps=1.0, width=160, height=160, frame_count=len(detections)),
    )
    class_ids: list[int] = []

    def load_pipeline(**kwargs: object) -> SimpleNamespace:
        class_ids.extend(kwargs["post_cfg"].class_ids)
        return SimpleNamespace(backend_name="stub")

    monkeypatch.setattr(runner_mvp, "load_pipeline", load_pipeline)
    artifacts_before_frame: dict[int, tuple[int, int]] = {}

    def infer(pipeline: object, frame: np.ndarray) -> tuple[list[Detection], tuple[float, ...]]:
        artifacts_before_frame[capture.frames_read] = (
            len(list(out_dir.glob("sessions/*/session_*"))),
            len(list(out_dir.glob("sessions/*/session_*/checklist.json"))),
        )
        return detections[capture.frames_read - 1], (0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(runner_mvp, "_run_pipeline_timed", infer)
    assert runner_mvp.run_mvp(
        args, args_raw=vars(args).copy(), config_path=None, config_payload=None,
    ) == 0
    assert capture.released
    return _Run(out_dir, args, capture, class_ids, artifacts_before_frame)


def _roll_and_tools() -> list[Detection]:
    return [
        Detection(x1=20, y1=20, x2=120, y2=120, score=0.9, class_id=2),
        Detection(x1=30, y1=30, x2=45, y2=45, score=0.9, class_id=3),
        Detection(x1=70, y1=70, x2=90, y2=90, score=0.9, class_id=4),
    ]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_default_roll_runner_survives_occlusion_and_finalizes_after_five_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Start on frame 3, survive frames 4-7 hidden, recover on 8, end on 13.
    run = _run(
        monkeypatch, tmp_path,
        [_roll_and_tools()] * 3 + [[]] * 4 + [_roll_and_tools()] + [[]] * 6,
    )

    assert (run.args.start_s, run.args.end_s) == (3.0, 5.0)
    assert run.class_ids == [2, 3, 4]
    assert run.artifacts_before_frame[3] == (0, 0)
    assert run.artifacts_before_frame[4] == (1, 0)
    assert run.artifacts_before_frame[8] == (1, 0)
    assert run.artifacts_before_frame[13] == (1, 0)
    assert run.artifacts_before_frame[14] == (1, 1)
    checklists = list(run.out_dir.glob("sessions/*/session_*/checklist.json"))
    assert len(checklists) == 1
    session = _json(checklists[0])
    assert (session["start_time_s"], session["end_time_s"]) == (3.0, 13.0)
    assert session["sop_profile"] == "roll_sop_v1"
    assert session["cleaned"] == session["labeled"] == "DONE"
    assert session["overall_status"] == "SESUAI SOP"
    report = _json(next(run.out_dir.glob("reports/*/daily_report.json")))
    assert report["total_sessions"] == report["overall_compliant"] == 1
    with next(run.out_dir.glob("reports/*/sessions.csv")).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["session_id"] == session["session_id"]
    config = _json(next(run.out_dir.glob("reports/*/run_config.json")))
    assert config["sessionization"]["start_seconds"] == 3.0
    assert config["sessionization"]["end_seconds"] == 5.0


def test_roll_runner_flushes_active_session_at_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run = _run(monkeypatch, tmp_path, [_roll_and_tools()] * 4)

    assert run.artifacts_before_frame[4] == (1, 0)
    checklists = list(run.out_dir.glob("sessions/*/session_*/checklist.json"))
    assert len(checklists) == 1
    session = _json(checklists[0])
    assert (session["start_time_s"], session["end_time_s"]) == (3.0, 4.0)
    assert _json(next(run.out_dir.glob("reports/*/daily_report.json")))["total_sessions"] == 1


def test_person_without_roll_emits_independent_helmet_alert_and_no_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    person = Detection(x1=10, y1=10, x2=100, y2=150, score=0.9, class_id=0)
    run = _run(monkeypatch, tmp_path, [[person]] * 11, "--enable-helmet-alerts")

    assert run.class_ids == [0, 1, 2, 3, 4]
    assert not list(run.out_dir.glob("sessions/*/session_*"))
    alerts = list(run.out_dir.glob("alerts/*/*/alert.json"))
    assert len(alerts) == 1
    alert = _json(alerts[0])
    assert alert["alert_type"] == "NO_HELMET"
    assert alert["related_session_uid"] is None
    report = _json(next(run.out_dir.glob("reports/*/daily_report.json")))
    assert report["sop_profile"] == "roll_sop_v1"
    assert report["total_sessions"] == 0


@pytest.mark.parametrize("reconnect_after", [None, 1])
@pytest.mark.parametrize("new_size", [(160, 160), (80, 80), (320, 320)])
def test_legacy_helmet_roi_uses_decoded_dimensions_without_rescaling_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    new_size: tuple[int, int], reconnect_after: int | None,
) -> None:
    points = [[0, 0], [159, 0], [159, 159], [0, 159]]
    helmet_roi = tmp_path / "helmet_legacy.json"
    helmet_roi.write_text(json.dumps({"polygon": points}), encoding="utf-8")
    person = Detection(
        x1=10, y1=10, x2=min(159, new_size[0] - 5),
        y2=min(150, new_size[1] - 5), score=0.9, class_id=0,
    )
    run = _run(
        monkeypatch, tmp_path, [[]] + [[person]] * 3,
        "--enable-helmet-alerts", "--helmet-alert-s", "2",
        "--helmet-alert-min-person-height", "1",
        "--helmet-alert-roi", str(helmet_roi),
        frame_sizes=[(160, 160)] + [new_size] * 3,
        reconnect_after=reconnect_after,
    )

    alerts = list(run.out_dir.glob("alerts/*/*/alert.json"))
    assert len(alerts) == 1
    assert _json(alerts[0])["trigger_frame_idx"] == 3
    config = _json(next(run.out_dir.glob("reports/*/run_config.json")))
    assert config["helmet_alerts"]["roi"]["frame_size"] is None
    assert config["helmet_alerts"]["roi_resolved"] == {
        "frame_size": list(new_size), "points": points,
    }
    assert _json(helmet_roi) == {"polygon": points}


@pytest.mark.parametrize("box", [(0, 10, 100, 150), (10, 0, 100, 150)])
def test_legacy_helmet_roi_still_rejects_cropped_heads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    box: tuple[int, int, int, int],
) -> None:
    helmet_roi = tmp_path / "helmet_legacy.json"
    helmet_roi.write_text(json.dumps({
        "polygon": [[0, 0], [159, 0], [159, 159], [0, 159]],
    }), encoding="utf-8")
    person = Detection(
        x1=box[0], y1=box[1], x2=box[2], y2=box[3], score=0.9, class_id=0,
    )
    run = _run(
        monkeypatch, tmp_path, [[person]] * 3,
        "--enable-helmet-alerts", "--helmet-alert-s", "1",
        "--helmet-alert-roi", str(helmet_roi),
    )

    assert not list(run.out_dir.glob("alerts/*/*/alert.json"))


@pytest.mark.parametrize("reconnect_after", [None, 1])
@pytest.mark.parametrize("stream_sizes", [
    ((160, 160), (80, 80)),
    ((160, 160), (320, 320)),
    ((1920, 1080), (640, 360)),
])
@pytest.mark.parametrize("helmet_roi", [False, True])
def test_resolution_change_keeps_roll_and_helmet_detection_in_current_rois(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], reconnect_after: int | None,
    stream_sizes: tuple[tuple[int, int], tuple[int, int]], helmet_roi: bool,
) -> None:
    old_size, new_size = stream_sizes
    old_scale_x, old_scale_y = old_size[0] / 160, old_size[1] / 160
    scale_x, scale_y = new_size[0] / 160, new_size[1] / 160
    roll = Detection(
        x1=100 * old_scale_x, y1=100 * old_scale_y,
        x2=140 * old_scale_x, y2=140 * old_scale_y, score=0.9, class_id=2,
    )
    resized_roll = Detection(
        x1=100 * scale_x, y1=100 * scale_y, x2=140 * scale_x, y2=140 * scale_y,
        score=0.9, class_id=2,
    )
    person = Detection(
        x1=90 * scale_x, y1=85 * scale_y, x2=145 * scale_x, y2=149 * scale_y,
        score=0.9, class_id=0,
    )
    helmet_args = [
        "--enable-helmet-alerts", "--helmet-alert-s", "1",
        "--helmet-alert-min-person-height", "1",
    ]
    if helmet_roi:
        helmet_args.extend(["--helmet-alert-roi", str(tmp_path / "roi.json")])
    run = _run(
        monkeypatch, tmp_path, [[roll]] + [[resized_roll, person]] * 3,
        *helmet_args,
        frame_sizes=[old_size] + [new_size] * 3,
        reconnect_after=reconnect_after,
        roi_points=[[80, 80], [150, 80], [150, 150], [80, 150]],
    )

    checklists = list(run.out_dir.glob("sessions/*/session_*/checklist.json"))
    assert len(checklists) == 1
    assert _json(checklists[0])["start_time_s"] == 3.0
    assert len(list(run.out_dir.glob("alerts/*/*/alert.json"))) == 1
    config = _json(next(run.out_dir.glob("reports/*/run_config.json")))
    assert config["roi_resolved"]["frame_size"] == list(new_size)
    assert config["roi_resolved"]["points"][0] == [80 * scale_x, 80 * scale_y]
    assert config["helmet_alerts"]["roi_resolved"]["frame_size"] == list(new_size)
    if helmet_roi:
        assert config["helmet_alerts"]["roi_resolved"] == config["roi_resolved"]
    else:
        assert config["helmet_alerts"]["roi_resolved"]["points"][-2] == [new_size[0] - 1, new_size[1] - 1]
    assert config["reconnect_events"] == int(reconnect_after is not None)
    output = capsys.readouterr().out
    assert output.count("Stream resolution changed") == 1

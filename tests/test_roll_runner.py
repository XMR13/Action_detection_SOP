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
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.frames_read = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames_read == self.frame_count:
            return False, None
        self.frames_read += 1
        return True, np.zeros((160, 160, 3), dtype=np.uint8)

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
) -> _Run:
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        "names:\n  0: person\n  1: helmet\n  2: roll\n  3: cleaning_cloth\n  4: label\n",
        encoding="utf-8",
    )
    roi = tmp_path / "roi.json"
    roi.write_text(
        json.dumps({
            "polygon": [[0, 0], [159, 0], [159, 159], [0, 159]],
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
    capture = _Capture(len(detections))
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

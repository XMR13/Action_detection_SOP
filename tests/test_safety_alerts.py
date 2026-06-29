from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from Action_Detection_SOP.roi import RoiPolygon
from Action_Detection_SOP.safety_alerts import HelmetAlertConfig, HelmetAlertEngine, write_helmet_alert_artifacts
from yolo_kit.types import Detection


def _roi() -> RoiPolygon:
    return RoiPolygon(points=((0, 0), (400, 0), (400, 400), (0, 400)), frame_size=(400, 400))


def _person(*, x1: float = 50, y1: float = 20, x2: float = 150, y2: float = 220, score: float = 0.9) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=0)


def _helmet() -> Detection:
    return Detection(x1=80, y1=30, x2=120, y2=60, score=0.85, class_id=1)


def _engine(*, required_s: float = 5.0, cooldown_s: float = 0.0, min_height: int = 120) -> HelmetAlertEngine:
    return HelmetAlertEngine(
        HelmetAlertConfig(
            required_seconds=required_s,
            analysis_fps=1.0,
            recovery_seconds=2.0,
            absence_seconds=2.0,
            cooldown_seconds=cooldown_s,
            min_person_height_px=min_height,
            head_top_fraction=0.35,
            max_gap_frames=1,
            safety_area_id="helmet_area_test",
        ),
        source="camera-1",
        camera_id="cam_1",
    )


def test_alert_fires_after_sustained_no_helmet_only() -> None:
    engine = _engine(required_s=5.0)
    alerts = []
    for frame_idx in range(1, 5):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )
    assert alerts == []

    alerts.extend(engine.update(time_s=5.0, frame_idx=5, persons=[_person()], helmets=[], safety_roi=_roi()))

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.alert_type == "NO_HELMET"
    assert alert.start_time_s == 1.0
    assert alert.end_time_s == 5.0
    assert alert.primary.height_px == 200.0


def test_alert_does_not_recur_during_same_episode() -> None:
    engine = _engine(required_s=3.0)
    alerts = []
    for frame_idx in range(1, 10):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )

    assert len(alerts) == 1


def test_episode_closes_after_person_leaves_and_allows_new_alert() -> None:
    engine = _engine(required_s=3.0, cooldown_s=0.0)
    alerts = []
    for frame_idx in range(1, 4):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )
    for frame_idx in range(4, 6):
        alerts.extend(engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[], helmets=[], safety_roi=_roi()))
    for frame_idx in range(6, 9):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )

    assert len(alerts) == 2
    assert alerts[0].alert_uid != alerts[1].alert_uid


def test_episode_closes_after_helmet_recovery_and_allows_new_alert() -> None:
    engine = _engine(required_s=3.0, cooldown_s=0.0)
    alerts = []
    for frame_idx in range(1, 4):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )
    for frame_idx in range(4, 6):
        alerts.extend(
            engine.update(
                time_s=float(frame_idx),
                frame_idx=frame_idx,
                persons=[_person()],
                helmets=[_helmet()],
                safety_roi=_roi(),
            )
        )
    for frame_idx in range(6, 9):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )

    assert len(alerts) == 2


def test_ignores_person_below_min_height() -> None:
    engine = _engine(required_s=3.0, min_height=120)
    short_person = _person(y1=20, y2=100)
    alerts = []
    for frame_idx in range(1, 8):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[short_person], helmets=[], safety_roi=_roi())
        )

    assert alerts == []


def test_ignores_person_outside_safety_roi() -> None:
    engine = _engine(required_s=3.0)
    outside_person = _person(x1=500, y1=20, x2=620, y2=250)
    alerts = []
    for frame_idx in range(1, 8):
        alerts.extend(
            engine.update(
                time_s=float(frame_idx),
                frame_idx=frame_idx,
                persons=[outside_person],
                helmets=[],
                safety_roi=_roi(),
            )
        )

    assert alerts == []


def test_associated_helmet_prevents_no_helmet_alert() -> None:
    engine = _engine(required_s=3.0)
    alerts = []
    for frame_idx in range(1, 8):
        alerts.extend(
            engine.update(
                time_s=float(frame_idx),
                frame_idx=frame_idx,
                persons=[_person()],
                helmets=[_helmet()],
                safety_roi=_roi(),
            )
        )

    assert alerts == []


def test_multiple_people_create_one_scene_alert_with_primary_largest() -> None:
    engine = _engine(required_s=2.0)
    smaller = _person(x1=40, y1=40, x2=100, y2=180, score=0.95)
    larger = _person(x1=180, y1=20, x2=300, y2=260, score=0.8)
    alerts = []
    for frame_idx in range(1, 3):
        alerts.extend(
            engine.update(
                time_s=float(frame_idx),
                frame_idx=frame_idx,
                persons=[smaller, larger],
                helmets=[],
                safety_roi=_roi(),
            )
        )

    assert len(alerts) == 1
    payload = alerts[0].to_payload(run_start_dt=None, fallback_date="2026-06-29")
    assert payload["person_count"] == 2
    assert payload["person_height_px"] == 240.0


def test_write_helmet_alert_artifacts(tmp_path: Path) -> None:
    engine = _engine(required_s=1.0)
    alerts = engine.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[], safety_roi=_roi())
    assert len(alerts) == 1

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    alert_dir = write_helmet_alert_artifacts(
        out_dir=tmp_path,
        date="2026-06-29",
        alert=alerts[0],
        frame_bgr=frame,
        safety_roi=_roi(),
        run_start_dt=datetime(2026, 6, 29, 8, 0, 0),
    )

    assert alert_dir == tmp_path / "alerts" / "2026-06-29" / alerts[0].alert_uid
    assert (alert_dir / "thumbnail.jpg").exists()
    payload = json.loads((alert_dir / "alert.json").read_text(encoding="utf-8"))
    assert payload["alert_uid"] == alerts[0].alert_uid
    assert payload["status"] == "PENDING"
    assert payload["machine_status"] == "NO_HELMET"
    assert payload["artifacts"]["thumbnail"] == "thumbnail.jpg"
    assert payload["artifacts"]["clip"] is None

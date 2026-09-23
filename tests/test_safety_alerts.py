from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from Action_Detection_SOP.roi import RoiPolygon
from Action_Detection_SOP.safety_alerts import (
    DEFAULT_HELMET_ALERT_CONFIDENCE,
    HELMET_DIAGNOSTICS_SCHEMA_VERSION,
    HelmetDiagnosticAssociation,
    HelmetDiagnosticEvent,
    HelmetDiagnosticObservation,
    HelmetDiagnosticTracker,
    HelmetAlertConfig,
    HelmetAlertEngine,
    write_helmet_alert_artifacts,
)
from yolo_kit.types import Detection


def _roi() -> RoiPolygon:
    return RoiPolygon(points=((0, 0), (400, 0), (400, 400), (0, 400)), frame_size=(400, 400))


def _person(*, x1: float = 50, y1: float = 20, x2: float = 150, y2: float = 220, score: float = 0.9) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=0)


def _helmet(*, score: float = 0.85) -> Detection:
    return Detection(x1=80, y1=30, x2=120, y2=60, score=score, class_id=1)


def _engine(
    *,
    required_s: float = 5.0,
    cooldown_s: float = 0.0,
    min_height: int = 120,
    diagnostics: bool = False,
) -> HelmetAlertEngine:
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
        diagnostics_enabled=diagnostics,
    )


def _helmet_at(*, x1: float = 80, score: float = 0.85) -> Detection:
    return Detection(x1=x1, y1=30, x2=x1 + 40, y2=60, score=score, class_id=1)


def test_helmet_diagnostic_contract_is_versioned_and_stable() -> None:
    assert HELMET_DIAGNOSTICS_SCHEMA_VERSION == 2
    association = HelmetDiagnosticAssociation(
        helmet_box=(80.0, 30.0, 120.0, 60.0),
        helmet_score=0.33333,
        center_inside_person=True,
        center_inside_head=True,
        center_distance_to_person_px=0.5,
    )
    observation = HelmetDiagnosticObservation(
        diagnostic_track_id=1,
        frame_idx=7,
        time_s=1.23456,
        person_box=(50.0, 20.0, 150.0, 220.0),
        person_height_px=200.0,
        normalized_position=(0.25, 0.3),
        at_frame_edge=True,
        head_visible=True,
        helmet_score_history=(0.33333,),
        helmet_hit_count=1,
        helmet_observation_count=1,
        helmet_hit_rate=1.0,
        best_helmet_score=0.33333,
        associations=(association,),
    )
    event = HelmetDiagnosticEvent(
        event="alert_emitted",
        reason="sustained_no_helmet",
        frame_idx=7,
        time_s=1.23456,
        episode_start_frame_idx=1,
        episode_start_time_s=0.0,
    )

    payload = event.as_payload()
    assert payload["schema_version"] == HELMET_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["time_s"] == 1.235
    assert observation.as_payload()["associations"][0]["helmet_score"] == 0.333
    assert "helmet_center" not in association.as_payload()
    assert "observations" not in payload
    assert "source" not in payload
    assert json.dumps(payload, sort_keys=True) == json.dumps(event.as_payload(), sort_keys=True)

    with pytest.raises(ValueError, match="reason must not be empty"):
        HelmetDiagnosticEvent(
            event="alert_emitted",
            reason="",
            frame_idx=7,
            time_s=1.0,
            episode_start_frame_idx=1,
            episode_start_time_s=0.0,
        )


def test_diagnostic_tracker_matches_people_and_bounds_history() -> None:
    tracker = HelmetDiagnosticTracker(max_missed_frames=1, score_history_limit=2)
    first = tracker.update(
        persons=[_person()],
        helmets=[_helmet_at(score=0.2)],
        frame_idx=1,
        time_s=1.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )
    second = tracker.update(
        persons=[_person(x1=52, x2=152)],
        helmets=[_helmet_at(x1=82, score=0.3)],
        frame_idx=2,
        time_s=2.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )
    third = tracker.update(
        persons=[_person(x1=54, x2=154)],
        helmets=[_helmet_at(x1=84, score=0.4)],
        frame_idx=3,
        time_s=3.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )

    assert first[0].diagnostic_track_id == second[0].diagnostic_track_id == third[0].diagnostic_track_id
    assert third[0].helmet_observation_count == 3
    assert third[0].helmet_hit_count == 3
    assert third[0].helmet_hit_rate == 1.0
    assert third[0].best_helmet_score == 0.4
    assert third[0].helmet_score_history == (0.3, 0.4)
    assert third[0].track_history_truncated is True
    assert third[0].associations[0].center_inside_head is True
    assert tracker.active_track_count == 1


def test_diagnostic_tracker_preserves_associated_evidence_when_capping_payload() -> None:
    tracker = HelmetDiagnosticTracker()
    unrelated = [
        Detection(
            x1=200 + (index * 20),
            y1=200,
            x2=220 + (index * 20),
            y2=230,
            score=0.9,
            class_id=1,
        )
        for index in range(8)
    ]
    observations = tracker.update(
        persons=[_person()],
        helmets=unrelated + [_helmet_at(score=0.2)],
        frame_idx=1,
        time_s=1.0,
        frame_size=(1000, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )

    observation = observations[0]
    assert observation.helmet_hit_count == 1
    assert observation.best_helmet_score == 0.2
    assert len(observation.associations) == 8
    assert any(item.center_inside_head for item in observation.associations)


def test_diagnostic_tracker_uses_helmet_box_center_for_association() -> None:
    tracker = HelmetDiagnosticTracker()
    observation = tracker.update(
        persons=[_person(x1=100, x2=200)],
        helmets=[Detection(x1=80, y1=30, x2=110, y2=60, score=0.9, class_id=1)],
        frame_idx=1,
        time_s=1.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )[0]

    association = observation.associations[0]
    assert association.center_inside_person is False
    assert association.center_inside_head is False
    assert observation.helmet_hit_count == 0
    assert observation.best_helmet_score is None


def test_diagnostic_tracker_rejects_history_limit_above_contract_bound() -> None:
    with pytest.raises(ValueError, match="score_history_limit must be <= 32"):
        HelmetDiagnosticTracker(score_history_limit=33)

    with pytest.raises(ValueError, match="diagnostic_tracker requires diagnostics_enabled=True"):
        HelmetAlertEngine(
            HelmetAlertConfig(),
            source="camera-1",
            diagnostic_tracker=HelmetDiagnosticTracker(),
        )


def test_diagnostic_sidecar_isolates_malformed_diagnostic_inputs() -> None:
    disabled = _engine(diagnostics=False)
    enabled = _engine(diagnostics=True)
    kwargs = {
        "time_s": "not-a-time",
        "frame_idx": "not-a-frame",
        "persons": [],
        "helmets": [],
        "safety_roi": _roi(),
    }

    assert disabled.update(**kwargs) == ()
    assert enabled.update(**kwargs) == ()
    assert enabled.diagnostic_error_count == 1


def test_diagnostic_tracker_expires_missed_tracks_without_reusing_ids() -> None:
    tracker = HelmetDiagnosticTracker(max_missed_frames=1)
    first = tracker.update(
        persons=[_person()],
        helmets=[],
        frame_idx=1,
        time_s=1.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )
    tracker.update(
        persons=[],
        helmets=[],
        frame_idx=2,
        time_s=2.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )
    tracker.update(
        persons=[],
        helmets=[],
        frame_idx=3,
        time_s=3.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )
    replacement = tracker.update(
        persons=[_person()],
        helmets=[],
        frame_idx=4,
        time_s=4.0,
        frame_size=(400, 400),
        head_top_fraction=0.35,
        verification_confidence=0.15,
    )

    assert first[0].diagnostic_track_id == 1
    assert replacement[0].diagnostic_track_id == 2
    assert tracker.active_track_count == 1


def test_diagnostics_do_not_change_alert_decisions_or_timing() -> None:
    disabled = _engine(required_s=3.0, diagnostics=False)
    enabled = _engine(required_s=3.0, diagnostics=True)
    disabled_alerts = []
    enabled_alerts = []
    for frame_idx in range(1, 4):
        kwargs = {
            "time_s": float(frame_idx),
            "frame_idx": frame_idx,
            "persons": [_person()],
            "helmets": [],
            "safety_roi": _roi(),
        }
        disabled_alerts.extend(disabled.update(**kwargs))
        enabled_alerts.extend(enabled.update(**kwargs))

    assert disabled_alerts == enabled_alerts
    assert [(alert.start_time_s, alert.end_time_s, alert.trigger_frame_idx) for alert in disabled_alerts] == [
        (1.0, 3.0, 3)
    ]
    observations = enabled.pop_diagnostic_observations()
    assert len(observations) == 3
    diagnostic_events = enabled.pop_diagnostic_events()
    assert [event.event for event in diagnostic_events] == ["episode_started", "alert_emitted"]
    assert diagnostic_events[-1].alert_uid == enabled_alerts[0].alert_uid
    assert enabled.diagnostic_error_count == 0


def test_diagnostic_events_explain_weak_helmet_cancel_and_recovery() -> None:
    cancelled = _engine(required_s=2.0, diagnostics=True)
    cancelled.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[], safety_roi=_roi())
    cancelled.update(
        time_s=2.0,
        frame_idx=2,
        persons=[_person()],
        helmets=[_helmet(score=0.2)],
        safety_roi=_roi(),
    )
    cancellation_events = cancelled.pop_diagnostic_events()
    assert [(event.event, event.reason) for event in cancellation_events] == [
        ("episode_started", "no_helmet_candidate"),
        ("episode_cancelled", "weak_helmet_verified"),
    ]

    recovered = _engine(required_s=1.0, diagnostics=True)
    alerts = recovered.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[], safety_roi=_roi())
    recovered.update(time_s=2.0, frame_idx=2, persons=[_person()], helmets=[_helmet()], safety_roi=_roi())
    recovered.update(time_s=3.0, frame_idx=3, persons=[_person()], helmets=[_helmet()], safety_roi=_roi())

    events = recovered.pop_diagnostic_events()
    assert len(alerts) == 1
    assert [(event.event, event.reason) for event in events] == [
        ("episode_started", "no_helmet_candidate"),
        ("alert_emitted", "sustained_no_helmet"),
        ("episode_closed", "helmet_recovered"),
    ]


def test_alert_fires_only_after_ten_seconds_without_helmet() -> None:
    assert HelmetAlertConfig().required_seconds == 10.0
    assert HelmetAlertConfig().verification_confidence == DEFAULT_HELMET_ALERT_CONFIDENCE
    engine = _engine(required_s=10.0)
    alerts = []
    for frame_idx in range(1, 10):
        alerts.extend(
            engine.update(time_s=float(frame_idx), frame_idx=frame_idx, persons=[_person()], helmets=[], safety_roi=_roi())
        )
    assert alerts == []

    alerts.extend(engine.update(time_s=10.0, frame_idx=10, persons=[_person()], helmets=[], safety_roi=_roi()))

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.alert_type == "NO_HELMET"
    assert alert.start_time_s == 1.0
    assert alert.end_time_s == 10.0
    assert alert.primary.height_px == 200.0


def test_weak_helmet_is_checked_at_alert_boundary() -> None:
    engine = _engine(required_s=2.0)
    weak_helmet = _helmet(score=0.2)

    assert engine.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[weak_helmet], safety_roi=_roi()) == ()
    assert engine.update(time_s=2.0, frame_idx=2, persons=[_person()], helmets=[weak_helmet], safety_roi=_roi()) == ()


def test_alert_payload_logs_weak_helmet_score_and_thresholds() -> None:
    engine = _engine(required_s=2.0)
    weak_helmet = _helmet(score=0.2)

    engine.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[weak_helmet], safety_roi=_roi())
    alerts = engine.update(time_s=2.0, frame_idx=2, persons=[_person()], helmets=[], safety_roi=_roi())

    assert len(alerts) == 1
    payload = alerts[0].to_payload(run_start_dt=None, fallback_date="2026-09-14")
    assert payload["best_helmet_score"] == 0.2
    assert payload["helmet_confidence_floor"] == DEFAULT_HELMET_ALERT_CONFIDENCE
    assert payload["helmet_strong_confidence"] == 0.35


def test_alert_can_use_wall_clock_timestamps_for_live_sources() -> None:
    engine = _engine(required_s=2.0)
    start_wall = datetime(2026, 9, 7, 23, 59, 59)
    engine.update(
        time_s=1.0,
        frame_idx=1,
        persons=[_person()],
        helmets=[],
        safety_roi=_roi(),
        wall_dt=start_wall,
    )
    alerts = engine.update(
        time_s=2.0,
        frame_idx=2,
        persons=[_person()],
        helmets=[],
        safety_roi=_roi(),
        wall_dt=datetime(2026, 9, 8, 0, 0, 1),
    )

    assert len(alerts) == 1
    payload = alerts[0].to_payload(run_start_dt=None, fallback_date="unknown")
    assert payload["start_date"] == "2026-09-07"
    assert payload["end_date"] == "2026-09-08"


def test_alert_redacts_credentials_from_source_payload_and_uid() -> None:
    engine = HelmetAlertEngine(
        HelmetAlertConfig(required_seconds=1.0, analysis_fps=1.0),
        source="rtsp://camera_user:camera_password@10.77.77.1:554/Streaming/Channels/1601?token=secret",
    )

    alerts = engine.update(
        time_s=1.0,
        frame_idx=1,
        persons=[_person()],
        helmets=[],
        safety_roi=_roi(),
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.source == "rtsp://10.77.77.1:554/Streaming/Channels/1601"
    assert alert.alert_uid.startswith("alert_no_helmet_000001_")
    assert "camera_user" not in alert.alert_uid
    assert "camera_password" not in alert.alert_uid
    assert "secret" not in json.dumps(alert.to_payload(run_start_dt=None, fallback_date="2026-09-11"))


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

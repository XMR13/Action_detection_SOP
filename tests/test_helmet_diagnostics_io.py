import json
from pathlib import Path

from Action_Detection_SOP.helmet_diagnostics_capture import HelmetDiagnosticCapture
from Action_Detection_SOP.helmet_diagnostics_io import HelmetDiagnosticJsonlWriter
from Action_Detection_SOP.roi import RoiPolygon
from Action_Detection_SOP.safety_alerts import HelmetAlertConfig, HelmetAlertEngine
from yolo_kit.types import Detection


def _person() -> Detection:
    return Detection(x1=50, y1=20, x2=150, y2=220, score=0.9, class_id=0)


def _roi() -> RoiPolygon:
    return RoiPolygon(points=((0, 0), (400, 0), (400, 400), (0, 400)), frame_size=(400, 400))


def test_jsonl_writer_records_frames_events_and_redacts_sources(tmp_path: Path) -> None:
    engine = HelmetAlertEngine(
        HelmetAlertConfig(required_seconds=1.0, analysis_fps=1.0, min_person_height_px=120),
        source="camera-1",
        camera_id="cam_1",
        diagnostics_enabled=True,
    )
    alerts = engine.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[], safety_roi=_roi())
    observations = engine.pop_diagnostic_observations()
    events = engine.pop_diagnostic_events()

    writer = HelmetDiagnosticJsonlWriter(
        out_dir=tmp_path,
        date="2026-09-23",
        source="rtsp://alice:secret@camera.local/stream?token=private",
        camera_id="rtsp://operator:secret@camera.local/camera",
        context={"model_sha256": "abc123"},
        max_total_bytes=64 * 1024,
    )
    assert writer.write_segment(segment_id=0, reason="initial_capture", frame_idx=0, time_s=0.0)
    assert writer.write_frame(
        frame_idx=1,
        time_s=1.0,
        segment_id=0,
        observations=observations,
        events=events,
        alert_uids=[alert.alert_uid for alert in alerts],
    )
    assert writer.write_frame(
        frame_idx=2,
        time_s=2.0,
        segment_id=0,
        observations=(),
        events=(),
    )
    writer.close()

    records = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["record_type"] == "run_start"
    assert records[0]["source"] == "rtsp://camera.local/stream"
    assert records[0]["camera_id"] == "rtsp://camera.local/camera"
    assert records[1]["record_type"] == "capture_segment"
    assert records[2]["observations"][0]["diagnostic_track_id"] == 1
    assert records[2]["events"][0]["event"] == "episode_started"
    assert "observations" not in records[2]["events"][0]
    assert "source" not in records[2]["events"][0]
    assert records[2]["alert_uids"] == [alerts[0].alert_uid]
    assert records[3]["observations"] == []
    assert records[-1]["record_type"] == "run_end"
    assert "secret" not in writer.path.read_text(encoding="utf-8")
    assert "token=private" not in writer.path.read_text(encoding="utf-8")
    assert "alerts" not in writer.path.parts


def test_jsonl_writer_stops_at_global_storage_cap(tmp_path: Path) -> None:
    writer = HelmetDiagnosticJsonlWriter(
        out_dir=tmp_path,
        date="2026-09-23",
        source="camera-1",
        camera_id=None,
        context={},
        max_total_bytes=4096,
    )
    accepted_frames = 0
    for frame_idx in range(100):
        if not writer.write_frame(
            frame_idx=frame_idx,
            time_s=float(frame_idx),
            segment_id=0,
            observations=(),
            events=(),
        ):
            break
        accepted_frames += 1
    writer.close()

    assert accepted_frames > 0
    assert writer.truncated is True
    assert writer.summary()["disabled_reason"] == "storage_limit"
    assert writer.path.stat().st_size <= 4096
    assert json.loads(writer.path.read_text(encoding="utf-8").splitlines()[-1])["truncated"] is True


def test_diagnostic_capture_drains_frames_and_marks_new_segments(tmp_path: Path) -> None:
    engine = HelmetAlertEngine(
        HelmetAlertConfig(required_seconds=10.0, analysis_fps=1.0, min_person_height_px=120),
        source="camera-1",
        diagnostics_enabled=True,
    )
    capture = HelmetDiagnosticCapture(
        engine=engine,
        out_dir=tmp_path,
        date="2026-09-23",
        source="camera-1",
        camera_id=None,
        context={},
        max_total_bytes=64 * 1024,
    )

    assert capture.start()
    capture.persist_frame(frame_idx=0, time_s=0.0)
    engine.update(time_s=1.0, frame_idx=1, persons=[_person()], helmets=[], safety_roi=_roi())
    capture.persist_frame(frame_idx=1, time_s=1.0)
    capture.start_segment(reason="capture_reconnected", frame_idx=2, time_s=2.0)
    engine.update(time_s=2.0, frame_idx=2, persons=[_person()], helmets=[], safety_roi=_roi())
    capture.persist_frame(frame_idx=2, time_s=2.0)
    engine.flush(time_s=3.0, frame_idx=3)
    capture.close(frame_idx=3, time_s=3.0)

    assert capture.path is not None
    records = [json.loads(line) for line in capture.path.read_text(encoding="utf-8").splitlines()]
    assert [record["record_type"] for record in records] == [
        "run_start",
        "frame",
        "capture_segment",
        "frame",
        "final_drain",
        "run_end",
    ]
    assert records[2]["reason"] == "capture_reconnected"
    assert records[1]["observations"][0]["diagnostic_track_id"] == 1
    assert records[3]["observations"][0]["diagnostic_track_id"] == 1
    assert records[4]["events"][0]["reason"] == "runner_flush"
    assert capture.summary()["enabled"] is True

from __future__ import annotations

import json
from pathlib import Path

from Scripts.analyze_helmet_diagnostics import TrackSummary, _analyze_file, main


SCHEMA_VERSION = 2


def _observation(
    *,
    track_id: int = 1,
    frame_idx: int = 1,
    time_s: float = 1.0,
    height: float = 180.0,
    position: tuple[float, float] = (0.25, 0.6),
    edge: bool = True,
    head_visible: bool = False,
    hits: int = 1,
    observations: int = 4,
    hit_rate: float = 0.25,
    best_score: float = 0.42,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_track_id": track_id,
        "frame_idx": frame_idx,
        "time_s": time_s,
        "person_box": [10.0, 20.0, 100.0, 200.0],
        "person_height_px": height,
        "normalized_position": list(position),
        "at_frame_edge": edge,
        "head_visible": head_visible,
        "helmet_score_history": [0.22, best_score],
        "helmet_hit_count": hits,
        "helmet_observation_count": observations,
        "helmet_hit_rate": hit_rate,
        "best_helmet_score": best_score,
        "associations": [
            {
                "helmet_box": [15.0, 20.0, 35.0, 40.0],
                "helmet_score": best_score,
                "center_inside_person": True,
                "center_inside_head": False,
                "center_distance_to_person_px": 4.0,
            }
        ],
        "history": {"length": 4, "limit": 32, "truncated": False},
    }


def _event() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "event": "episode_started",
        "reason": "no_helmet_candidate",
        "frame_idx": 1,
        "time_s": 1.0,
        "episode_start_frame_idx": 1,
        "episode_start_time_s": 1.0,
        "alert_uid": None,
    }


def _run_records(day: str, *, run_id: str = "run-one") -> list[dict]:
    return [
        {
            "record_type": "run_start",
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "started_at_utc": f"{day}T00:00:00+00:00",
            "source": "camera-1",
            "camera_id": None,
            "context": {},
            "max_total_bytes": 100000,
        },
        {
            "record_type": "capture_segment",
            "logged_at_utc": f"{day}T00:00:01+00:00",
            "segment_id": 0,
            "reason": "initial_capture",
            "frame_idx": 0,
            "time_s": 0.0,
        },
        {
            "record_type": "frame",
            "logged_at_utc": f"{day}T00:00:04+00:00",
            "segment_id": 0,
            "frame_idx": 1,
            "time_s": 1.0,
            "alert_uids": [],
            "observations": [_observation()],
            "events": [_event()],
        },
        {
            "record_type": "frame",
            "logged_at_utc": f"{day}T00:00:07+00:00",
            "segment_id": 0,
            "frame_idx": 2,
            "time_s": 2.0,
            "alert_uids": [],
            "observations": [
                _observation(
                    frame_idx=2,
                    time_s=2.0,
                    height=200.0,
                    position=(0.75, 0.4),
                    edge=False,
                    head_visible=True,
                    hits=2,
                    observations=5,
                    hit_rate=0.4,
                    best_score=0.51,
                )
            ],
            "events": [],
        },
        {
            "record_type": "run_end",
            "ended_at_utc": f"{day}T00:00:08+00:00",
            "frames_written": 2,
            "observations_written": 2,
            "events_written": 1,
            "truncated": False,
            "failed": False,
            "disabled_reason": None,
        },
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records), encoding="utf-8")


def test_analyzer_summarizes_run_events_and_track_traits(tmp_path: Path, capsys) -> None:
    path = tmp_path / "2026-09-23" / "run_one.jsonl"
    _write_jsonl(path, _run_records("2026-09-23"))

    assert main([str(path)]) == 0
    output = capsys.readouterr().out

    assert "1 file(s), 1 complete, 0 incomplete" in output
    assert "duration 00:00:08" in output
    assert "2 observations" in output
    assert "episode_started/no_helmet_candidate=1" in output
    assert "run_start=1" in output and "frame=2" in output and "run_end=1" in output
    assert "run_one.jsonl | 0:1 | 2 | 190 [180-200] | 0.50,0.50" in output
    assert "50%/50% | 2/5 (40%) | 0.510 | 100%/0%" in output
    assert "descriptive diagnostics only" in output.lower()


def test_run_start_requires_integer_supported_schema_version(tmp_path: Path, capsys) -> None:
    records = _run_records("2026-09-23")
    records[0]["schema_version"] = 2.0
    path = tmp_path / "float_schema.jsonl"
    _write_jsonl(path, records)

    assert main([str(path)]) == 0
    output = capsys.readouterr().out
    run_line = next(line for line in output.splitlines() if line.startswith("float_schema.jsonl |"))

    assert "0 complete, 1 incomplete" in output
    assert "INVALID_START" in run_line
    assert "SCHEMA:missing/invalid" in run_line
    assert not run_line.endswith("| OK")


def test_extreme_timestamp_offsets_are_reported_without_aborting_scan(tmp_path: Path, capsys) -> None:
    root = tmp_path / "extreme_timestamps"
    cases = [
        (
            "minimum_start",
            0,
            "started_at_utc",
            "0001-01-01T00:00:00+23:59",
            "run_start has invalid started_at_utc",
            "INVALID_START",
        ),
        (
            "maximum_end",
            -1,
            "ended_at_utc",
            "9999-12-31T23:59:59-23:59",
            "run_end has invalid ended_at_utc",
            "INVALID_END",
        ),
    ]
    for name, record_index, field_name, timestamp, expected_detail, expected_status in cases:
        records = _run_records("2026-09-23")
        records[record_index][field_name] = timestamp
        _write_jsonl(root / f"{name}.jsonl", records)

    assert main([str(root)]) == 0
    output = capsys.readouterr().out

    assert "2 file(s), 0 complete, 2 incomplete" in output
    for name, _, _, _, expected_detail, expected_status in cases:
        run_line = next(line for line in output.splitlines() if line.startswith(f"{name}.jsonl |"))
        assert expected_detail in output
        assert expected_status in run_line
        assert "MALFORMED" in run_line


def test_track_summary_uses_streaming_aggregates() -> None:
    track = TrackSummary(segment_id=0, track_id=1)
    observation = _observation()
    for _ in range(2000):
        track.add(observation)

    assert track.sample_count == 2000
    assert track.height_mean == 180.0
    assert track.height_min == track.height_max == 180.0
    assert track.x_mean == 0.25 and track.y_mean == 0.6
    assert not hasattr(track, "heights")
    assert not hasattr(track, "association_scores")


def test_analyzer_reports_malformed_truncated_and_missing_run_end(tmp_path: Path, capsys) -> None:
    path = tmp_path / "2026-09-23" / "partial.jsonl"
    path.parent.mkdir(parents=True)
    header = _run_records("2026-09-23")[0]
    malformed_observation = {"schema_version": SCHEMA_VERSION, "diagnostic_track_id": 1}
    frame = {
        "record_type": "frame",
        "logged_at_utc": "2026-09-23T00:00:03+00:00",
        "segment_id": 0,
        "frame_idx": 1,
        "time_s": 2.0,
        "observations": [malformed_observation],
        "events": [],
    }
    path.write_text(
        json.dumps(header) + "\n" + json.dumps(frame) + "\n" + '{"record_type":"frame","observations":',
        encoding="utf-8",
    )

    assert main([str(path)]) == 0
    output = capsys.readouterr().out

    assert "0 complete, 1 incomplete" in output
    assert "NO_END" in output and "OPEN_TAIL" in output
    assert "MALFORMED:2" in output
    assert "invalid observation[0]" in output
    assert "invalid JSON (JSONDecodeError)" in output
    assert "~00:00:03" in output


def test_analyzer_recursively_aggregates_multiple_days(tmp_path: Path, capsys) -> None:
    root = tmp_path / "diagnostics"
    _write_jsonl(root / "2026-09-22" / "run_a.jsonl", _run_records("2026-09-22", run_id="a"))
    _write_jsonl(root / "2026-09-23" / "run_b.jsonl", _run_records("2026-09-23", run_id="b"))

    assert main([str(root), "--max-tracks", "0"]) == 0
    output = capsys.readouterr().out

    assert "2 file(s), 2 complete, 0 incomplete" in output
    assert "2026-09-22 | 1 1 | 00:00:08" in output
    assert "2026-09-23 | 1 1 | 00:00:08" in output
    assert "episode_started/no_helmet_candidate=2" in output
    assert "run_a.jsonl" in output and "run_b.jsonl" in output
    assert "top 2 of 2" in output.lower()


def test_analyzer_rejects_reversed_duplicate_and_post_end_records(tmp_path: Path, capsys) -> None:
    base = _run_records("2026-09-23")
    run_end_first = list(base)
    terminal = run_end_first.pop()
    run_end_first.insert(0, terminal)
    after_end = list(base) + [{"record_type": "trailing_marker"}]
    duplicate_start = list(base)
    duplicate_start.insert(1, dict(duplicate_start[0]))
    duplicate_end = list(base) + [dict(base[-1])]

    scenarios = [
        ("run_end_first", run_end_first, "record appears before run_start"),
        ("after_end", after_end, "nonblank record follows run_end"),
        ("duplicate_start", duplicate_start, "duplicate run_start"),
        ("duplicate_end", duplicate_end, "MULTIPLE_ENDS"),
    ]
    for name, records, expected_detail in scenarios:
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, records)

        assert main([str(path)]) == 0
        output = capsys.readouterr().out
        assert "0 complete, 1 incomplete" in output
        assert "ORDERING:" in output
        assert expected_detail in output


def test_frame_may_follow_run_start_without_capture_segment(tmp_path: Path, capsys) -> None:
    records = [record for record in _run_records("2026-09-23") if record["record_type"] != "capture_segment"]
    path = tmp_path / "no_segment_marker.jsonl"
    _write_jsonl(path, records)

    assert main([str(path)]) == 0
    output = capsys.readouterr().out
    run_line = next(line for line in output.splitlines() if line.startswith("no_segment_marker.jsonl |"))

    assert "1 complete, 0 incomplete" in output
    assert run_line.endswith("| OK")


def test_analyzer_groups_full_overnight_duration_to_run_start_day(tmp_path: Path, capsys) -> None:
    records = _run_records("2026-09-23")
    records[0]["started_at_utc"] = "2026-09-23T23:30:00+00:00"
    records[-1]["ended_at_utc"] = "2026-09-24T00:30:00+00:00"
    path = tmp_path / "overnight.jsonl"
    _write_jsonl(path, records)

    assert main([str(path)]) == 0
    output = capsys.readouterr().out
    daily = output.split("By run start UTC day", maxsplit=1)[1].split("\nRecord types:", maxsplit=1)[0]

    assert "full duration grouped to start day" in output
    assert "2026-09-23 | 1 1 | 01:00:00" in daily
    assert "2026-09-24 |" not in daily


def test_analyzer_rejects_oversized_integer_and_continues_scanning(tmp_path: Path, capsys) -> None:
    records = _run_records("2026-09-23")
    records[2]["observations"][0]["person_height_px"] = 10**400
    path = tmp_path / "oversized.jsonl"
    _write_jsonl(path, records)

    assert main([str(path)]) == 0
    output = capsys.readouterr().out

    assert "1 observations" in output
    assert "run_end count mismatch: observations_written=2, parsed=1" in output
    assert "MALFORMED:2" in output
    assert "COUNTS_MISMATCH" in output
    assert "oversized.jsonl | 0:1 | 1 | 200 [200-200]" in output


def test_invalid_run_end_metadata_is_not_complete(tmp_path: Path, capsys) -> None:
    records = _run_records("2026-09-23")
    records[-1].update(
        {
            "ended_at_utc": "not-a-timestamp",
            "truncated": "false",
            "failed": 0,
            "frames_written": -1,
            "observations_written": "2",
        }
    )
    path = tmp_path / "invalid_end.jsonl"
    _write_jsonl(path, records)

    assert main([str(path)]) == 0
    output = capsys.readouterr().out

    assert "0 complete, 1 incomplete" in output
    assert "INVALID_END" in output
    assert "NO_END" not in output
    assert "run_end has invalid ended_at_utc" in output
    summary = _analyze_file(path, path.name)
    assert summary.valid_run_end_count == 0
    assert summary.terminal_counts_match is False
    assert summary.malformed_count == 5


def test_run_end_count_mismatch_is_not_complete_but_truncated_and_failed_can_be(tmp_path: Path, capsys) -> None:
    mismatched = _run_records("2026-09-23")
    mismatched[-1]["events_written"] = 3
    mismatch_path = tmp_path / "mismatch.jsonl"
    _write_jsonl(mismatch_path, mismatched)

    assert main([str(mismatch_path)]) == 0
    mismatch_output = capsys.readouterr().out
    assert "0 complete, 1 incomplete" in mismatch_output
    assert "COUNTS_MISMATCH" in mismatch_output
    assert "events_written=3, parsed=1" in mismatch_output

    terminated = _run_records("2026-09-23")
    terminated[-1]["truncated"] = True
    terminated[-1]["failed"] = True
    terminated_path = tmp_path / "terminated.jsonl"
    _write_jsonl(terminated_path, terminated)

    assert main([str(terminated_path)]) == 0
    terminated_output = capsys.readouterr().out
    assert "1 complete, 0 incomplete" in terminated_output
    assert "TRUNCATED,FAILED" in terminated_output

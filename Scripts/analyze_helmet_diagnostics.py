"""Summarize helmet-diagnostics JSONL files without changing any source data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional, Sequence


# Keep this reader independent of OpenCV/TensorRT runtime imports. Bump this
# contract marker when the writer's diagnostics payload schema changes.
HELMET_DIAGNOSTICS_SCHEMA_VERSION = 2
_RECORD_TYPES = {"run_start", "frame", "final_drain", "capture_segment", "run_end"}


@dataclass
class TrackSummary:
    segment_id: int
    track_id: int
    sample_count: int = 0
    height_mean: float = 0.0
    height_min: Optional[float] = None
    height_max: Optional[float] = None
    x_mean: float = 0.0
    y_mean: float = 0.0
    edge_count: int = 0
    head_visible_count: int = 0
    association_count: int = 0
    association_inside_person: int = 0
    association_inside_head: int = 0
    latest_order: tuple[float, int] = (-1.0, -1)
    latest_hit_count: int = 0
    latest_observation_count: int = 0
    latest_hit_rate: Optional[float] = None
    latest_best_score: Optional[float] = None

    def add(self, observation: dict[str, Any]) -> None:
        height = float(observation["person_height_px"])
        x, y = observation["normalized_position"]
        self.sample_count += 1
        self.height_mean += (height - self.height_mean) / self.sample_count
        self.height_min = height if self.height_min is None else min(self.height_min, height)
        self.height_max = height if self.height_max is None else max(self.height_max, height)
        self.x_mean += (float(x) - self.x_mean) / self.sample_count
        self.y_mean += (float(y) - self.y_mean) / self.sample_count
        self.edge_count += int(observation["at_frame_edge"])
        self.head_visible_count += int(observation["head_visible"])

        time_s = float(observation["time_s"])
        frame_idx = int(observation["frame_idx"])
        order = (time_s, frame_idx)
        if order >= self.latest_order:
            self.latest_order = order
            self.latest_hit_count = int(observation["helmet_hit_count"])
            self.latest_observation_count = int(observation["helmet_observation_count"])
            self.latest_hit_rate = observation["helmet_hit_rate"]
            self.latest_best_score = observation["best_helmet_score"]

        for association in observation["associations"]:
            self.association_count += 1
            self.association_inside_person += int(association["center_inside_person"])
            self.association_inside_head += int(association["center_inside_head"])

    def hit_rate(self) -> Optional[float]:
        if self.latest_hit_rate is not None:
            return float(self.latest_hit_rate)
        if self.latest_observation_count:
            return self.latest_hit_count / self.latest_observation_count
        return None


@dataclass
class RunSummary:
    path: Path
    display_path: str
    size_bytes: int = 0
    record_counts: Counter[str] = field(default_factory=Counter)
    schema_versions: Counter[str] = field(default_factory=Counter)
    unsupported_schemas: set[str] = field(default_factory=set)
    event_reasons: Counter[tuple[str, str]] = field(default_factory=Counter)
    segment_ids: set[int] = field(default_factory=set)
    tracks: dict[tuple[int, int], TrackSummary] = field(default_factory=dict)
    observation_count: int = 0
    event_count: int = 0
    malformed_count: int = 0
    parse_error_count: int = 0
    blank_line_count: int = 0
    unknown_record_count: int = 0
    unterminated_final_line: bool = False
    ordering_error_count: int = 0
    run_start_count: int = 0
    run_end_count: int = 0
    valid_run_start_count: int = 0
    valid_run_end_count: int = 0
    terminal_counts: Optional[dict[str, int]] = None
    terminal_line_number: Optional[int] = None
    terminal_count_mismatches: list[str] = field(default_factory=list)
    terminal_counts_match: Optional[bool] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    last_logged_at: Optional[datetime] = None
    min_time_s: Optional[float] = None
    max_time_s: Optional[float] = None
    declared_truncated: bool = False
    declared_failed: bool = False
    disabled_reasons: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    io_error: Optional[str] = None

    @property
    def record_count(self) -> int:
        return sum(self.record_counts.values())

    @property
    def complete(self) -> bool:
        return (
            self.run_start_count == 1
            and self.valid_run_start_count == 1
            and self.run_end_count == 1
            and self.valid_run_end_count == 1
            and self.terminal_counts_match is True
            and self.ordering_error_count == 0
            and self.io_error is None
        )

    @property
    def day(self) -> str:
        if self.start_at is not None:
            return self.start_at.astimezone(timezone.utc).date().isoformat()
        for part in reversed(self.path.parts):
            try:
                datetime.strptime(part, "%Y-%m-%d")
                return part
            except ValueError:
                pass
        return "unknown"

    def duration(self) -> tuple[Optional[float], bool]:
        """Return seconds and whether the value is an observed lower bound."""
        if self.start_at is not None and self.end_at is not None:
            seconds = (self.end_at - self.start_at).total_seconds()
            if seconds >= 0:
                return seconds, False
        if self.start_at is not None and self.last_logged_at is not None:
            seconds = (self.last_logged_at - self.start_at).total_seconds()
            if seconds >= 0:
                return seconds, True
        if self.min_time_s is not None and self.max_time_s is not None:
            return max(0.0, self.max_time_s - self.min_time_s), True
        return None, True

    def mark_malformed(self, line_number: int, message: str) -> None:
        self.malformed_count += 1
        if len(self.examples) < 3:
            self.examples.append(f"line {line_number}: {message}")

    def mark_ordering_error(self, line_number: int, message: str) -> None:
        self.ordering_error_count += 1
        self.mark_malformed(line_number, message)


def _parse_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number {value}")


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _is_int(value: Any, *, minimum: Optional[int] = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _is_number(value: Any, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return (
        math.isfinite(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def _is_box(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(_is_number(item) for item in value)


def _schema_value(value: Any) -> str:
    if _is_int(value, minimum=1):
        return str(value)
    return "missing/invalid"


def _valid_association(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    distance = value.get("center_distance_to_person_px")
    return (
        _is_box(value.get("helmet_box"))
        and _is_number(value.get("helmet_score"), minimum=0.0, maximum=1.0)
        and isinstance(value.get("center_inside_person"), bool)
        and isinstance(value.get("center_inside_head"), bool)
        and (distance is None or _is_number(distance, minimum=0.0))
    )


def _valid_observation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    hit_count = value.get("helmet_hit_count")
    observation_count = value.get("helmet_observation_count")
    hit_rate = value.get("helmet_hit_rate")
    best_score = value.get("best_helmet_score")
    position = value.get("normalized_position")
    history = value.get("history")
    score_history = value.get("helmet_score_history")
    associations = value.get("associations")
    return (
        _is_int(value.get("schema_version"), minimum=1)
        and value.get("schema_version") == HELMET_DIAGNOSTICS_SCHEMA_VERSION
        and _is_int(value.get("diagnostic_track_id"), minimum=1)
        and _is_int(value.get("frame_idx"), minimum=0)
        and _is_number(value.get("time_s"), minimum=0.0)
        and _is_box(value.get("person_box"))
        and _is_number(value.get("person_height_px"), minimum=0.0)
        and isinstance(position, list)
        and len(position) == 2
        and all(_is_number(item, minimum=0.0, maximum=1.0) for item in position)
        and isinstance(value.get("at_frame_edge"), bool)
        and isinstance(value.get("head_visible"), bool)
        and isinstance(score_history, list)
        and all(_is_number(item, minimum=0.0, maximum=1.0) for item in score_history)
        and _is_int(hit_count, minimum=0)
        and _is_int(observation_count, minimum=0)
        and hit_count <= observation_count
        and len(score_history) <= observation_count
        and (hit_rate is None or _is_number(hit_rate, minimum=0.0, maximum=1.0))
        and (best_score is None or _is_number(best_score, minimum=0.0, maximum=1.0))
        and isinstance(associations, list)
        and all(_valid_association(item) for item in associations)
        and isinstance(history, dict)
        and _is_int(history.get("length"), minimum=0)
        and _is_int(history.get("limit"), minimum=1)
        and isinstance(history.get("truncated"), bool)
    )


def _valid_event(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    alert_uid = value.get("alert_uid")
    return (
        _is_int(value.get("schema_version"), minimum=1)
        and value.get("schema_version") == HELMET_DIAGNOSTICS_SCHEMA_VERSION
        and isinstance(value.get("event"), str)
        and bool(value["event"])
        and isinstance(value.get("reason"), str)
        and bool(value["reason"])
        and _is_int(value.get("frame_idx"), minimum=0)
        and _is_number(value.get("time_s"), minimum=0.0)
        and _is_int(value.get("episode_start_frame_idx"), minimum=0)
        and _is_number(value.get("episode_start_time_s"), minimum=0.0)
        and (alert_uid is None or isinstance(alert_uid, str))
    )


def _add_time(run: RunSummary, value: Any) -> None:
    parsed = _parse_datetime(value)
    if parsed is not None and (run.last_logged_at is None or parsed > run.last_logged_at):
        run.last_logged_at = parsed


def _add_offset(run: RunSummary, value: Any) -> None:
    if not _is_number(value, minimum=0.0):
        return
    number = float(value)
    run.min_time_s = number if run.min_time_s is None else min(run.min_time_s, number)
    run.max_time_s = number if run.max_time_s is None else max(run.max_time_s, number)


def _add_track(run: RunSummary, segment_id: int, observation: dict[str, Any]) -> None:
    track_id = int(observation["diagnostic_track_id"])
    key = (segment_id, track_id)
    track = run.tracks.get(key)
    if track is None:
        track = TrackSummary(segment_id=segment_id, track_id=track_id)
        run.tracks[key] = track
    track.add(observation)


def _process_record(run: RunSummary, record: dict[str, Any], line_number: int) -> None:
    record_type = record.get("record_type")
    if run.run_start_count == 0 and record_type != "run_start":
        run.mark_ordering_error(line_number, "record appears before run_start")
    if not isinstance(record_type, str) or not record_type:
        run.mark_malformed(line_number, "missing record_type")
        return
    run.record_counts[record_type] += 1
    if record_type not in _RECORD_TYPES:
        run.unknown_record_count += 1
        return

    if record_type == "run_start":
        if run.run_start_count > 0:
            run.mark_ordering_error(line_number, "duplicate run_start")
        run.run_start_count += 1
        start_valid = True
        version = record.get("schema_version")
        version_key = _schema_value(version)
        run.schema_versions[version_key] += 1
        if (
            not _is_int(version, minimum=1)
            or version != HELMET_DIAGNOSTICS_SCHEMA_VERSION
        ):
            run.unsupported_schemas.add(version_key)
            run.mark_malformed(line_number, f"unsupported run schema {version_key}")
            start_valid = False
        started_at = _parse_datetime(record.get("started_at_utc"))
        if started_at is None:
            run.mark_malformed(line_number, "run_start has invalid started_at_utc")
            start_valid = False
        elif run.start_at is None:
            run.start_at = started_at
        if not isinstance(record.get("run_id"), str) or not record.get("run_id"):
            run.mark_malformed(line_number, "run_start has invalid run_id")
            start_valid = False
        if not isinstance(record.get("context"), dict):
            run.mark_malformed(line_number, "run_start has invalid context")
            start_valid = False
        if not _is_int(record.get("max_total_bytes"), minimum=1):
            run.mark_malformed(line_number, "run_start has invalid max_total_bytes")
            start_valid = False
        if start_valid:
            run.valid_run_start_count += 1
        return

    if record_type == "run_end":
        run.run_end_count += 1
        end_valid = True
        ended_at = _parse_datetime(record.get("ended_at_utc"))
        if ended_at is None:
            run.mark_malformed(line_number, "run_end has invalid ended_at_utc")
            end_valid = False
        else:
            run.end_at = ended_at
        truncated = record.get("truncated")
        failed = record.get("failed")
        if isinstance(truncated, bool):
            run.declared_truncated |= truncated
        else:
            run.mark_malformed(line_number, "run_end has invalid truncated flag")
            end_valid = False
        if isinstance(failed, bool):
            run.declared_failed |= failed
        else:
            run.mark_malformed(line_number, "run_end has invalid failed flag")
            end_valid = False
        terminal_counts: dict[str, int] = {}
        for field_name in ("frames_written", "observations_written", "events_written"):
            value = record.get(field_name)
            if _is_int(value, minimum=0):
                terminal_counts[field_name] = value
            else:
                run.mark_malformed(line_number, f"run_end has invalid {field_name}")
                end_valid = False
        reason = record.get("disabled_reason")
        if isinstance(reason, str) and reason:
            run.disabled_reasons[reason] += 1
        elif reason is not None:
            run.mark_malformed(line_number, "run_end has invalid disabled_reason")
            end_valid = False
        if end_valid:
            run.valid_run_end_count += 1
        if len(terminal_counts) == 3 and run.terminal_counts is None:
            run.terminal_line_number = line_number
            run.terminal_counts = terminal_counts
        return

    if record_type == "capture_segment":
        segment_id = record.get("segment_id")
        if not _is_int(segment_id, minimum=0):
            run.mark_malformed(line_number, "capture_segment has invalid segment_id")
        else:
            run.segment_ids.add(segment_id)
        if not isinstance(record.get("reason"), str):
            run.mark_malformed(line_number, "capture_segment has invalid reason")
        if not _is_int(record.get("frame_idx"), minimum=0) or not _is_number(record.get("time_s"), minimum=0.0):
            run.mark_malformed(line_number, "capture_segment has invalid frame/time")
        _add_offset(run, record.get("time_s"))
        _add_time(run, record.get("logged_at_utc"))
        return

    segment_id = record.get("segment_id")
    valid_segment = _is_int(segment_id, minimum=0)
    if not valid_segment:
        run.mark_malformed(line_number, f"{record_type} has invalid segment_id")
    else:
        run.segment_ids.add(segment_id)
    _add_offset(run, record.get("time_s"))
    _add_time(run, record.get("logged_at_utc"))
    if not _is_int(record.get("frame_idx"), minimum=0) or not _is_number(record.get("time_s"), minimum=0.0):
        run.mark_malformed(line_number, f"{record_type} has invalid frame/time")

    observations = record.get("observations")
    if not isinstance(observations, list):
        run.mark_malformed(line_number, f"{record_type} observations is not a list")
    elif valid_segment:
        for item_index, observation in enumerate(observations):
            if not _valid_observation(observation):
                version_key = _schema_value(observation.get("schema_version")) if isinstance(observation, dict) else "missing/invalid"
                if version_key != str(HELMET_DIAGNOSTICS_SCHEMA_VERSION):
                    run.schema_versions[version_key] += 1
                    run.unsupported_schemas.add(version_key)
                run.mark_malformed(line_number, f"invalid observation[{item_index}]")
                continue
            run.schema_versions[str(observation["schema_version"])] += 1
            run.observation_count += 1
            _add_offset(run, observation["time_s"])
            _add_track(run, segment_id, observation)

    events = record.get("events")
    if not isinstance(events, list):
        run.mark_malformed(line_number, f"{record_type} events is not a list")
    else:
        for item_index, event in enumerate(events):
            if not _valid_event(event):
                version_key = _schema_value(event.get("schema_version")) if isinstance(event, dict) else "missing/invalid"
                if version_key != str(HELMET_DIAGNOSTICS_SCHEMA_VERSION):
                    run.schema_versions[version_key] += 1
                    run.unsupported_schemas.add(version_key)
                run.mark_malformed(line_number, f"invalid event[{item_index}]")
                continue
            run.schema_versions[str(event["schema_version"])] += 1
            run.event_count += 1
            _add_offset(run, event["time_s"])
            run.event_reasons[(event["event"], event["reason"])] += 1


def _finalize_run(run: RunSummary) -> None:
    if run.run_start_count > 1:
        run.mark_malformed(0, f"expected one run_start, found {run.run_start_count}")
    if run.run_end_count > 1:
        run.mark_malformed(0, f"expected one run_end, found {run.run_end_count}")

    expected_counts = {
        "frames_written": run.record_counts["frame"],
        "observations_written": run.observation_count,
        "events_written": run.event_count,
    }
    if run.terminal_counts is not None and run.terminal_line_number is not None:
        for field_name, expected in expected_counts.items():
            declared = run.terminal_counts[field_name]
            if declared != expected:
                mismatch = f"{field_name}={declared}, parsed={expected}"
                run.terminal_count_mismatches.append(mismatch)
                run.mark_malformed(run.terminal_line_number, f"run_end count mismatch: {mismatch}")

    run.terminal_counts_match = (
        run.run_end_count == 1
        and run.valid_run_end_count == 1
        and run.terminal_counts is not None
        and not run.terminal_count_mismatches
    )


def _analyze_file(path: Path, display_path: str) -> RunSummary:
    run = RunSummary(path=path, display_path=display_path)
    try:
        run.size_bytes = path.stat().st_size
        stream: BinaryIO
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    run.blank_line_count += 1
                    continue
                if run.run_end_count > 0:
                    run.mark_ordering_error(line_number, "nonblank record follows run_end")
                if not raw_line.endswith(b"\n"):
                    run.unterminated_final_line = True
                try:
                    decoded = raw_line.decode("utf-8").strip()
                    record = json.loads(decoded, parse_constant=_parse_constant)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    if run.run_start_count == 0:
                        run.mark_ordering_error(line_number, "nonblank record appears before run_start")
                    run.parse_error_count += 1
                    if len(run.examples) < 3:
                        run.examples.append(f"line {line_number}: invalid JSON ({type(exc).__name__})")
                    continue
                if not isinstance(record, dict):
                    if run.run_start_count == 0:
                        run.mark_ordering_error(line_number, "record appears before run_start")
                    run.mark_malformed(line_number, "JSON value is not an object")
                    continue
                _process_record(run, record, line_number)
    except OSError as exc:
        run.io_error = f"{type(exc).__name__}: {exc}"
    _finalize_run(run)
    return run


def _discover(input_path: Path) -> tuple[list[Path], Path]:
    if input_path.is_file():
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise ValueError(f"input path does not exist or is not readable: {input_path}")
    files = sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())
    if not files:
        raise ValueError(f"no .jsonl files found under: {input_path}")
    return files, input_path


def _display_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _format_bytes(size: int) -> str:
    return f"{size / (1024 * 1024):.2f} MiB"


def _format_duration(seconds: Optional[float], approximate: bool = False) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    whole = max(0, int(round(seconds)))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    prefix = "~" if approximate else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_percent(numerator: int, denominator: int) -> str:
    return "-" if denominator == 0 else f"{100.0 * numerator / denominator:.0f}%"


def _run_status(run: RunSummary) -> str:
    status: list[str] = []
    if run.declared_truncated:
        status.append("TRUNCATED")
    if run.declared_failed:
        status.append("FAILED")
    if run.run_start_count == 0:
        status.append("NO_START")
    elif run.run_start_count > 1:
        status.append("MULTIPLE_STARTS")
    elif run.valid_run_start_count != 1:
        status.append("INVALID_START")
    if run.run_end_count == 0:
        status.append("NO_END")
    elif run.run_end_count > 1:
        status.append("MULTIPLE_ENDS")
    elif run.valid_run_end_count != 1:
        status.append("INVALID_END")
    if run.ordering_error_count:
        status.append(f"ORDERING:{run.ordering_error_count}")
    if run.terminal_count_mismatches:
        status.append("COUNTS_MISMATCH")
    if run.parse_error_count or run.malformed_count:
        status.append(f"MALFORMED:{run.parse_error_count + run.malformed_count}")
    if run.unknown_record_count:
        status.append(f"UNKNOWN:{run.unknown_record_count}")
    if run.unsupported_schemas:
        versions = ",".join(sorted(run.unsupported_schemas))
        status.append(f"SCHEMA:{versions}")
    if run.unterminated_final_line:
        status.append("OPEN_TAIL")
    if run.blank_line_count:
        status.append(f"BLANK:{run.blank_line_count}")
    if run.io_error:
        status.append("UNREADABLE")
    if run.disabled_reasons:
        status.append("DISABLED:" + ",".join(sorted(run.disabled_reasons)))
    return ",".join(status) if status else "OK"


def _print_summary(runs: Sequence[RunSummary], max_tracks: int) -> None:
    total_bytes = sum(run.size_bytes for run in runs)
    total_records = sum(run.record_count for run in runs)
    total_observations = sum(run.observation_count for run in runs)
    total_events = sum(run.event_count for run in runs)
    total_segments = sum(len(run.segment_ids) for run in runs)
    complete_count = sum(run.complete for run in runs)
    duration_total = 0.0
    duration_count = 0
    duration_approximate = False
    for run in runs:
        duration, approximate = run.duration()
        if duration is not None:
            duration_total += duration
            duration_count += 1
            duration_approximate |= approximate

    total_duration = _format_duration(duration_total if duration_count else None, duration_approximate)
    print(
        f"Helmet diagnostics: {len(runs)} file(s), {complete_count} complete, "
        f"{len(runs) - complete_count} incomplete; {_format_bytes(total_bytes)}; "
        f"duration {total_duration}; {total_records} records, {total_observations} observations, "
        f"{total_events} events, {total_segments} segments"
    )

    daily: dict[str, list[RunSummary]] = defaultdict(list)
    for run in runs:
        daily[run.day].append(run)
    print(
        "\nBy run start UTC day (full duration grouped to start day; path date fallback): "
        "day | runs complete | duration | MiB | records | observations | events | segments"
    )
    for day in sorted(daily):
        day_runs = daily[day]
        day_duration = 0.0
        day_duration_count = 0
        day_approx = False
        for run in day_runs:
            duration, approximate = run.duration()
            if duration is not None:
                day_duration += duration
                day_duration_count += 1
                day_approx |= approximate
        print(
            f"{day} | {len(day_runs)} {sum(run.complete for run in day_runs)} | "
            f"{_format_duration(day_duration if day_duration_count else None, day_approx)} | "
            f"{_format_bytes(sum(run.size_bytes for run in day_runs))} | "
            f"{sum(run.record_count for run in day_runs)} | "
            f"{sum(run.observation_count for run in day_runs)} | "
            f"{sum(run.event_count for run in day_runs)} | "
            f"{sum(len(run.segment_ids) for run in day_runs)}"
        )

    record_types: Counter[str] = Counter()
    event_reasons: Counter[tuple[str, str]] = Counter()
    for run in runs:
        record_types.update(run.record_counts)
        event_reasons.update(run.event_reasons)
    print("\nRecord types: " + (", ".join(f"{key}={record_types[key]}" for key in sorted(record_types)) or "none"))
    event_reason_text = ", ".join(
        f"{event}/{reason}={count}" for (event, reason), count in sorted(event_reasons.items())
    ) or "none"
    print("Events/reasons: " + event_reason_text)

    print("\nRuns: file | UTC start | duration | MiB | records | obs | events | segments | status")
    for run in runs:
        duration, approximate = run.duration()
        start = run.start_at.isoformat(timespec="seconds") if run.start_at else "-"
        print(
            f"{run.display_path} | {start} | {_format_duration(duration, approximate)} | "
            f"{_format_bytes(run.size_bytes)} | {run.record_count} | {run.observation_count} | "
            f"{run.event_count} | {len(run.segment_ids)} | {_run_status(run)}"
        )
        if run.io_error:
            print(f"  read error: {run.io_error}")
        for example in run.examples:
            print(f"  {example}")

    tracks: list[tuple[str, TrackSummary]] = []
    for run in runs:
        tracks.extend((run.display_path, track) for track in run.tracks.values())
    tracks.sort(key=lambda item: (-item[1].sample_count, item[0], item[1].segment_id, item[1].track_id))
    if tracks:
        shown = len(tracks) if max_tracks == 0 else min(max_tracks, len(tracks))
        print(
            f"\nPer-track traits (top {shown} of {len(tracks)}; grouped by segment and temporary track ID): "
            "file | seg:track | samples | height px mean [min-max] | position x,y mean | edge/head visible | "
            "helmet hits/latest observations | latest best score | candidates inside person/head"
        )
        for filename, track in tracks[:shown]:
            height_min = track.height_min if track.height_min is not None else 0.0
            height_max = track.height_max if track.height_max is not None else 0.0
            position_mean = f"{track.x_mean:.2f},{track.y_mean:.2f}"
            hit_rate = track.hit_rate()
            hits = "-" if not track.latest_observation_count else (
                f"{track.latest_hit_count}/{track.latest_observation_count}"
                f" ({100.0 * hit_rate:.0f}%)" if hit_rate is not None else
                f"{track.latest_hit_count}/{track.latest_observation_count}"
            )
            best = "-" if track.latest_best_score is None else f"{track.latest_best_score:.3f}"
            inside_person = _format_percent(track.association_inside_person, track.association_count)
            inside_head = _format_percent(track.association_inside_head, track.association_count)
            print(
                f"{filename} | {track.segment_id}:{track.track_id} | {track.sample_count} | "
                f"{track.height_mean:.0f} [{height_min:.0f}-{height_max:.0f}] | "
                f"{position_mean} | {_format_percent(track.edge_count, track.sample_count)}/"
                f"{_format_percent(track.head_visible_count, track.sample_count)} | {hits} | {best} | "
                f"{inside_person}/{inside_head}"
            )
        if shown < len(tracks):
            print(f"  {len(tracks) - shown} additional tracks omitted; pass --max-tracks 0 to show all.")

    print("\nDescriptive diagnostics only; track traits do not establish alert causality or change alert behavior.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and summarize helmet diagnostics JSONL runs; files are never modified."
    )
    parser.add_argument("path", type=Path, help="One JSONL file or a directory searched recursively.")
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=25,
        help="Maximum per-track rows to print (default: 25; use 0 for all).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_tracks < 0:
        parser.error("--max-tracks must be >= 0")
    try:
        paths, root = _discover(args.path)
    except ValueError as exc:
        parser.error(str(exc))
    runs = [
        _analyze_file(path, _display_name(path, root))
        for path in paths
    ]
    _print_summary(runs, max_tracks=args.max_tracks)
    return 1 if any(run.io_error for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())

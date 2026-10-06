from __future__ import annotations

from datetime import datetime

import pytest

from Action_Detection_SOP.shifts import assign_shift_for_interval, assign_shift_for_time, parse_iso_datetime


def test_shift_boundaries() -> None:
    # Shift 1: 07:30-15:30
    # Shift 2: 15:30-23:30
    # Shift 3: 23:30-07:30 (next day)

    a = assign_shift_for_time(datetime(2026, 3, 4, 7, 29, 59))
    assert a is not None
    assert a.shift_id == "S3"
    assert a.shift_date == "2026-03-03"

    a = assign_shift_for_time(datetime(2026, 3, 4, 7, 30, 0))
    assert a is not None
    assert a.shift_id == "S1"
    assert a.shift_date == "2026-03-04"

    a = assign_shift_for_time(datetime(2026, 3, 4, 15, 29, 59))
    assert a is not None
    assert a.shift_id == "S1"

    a = assign_shift_for_time(datetime(2026, 3, 4, 15, 30, 0))
    assert a is not None
    assert a.shift_id == "S2"

    a = assign_shift_for_time(datetime(2026, 3, 4, 23, 29, 59))
    assert a is not None
    assert a.shift_id == "S2"

    a = assign_shift_for_time(datetime(2026, 3, 4, 23, 30, 0))
    assert a is not None
    assert a.shift_id == "S3"
    assert a.shift_date == "2026-03-04"

    a = assign_shift_for_time(datetime(2026, 3, 5, 0, 1, 0))
    assert a is not None
    assert a.shift_id == "S3"
    assert a.shift_date == "2026-03-04"


def test_parse_iso_datetime_supports_z_suffix() -> None:
    dt = parse_iso_datetime("2026-03-04T00:00:00Z")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 3
    assert dt.day == 4


@pytest.mark.parametrize("timestamp, shift_id, shift_date", [
    ("2026-10-04T03:00:00", "S3", "2026-10-03"),
    ("2026-10-04T07:29:59.999999+07:00", "S3", "2026-10-03"),
    ("2026-10-04T07:30:00+07:00", "S1", "2026-10-04"),
    ("2026-10-04T15:29:59+07:00", "S1", "2026-10-04"),
    ("2026-10-04T15:30:00+07:00", "S2", "2026-10-04"),
    ("2026-10-04T23:29:59+07:00", "S2", "2026-10-04"),
    ("2026-10-04T23:30:00+07:00", "S3", "2026-10-04"),
    ("2026-10-04T20:00:00Z", "S3", "2026-10-04"),  # 5 Oct 03:00 WIB
    ("2026-10-04T00:30:00Z", "S1", "2026-10-04"),
    ("2026-01-01T03:00:00", "S3", "2025-12-31"),
    ("2026-03-01T03:00:00", "S3", "2026-02-28"),
])
def test_reporting_shift_dates_in_wib(timestamp: str, shift_id: str, shift_date: str) -> None:
    assignment = assign_shift_for_time(datetime.fromisoformat(timestamp.replace("Z", "+00:00")))
    assert assignment is not None
    assert (assignment.shift_id, assignment.shift_date) == (shift_id, shift_date)
    assert assignment.shift_start_dt.utcoffset().total_seconds() == 7 * 3600


@pytest.mark.parametrize("start, end, shift_id, shift_date", [
    ("2026-10-03T23:59:00", "2026-10-04T00:01:00", "S3", "2026-10-03"),
    ("2026-10-04T07:29:00", "2026-10-04T07:32:00", "S1", "2026-10-04"),
    ("2026-10-04T07:29:00", "2026-10-04T07:31:00", "S3", "2026-10-03"),
    ("2026-10-04T07:32:00+07:00", "2026-10-04T00:29:00Z", "S1", "2026-10-04"),
])
def test_whole_session_uses_largest_overlap(
    start: str, end: str, shift_id: str, shift_date: str,
) -> None:
    assignment = assign_shift_for_interval(
        start_dt=datetime.fromisoformat(start), end_dt=datetime.fromisoformat(end.replace("Z", "+00:00")),
    )
    assert assignment is not None
    assert (assignment.shift_id, assignment.shift_date) == (shift_id, shift_date)


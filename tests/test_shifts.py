from __future__ import annotations

from datetime import datetime

from Action_Detection_SOP.shifts import assign_shift_for_time, parse_iso_datetime


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


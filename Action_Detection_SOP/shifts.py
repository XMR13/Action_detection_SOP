from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterable, Optional, Tuple

"""Scirpt for getting the shift data into the reports"""

#mke the shift date immutable
@dataclass(frozen=True)
class ShiftDef:
    shift_id: str
    name: str
    start: time
    end: time

    @property
    def crosses_midnight(self) -> bool:
        # Treat equal as "crosses" to avoid zero-length intervals.
        return self.end <= self.start


@dataclass(frozen=True)
class ShiftAssignment:
    shift_id: str
    shift_name: str
    shift_date: str
    shift_start_dt: datetime
    shift_end_dt: datetime

    def to_iso_fields(self) -> dict[str, str]:
        return {
            "shift_id": self.shift_id,
            "shift_name": self.shift_name,
            "shift_date": self.shift_date,
            "shift_start_time_iso": self.shift_start_dt.isoformat(timespec="seconds"),
            "shift_end_time_iso": self.shift_end_dt.isoformat(timespec="seconds"),
        }


def default_times_shitft() -> Tuple[ShiftDef, ShiftDef, ShiftDef]: #return back kthe asthe dataclass
    #jadwal untuk shift akan ditentukan sebagai berikut
    # Shift 1 : 07:30 - 15:30
    # Shift 2 : 15:30 - 23:30
    # Shift 3 : 23:30 - 07:30 (next day) make sure the time date is gone
    return (
        ShiftDef("S1", "Shift 1", start=time(7,30), end=time(15,30)),
        ShiftDef("S2", "Shift 2", start=time(15,30), end=time(23,30)),
        ShiftDef("S3", "Shift 3", start=time(23, 30), end=time(7, 30)),
    )



def parse_iso_datetime(value: object) -> Optional[datetime]:
    #get the parse iso datetime
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    # Support trailing "Z" (UTC) in case future writers emit it.
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except Exception:
        return None


def _coerce_shift_clock(value: object) -> time:
    """Normalize shift clock values to datetime.time.

    Accepts:
    - datetime.time
    - (hour, minute) tuple/list
    """
    if isinstance(value, time):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        hour, minute = value
        if isinstance(hour, int) and isinstance(minute, int):
            return time(hour, minute)
    raise TypeError(f"Shift clock must be datetime.time or (hour, minute), got {type(value).__name__}")


def _shift_window_for_day(shift: ShiftDef, *, day: datetime) -> tuple[datetime, datetime]:
    #continuing the crossing day date time
    start_clock = _coerce_shift_clock(shift.start)
    end_clock = _coerce_shift_clock(shift.end)

    start_dt = datetime.combine(day.date(), start_clock, tzinfo=day.tzinfo)
    end_dt = datetime.combine(day.date(), end_clock, tzinfo=day.tzinfo)
    if shift.crosses_midnight:
        end_dt = end_dt + timedelta(days=1)
    return start_dt, end_dt


def _interval_overlap_s(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return 0.0
    return float((hi - lo).total_seconds())


def assign_shift_for_time(
    dt: datetime,
    *,
    shifts: Optional[Iterable[ShiftDef]] = None,
) -> Optional[ShiftAssignment]:
    return assign_shift_for_interval(start_dt=dt, end_dt=dt, shifts=shifts)


def assign_shift_for_interval(
    *,
    start_dt: datetime,
    end_dt: datetime,
    shifts: Optional[Iterable[ShiftDef]] = None, #optional shifts record
) -> Optional[ShiftAssignment]:
    """
    Assign a shift for a session interval.

    - If the interval crosses a boundary, we pick the shift with the largest time overlap.
    - For overnight shift (23:30-07:30), `shift_date` is the date when the shift starts
      (so 01:00 belongs to the previous day's Shift 3).
    """
    #cases when start date change interchangebly
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt


    #load shift definitions based on previous shifts definition
    shift_defs = tuple(shifts) if shifts is not None else default_times_shitft()
    if not shift_defs:
        return None

    # Build candidate windows around the interval to cover overnight spans.
    days = {
        start_dt.replace(hour=0, minute=0, second=0, microsecond=0),
        end_dt.replace(hour=0, minute=0, second=0, microsecond=0),
    }
    for base in list(days):
        days.add(base - timedelta(days=1))
        days.add(base + timedelta(days=1))

    best: Optional[tuple[float, datetime, ShiftDef, datetime, datetime]] = None
    for day in sorted(days):
        for sh in shift_defs:
            w0, w1 = _shift_window_for_day(sh, day=day)
            overlap = _interval_overlap_s(start_dt, end_dt, w0, w1)
            if overlap <= 0:
                continue
            # Pick highest overlap; tie-breaker is earliest shift window start.
            candidate = (overlap, w0, sh, w0, w1)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate

    if best is None:
        # If the interval doesn't overlap any window (shouldn't happen), fall back to time-of-day rules.
        t = start_dt.timetz() if start_dt.tzinfo else start_dt.time()
        for sh in shift_defs:
            if not sh.crosses_midnight:
                if sh.start <= t < sh.end:
                    w0, w1 = _shift_window_for_day(sh, day=start_dt)
                    return ShiftAssignment(sh.shift_id, sh.name, w0.date().isoformat(), w0, w1)
            else:
                # Overnight: [start..24h) U [0..end)
                if t >= sh.start or t < sh.end:
                    # If we're after midnight, shift started yesterday.
                    base = start_dt - timedelta(days=1) if t < sh.end else start_dt
                    w0, w1 = _shift_window_for_day(sh, day=base)
                    return ShiftAssignment(sh.shift_id, sh.name, w0.date().isoformat(), w0, w1)
        return None

    _, _, sh, w0, w1 = best
    return ShiftAssignment(sh.shift_id, sh.name, w0.date().isoformat(), w0, w1)

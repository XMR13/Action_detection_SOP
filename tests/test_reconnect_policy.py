from __future__ import annotations

import pytest

from Action_Detection_SOP.reconnect_policy import reconnect_wait_seconds


def test_reconnect_wait_no_backoff() -> None:
    assert reconnect_wait_seconds(attempt=1, base_wait_s=1.0, backoff=1.0, wait_cap_s=0.0) == 1.0
    assert reconnect_wait_seconds(attempt=3, base_wait_s=1.0, backoff=1.0, wait_cap_s=0.0) == 1.0


def test_reconnect_wait_with_backoff_and_cap() -> None:
    assert reconnect_wait_seconds(attempt=1, base_wait_s=1.0, backoff=2.0, wait_cap_s=10.0) == 1.0
    assert reconnect_wait_seconds(attempt=2, base_wait_s=1.0, backoff=2.0, wait_cap_s=10.0) == 2.0
    assert reconnect_wait_seconds(attempt=5, base_wait_s=1.0, backoff=2.0, wait_cap_s=10.0) == 10.0


def test_reconnect_wait_zero_base_is_zero() -> None:
    assert reconnect_wait_seconds(attempt=4, base_wait_s=0.0, backoff=2.0, wait_cap_s=10.0) == 0.0


@pytest.mark.parametrize(
    "attempt,base_wait_s,backoff,wait_cap_s",
    [
        (0, 1.0, 1.0, 0.0),
        (1, -1.0, 1.0, 0.0),
        (1, 1.0, 0.9, 0.0),
        (1, 1.0, 1.0, -1.0),
    ],
)
def test_reconnect_wait_invalid_inputs(
    attempt: int,
    base_wait_s: float,
    backoff: float,
    wait_cap_s: float,
) -> None:
    with pytest.raises(ValueError):
        reconnect_wait_seconds(
            attempt=attempt,
            base_wait_s=base_wait_s,
            backoff=backoff,
            wait_cap_s=wait_cap_s,
        )

from __future__ import annotations
        
def reconnect_wait_seconds(*, attempt: int, base_wait_s: float, backoff: float, wait_cap_s: float) -> float:
    """Compute reconnect sleep with exponential backoff and optional cap."""
    if attempt <= 0:
        raise ValueError("attempt must be >= 1")
    if base_wait_s < 0:
        raise ValueError("base_wait_s must be >= 0")
    if backoff < 1.0:
        raise ValueError("backoff must be >= 1.0")
    if wait_cap_s < 0:
        raise ValueError("wait_cap_s must be >= 0")
    if base_wait_s == 0:
        return 0.0

    wait_s = float(base_wait_s) * (float(backoff) ** float(attempt - 1))
    if wait_cap_s > 0:
        wait_s = min(wait_s, float(wait_cap_s))
    return float(wait_s)

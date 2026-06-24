from __future__ import annotations

from types import SimpleNamespace

from Action_Detection_SOP.web_mvp.app import _disk_health


def test_disk_health_classifies_ok_warning_and_critical() -> None:
    """
    Basic test disk heatlh test
    """
    disk_ok = SimpleNamespace(total=100, used=50, free=50)
    assert _disk_health(disk_ok, warning_used_pct=75.0, critical_used_pct=85.0)["status"] == "ok"

    disk_warning = SimpleNamespace(total=100, used=75, free=25)
    warning = _disk_health(disk_warning, warning_used_pct=75.0, critical_used_pct=85.0)
    assert warning["status"] == "warning"
    assert warning["issues"] == ["disk_used_pct_warning"]

    disk_critical = SimpleNamespace(total=100, used=85, free=15)
    critical = _disk_health(disk_critical, warning_used_pct=75.0, critical_used_pct=85.0)
    assert critical["status"] == "critical"
    assert critical["issues"] == ["disk_used_pct_critical"]

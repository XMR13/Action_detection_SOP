from __future__ import annotations

import json
from pathlib import Path

import pytest

from Action_Detection_SOP.web_mvp.session_index import SessionIndex


@pytest.mark.parametrize("nested_run", [False, True])
def test_session_appears_only_after_checklist_is_finalized(
    tmp_path: Path, nested_run: bool
) -> None:
    data_dir = tmp_path / "data"
    run_root = data_dir / "camera_run" if nested_run else data_dir
    session_dir = run_root / "sessions" / "2026-09-28" / "session_0001"
    session_dir.mkdir(parents=True)
    (session_dir / "run_config.json").write_text("{}", encoding="utf-8")

    index = SessionIndex(data_dir=data_dir)
    index.refresh()
    assert index.list() == []

    checklist_path = session_dir / "checklist.json"
    checklist_path.write_text('{"session_id":', encoding="utf-8")
    index.refresh()
    assert index.list() == []

    checklist_path.write_text(
        json.dumps({"session_id": "0001", "sop_profile": "roll_sop_v1"}),
        encoding="utf-8",
    )
    index.refresh()
    sessions = index.list()
    assert len(sessions) == 1
    assert sessions[0].session_id == "0001"
    assert sessions[0].paths.session_dir == session_dir

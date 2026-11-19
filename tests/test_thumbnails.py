from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from Action_Detection_SOP.thumbnails import ThumbnailSpec, encode_thumbnail, resize_for_thumbnail, write_thumbnail
from Scripts.shrink_thumbnails import main, shrink_thumbnails


def _noisy_frame(h: int = 1080, w: int = 1920) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_resize_keeps_aspect_and_never_upscales() -> None:
    out = resize_for_thumbnail(_noisy_frame(1080, 1920), ThumbnailSpec(max_width=960))
    assert out.shape == (540, 960, 3)
    small = _noisy_frame(240, 320)
    assert resize_for_thumbnail(small, ThumbnailSpec(max_width=960)) is small


def test_spec_validates() -> None:
    with pytest.raises(ValueError):
        ThumbnailSpec(jpeg_quality=0)
    with pytest.raises(ValueError):
        ThumbnailSpec(max_width=10)


def test_write_thumbnail_is_decodable_and_small(tmp_path: Path) -> None:
    frame = _noisy_frame()
    path = tmp_path / "thumbnail.jpg"
    n = write_thumbnail(path, frame)
    assert path.stat().st_size == n
    decoded = cv2.imread(str(path))
    assert decoded.shape == (540, 960, 3)
    full = cv2.imencode(".jpg", frame)[1].nbytes
    assert n < full
    assert not list(tmp_path.glob(".*.tmp"))


def _write_full_thumb(path: Path, *, mtime: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _noisy_frame())
    os.utime(path, (mtime, mtime))
    return path.stat().st_size


def test_shrink_dry_run_then_apply_preserves_mtime(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    thumb = sessions / "2026-10-01" / "session_000001" / "thumbnail.jpg"
    original_size = _write_full_thumb(thumb, mtime=1_700_000_000)
    other = sessions / "2026-10-01" / "session_000001" / "evidence" / "frame.jpg"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"not a thumbnail")

    dry = shrink_thumbnails(sessions_dir=sessions, spec=ThumbnailSpec(), dry_run=True, min_size_bytes=0)
    assert dry.scanned == 1 and dry.shrunk == 1 and dry.bytes_saved > 0
    assert thumb.stat().st_size == original_size

    applied = shrink_thumbnails(sessions_dir=sessions, spec=ThumbnailSpec(), dry_run=False, min_size_bytes=0)
    assert applied.shrunk == 1
    assert thumb.stat().st_size < original_size
    assert thumb.stat().st_mtime == 1_700_000_000
    assert cv2.imread(str(thumb)).shape[1] == 960
    assert other.read_bytes() == b"not a thumbnail"

    again = shrink_thumbnails(sessions_dir=sessions, spec=ThumbnailSpec(), dry_run=False, min_size_bytes=0)
    assert again.shrunk == 0 and again.skipped_already_small == 1


def test_shrink_counts_corrupt_file_as_failed(tmp_path: Path, capsys) -> None:
    thumb = tmp_path / "sessions" / "2026-10-01" / "session_000002" / "thumbnail.jpg"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"\xff" * 200_000)
    rc = main(["--data-dir", str(tmp_path), "--apply"])
    assert rc == 1
    assert thumb.read_bytes() == b"\xff" * 200_000
    assert "unreadable JPEG" in capsys.readouterr().out

"""Verify one source evidence file, frame overlap removal, and upload readiness."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from Action_Detection_SOP.evidence import EvidenceClipConfig, EvidenceClipper
from Action_Detection_SOP.evidence_io import write_evidence_clip
from Action_Detection_SOP.session_evidence import SessionEvidenceWriter
from Scripts import sop_uploader


def _data(name: str, times: list[float], event: float, *, size=(96, 64)):
    clipper = EvidenceClipper(EvidenceClipConfig(pre_seconds=10, post_seconds=10,
                                               max_seconds=20, analysis_fps=2))
    for t in times:
        clipper.add_frame(time_s=t, frame=np.full((size[1], size[0], 3), round(t * 12), dtype=np.uint8))
    clipper.trigger(name=name, time_s=event, frame_idx=round(event * 2))
    return clipper.flush()[0]


def _clip(root: Path, name: str, times: list[float], event: float, **kwargs) -> dict:
    return write_evidence_clip(clip_data=_data(name, times, event), session_dir=root, index=1)


def _final(root: Path) -> dict:
    writer = SessionEvidenceWriter(session_dir=root, fps=2)
    writer.append(_data('cleaned_done', [1, 1.5, 2], 1.5))
    writer.append(_data('labeled_done', [1.5, 2, 2.5], 2))
    return writer.finalize()


def test_overlapping_nested_and_disjoint_windows_produce_one_source_video(tmp_path: Path) -> None:
    writer = SessionEvidenceWriter(session_dir=tmp_path, fps=2)
    writer.append(_data('roll_entered', [1, 1.5, 2, 2.5, 3], 1))
    writer.append(_data('cleaned_done', [1.5, 2, 2.5], 2))
    writer.append(_data('labeled_done', [2, 2.5, 3, 3.5, 4], 3))
    writer.append(_data('roll_left', [7, 7.5], 7.5))
    assert not (tmp_path / 'evidence.json').exists()
    assert not list(sop_uploader._artifact_paths(tmp_path))
    video = writer.finalize()
    assert video['frame_times_s'] == [1, 1.5, 2, 2.5, 3, 3.5, 4, 7, 7.5]
    assert video['duration_s'] == 4.5
    assert [e['playback_time_s'] for e in video['events']] == [0, 1, 2, 4]
    assert [e['label'] for e in video['events']] == ['Roll masuk', 'Pembersihan selesai', 'Label terpasang', 'Roll keluar']
    assert len(list((tmp_path / 'evidence').iterdir())) == 1
    assert not list(tmp_path.rglob('*.mp4'))
    assert len([p for p in sop_uploader._artifact_paths(tmp_path) if p.suffix == '.webm']) == 1
    assert json.loads((tmp_path / 'evidence.json').read_text())['clips'] == []
    assert writer.finalize() == video
    cap = cv2.VideoCapture(str(tmp_path / video['file']))
    values = []
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(2)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            values.append(float(frame.mean()))
    finally:
        cap.release()
    assert len(values) == 9
    assert values == pytest.approx([12, 18, 24, 30, 36, 42, 48, 84, 90], abs=4)
    assert not list(tmp_path.glob('.evidence-*'))


def test_resolution_change_preserves_canvas_and_aspect_ratio(tmp_path: Path) -> None:
    writer = SessionEvidenceWriter(session_dir=tmp_path, fps=2)
    writer.append(_data('roll_entered', [1, 1.5], 1))
    writer.append(_data('cleaned_done', [3, 3.5], 3, size=(48, 48)))
    video = writer.finalize()
    cap = cv2.VideoCapture(str(tmp_path / video['file']))
    try:
        for _ in range(4):
            ok, frame = cap.read()
            assert ok and frame.shape == (64, 96, 3)
        assert frame[:, 70:].mean() < 5  # padded instead of stretched/cropped
    finally:
        cap.release()


def test_abort_removes_unfinished_video_and_does_not_publish_manifest(tmp_path: Path) -> None:
    writer = SessionEvidenceWriter(session_dir=tmp_path, fps=2)
    writer.append(_data('roll_entered', [1, 1.5], 1))
    writer.abort()
    assert not list(tmp_path.rglob('*.webm'))
    assert not (tmp_path / 'evidence.json').exists()
    with pytest.raises(RuntimeError, match='closed'):
        writer.append(_data('cleaned_done', [2, 2.5], 2))


@pytest.mark.parametrize('fps', [0, -1, float('nan'), 241])
def test_invalid_fps_rejected(tmp_path: Path, fps: float) -> None:
    with pytest.raises(ValueError):
        SessionEvidenceWriter(session_dir=tmp_path, fps=fps)


def test_out_of_order_windows_do_not_silently_drop_frames(tmp_path: Path) -> None:
    writer = SessionEvidenceWriter(session_dir=tmp_path, fps=2)
    writer.append(_data('labeled_done', [4, 4.5], 4))
    try:
        with pytest.raises(ValueError, match='order'):
            writer.append(_data('cleaned_done', [1, 1.5], 1))
    finally:
        writer.abort()


def test_uploader_enqueues_one_video_and_ignores_unfinished_files(tmp_path: Path) -> None:
    from test_sop_uploader import _build_session_dir, _spool
    root, session = _build_session_dir(tmp_path)
    _final(session)
    (session / 'evidence/.unfinished.webm').write_bytes(b'partial')
    (session / 'evidence/unfinished.tmp').write_bytes(b'partial')
    spool = _spool(root)
    sop_uploader._enqueue_session_tasks(sessions=sop_uploader._iter_sessions(root), spool=spool, dry_run=False)
    tasks = [json.loads(p.read_text()) for p in spool.pending.glob('*.json')]
    paths = [t.get('rel_path') for t in tasks]
    assert paths.count('evidence/session.webm') == 1
    assert not any(p and ('unfinished' in p or p.endswith('.mp4')) for p in paths)


def test_empty_writer_has_no_video_or_manifest(tmp_path: Path) -> None:
    assert SessionEvidenceWriter(session_dir=tmp_path, fps=2).finalize() is None
    assert not list(tmp_path.iterdir())


def test_encoder_does_not_block_producer_or_publish_before_verified_output(tmp_path: Path, monkeypatch) -> None:
    import threading
    from Action_Detection_SOP.session_evidence import EvidenceEncoder
    writer = SessionEvidenceWriter(session_dir=tmp_path, fps=2)
    original = writer.append
    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    def delayed(data):
        entered.set()
        assert release.wait(5)
        original(data)
    monkeypatch.setattr(writer, 'append', delayed)
    encoder = EvidenceEncoder()
    try:
        encoder.append(writer, _data('roll_entered', [1, 1.5], 1))
        assert entered.wait(5)
        encoder.finish(writer, published.set)
        assert not published.is_set()
        assert not (tmp_path / 'evidence.json').exists()
        release.set()
    finally:
        release.set()
        encoder.close()
    assert published.is_set()
    assert (tmp_path / 'evidence/session.webm').exists()


def test_encoder_failure_never_publishes_session_and_cleans_partial_output(tmp_path: Path, monkeypatch) -> None:
    from Action_Detection_SOP.session_evidence import EvidenceEncoder
    import threading
    writer = SessionEvidenceWriter(session_dir=tmp_path, fps=2)
    entered, release = threading.Event(), threading.Event()
    def fail(data):
        entered.set()
        release.wait(5)
        raise RuntimeError('encoder failure')
    monkeypatch.setattr(writer, 'append', fail)
    published = []
    encoder = EvidenceEncoder()
    encoder.append(writer, _data('roll_entered', [1, 1.5], 1))
    assert entered.wait(5)
    encoder.finish(writer, lambda: published.append(True))
    release.set()
    with pytest.raises(RuntimeError, match='preparation failed'):
        encoder.close()
    assert not published
    assert not (tmp_path / 'evidence.json').exists()

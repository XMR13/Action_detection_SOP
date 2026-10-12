"""Verify capture-host VP8 support using the real evidence writer and worker."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

import numpy as np

from Action_Detection_SOP.evidence import EvidenceClipConfig, EvidenceClipper
from Action_Detection_SOP.session_evidence import EvidenceEncoder, SessionEvidenceWriter


def main() -> int:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sop-video-probe-") as folder:
        root = Path(folder)
        clipper = EvidenceClipper(EvidenceClipConfig(analysis_fps=5))
        for index in range(3):
            frame = np.full((96, 160, 3), 50 * (index + 1), dtype=np.uint8)
            clipper.add_frame(time_s=index / 5, frame=frame)
        clipper.trigger(name="codec_probe", time_s=0.2, frame_idx=1)
        writer = SessionEvidenceWriter(session_dir=root, fps=5)
        encoder = EvidenceEncoder()
        try:
            for clip in clipper.flush():
                encoder.append(writer, clip)
            encoder.finish(writer, lambda: None)
        finally:
            encoder.close()
        manifest = json.loads((root / "evidence.json").read_text())
        files = list((root / "evidence").iterdir())
        if len(files) != 1 or manifest["video"]["frame_count"] != 3:
            raise RuntimeError("Single-video codec verification failed")
        print(json.dumps({"status": "passed", "codec": "VP8/WebM", "files": 1,
                          "verified_frames": 3, "video_bytes": files[0].stat().st_size,
                          "probe_ms": round((time.perf_counter() - started) * 1000, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

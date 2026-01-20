import unittest

from Action_Detection_SOP.evidence import EvidenceClipConfig, EvidenceClipper


class TestEvidenceClipper(unittest.TestCase):
    def test_clip_includes_pre_and_post_frames(self) -> None:
        cfg = EvidenceClipConfig(pre_seconds=2.0, post_seconds=2.0, max_seconds=10.0, analysis_fps=1.0)
        clipper = EvidenceClipper(cfg)
        for t in [1, 2, 3, 4, 5]:
            self.assertEqual(clipper.add_frame(time_s=float(t), frame=t), [])
        clipper.trigger(name="roi_dwell_done", time_s=5.0, frame_idx=5)
        completed = []
        for t in [6, 7]:
            completed.extend(clipper.add_frame(time_s=float(t), frame=t))
        self.assertEqual(len(completed), 1)
        clip = completed[0]
        self.assertEqual(clip.frames, [3, 4, 5, 6, 7])
        self.assertEqual(clip.clip.frame_count, 5)

    def test_flush_without_post_frames(self) -> None:
        cfg = EvidenceClipConfig(pre_seconds=2.0, post_seconds=2.0, max_seconds=10.0, analysis_fps=1.0)
        clipper = EvidenceClipper(cfg)
        for t in [1, 2, 3, 4, 5]:
            clipper.add_frame(time_s=float(t), frame=t)
        clipper.trigger(name="helmet_done", time_s=5.0, frame_idx=5)
        completed = clipper.flush()
        self.assertEqual(len(completed), 1)
        clip = completed[0]
        self.assertEqual(clip.frames, [3, 4, 5])
        self.assertEqual(clip.clip.frame_count, 3)

    def test_max_seconds_clamps_post_window(self) -> None:
        cfg = EvidenceClipConfig(pre_seconds=4.0, post_seconds=4.0, max_seconds=5.0, analysis_fps=1.0)
        pre, post = cfg.resolved_window()
        self.assertEqual(pre, 4.0)
        self.assertEqual(post, 1.0)


if __name__ == "__main__":
    unittest.main()

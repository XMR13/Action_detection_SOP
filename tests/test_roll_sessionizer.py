import unittest

from Action_Detection_SOP.session import RollSessionConfig, RollSessionizer


class TestRollSessionizer(unittest.TestCase):
    def test_occlusion_keeps_session_open_and_detection_resets_exit_delay(self) -> None:
        cfg = RollSessionConfig(start_seconds=0.2, analysis_fps=5.0)
        sessionizer = RollSessionizer(cfg)
        self.assertEqual(sessionizer.update(True), "start")

        # Four seconds of occlusion exceeds the old runner delay of two seconds.
        for _ in range(20):
            self.assertIsNone(sessionizer.update(False))
        self.assertTrue(sessionizer.active)
        self.assertIsNone(sessionizer.update(True))

        # Reappearance resets the delay: require a fresh five-second absence.
        for _ in range(24):
            self.assertIsNone(sessionizer.update(False))
        self.assertTrue(sessionizer.active)
        self.assertEqual(sessionizer.update(False), "end")
        self.assertFalse(sessionizer.active)

    def test_start_and_end(self) -> None:
        cfg = RollSessionConfig(start_seconds=1.0, end_seconds=1.0, analysis_fps=2.0)
        sessionizer = RollSessionizer(cfg)
        events = [
            sessionizer.update(True),
            sessionizer.update(True),
            sessionizer.update(False),
            sessionizer.update(False),
        ]
        self.assertEqual(events, [None, "start", None, "end"])
        self.assertFalse(sessionizer.active)

    def test_reset_clears_state(self) -> None:
        cfg = RollSessionConfig(start_seconds=0.5, end_seconds=0.5, analysis_fps=2.0)
        sessionizer = RollSessionizer(cfg)
        sessionizer.update(True)
        sessionizer.update(True)
        self.assertTrue(sessionizer.active)
        sessionizer.reset()
        self.assertFalse(sessionizer.active)
        self.assertIsNone(sessionizer.update(False))


if __name__ == "__main__":
    unittest.main()

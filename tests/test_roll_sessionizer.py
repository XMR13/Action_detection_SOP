import unittest

from Action_Detection_SOP.session import RollSessionConfig, RollSessionizer


class TestRollSessionizer(unittest.TestCase):
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

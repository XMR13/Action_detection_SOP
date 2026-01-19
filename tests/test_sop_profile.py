import json
import tempfile
import unittest
from pathlib import Path

from Action_Detection_SOP.config import SopProfile, load_sop_profile


class TestSopProfile(unittest.TestCase):
    def _write_profile(self, payload: dict) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "profile.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_ok(self) -> None:
        path = self._write_profile(
            {
                "schema_version": 1,
                "session_start_seconds": 1.5,
                "session_end_seconds": 2.5,
                "min_session_seconds": 1.0,
                "roi_dwell_seconds": 7.0,
                "notes": "test",
            }
        )
        profile = load_sop_profile(path)
        self.assertIsInstance(profile, SopProfile)
        self.assertEqual(profile.session_start_seconds, 1.5)
        self.assertEqual(profile.session_end_seconds, 2.5)
        self.assertEqual(profile.min_session_seconds, 1.0)
        self.assertEqual(profile.roi_dwell_seconds, 7.0)
        self.assertEqual(profile.notes, "test")

    def test_unknown_keys_rejected(self) -> None:
        path = self._write_profile(
            {
                "schema_version": 1,
                "session_start_seconds": 1.0,
                "session_end_seconds": 2.0,
                "roi_dwell_seconds": 3.0,
                "extra": 123,
            }
        )
        with self.assertRaises(ValueError):
            load_sop_profile(path)

    def test_invalid_seconds_rejected(self) -> None:
        path = self._write_profile(
            {
                "schema_version": 1,
                "session_start_seconds": 0,
                "session_end_seconds": 2.0,
                "roi_dwell_seconds": 3.0,
            }
        )
        with self.assertRaises(ValueError):
            load_sop_profile(path)

    def test_missing_min_session_defaults_to_zero(self) -> None:
        path = self._write_profile(
            {
                "schema_version": 1,
                "session_start_seconds": 1.0,
                "session_end_seconds": 2.0,
                "roi_dwell_seconds": 3.0,
            }
        )
        profile = load_sop_profile(path)
        self.assertEqual(profile.min_session_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()

import unittest

from Action_Detection_SOP.sop_engine import (
    HelmetRuleConfig,
    RoiDwellRuleConfig,
    SessionizationConfig,
    SopEngine,
    SopEngineConfig,
    StepStatus,
)
from yolo_kit.types import Detection


def _person_det() -> Detection:
    # Tall enough person box so helmet association has a meaningful head region.
    return Detection(x1=10, y1=10, x2=110, y2=210, score=0.9, class_id=0)


def _helmet_det() -> Detection:
    # Center (40, 35) is inside the person's top region when head_top_fraction ~= 0.35.
    return Detection(x1=20, y1=20, x2=60, y2=50, score=0.9, class_id=1)


class TestSopEngineMvpA(unittest.TestCase):
    def _run(
        self,
        *,
        engine: SopEngine,
        analysis_fps: float,
        present_frames: int,
        absent_frames: int,
        helmet_present: bool,
        start_frame_idx: int = 1,
    ):
        results = []
        frame_idx = start_frame_idx
        for _ in range(present_frames):
            t_s = frame_idx / analysis_fps
            persons = [_person_det()]
            res = engine.update(
                time_s=t_s,
                frame_idx=frame_idx,
                persons_in_roi=persons,
                persons_all=persons,
                helmets_all=[_helmet_det()] if helmet_present else [],
            )
            if res is not None:
                results.append(res)
            frame_idx += 1

        for _ in range(absent_frames):
            t_s = frame_idx / analysis_fps
            res = engine.update(
                time_s=t_s,
                frame_idx=frame_idx,
                persons_in_roi=[],
                persons_all=[],
                helmets_all=[],
            )
            if res is not None:
                results.append(res)
            frame_idx += 1
        return results

    def test_can_run_with_helmet_disabled(self) -> None:
        fps = 5.0
        engine = SopEngine(
            SopEngineConfig(
                session=SessionizationConfig(start_seconds=1.0, end_seconds=1.0, analysis_fps=fps),
                helmet=None,
                roi_dwell=RoiDwellRuleConfig(required_seconds=1.0, analysis_fps=fps),
            )
        )
        results = self._run(engine=engine, analysis_fps=fps, present_frames=6, absent_frames=5, helmet_present=False)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.operator_present, StepStatus.DONE)
        self.assertEqual(r.helmet, StepStatus.UNKNOWN)
        self.assertIn("helmet_check_disabled", set(r.notes))

    def test_helmet_done_with_sustained_association(self) -> None:
        fps = 5.0
        engine = SopEngine(
            SopEngineConfig(
                session=SessionizationConfig(start_seconds=1.0, end_seconds=1.0, analysis_fps=fps),
                helmet=HelmetRuleConfig(required_seconds=1.0, analysis_fps=fps, head_top_fraction=0.35, max_gap_frames=0),
                roi_dwell=RoiDwellRuleConfig(required_seconds=1.0, analysis_fps=fps),
            )
        )
        results = self._run(engine=engine, analysis_fps=fps, present_frames=10, absent_frames=5, helmet_present=True)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.operator_present, StepStatus.DONE)
        self.assertEqual(r.helmet, StepStatus.DONE)
        self.assertGreaterEqual(r.helmet_positive_frames, 5)

    def test_short_session_is_unknown(self) -> None:
        fps = 5.0
        engine = SopEngine(
            SopEngineConfig(
                session=SessionizationConfig(start_seconds=0.2, end_seconds=0.2, analysis_fps=fps),
                helmet=HelmetRuleConfig(required_seconds=1.0, analysis_fps=fps, max_gap_frames=0, short_session_is_unknown=True),
                roi_dwell=RoiDwellRuleConfig(required_seconds=1.0, analysis_fps=fps),
            )
        )
        results = self._run(engine=engine, analysis_fps=fps, present_frames=1, absent_frames=1, helmet_present=False)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.helmet, StepStatus.UNKNOWN)
        self.assertIn("session_too_short_for_helmet_decision", set(r.notes))

    def test_roi_dwell_done_with_gap_tolerance(self) -> None:
        fps = 5.0
        engine = SopEngine(
            SopEngineConfig(
                session=SessionizationConfig(start_seconds=0.2, end_seconds=1.0, analysis_fps=fps),
                helmet=None,
                roi_dwell=RoiDwellRuleConfig(
                    required_seconds=1.0,
                    analysis_fps=fps,
                    max_gap_frames=1,
                    max_track_missed=2,
                    iou_match_threshold=0.2,
                ),
            )
        )
        # 5 frames required; provide 3, miss 1, provide 2 => should still reach 5 with gap tolerance.
        results = []
        frame_idx = 1
        for present in [True, True, True, False, True, True, False, False]:
            t_s = frame_idx / fps
            persons = [_person_det()] if present else []
            res = engine.update(
                time_s=t_s,
                frame_idx=frame_idx,
                persons_in_roi=persons,
                persons_all=persons,
                helmets_all=[],
            )
            if res is not None:
                results.append(res)
            frame_idx += 1
        flush_res = engine.flush(time_s=frame_idx / fps)
        if flush_res is not None:
            results.append(flush_res)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].roi_dwell, StepStatus.DONE)


if __name__ == "__main__":
    unittest.main()

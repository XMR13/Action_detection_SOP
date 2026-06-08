import unittest

from Action_Detection_SOP.roll_sop_engine import (
    RollComplianceStatus,
    RollEvidenceRuleConfig,
    RollSopEngine,
    RollSopEngineConfig,
)
from Action_Detection_SOP.session import RollSessionConfig
from Action_Detection_SOP.sop_engine import StepStatus
from yolo_kit.types import Detection


def _roll() -> Detection:
    return Detection(x1=0, y1=0, x2=100, y2=100, score=0.9, class_id=2)


def _cloth_on_roll() -> Detection:
    return Detection(x1=10, y1=10, x2=30, y2=30, score=0.9, class_id=3)


def _label_on_roll() -> Detection:
    return Detection(x1=45, y1=45, x2=75, y2=75, score=0.9, class_id=4)


def _outside_tool(class_id: int) -> Detection:
    return Detection(x1=150, y1=150, x2=180, y2=180, score=0.9, class_id=class_id)


def _cfg(*, required_seconds: float = 0.4) -> RollSopEngineConfig:
    fps = 5.0
    return RollSopEngineConfig(
        session=RollSessionConfig(start_seconds=0.2, end_seconds=0.4, analysis_fps=fps),
        cleaning=RollEvidenceRuleConfig(required_seconds=required_seconds, analysis_fps=fps, max_gap_frames=0),
        labeling=RollEvidenceRuleConfig(required_seconds=required_seconds, analysis_fps=fps, max_gap_frames=0),
    )


class TestRollSopEngine(unittest.TestCase):
    def _run_session(
        self,
        *,
        engine: RollSopEngine,
        present_frames: int,
        absent_frames: int,
        cloth_frames: int,
        label_frames: int,
    ):
        results = []
        frame_idx = 1
        fps = engine.cfg.session.analysis_fps
        for i in range(present_frames):
            res = engine.update(
                time_s=frame_idx / fps,
                frame_idx=frame_idx,
                rolls=[_roll()],
                cleaning_cloths=[_cloth_on_roll()] if i < cloth_frames else [],
                labels=[_label_on_roll()] if i < label_frames else [],
            )
            if res is not None:
                results.append(res)
            frame_idx += 1

        for _ in range(absent_frames):
            res = engine.update(
                time_s=frame_idx / fps,
                frame_idx=frame_idx,
                rolls=[],
                cleaning_cloths=[],
                labels=[],
            )
            if res is not None:
                results.append(res)
            frame_idx += 1
        return results

    def test_done_session_finalizes_with_roll_sop_fields_and_events(self) -> None:
        engine = RollSopEngine(_cfg())

        results = self._run_session(
            engine=engine,
            present_frames=3,
            absent_frames=2,
            cloth_frames=2,
            label_frames=2,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.session_id, "000001")
        self.assertEqual(result.sop_profile, "roll_sop_v1")
        self.assertEqual(result.start_time_s, 0.2)
        self.assertEqual(result.end_time_s, 1.0)
        self.assertEqual(result.cleaned, StepStatus.DONE)
        self.assertEqual(result.labeled, StepStatus.DONE)
        self.assertEqual(result.overall_status, RollComplianceStatus.COMPLIANT)
        self.assertEqual(result.cleaning_positive_frames, 2)
        self.assertEqual(result.labeling_positive_frames, 2)

        names = [event.name for event in engine.pop_events()]
        self.assertEqual(names, ["roll_entered", "cleaned_done", "labeled_done", "roll_left"])

    def test_missing_evidence_is_non_compliant(self) -> None:
        engine = RollSopEngine(_cfg())

        results = self._run_session(
            engine=engine,
            present_frames=3,
            absent_frames=2,
            cloth_frames=0,
            label_frames=2,
        )

        result = results[0]
        self.assertEqual(result.cleaned, StepStatus.NOT_DONE)
        self.assertEqual(result.labeled, StepStatus.DONE)
        self.assertEqual(result.overall_status, RollComplianceStatus.NON_COMPLIANT)

    def test_partial_evidence_is_unknown(self) -> None:
        engine = RollSopEngine(_cfg(required_seconds=0.6))

        results = self._run_session(
            engine=engine,
            present_frames=4,
            absent_frames=2,
            cloth_frames=1,
            label_frames=3,
        )

        result = results[0]
        self.assertEqual(result.cleaned, StepStatus.UNKNOWN)
        self.assertEqual(result.labeled, StepStatus.DONE)
        self.assertEqual(result.overall_status, RollComplianceStatus.UNKNOWN)
        self.assertIn("insufficient_cleaning_evidence", set(result.notes))

    def test_evidence_must_overlap_roll(self) -> None:
        engine = RollSopEngine(_cfg())
        frame_idx = 1
        fps = engine.cfg.session.analysis_fps

        for _ in range(3):
            engine.update(
                time_s=frame_idx / fps,
                frame_idx=frame_idx,
                rolls=[_roll()],
                cleaning_cloths=[_outside_tool(3)],
                labels=[_outside_tool(4)],
            )
            frame_idx += 1

        result = engine.flush(time_s=frame_idx / fps, frame_idx=frame_idx)
        assert result is not None
        self.assertEqual(result.cleaned, StepStatus.NOT_DONE)
        self.assertEqual(result.labeled, StepStatus.NOT_DONE)
        self.assertEqual(result.overall_status, RollComplianceStatus.NON_COMPLIANT)

    def test_flush_finalizes_active_session(self) -> None:
        engine = RollSopEngine(_cfg())
        result = engine.update(
            time_s=0.2,
            frame_idx=1,
            rolls=[_roll()],
            cleaning_cloths=[_cloth_on_roll()],
            labels=[_label_on_roll()],
        )
        self.assertIsNone(result)
        self.assertEqual(engine.active_session_id, "000001")

        flushed = engine.flush(time_s=0.4, frame_idx=2)

        assert flushed is not None
        self.assertEqual(flushed.session_id, "000001")
        self.assertIsNone(engine.active_session_id)
        self.assertEqual([event.name for event in engine.pop_events()], ["roll_entered", "roll_left"])


if __name__ == "__main__":
    unittest.main()

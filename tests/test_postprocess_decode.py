import unittest

import numpy as np

from yolo_kit.postprocess import YoloPostConfig, YoloPostprocessor


class TestYoloPostprocessDecode(unittest.TestCase):
    def test_decode_decoded_nx6(self) -> None:
        #decode the classic nx6 format that yolo detections alog

        p = np.array(
            [
                [10, 20, 30, 40, 0.9, 1],
                [11, 21, 31, 41, 0.8, 3],
            ],
            dtype=np.float32,
        )
        post = YoloPostprocessor(YoloPostConfig())
        boxes, scores, class_ids = post._decode(p)
        self.assertEqual(boxes.shape, (2, 4))
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(class_ids.shape, (2,))
        self.assertTrue(np.allclose(scores, np.array([0.9, 0.8], dtype=np.float32)))
        self.assertTrue(np.array_equal(class_ids, np.array([1, 3], dtype=np.int32)))

    def test_decode_decoded_nx6_cxcywh_auto(self) -> None:
        # Some exporters store (N, 6) as [cx, cy, w, h, score, class_id] (often normalized).
        p = np.array([[0.5, 0.5, 0.5, 0.5, 0.9, 1]], dtype=np.float32)
        post = YoloPostprocessor(YoloPostConfig(decoded_box_format="auto"))
        boxes, scores, class_ids = post._decode(p, input_size=(640.0, 640.0))
        self.assertEqual(boxes.shape, (1, 4))
        # cxcywh -> xyxy: (0.25..0.75)
        self.assertTrue(np.allclose(boxes[0], np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32), atol=1e-6))
        self.assertTrue(np.allclose(scores, np.array([0.9], dtype=np.float32)))
        self.assertTrue(np.array_equal(class_ids, np.array([1], dtype=np.int32)))

    def test_decode_decoded_6xn(self) -> None:
        p = np.array(
            [
                [10, 11],
                [20, 21],
                [30, 31],
                [40, 41],
                [0.9, 0.8],
                [1, 3],
            ],
            dtype=np.float32,
        )
        post = YoloPostprocessor(YoloPostConfig())
        boxes, scores, class_ids = post._decode(p)
        self.assertEqual(boxes.shape, (2, 4))
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(class_ids.shape, (2,))
        self.assertTrue(np.allclose(scores, np.array([0.9, 0.8], dtype=np.float32)))
        self.assertTrue(np.array_equal(class_ids, np.array([1, 3], dtype=np.int32)))

    def test_decode_n_5_plus_c(self) -> None:
        # (N, 5 + C): [cx, cy, w, h, obj, class_scores...]
        # 2 boxes, 3 classes
        p = np.array(
            [
                [50, 60, 10, 20, 0.5, 0.1, 0.9, 0.2],  # class 1 (0.9)
                [55, 66, 12, 18, 0.8, 0.7, 0.1, 0.2],  # class 0 (0.7)
            ],
            dtype=np.float32,
        )
        post = YoloPostprocessor(YoloPostConfig())
        boxes, scores, class_ids = post._decode(p)
        self.assertEqual(boxes.shape, (2, 4))
        self.assertTrue(np.allclose(scores, np.array([0.5 * 0.9, 0.8 * 0.7], dtype=np.float32)))
        self.assertTrue(np.array_equal(class_ids, np.array([1, 0], dtype=np.int64)))

    def test_decode_anchors_layout_class_scores(self) -> None:
        # (C + 4, A) without explicit objectness
        # Use a realistic-ish anchor count so the heuristic classifies it as anchors layout.
        # 3 classes, 32 anchors
        a = 32
        boxes = np.zeros((4, a), dtype=np.float32)
        boxes[0, :] = 50  # cx
        boxes[1, :] = 60  # cy
        boxes[2, :] = 10  # w
        boxes[3, :] = 20  # h
        boxes[0, 1] = 55
        boxes[1, 1] = 66
        boxes[2, 1] = 12
        boxes[3, 1] = 18

        class_scores = np.zeros((3, a), dtype=np.float32)
        # Anchor 0: class 1 (0.9)
        class_scores[:, 0] = [0.1, 0.9, 0.2]
        # Anchor 1: class 0 (0.7)
        class_scores[:, 1] = [0.7, 0.1, 0.2]
        # Remaining anchors: class 1 (0.9)
        class_scores[:, 2:] = np.array([[0.1], [0.9], [0.2]], dtype=np.float32)

        p = np.vstack([boxes, class_scores])  # (7, 32)
        post = YoloPostprocessor(YoloPostConfig(anchors_has_objectness=False))
        boxes_out, scores, class_ids = post._decode(p)
        self.assertEqual(boxes_out.shape, (a, 4))
        self.assertTrue(np.allclose(scores[:2], np.array([0.9, 0.7], dtype=np.float32)))
        self.assertTrue(np.array_equal(class_ids[:2], np.array([1, 0], dtype=np.int64)))

    def test_decode_anchors_layout_score_and_class_id(self) -> None:
        # Some exporters emit (4 + 2, A) where the last 2 rows are:
        # - score (float)
        # - class_id (int-like)
        a = 32
        boxes = np.zeros((4, a), dtype=np.float32)
        boxes[0, :] = 50  # cx
        boxes[1, :] = 60  # cy
        boxes[2, :] = 10  # w
        boxes[3, :] = 20  # h

        scores = np.zeros((1, a), dtype=np.float32)
        scores[0, :] = 0.2
        scores[0, 0] = 0.9
        scores[0, 1] = 0.8

        class_ids = np.zeros((1, a), dtype=np.float32)
        class_ids[0, 0] = 1.0
        class_ids[0, 1] = 0.0

        p = np.vstack([boxes, scores, class_ids])  # (6, 32)
        post = YoloPostprocessor(YoloPostConfig(anchors_has_objectness=False))
        boxes_out, scores_out, class_ids_out = post._decode(p)
        self.assertEqual(boxes_out.shape, (a, 4))
        self.assertTrue(np.allclose(scores_out[:2], np.array([0.9, 0.8], dtype=np.float32)))
        self.assertTrue(np.array_equal(class_ids_out[:2], np.array([1, 0], dtype=np.int64)))

    def test_process_denormalizes_boxes_when_needed(self) -> None:
        # Some ONNX exports emit normalized cxcywh in anchors layout: (4 + C, A).
        # Ensure process() scales normalized boxes back to pixel coords.
        a = 32
        boxes = np.zeros((4, a), dtype=np.float32)
        # Normalized cxcywh for anchor 0: centered box spanning half the image.
        boxes[0, 0] = 0.5  # cx
        boxes[1, 0] = 0.5  # cy
        boxes[2, 0] = 0.5  # w
        boxes[3, 0] = 0.5  # h

        class_scores = np.zeros((2, a), dtype=np.float32)
        class_scores[:, 0] = [0.1, 0.9]
        p = np.vstack([boxes, class_scores])  # (6, 32)

        post = YoloPostprocessor(
            YoloPostConfig(
                conf_threshold=0.01,
                apply_nms=False,
                max_detections=1,
                anchors_has_objectness=False,
            )
        )
        dets = post.process(p, orig_size=(640, 640), pad=(0.0, 0.0), ratio=(1.0, 1.0))
        self.assertEqual(len(dets), 1)

        # Expected xyxy: (0.25..0.75) * 640 = (160..480)
        self.assertAlmostEqual(dets[0].x1, 160.0, delta=1.0)
        self.assertAlmostEqual(dets[0].y1, 160.0, delta=1.0)
        self.assertAlmostEqual(dets[0].x2, 480.0, delta=1.0)
        self.assertAlmostEqual(dets[0].y2, 480.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()

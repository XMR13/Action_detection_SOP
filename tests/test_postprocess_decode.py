import unittest

import numpy as np

from yolo_kit.postprocess import YoloPostConfig, YoloPostprocessor


class TestYoloPostprocessDecode(unittest.TestCase):
    def test_decode_decoded_nx6(self) -> None:
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


if __name__ == "__main__":
    unittest.main()

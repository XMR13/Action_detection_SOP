from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .nms import NMSConfig, nms
from .types import Detection


@dataclass
class YoloPostConfig:
    """
    Konfigturasi untuk YOLO post processing
    """
    conf_threshold: float = 0.25
    iou_threshold: float = 0.4
    max_detections: int = 50
    # If False, skip NMS and only keep top `max_detections` by score.
    apply_nms: bool = True
    # If True, NMS is class-agnostic (current default behavior).
    # If False, runs per-class NMS then merges results by score.
    class_agnostic_nms: bool = True
    # For (C+4, A) layouts, some exports include an objectness row: (C+5, A).
    # Set True/False to force interpretation; None keeps a simple default.
    anchors_has_objectness: Optional[bool] = None
    # Optional list of class IDs to keep; None keeps all.
    class_ids: Optional[Sequence[int]] = 0


class YoloPostprocessor:
    """
    Generic post-process untuk YOLO expoerts:

    Layout yang dibantuk (per image):
    - (N, 5 + C): [cx, cy, w, h, obj, class_scores...]
    - (C + 4, anchor) : contoh 84 x 8400 untuk yolov9/v8 (with anchors)
    - Telah didecoding (N, 6) : [x1, y1, x2, y2, score, class_id]

    Input bisa berupa Numpy: output torch harus dikeluarkan dengan torch.detach() dan dikonversikan 
    menjadi numpy

    """

    def __init__(self, cfg: YoloPostConfig):
        self.cfg = cfg

    def process(
        self,
        preds: np.ndarray,
        orig_size: Tuple[int, int],
        pad: Tuple[float, float] = (0.0, 0.0),
        ratio: Tuple[float, float] = (1.0, 1.0),
    ) -> List[Detection]:
        """
        Mengkonversikan raw output model menjadi filter yang telah didteksi dengan
        koordinat gambar yang original.

        Arg:
            preds: Output model untuk single image
            orig_size: (width, height) untuk gambar awal
            pad: (dw, dh) digunakan kektika letterbox (left/top)
            ratio: (rw, rh) scaling digunakan untuk resize
        """

        boxes_xyxy, scores, class_ids = self._decode(preds)
        if boxes_xyxy.size == 0:
            return []

        # Filter by score
        keep = scores >= self.cfg.conf_threshold
        boxes_xyxy, scores, class_ids = boxes_xyxy[keep], scores[keep], class_ids[keep]
        if boxes_xyxy.size == 0:
            return []

        # Optional class filter
        if self.cfg.class_ids is not None:
            mask = np.isin(class_ids, np.array(self.cfg.class_ids))
            boxes_xyxy, scores, class_ids = boxes_xyxy[mask], scores[mask], class_ids[mask]
            if boxes_xyxy.size == 0:
                return []

        # NMS (optional) / Top-K
        if self.cfg.apply_nms:
            boxes_xyxy, scores, class_ids = self._apply_nms(boxes_xyxy, scores, class_ids)
        else:
            boxes_xyxy, scores, class_ids = self._select_topk(boxes_xyxy, scores, class_ids)

        # Scale boxes back to original image
        boxes_xyxy = self._scale_boxes(boxes_xyxy, orig_size, pad, ratio)

        return [
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(score),
                class_id=int(cls_id),
            )
            for (x1, y1, x2, y2), score, cls_id in zip(boxes_xyxy, scores, class_ids)
        ]

    # ------------------------------------------------------------------ #
    # Helper internal
    # ------------------------------------------------------------------ #
    def _decode(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode variasi dari YOLO layout menjadi kotak dengan koordinat xyxy, score, dan class_ids.
        """

        p = np.asarray(preds)
        if p.ndim == 3:
            if p.shape[0] != 1:
                raise ValueError(f"Batch > 1 is not supported (got shape {p.shape}). Pass one image at a time.")
            p = p[0] #menghilangkan axis dim 0 
        p = np.squeeze(p)

        # kasus:terlah terdecode (N, 6) => [x1, y1, x2, y2, score, class_id]
        if p.ndim == 2 and p.shape[1] == 6:
            boxes = p[:, 0:4] #sisa duanya bukan box
            scores = p[:, 4] #box
            class_id = p[:, 4:].astype(int)
            return boxes, scores, class_id


        if p.ndim == 2:
            h, w = p.shape

            # Heuristic: YOLO "channels" dimension is usually small (<= ~512) and anchors dimension is large.
            small, large = (h, w) if h <= w else (w, h)
            is_channels_first = h <= w
            looks_like_anchors_layout = small >= 6 and small <= 512 and (large / max(small, 1)) >= 4

            # Kasus : (C+4, A) atau (C+5, A) (channel yang pertama)
            if looks_like_anchors_layout and is_channels_first:
                boxes = p[0:4, :].T  # (A, 4) as cx, cy, w, h
                rest = p[4:, :]  # (C or 1+C, A)

                if self.cfg.anchors_has_objectness is True:
                    if rest.shape[0] < 2:
                        raise ValueError(f"Expected objectness + class scores, got shape {p.shape}.")
                    objectness = rest[0, :]
                    class_scores = rest[1:, :]
                    class_ids = np.argmax(class_scores, axis=0)
                    class_conf = class_scores[class_ids, np.arange(class_scores.shape[1])]
                    scores = objectness * class_conf
                else:
                    # Default: treat remaining rows as class scores only.
                    class_scores = rest
                    class_ids = np.argmax(class_scores, axis=0)
                    scores = class_scores[class_ids, np.arange(class_scores.shape[1])]

            # jika memiliki seperti ni (N, 5 + C) (anchor)
            elif p.shape[1] >= 6:
                boxes = p[:, :4] 
                objectness = p[:, 4:5]
                class_scores = p[:, 5:]
                class_ids = np.argmax(class_scores, axis=0) #ambil dari axis 0
                class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
                scores = objectness[:, 0] * class_conf
            else:
                raise ValueError(f"Format yolo tidak dissuport {p.shape}")

        else:
            raise ValueError(f"Unsupported YOLO output shape: {p.shape}")

        # Convert cxcywh -> xyxy
        cx, cy, w_box, h_box = boxes.T
        x1 = cx - w_box / 2
        y1 = cy - h_box / 2
        x2 = cx + w_box / 2
        y2 = cy + h_box / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        return boxes_xyxy, scores, class_ids

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        orig_size: Tuple[int, int],
        pad: Tuple[float, float],
        ratio: Tuple[float, float],
    ) -> np.ndarray:
        """
        Map boxes dari letterboard ke original image.
        """

        dw, dh = pad
        rw, rh = ratio
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / rw
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / rh

        orig_w, orig_h = orig_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
        return boxes

    def _select_topk(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if boxes.size == 0:
            return boxes, scores, class_ids
        if scores.shape[0] <= self.cfg.max_detections:
            return boxes, scores, class_ids
        order = np.argsort(scores)[::-1][: self.cfg.max_detections]
        return boxes[order], scores[order], class_ids[order]

    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nms_cfg = NMSConfig(iou_threshold=self.cfg.iou_threshold, max_detections=self.cfg.max_detections)

        if self.cfg.class_agnostic_nms:
            keep_idx = nms(boxes, scores, nms_cfg)
            return boxes[keep_idx], scores[keep_idx], class_ids[keep_idx]

        kept: List[int] = []
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            if idx.size == 0:
                continue
            keep_local = nms(boxes[idx], scores[idx], nms_cfg)
            kept.extend(idx[keep_local].tolist())

        if not kept:
            return boxes[:0], scores[:0], class_ids[:0]

        kept = np.array(kept, dtype=np.int32)
        order = np.argsort(scores[kept])[::-1]
        kept = kept[order]
        if kept.size > self.cfg.max_detections:
            kept = kept[: self.cfg.max_detections]
        return boxes[kept], scores[kept], class_ids[kept]

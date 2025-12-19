from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from yolo_kit import LetterboxConfig, YoloPostConfig, YoloPostprocessor, load_pipeline


@dataclass(frozen=True)
class TimingSummary:
    n: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("No values provided.")
    if q < 0.0 or q > 100.0:
        raise ValueError("q must be in [0, 100].")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # Linear interpolation between closest ranks.
    pos = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    t = pos - lo
    return float(sorted_values[lo] * (1.0 - t) + sorted_values[hi] * t)


def _summarize_ms(values_s: List[float]) -> TimingSummary:
    ms = [v * 1000.0 for v in values_s]
    ms_sorted = sorted(ms)
    return TimingSummary(
        n=len(ms_sorted),
        mean_ms=float(statistics.fmean(ms_sorted)) if ms_sorted else 0.0,
        p50_ms=_percentile(ms_sorted, 50.0) if ms_sorted else 0.0,
        p90_ms=_percentile(ms_sorted, 90.0) if ms_sorted else 0.0,
        p95_ms=_percentile(ms_sorted, 95.0) if ms_sorted else 0.0,
    )


def _iter_frames(args: argparse.Namespace) -> Iterable[np.ndarray]:
    if args.image is not None:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not read image at path: {args.image}")
        for _ in range(int(args.repeats)):
            yield img
        return

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.video}")
    else:
        cam_index = 0 if args.webcam is None else int(args.webcam)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index: {cam_index}")

    frame_idx = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if (frame_idx - 1) % int(args.every) != 0:
                continue
            yield frame
            processed += 1
            if args.max_frames and processed >= int(args.max_frames):
                break
    finally:
        cap.release()


def _format_summary(label: str, s: TimingSummary) -> str:
    return (
        f"{label}: n={s.n} mean={s.mean_ms:.3f}ms p50={s.p50_ms:.3f}ms "
        f"p90={s.p90_ms:.3f}ms p95={s.p95_ms:.3f}ms"
    )


def _make_postprocessors(args: argparse.Namespace) -> Tuple[YoloPostprocessor, YoloPostprocessor]:
    base = YoloPostConfig(
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        max_detections=int(args.max_det),
        class_agnostic_nms=not bool(args.per_class_nms),
    )
    post_with_nms = YoloPostprocessor(base)
    post_no_nms = YoloPostprocessor(
        YoloPostConfig(
            conf_threshold=base.conf_threshold,
            iou_threshold=base.iou_threshold,
            max_detections=base.max_detections,
            apply_nms=False,
            class_agnostic_nms=base.class_agnostic_nms,
            anchors_has_objectness=base.anchors_has_objectness,
            class_ids=base.class_ids,
        )
    )
    return post_with_nms, post_no_nms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO postprocess latency with NMS vs without NMS (top-K only)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", default=None, help="Path to an input image (repeated N times).")
    src.add_argument("--video", default=None, help="Path to an input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")
    src.add_argument(
        "--synthetic-boxes",
        type=int,
        default=None,
        help="Run a model-free benchmark using N synthetic decoded boxes (skips preprocess/inference).",
    )

    parser.add_argument("--model", default="Models/yolov9-s_v2.onnx", help="Path to a YOLO model (.onnx/.engine/.pt).")
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument(
        "--onnx-providers",
        default=None,
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (pre-NMS).")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections to keep after NMS/top-K.")
    parser.add_argument("--per-class-nms", action="store_true", help="Use per-class NMS (default is class-agnostic).")
    parser.add_argument("--synthetic-classes", type=int, default=1, help="For --synthetic-boxes: number of classes.")

    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame for video/webcam.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames (0 = no limit).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames to run but not record.")
    parser.add_argument("--repeats", type=int, default=50, help="For --image only: number of repeats.")
    args = parser.parse_args()

    if args.every < 1:
        raise ValueError("--every must be >= 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_det < 1:
        raise ValueError("--max-det must be >= 1")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")

    post_with_nms, post_no_nms = _make_postprocessors(args)

    t_pre: List[float] = []
    t_inf: List[float] = []
    t_post_nms: List[float] = []
    t_post_no: List[float] = []

    if args.synthetic_boxes is not None:
        n = int(args.synthetic_boxes)
        if n < 1:
            raise ValueError("--synthetic-boxes must be >= 1")
        n_classes = int(args.synthetic_classes)
        if n_classes < 1:
            raise ValueError("--synthetic-classes must be >= 1")

        # Decoded layout: [x1, y1, x2, y2, score, class_id]
        rng = np.random.default_rng(0)
        x1y1 = rng.uniform(0, 600, size=(n, 2)).astype(np.float32)
        wh = rng.uniform(5, 80, size=(n, 2)).astype(np.float32)
        x2y2 = x1y1 + wh
        scores = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        class_ids = rng.integers(0, n_classes, size=(n, 1), dtype=np.int32).astype(np.float32)
        decoded = np.concatenate([x1y1, x2y2, scores, class_ids], axis=1)

        seen = 0
        warm = 0
        while True:
            seen += 1
            t0 = time.perf_counter()
            _ = post_with_nms.process(decoded, orig_size=(640, 640), pad=(0.0, 0.0), ratio=(1.0, 1.0))
            t1 = time.perf_counter()
            _ = post_no_nms.process(decoded, orig_size=(640, 640), pad=(0.0, 0.0), ratio=(1.0, 1.0))
            t2 = time.perf_counter()

            if warm < int(args.warmup):
                warm += 1
            else:
                t_post_nms.append(t1 - t0)
                t_post_no.append(t2 - t1)

            if args.max_frames and len(t_post_nms) >= int(args.max_frames):
                break
            if args.max_frames == 0 and len(t_post_nms) >= 200:
                break

        t_pre = [0.0] * len(t_post_nms)
        t_inf = [0.0] * len(t_post_nms)
        frames_seen = seen
        warmup = int(args.warmup)
    else:
        onnx_providers = None
        if args.onnx_providers:
            onnx_providers = [p.strip() for p in str(args.onnx_providers).split(",") if p.strip()]

        # We set apply_nms=True here only so `pipeline(frame)` works, but for the benchmark we
        # use raw preds and run postprocess ourselves (NMS on/off) for fair comparison.
        pipeline = load_pipeline(
            model_path=args.model,
            backend=args.backend,
            post_cfg=YoloPostConfig(
                conf_threshold=float(args.conf),
                iou_threshold=float(args.iou),
                max_detections=int(args.max_det),
            ),
            letterbox_cfg=LetterboxConfig(new_shape=(int(args.imgsz), int(args.imgsz))),
            onnx_providers=onnx_providers,
        )

        seen = 0
        warm = 0
        for frame in _iter_frames(args):
            seen += 1

            t0 = time.perf_counter()
            prep = pipeline.preprocess(frame)
            t1 = time.perf_counter()
            preds = pipeline._infer_fn(prep.blob)  # type: ignore[attr-defined]
            t2 = time.perf_counter()
            _ = post_with_nms.process(preds, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio)
            t3 = time.perf_counter()
            _ = post_no_nms.process(preds, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio)
            t4 = time.perf_counter()

            if warm < int(args.warmup):
                warm += 1
                continue

            t_pre.append(t1 - t0)
            t_inf.append(t2 - t1)
            t_post_nms.append(t3 - t2)
            t_post_no.append(t4 - t3)

            if args.max_frames and len(t_pre) >= int(args.max_frames):
                break

        frames_seen = seen
        warmup = int(args.warmup)

    if not t_pre:
        raise RuntimeError("No benchmark samples collected (check input source / max-frames / warmup).")

    pre_s = _summarize_ms(t_pre)
    inf_s = _summarize_ms(t_inf)
    post_nms_s = _summarize_ms(t_post_nms)
    post_no_s = _summarize_ms(t_post_no)

    print(_format_summary("preprocess", pre_s))
    print(_format_summary("inference", inf_s))
    print(_format_summary("postprocess_with_nms", post_nms_s))
    print(_format_summary("postprocess_no_nms_topk", post_no_s))
    print(f"frames_seen={frames_seen} samples_recorded={len(t_pre)} warmup={warmup}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

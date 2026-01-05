from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from yolo_kit import LetterboxConfig, YoloPostConfig, load_pipeline


@dataclass(frozen=True)
class TimingSummary:
    n: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float


@dataclass(frozen=True)
class CaseSpec:
    label: str
    model: str
    imgsz: int


@dataclass(frozen=True)
class CaseResult:
    case: CaseSpec
    preprocess: TimingSummary
    inference: TimingSummary
    postprocess: TimingSummary
    total: TimingSummary


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("No values provided.")
    if q < 0.0 or q > 100.0:
        raise ValueError("q must be in [0, 100].")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    t = pos - lo
    return float(sorted_values[lo] * (1.0 - t) + sorted_values[hi] * t)


def _summarize_s(values_s: List[float]) -> TimingSummary:
    ms_sorted = sorted([v * 1000.0 for v in values_s])
    return TimingSummary(
        n=len(ms_sorted),
        mean_ms=float(statistics.fmean(ms_sorted)) if ms_sorted else 0.0,
        p50_ms=_percentile(ms_sorted, 50.0) if ms_sorted else 0.0,
        p90_ms=_percentile(ms_sorted, 90.0) if ms_sorted else 0.0,
        p95_ms=_percentile(ms_sorted, 95.0) if ms_sorted else 0.0,
    )


def _parse_case(raw: str) -> CaseSpec:
    """
    Format: LABEL:MODEL_PATH:IMGSZ
    Example: "640:Models/action_model.onnx:640"
    """
    parts = [p.strip() for p in str(raw).split(":", maxsplit=2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError('Invalid --case. Expected format "LABEL:MODEL_PATH:IMGSZ".')
    label, model, imgsz_s = parts
    imgsz = int(imgsz_s)
    if imgsz < 32:
        raise ValueError("IMGSZ in --case must be >= 32")
    return CaseSpec(label=label, model=model, imgsz=imgsz)


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


def _format_ms_triplet(s: TimingSummary) -> str:
    return f"{s.mean_ms:.3f}/{s.p50_ms:.3f}/{s.p95_ms:.3f}"


def _print_table(results: Sequence[CaseResult]) -> None:
    rows: List[Tuple[str, str, str, str, str]] = []
    for r in results:
        total_fps = (1000.0 / r.total.mean_ms) if r.total.mean_ms > 0 else 0.0
        rows.append(
            (
                r.case.label,
                f"{Path(r.case.model).name}@{r.case.imgsz}",
                _format_ms_triplet(r.inference),
                _format_ms_triplet(r.total),
                f"{total_fps:.2f}",
            )
        )

    headers = ("case", "model@imgsz", "infer(mean/p50/p95 ms)", "total(mean/p50/p95 ms)", "fps(mean_total)")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: Tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _run_case(args: argparse.Namespace, case: CaseSpec) -> CaseResult:
    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = [p.strip() for p in str(args.onnx_providers).split(",") if p.strip()]

    pipeline = load_pipeline(
        model_path=case.model,
        backend=args.backend,
        post_cfg=YoloPostConfig(
            conf_threshold=float(args.conf),
            iou_threshold=float(args.iou),
            apply_nms=not bool(args.no_nms),
            class_agnostic_nms=not bool(args.per_class_nms),
            max_detections=int(args.max_det),
        ),
        letterbox_cfg=LetterboxConfig(new_shape=(int(case.imgsz), int(case.imgsz))),
        onnx_providers=onnx_providers,
        onnx_input_name=args.onnx_input_name,
        onnx_output_name=args.onnx_output_name,
    )

    t_pre: List[float] = []
    t_inf: List[float] = []
    t_post: List[float] = []
    t_total: List[float] = []

    warm = 0
    seen = 0
    for frame in _iter_frames(args):
        seen += 1

        t0 = time.perf_counter()
        prep = pipeline.preprocess(frame)
        t1 = time.perf_counter()
        preds = pipeline._infer_fn(prep.blob)  # type: ignore[attr-defined]
        t2 = time.perf_counter()
        _ = pipeline.post.process(preds, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio)
        t3 = time.perf_counter()

        if warm < int(args.warmup):
            warm += 1
            continue

        t_pre.append(t1 - t0)
        t_inf.append(t2 - t1)
        t_post.append(t3 - t2)
        t_total.append(t3 - t0)

        if args.max_frames and len(t_total) >= int(args.max_frames):
            break

    if not t_total:
        raise RuntimeError(
            f"No samples recorded for case {case.label!r}. "
            "Check your source, --warmup, and --max-frames/--repeats."
        )

    return CaseResult(
        case=case,
        preprocess=_summarize_s(t_pre),
        inference=_summarize_s(t_inf),
        postprocess=_summarize_s(t_post),
        total=_summarize_s(t_total),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare end-to-end latency across multiple YOLO models/exports.\n\n"
            'Repeat --case with format "LABEL:MODEL_PATH:IMGSZ" (e.g. 640:Models/action_model.onnx:640).'
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", default=None, help="Path to an input image (repeated N times).")
    src.add_argument("--video", default=None, help="Path to an input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")

    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help='Benchmark case in format "LABEL:MODEL_PATH:IMGSZ". Repeatable.',
    )
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument(
        "--onnx-providers",
        default=None,
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument(
        "--onnx-input-name",
        default=None,
        help='Override ONNX input name (default: first input, often "images").',
    )
    parser.add_argument(
        "--onnx-output-name",
        default=None,
        help='Override ONNX output name (default: first output, e.g. "output0").',
    )

    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections to keep after NMS/top-K.")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS and only keep top-K detections by score.")
    parser.add_argument("--per-class-nms", action="store_true", help="Use per-class NMS (default is class-agnostic).")

    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame for video/webcam.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames (0 = no limit).")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames to run but not record.")
    parser.add_argument("--repeats", type=int, default=200, help="For --image only: number of repeats.")
    parser.add_argument("--json-out", default=None, help="Optional output path to write results JSON.")
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
    if args.conf < 0 or args.conf > 1:
        raise ValueError("--conf must be in [0, 1]")
    if args.iou < 0 or args.iou > 1:
        raise ValueError("--iou must be in [0, 1]")

    cases = [_parse_case(c) for c in args.case]

    results: List[CaseResult] = []
    for case in cases:
        print(f"running case={case.label!r} model={case.model!r} imgsz={case.imgsz} ...")
        results.append(_run_case(args, case))

    _print_table(results)

    if args.json_out:
        payload: Dict[str, Any] = {"results": [asdict(r) for r in results]}
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


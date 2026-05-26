import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from Action_Detection_SOP.roi import RoiPolygon, draw_roi, load_roi_json, resolve_roi_for_frame
from yolo_kit import LetterboxConfig, YoloPostConfig, draw_detections, load_class_names, load_pipeline


def _parse_label_conf(raw_values, *, class_names):
    if not raw_values:
        return {}
    if not class_names:
        raise ValueError("--label-conf requires --metadata so label names can be resolved.")

    by_name = {str(name).strip().lower(): int(class_id) for class_id, name in class_names.items()}
    out = {}
    for raw in raw_values:
        if "=" not in str(raw):
            raise ValueError(f"--label-conf must use label=value format, got: {raw!r}")
        raw_label, raw_conf = str(raw).split("=", 1)
        label = raw_label.strip().lower()
        if not label:
            raise ValueError(f"--label-conf has an empty label: {raw!r}")
        if label not in by_name:
            raise ValueError(f"--label-conf label {label!r} is not in metadata names: {sorted(by_name)}")
        try:
            conf = float(raw_conf)
        except ValueError as exc:
            raise ValueError(f"--label-conf value must be a number within [0, 1], got: {raw!r}") from exc
        if conf < 0.0 or conf > 1.0:
            raise ValueError(f"--label-conf value must be within [0, 1], got: {raw!r}")
        out[int(by_name[label])] = conf
    return out


def _filter_by_class_conf(detections, *, default_conf: float, class_conf):
    out = []
    for det in detections:
        class_id = int(det.class_id) if det.class_id is not None else None
        min_conf = float(class_conf.get(class_id, default_conf))
        if float(det.score) >= min_conf:
            out.append(det)
    return out


def _maybe_resize_max_side(image_bgr, *, max_side: int):
    if max_side <= 0:
        return image_bgr
    if image_bgr is None or not hasattr(image_bgr, "shape"):
        raise TypeError("image_bgr must be a NumPy array.")
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {getattr(image_bgr, 'shape', None)}")
    cur_max = max(h, w)
    if cur_max <= max_side:
        return image_bgr

    scale = float(max_side) / float(cur_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _processed_total_frames_from_capture(cap: cv2.VideoCapture, *, every: int):
    if every < 1:
        return None
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total is None or total <= 0:
        return None
    total_i = int(total)
    if total_i <= 0:
        return None
    return (total_i + every - 1) // every


def _draw_roi_overlay(image_bgr, roi: RoiPolygon | None):
    if roi is None:
        return image_bgr
    resolved = resolve_roi_for_frame(
        roi,
        frame_width=int(image_bgr.shape[1]),
        frame_height=int(image_bgr.shape[0]),
    )
    return draw_roi(image_bgr, resolved)


def _compress_video_with_ffmpeg(src: Path, dst: Path, *, crf: int, preset: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO detection and visualize bounding boxes + labels.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", default=None, help="Path to an input image.")
    src.add_argument("--video", default=None, help="Path to an input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")
    parser.add_argument("--model", default="Models/yolov9-s_v2.onnx", help="Path to a YOLO model (.onnx/.engine/.pt).")
    parser.add_argument("--metadata", default="configs/metadata.yaml", help="Path to class metadata (names mapping).")
    parser.add_argument(
        "--input-max-side",
        type=int,
        default=0,
        help="Optional downscale before inference/visualization: resize so max(H,W)=N (0=disabled).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--label-conf",
        action="append",
        default=[],
        help=(
            "Per-label confidence override, e.g. --label-conf cleaning_cloth=0.05. "
            "Other classes keep --conf."
        ),
    )
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS and only keep top-K detections by score.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections to keep after NMS/top-K.")
    parser.add_argument(
        "--class-ids",
        default=None,
        help='Optional class filter, e.g. "1" or "0,1". Useful to force helmet-only visualization.',
    )
    parser.add_argument(
        "--class-aware-nms",
        action="store_true",
        help="Run per-class NMS (recommended for nested objects like helmet inside person).",
    )
    parser.add_argument(
        "--anchors-box-format",
        default="auto",
        choices=["auto", "cxcywh", "x1y1wh", "xyxy"],
        help='Anchors layout box format (4+C,A). Use "auto" unless you know exporter specifics.',
    )
    parser.add_argument(
        "--decoded-box-format",
        default="auto",
        choices=["auto", "cxcywh", "x1y1wh", "xyxy"],
        help='Decoded layout box format (N,6). Use "auto" unless you know exporter specifics.',
    )
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument(
        "--onnx-providers",
        default="CUDAExecutionProvider",
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
    parser.add_argument("--show", action="store_true", help="Show a window with visualized detections.")
    parser.add_argument("--out", default=None, help="Optional output path (image or video) to save the visualization.")
    parser.add_argument(
        "--out-max-side",
        type=int,
        default=0,
        help="Optional resize for saved visualization only: max(H,W)=N (0=disabled).",
    )
    parser.add_argument(
        "--out-codec",
        default="mp4v",
        help='OpenCV video writer codec for the intermediate/fallback output, e.g. "mp4v" or "XVID".',
    )
    parser.add_argument(
        "--compress-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For video outputs, transcode with ffmpeg/libx264 after writing when ffmpeg is available.",
    )
    parser.add_argument("--out-crf", type=int, default=30, help="ffmpeg CRF for --compress-out (higher = smaller/lower quality).")
    parser.add_argument("--out-preset", default="veryfast", help="ffmpeg x264 preset for --compress-out.")
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI polygon JSON to draw on the visualization (from Scripts/calibrate_roi.py).",
    )
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame for video/webcam.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit).")
    parser.add_argument(
        "--debug-post",
        action="store_true",
        help="Print preprocess/postprocess debug stats (useful when boxes look misplaced).",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.metadata) if args.metadata else {}
    roi = load_roi_json(Path(args.roi)) if args.roi else None

    if args.input_max_side < 0:
        raise ValueError("--input-max-side must be >= 0")
    if args.out_max_side < 0:
        raise ValueError("--out-max-side must be >= 0")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    if args.conf < 0.0 or args.conf > 1.0:
        raise ValueError("--conf must be within [0, 1]")
    if not (0 <= int(args.out_crf) <= 51):
        raise ValueError("--out-crf must be within [0, 51]")
    if len(str(args.out_codec)) != 4:
        raise ValueError("--out-codec must be a four-character OpenCV codec such as mp4v")
    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = [p.strip() for p in str(args.onnx_providers).split(",") if p.strip()]

    class_ids = None
    if args.class_ids:
        class_ids = [int(x.strip()) for x in str(args.class_ids).split(",") if x.strip()]

    class_conf = _parse_label_conf(args.label_conf, class_names=class_names)
    if class_ids is not None:
        selected = set(int(x) for x in class_ids)
        unused_overrides = sorted(set(class_conf) - selected)
        if unused_overrides:
            names = [class_names.get(class_id, str(class_id)) for class_id in unused_overrides]
            raise ValueError(
                "--label-conf includes classes excluded by --class-ids: "
                f"{names}. Add them to --class-ids or remove the override."
            )
    post_conf = min([float(args.conf), *[float(v) for v in class_conf.values()]])

    pipeline = load_pipeline(
        model_path=args.model,
        backend=args.backend,
        post_cfg=YoloPostConfig(
            conf_threshold=post_conf,
            iou_threshold=args.iou,
            apply_nms=not bool(args.no_nms),
            class_agnostic_nms=not bool(args.class_aware_nms),
            anchors_box_format=str(args.anchors_box_format),
            decoded_box_format=str(args.decoded_box_format),
            max_detections=int(args.max_det),
            class_ids=class_ids,
        ),
        letterbox_cfg=LetterboxConfig(new_shape=(int(args.imgsz), int(args.imgsz))),
        onnx_providers=onnx_providers,
        onnx_input_name=args.onnx_input_name,
        onnx_output_name=args.onnx_output_name,
    )

    # Default behavior stays image-based (backward compatible) when no source is provided.
    image_path = args.image or (None if (args.video is not None or args.webcam is not None) else "Media/pedestrian.png")

    if image_path is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at path: {image_path}")
        img = _maybe_resize_max_side(img, max_side=int(args.input_max_side))

        if args.debug_post:
            import numpy as np

            prep = pipeline.preprocess(img)
            input_w = float(prep.orig_size[0] * prep.ratio[0] + 2.0 * prep.pad[0])
            input_h = float(prep.orig_size[1] * prep.ratio[1] + 2.0 * prep.pad[1])
            print("orig_size(w,h)=", prep.orig_size, "imgsz=", args.imgsz)
            print("ratio(w,h)=", prep.ratio, "pad(dw,dh)=", prep.pad)
            print("input_size(w,h)=", (input_w, input_h))
            print("blob shape/min/max=", prep.blob.shape, float(prep.blob.min()), float(prep.blob.max()))

            preds = pipeline._infer_fn(prep.blob)  # type: ignore[attr-defined]
            preds_np = preds  # already ndarray
            print("preds shape/min/max=", preds_np.shape, float(preds_np.min()), float(preds_np.max()))

            # Per-channel stats for anchors-layout debugging (e.g., (1, 6, 8400)).
            p = np.asarray(preds_np)
            if p.ndim == 3 and p.shape[0] == 1:
                p = p[0]
            if p.ndim == 2:
                h, w = p.shape
                # Interpret as (C, A) when the smaller dim is "channels".
                small, large = (h, w) if h <= w else (w, h)
                is_channels_first = h <= w
                looks_like_channels = small >= 6 and small <= 512 and (large / max(small, 1)) >= 4
                if looks_like_channels:
                    if is_channels_first:
                        channels, anchors = h, w
                        p_ca = p
                    else:
                        channels, anchors = w, h
                        p_ca = p.T
                    print("anchors layout guess: channels=", int(channels), "anchors=", int(anchors))
                    for ci in range(min(int(channels), 8)):
                        v = p_ca[ci, :]
                        print(f"ch[{ci}] min/max=", float(np.min(v)), float(np.max(v)))
                    if channels >= 6:
                        c4 = p_ca[4, :]
                        c5 = p_ca[5, :]
                        print("ch[4] vs ch[5] max(ch5-ch4)=", float(np.max(c5 - c4)))
                        print(
                            "count(ch5>ch4)=",
                            int(np.sum(c5 > c4)),
                            "count(ch5>0.1)=",
                            int(np.sum(c5 > 0.1)),
                            "count(ch4>0.1)=",
                            int(np.sum(c4 > 0.1)),
                        )

            boxes_xyxy, scores, class_ids = pipeline.post._decode(preds_np, input_size=(input_w, input_h))  # type: ignore[attr-defined]
            print(
                "decoded boxes(min/max)=",
                tuple(map(float, boxes_xyxy.min(axis=0))) if boxes_xyxy.size else None,
                tuple(map(float, boxes_xyxy.max(axis=0))) if boxes_xyxy.size else None,
            )
            print("decoded scores(min/max)=", (float(scores.min()), float(scores.max())) if scores.size else None)
            print("decoded class_ids(unique)=", sorted(set(map(int, class_ids.tolist())))[:20] if class_ids.size else None)

            boxes_xyxy = pipeline.post._maybe_denormalize_boxes_xyxy(  # type: ignore[attr-defined]
                boxes_xyxy, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio
            )
            boxes_scaled = pipeline.post._scale_boxes(boxes_xyxy.copy(), prep.orig_size, prep.pad, prep.ratio)  # type: ignore[attr-defined]
            print(
                "scaled boxes(min/max)=",
                tuple(map(float, boxes_scaled.min(axis=0))) if boxes_scaled.size else None,
                tuple(map(float, boxes_scaled.max(axis=0))) if boxes_scaled.size else None,
            )

            detections = pipeline.post.process(preds_np, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio)
        else:
            detections = pipeline(img)
        detections = _filter_by_class_conf(detections, default_conf=float(args.conf), class_conf=class_conf)
        vis = draw_detections(img, detections, class_names=class_names, show_score=True)
        vis = _draw_roi_overlay(vis, roi)
        vis = _maybe_resize_max_side(vis, max_side=int(args.out_max_side))
        if args.out:
            ok = cv2.imwrite(args.out, vis)
            if not ok:
                raise RuntimeError(f"Failed to write output image: {args.out}")

        if args.show:
            cv2.imshow("detections", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for det in detections:
            name = class_names.get(det.class_id, str(det.class_id)) if det.class_id is not None else "object"
            print(name, det.score, det.as_xyxy())

        return 0

    # Video/webcam path
    if args.every < 1:
        raise ValueError("--every must be >= 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.video}")
    else:
        cam_index = 0 if args.webcam is None else int(args.webcam)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index: {cam_index}")

    writer = None
    writer_path = Path(args.out) if args.out else None
    final_out_path = Path(args.out) if args.out else None
    frame_idx = 0
    processed = 0
    pbar = None

    try:
        if args.out and bool(args.compress_out):
            suffix = final_out_path.suffix if final_out_path is not None and final_out_path.suffix else ".mp4"
            tmp = tempfile.NamedTemporaryFile(prefix="visualize_raw_", suffix=suffix, delete=False)
            tmp.close()
            writer_path = Path(tmp.name)

        # Default: when writing a video output, show progress (best-effort).
        # For offline videos we can usually estimate total frames; for webcam the bar is indeterminate.
        if args.out:
            try:
                from tqdm import tqdm  # type: ignore
            except Exception:
                tqdm = None  # type: ignore[assignment]

            if tqdm is not None:
                total = None
                if args.video is not None:
                    total = _processed_total_frames_from_capture(cap, every=int(args.every))
                pbar = tqdm(total=total, unit="frame", desc="visualize")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            if (frame_idx - 1) % args.every != 0:
                continue

            frame = _maybe_resize_max_side(frame, max_side=int(args.input_max_side))
            detections = pipeline(frame)
            detections = _filter_by_class_conf(detections, default_conf=float(args.conf), class_conf=class_conf)
            vis = draw_detections(frame, detections, class_names=class_names, show_score=True)
            vis = _draw_roi_overlay(vis, roi)
            vis = _maybe_resize_max_side(vis, max_side=int(args.out_max_side))

            if args.out and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    fps = 30.0
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*str(args.out_codec))
                writer = cv2.VideoWriter(str(writer_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {writer_path}")

            if writer is not None:
                writer.write(vis)

            if args.show:
                cv2.imshow("detections", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            processed += 1
            if pbar is not None:
                pbar.update(1)
            if args.max_frames and processed >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            writer = None
        if (
            args.out
            and bool(args.compress_out)
            and writer_path is not None
            and final_out_path is not None
            and writer_path.exists()
            and writer_path.stat().st_size > 0
        ):
            compressed = _compress_video_with_ffmpeg(
                writer_path,
                final_out_path,
                crf=int(args.out_crf),
                preset=str(args.out_preset),
            )
            if compressed:
                try:
                    writer_path.unlink()
                except OSError:
                    pass
            else:
                if writer_path != final_out_path:
                    shutil.move(str(writer_path), str(final_out_path))
                    print("Note: ffmpeg compression unavailable/failed; kept OpenCV mp4v output.")
        if pbar is not None:
            pbar.close()
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

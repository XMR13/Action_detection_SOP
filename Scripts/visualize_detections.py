import argparse

import cv2

from yolo_kit import LetterboxConfig, YoloPostConfig, draw_detections, load_class_names, load_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO detection and visualize bounding boxes + labels.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", default=None, help="Path to an input image.")
    src.add_argument("--video", default=None, help="Path to an input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")
    parser.add_argument("--model", default="Models/yolov9-s_v2.onnx", help="Path to a YOLO model (.onnx/.engine/.pt).")
    parser.add_argument("--metadata", default="Models/metadata_PPE.yaml", help="Path to class metadata (names mapping).")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
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
    parser.add_argument("--show", action="store_true", help="Show a window with visualized detections.")
    parser.add_argument("--out", default=None, help="Optional output path (image or video) to save the visualization.")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame for video/webcam.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit).")
    parser.add_argument(
        "--debug-post",
        action="store_true",
        help="Print preprocess/postprocess debug stats (useful when boxes look misplaced).",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.metadata) if args.metadata else {}

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = [p.strip() for p in str(args.onnx_providers).split(",") if p.strip()]

    class_ids = None
    if args.class_ids:
        class_ids = [int(x.strip()) for x in str(args.class_ids).split(",") if x.strip()]

    pipeline = load_pipeline(
        model_path=args.model,
        backend=args.backend,
        post_cfg=YoloPostConfig(
            conf_threshold=args.conf,
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
        vis = draw_detections(img, detections, class_names=class_names, show_score=True)
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
    frame_idx = 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            if (frame_idx - 1) % args.every != 0:
                continue

            detections = pipeline(frame)
            vis = draw_detections(frame, detections, class_names=class_names, show_score=True)

            if args.out and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    fps = 30.0
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {args.out}")

            if writer is not None:
                writer.write(vis)

            if args.show:
                cv2.imshow("detections", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

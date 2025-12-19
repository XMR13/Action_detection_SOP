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
    parser.add_argument("--metadata", default="Models/metadata.yaml", help="Path to class metadata (names mapping).")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS and only keep top-K detections by score.")
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument(
        "--onnx-providers",
        default=None,
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument("--show", action="store_true", help="Show a window with visualized detections.")
    parser.add_argument("--out", default=None, help="Optional output path (image or video) to save the visualization.")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame for video/webcam.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit).")
    args = parser.parse_args()

    class_names = load_class_names(args.metadata) if args.metadata else {}

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = [p.strip() for p in str(args.onnx_providers).split(",") if p.strip()]

    pipeline = load_pipeline(
        model_path=args.model,
        backend=args.backend,
        post_cfg=YoloPostConfig(conf_threshold=args.conf, iou_threshold=args.iou, apply_nms=not bool(args.no_nms)),
        letterbox_cfg=LetterboxConfig(new_shape=(int(args.imgsz), int(args.imgsz))),
        onnx_providers=onnx_providers,
    )

    # Default behavior stays image-based (backward compatible) when no source is provided.
    image_path = args.image or (None if (args.video is not None or args.webcam is not None) else "Media/pedestrian.png")

    if image_path is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at path: {image_path}")

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

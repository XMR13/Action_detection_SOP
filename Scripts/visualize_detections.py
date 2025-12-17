import argparse

import cv2

from yolo_kit import YoloPostConfig, draw_detections, load_class_names, load_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO detection and visualize bounding boxes + labels.")
    parser.add_argument("--image", default="Media/pedestrian.png", help="Path to an input image.")
    parser.add_argument("--model", default="Models/yolov9-s_v2.onnx", help="Path to a YOLO model (.onnx/.engine/.pt).")
    parser.add_argument("--metadata", default="Models/metadata.yaml", help="Path to class metadata (names mapping).")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument("--show", action="store_true", help="Show a window with visualized detections.")
    parser.add_argument("--out", default=None, help="Optional output image path to save the visualization.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {args.image}")

    class_names = load_class_names(args.metadata) if args.metadata else {}

    pipeline = load_pipeline(
        model_path=args.model,
        backend=args.backend,
        post_cfg=YoloPostConfig(conf_threshold=args.conf, iou_threshold=args.iou),
    )

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

    # Useful output even when not showing/saving.
    for det in detections:
        name = class_names.get(det.class_id, str(det.class_id)) if det.class_id is not None else "object"
        print(name, det.score, det.as_xyxy())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

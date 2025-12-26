from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

from yolo_kit import LetterboxConfig, YoloPostConfig, load_class_names, load_pipeline
from yolo_kit.types import Detection

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class ExportConfig:
    images_dir: Path
    out: Path
    model: Path
    metadata: Optional[Path]
    backend: Optional[str]
    imgsz: int
    conf: float
    iou: float
    no_nms: bool
    onnx_providers: Optional[Tuple[str, ...]]
    person_label: Tuple[str, ...]
    include_empty_helmet_category: bool
    recursive: bool
    exts: Tuple[str, ...]
    max_images: int
    file_name_mode: str


def _iter_image_paths(images_dir: Path, *, recursive: bool, exts: Sequence[str]) -> List[Path]:
    patterns = [f"*.{e.lstrip('.').lower()}" for e in exts]
    paths: List[Path] = []
    if recursive:
        for p in images_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower().lstrip(".") in {e.lstrip(".").lower() for e in exts}:
                paths.append(p)
    else:
        for pat in patterns:
            paths.extend(images_dir.glob(pat))
    paths = sorted({p.resolve() for p in paths})
    return paths


def _name_to_ids(class_names: Dict[int, str], labels: Sequence[str]) -> List[int]:
    wanted = {str(s).strip().lower() for s in labels if str(s).strip()}
    ids: List[int] = []
    for cid, name in class_names.items():
        if str(name).strip().lower() in wanted:
            ids.append(int(cid))
    return sorted(set(ids))


def _clamp_xyxy(
    xyxy: Tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _to_coco_bbox(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _file_name_for_coco(path: Path, *, images_dir: Path, mode: str) -> str:
    if mode == "basename":
        return path.name
    rel = path.resolve().relative_to(images_dir.resolve())
    return rel.as_posix()


def _parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-annotate 'person' bounding boxes using a detector model, and export COCO JSON for CVAT import. "
            "Typical workflow: pre-annotate person -> import to CVAT -> manually label helmet -> export YOLO for training."
        )
    )
    parser.add_argument(
        "--images-dir",
        default="datasets",
        help="Directory containing images to annotate (e.g., datasets/cvat_frames/<video_stem>/).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output COCO JSON path (default: <images-dir>/person_auto_coco.json).",
    )
    parser.add_argument("--model", default="Models/yolo10s-PPE.onnx", help="Path to detector (.onnx/.engine/.pt).")
    parser.add_argument(
        "--metadata",
        default="Models/metadata_PPE.yaml",
        help="Path to class metadata yaml (names mapping). Set empty to skip name resolution.",
    )
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS and only keep top-K detections by score.")
    parser.add_argument(
        "--onnx-providers",
        default=None,
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument(
        "--person-label",
        action="append",
        default=["person"],
        help="Class name for person in metadata (repeatable).",
    )
    parser.add_argument(
        "--include-empty-helmet-category",
        action="store_true",
        help="Include a 'helmet' category with no annotations (helps when your CVAT project already has helmet label).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images under --images-dir.")
    parser.add_argument(
        "--ext",
        action="append",
        default=["jpg", "jpeg", "png"],
        help="Image extension(s) to include (repeatable).",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Stop after N images (0 = no limit).")
    parser.add_argument(
        "--file-name-mode",
        choices=["basename", "relative"],
        default="basename",
        help=(
            "How to write COCO images[].file_name. "
            "Use 'basename' if you upload images as flat files; use 'relative' if you upload a zip preserving folders."
        ),
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"--images-dir not found or not a directory: {images_dir}")

    out = Path(args.out) if args.out else (images_dir / "person_auto_coco.json")

    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")

    metadata: Optional[Path]
    if args.metadata and str(args.metadata).strip():
        metadata = Path(args.metadata)
        if not metadata.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata}")
    else:
        metadata = None

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    if args.conf < 0 or args.conf > 1:
        raise ValueError("--conf must be within [0, 1]")
    if args.iou <= 0 or args.iou > 1:
        raise ValueError("--iou must be within (0, 1]")
    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0")

    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = tuple(p.strip() for p in str(args.onnx_providers).split(",") if p.strip())

    exts = tuple(str(e).lower().lstrip(".") for e in args.ext if str(e).strip())
    if not exts:
        raise ValueError("At least one --ext must be provided.")

    return ExportConfig(
        images_dir=images_dir,
        out=out,
        model=model,
        metadata=metadata,
        backend=args.backend,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        no_nms=bool(args.no_nms),
        onnx_providers=onnx_providers,
        person_label=tuple(args.person_label),
        include_empty_helmet_category=bool(args.include_empty_helmet_category),
        recursive=bool(args.recursive),
        exts=exts,
        max_images=int(args.max_images),
        file_name_mode=str(args.file_name_mode),
    )


def _iter_person_detections(dets: Sequence[Detection], *, person_ids: Sequence[int]) -> Iterable[Detection]:
    wanted = set(int(x) for x in person_ids)
    for d in dets:
        if d.class_id is None:
            continue
        if int(d.class_id) in wanted:
            yield d


def main() -> int:
    cfg = _parse_args()

    class_names: Dict[int, str] = load_class_names(str(cfg.metadata)) if cfg.metadata else {}
    person_ids = _name_to_ids(class_names, cfg.person_label) if class_names else [0]
    if not person_ids:
        raise ValueError(
            f"Could not resolve person class ids from labels: {cfg.person_label!r}. "
            "Fix --metadata/--person-label, or pass --metadata '' to default person id=0."
        )

    pipeline = load_pipeline(
        model_path=str(cfg.model),
        backend=cfg.backend,
        post_cfg=YoloPostConfig(
            conf_threshold=float(cfg.conf),
            iou_threshold=float(cfg.iou),
            apply_nms=not bool(cfg.no_nms),
            class_ids=person_ids,
        ),
        letterbox_cfg=LetterboxConfig(new_shape=(int(cfg.imgsz), int(cfg.imgsz))),
        onnx_providers=list(cfg.onnx_providers) if cfg.onnx_providers else None,
    )

    image_paths = _iter_image_paths(cfg.images_dir, recursive=cfg.recursive, exts=cfg.exts)
    if cfg.max_images:
        image_paths = image_paths[: int(cfg.max_images)]
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {cfg.images_dir} with extensions {list(cfg.exts)} (recursive={cfg.recursive})."
        )

    categories: List[Dict[str, object]] = [{"id": 1, "name": "person"}]
    category_id_person = 1
    if cfg.include_empty_helmet_category:
        categories.append({"id": 2, "name": "helmet"})

    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []

    ann_id = 1
    img_id = 1

    if tqdm is None:
        print("Note: tqdm is not installed; progress bar disabled. (Install via uv if you want a bar.)")

    iterator = tqdm(image_paths, unit="img") if tqdm is not None else image_paths
    for p in iterator:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        h, w = img.shape[:2]

        images.append(
            {
                "id": img_id,
                "file_name": _file_name_for_coco(p, images_dir=cfg.images_dir, mode=cfg.file_name_mode),
                "width": int(w),
                "height": int(h),
            }
        )

        dets = pipeline(img)
        for d in _iter_person_detections(dets, person_ids=person_ids):
            clamped = _clamp_xyxy(d.as_xyxy(), width=w, height=h)
            if clamped is None:
                continue
            x, y, bw, bh = _to_coco_bbox(clamped)
            area = float(bw * bh)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id_person,
                    "bbox": [x, y, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    payload = {
        "info": {"description": "Auto-annotated person boxes", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

    print(f"Wrote COCO JSON: {cfg.out}")
    print(f"Images: {len(images)}")
    print(f"Annotations (person): {len(annotations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

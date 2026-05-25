from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Action_Detection_SOP.roi import RoiPolygon, load_roi_json, resolve_roi_for_frame
from yolo_kit import LetterboxConfig, YoloPostConfig, load_class_names, load_pipeline
from yolo_kit.types import Detection

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class MineConfig:
    video: Path
    out_dir: Path
    model: Path
    metadata: Path
    backend: Optional[str]
    imgsz: int
    conf: float
    iou: float
    max_det: int
    class_aware_nms: bool
    onnx_providers: Optional[Tuple[str, ...]]
    roll_label: str
    person_label: str
    roll_conf: float
    person_conf: float
    roi: Optional[RoiPolygon]
    scan_fps: float
    extract_fps: float
    pre_roll_s: float
    post_roll_s: float
    merge_gap_s: float
    min_window_s: float
    max_windows: int
    max_images: int
    ext: str
    jpg_quality: int
    name_template: str
    motion_threshold: float
    motion_weight: float


@dataclass(frozen=True)
class CandidateHit:
    frame_idx: int
    time_s: float
    roll_count: int
    person_count: int
    best_roll_score: float
    best_person_score: float
    motion_score: float
    reason: str


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float
    first_hit_s: float
    last_hit_s: float
    hit_count: int
    max_roll_score: float
    max_person_score: float
    max_motion_score: float
    reasons: Tuple[str, ...]


def _parse_args() -> MineConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Mine high-value labeling frames from long SOP videos. "
            "The script scans at low FPS with the current detector, creates windows where "
            "roll/person activity is likely, then extracts higher-FPS frame bursts for CVAT."
        )
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: datasets/roll_sop_v1/frames/<video_stem>_bursts).",
    )
    parser.add_argument("--model", required=True, help="Path to current roll SOP detector (.onnx/.engine/.pt).")
    parser.add_argument("--metadata", default="configs/metadata_roll_sop_v1.yaml", help="Class metadata YAML.")
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Base detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections after NMS.")
    parser.add_argument("--class-aware-nms", action="store_true", help="Run per-class NMS.")
    parser.add_argument(
        "--onnx-providers",
        default=None,
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument("--roll-label", default="roll", help="Metadata class name for roll.")
    parser.add_argument("--person-label", default="person", help="Metadata class name for person.")
    parser.add_argument("--roll-conf", type=float, default=0.45, help="Minimum roll confidence for candidate hits.")
    parser.add_argument("--person-conf", type=float, default=0.45, help="Minimum person confidence for candidate hits.")
    parser.add_argument("--roi", default=None, help="Optional ROI polygon JSON from Scripts/calibrate_roi.py.")
    parser.add_argument("--scan-fps", type=float, default=2.0, help="Detector scan rate over the full video.")
    parser.add_argument("--extract-fps", type=float, default=5.0, help="Frame extraction rate inside mined windows.")
    parser.add_argument("--pre-roll-s", type=float, default=3.0, help="Seconds to include before each candidate hit.")
    parser.add_argument("--post-roll-s", type=float, default=4.0, help="Seconds to include after each candidate hit.")
    parser.add_argument("--merge-gap-s", type=float, default=8.0, help="Merge candidate windows separated by <= this gap.")
    parser.add_argument("--min-window-s", type=float, default=1.0, help="Drop merged windows shorter than this.")
    parser.add_argument("--max-windows", type=int, default=0, help="Keep only the first N windows (0 = no limit).")
    parser.add_argument("--max-images", type=int, default=0, help="Stop after saving N images (0 = no limit).")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "jpeg", "png"], help="Output image extension.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality.")
    parser.add_argument(
        "--name-template",
        default="{stem}__burst{window:03d}__f{frame:06d}__t{t_s:06.2f}s",
        help="Filename template without extension. Fields: {stem}, {window}, {frame}, {t_s}.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=4.0,
        help=(
            "Optional ROI motion trigger. Mean abs diff above this value can create hits when a roll is present. "
            "Set 0 to disable motion-based hits."
        ),
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to the motion score before comparing with --motion-threshold.",
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    metadata = Path(args.metadata)
    if not metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata}")

    if args.out_dir is None:
        out_dir = Path("datasets") / "roll_sop_v1" / "frames" / f"{video.stem}_bursts"
    else:
        out_dir = Path(args.out_dir)

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    for name in ("conf", "iou", "roll_conf", "person_conf"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be within [0, 1]")
    if args.max_det <= 0:
        raise ValueError("--max-det must be > 0")
    if args.scan_fps <= 0 or args.extract_fps <= 0:
        raise ValueError("--scan-fps and --extract-fps must be > 0")
    if args.pre_roll_s < 0 or args.post_roll_s < 0:
        raise ValueError("--pre-roll-s and --post-roll-s must be >= 0")
    if args.merge_gap_s < 0 or args.min_window_s < 0:
        raise ValueError("--merge-gap-s and --min-window-s must be >= 0")
    if args.max_windows < 0 or args.max_images < 0:
        raise ValueError("--max-windows and --max-images must be >= 0")
    if not (1 <= int(args.jpg_quality) <= 100):
        raise ValueError("--jpg-quality must be in [1, 100]")
    if args.motion_threshold < 0 or args.motion_weight < 0:
        raise ValueError("--motion-threshold and --motion-weight must be >= 0")

    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = tuple(p.strip() for p in str(args.onnx_providers).split(",") if p.strip())

    roi = load_roi_json(Path(args.roi)) if args.roi else None

    return MineConfig(
        video=video,
        out_dir=out_dir,
        model=model,
        metadata=metadata,
        backend=args.backend,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        class_aware_nms=bool(args.class_aware_nms),
        onnx_providers=onnx_providers,
        roll_label=str(args.roll_label).strip().lower(),
        person_label=str(args.person_label).strip().lower(),
        roll_conf=float(args.roll_conf),
        person_conf=float(args.person_conf),
        roi=roi,
        scan_fps=float(args.scan_fps),
        extract_fps=float(args.extract_fps),
        pre_roll_s=float(args.pre_roll_s),
        post_roll_s=float(args.post_roll_s),
        merge_gap_s=float(args.merge_gap_s),
        min_window_s=float(args.min_window_s),
        max_windows=int(args.max_windows),
        max_images=int(args.max_images),
        ext=str(args.ext).lower(),
        jpg_quality=int(args.jpg_quality),
        name_template=str(args.name_template),
        motion_threshold=float(args.motion_threshold),
        motion_weight=float(args.motion_weight),
    )


def _name_to_id(class_names: Dict[int, str], label: str) -> int:
    wanted = label.strip().lower()
    for class_id, name in class_names.items():
        if str(name).strip().lower() == wanted:
            return int(class_id)
    raise ValueError(f"Could not resolve label {label!r} from metadata names: {class_names}")


def _det_center(d: Detection) -> Tuple[float, float]:
    return (float(d.x1 + d.x2) * 0.5, float(d.y1 + d.y2) * 0.5)


def _filter_by_roi(dets: Iterable[Detection], roi: Optional[RoiPolygon]) -> List[Detection]:
    if roi is None:
        return list(dets)
    out: List[Detection] = []
    for d in dets:
        cx, cy = _det_center(d)
        if roi.contains_point(cx, cy):
            out.append(d)
    return out


def _selected_dets(
    dets: Sequence[Detection],
    *,
    class_id: int,
    min_conf: float,
    roi: Optional[RoiPolygon],
) -> List[Detection]:
    selected = [d for d in dets if d.class_id == class_id and float(d.score) >= min_conf]
    return _filter_by_roi(selected, roi)


def _frame_time_s(*, frame_idx: int, fps: float, pos_msec: float) -> float:
    if fps > 0:
        return float(frame_idx) / float(fps)
    if pos_msec > 0:
        return float(pos_msec) / 1000.0
    return 0.0


def _gray_signature(frame: np.ndarray, roi: Optional[RoiPolygon]) -> np.ndarray:
    if roi is not None:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi.as_contour()], 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def _motion_score(cur_sig: np.ndarray, prev_sig: Optional[np.ndarray]) -> float:
    if prev_sig is None:
        return 0.0
    return float(np.mean(np.abs(cur_sig - prev_sig)))


def _scan_hits(cfg: MineConfig, *, roll_id: int, person_id: int) -> Tuple[List[CandidateHit], float, int]:
    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = 1 if fps <= 0 else max(1, int(round(fps / cfg.scan_fps)))
    total = None if frame_count <= 0 else (frame_count + every - 1) // every

    pipeline = load_pipeline(
        model_path=str(cfg.model),
        backend=cfg.backend,
        post_cfg=YoloPostConfig(
            conf_threshold=min(cfg.conf, cfg.roll_conf, cfg.person_conf),
            iou_threshold=cfg.iou,
            max_detections=cfg.max_det,
            apply_nms=True,
            class_agnostic_nms=not cfg.class_aware_nms,
            class_ids=[roll_id, person_id],
        ),
        letterbox_cfg=LetterboxConfig(new_shape=(cfg.imgsz, cfg.imgsz)),
        onnx_providers=list(cfg.onnx_providers) if cfg.onnx_providers else None,
    )

    if tqdm is None:
        print("Note: tqdm is not installed; progress bar disabled. (Install via uv if you want a bar.)")

    hits: List[CandidateHit] = []
    pbar = tqdm(total=total, unit="scan") if tqdm is not None else None
    prev_sig: Optional[np.ndarray] = None
    roi_active = cfg.roi
    roi_resolved = False
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % every != 0:
                frame_idx += 1
                continue

            if roi_active is not None and not roi_resolved:
                roi_active = resolve_roi_for_frame(
                    roi_active,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )
                roi_resolved = True

            sig = _gray_signature(frame, roi_active)
            motion = _motion_score(sig, prev_sig) * cfg.motion_weight
            prev_sig = sig

            dets = pipeline(frame)
            rolls = _selected_dets(dets, class_id=roll_id, min_conf=cfg.roll_conf, roi=roi_active)
            people = _selected_dets(dets, class_id=person_id, min_conf=cfg.person_conf, roi=roi_active)

            roll_count = len(rolls)
            person_count = len(people)
            best_roll = max((float(d.score) for d in rolls), default=0.0)
            best_person = max((float(d.score) for d in people), default=0.0)

            reason = ""
            if roll_count > 0 and person_count > 0:
                reason = "roll_and_person"
            elif roll_count > 0 and cfg.motion_threshold > 0 and motion >= cfg.motion_threshold:
                reason = "roll_and_motion"

            if reason:
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                t_s = _frame_time_s(frame_idx=frame_idx, fps=fps, pos_msec=pos_msec)
                hits.append(
                    CandidateHit(
                        frame_idx=int(frame_idx),
                        time_s=float(t_s),
                        roll_count=int(roll_count),
                        person_count=int(person_count),
                        best_roll_score=best_roll,
                        best_person_score=best_person,
                        motion_score=float(motion),
                        reason=reason,
                    )
                )

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(hits=len(hits))

            frame_idx += 1
    finally:
        cap.release()
        if pbar is not None:
            pbar.close()

    return hits, fps, frame_count


def _merge_windows(hits: Sequence[CandidateHit], cfg: MineConfig, *, duration_s: Optional[float]) -> List[Window]:
    raw: List[Tuple[float, float, CandidateHit]] = []
    for h in hits:
        start = max(0.0, float(h.time_s) - cfg.pre_roll_s)
        end = float(h.time_s) + cfg.post_roll_s
        if duration_s is not None:
            end = min(float(duration_s), end)
        if end >= start:
            raw.append((start, end, h))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0])
    windows: List[Window] = []
    cur_start, cur_end, first_hit = raw[0]
    cur_hits: List[CandidateHit] = [first_hit]

    for start, end, hit in raw[1:]:
        if start <= cur_end + cfg.merge_gap_s:
            cur_end = max(cur_end, end)
            cur_hits.append(hit)
            continue
        window = _window_from_hits(cur_start, cur_end, cur_hits)
        if window.end_s - window.start_s >= cfg.min_window_s:
            windows.append(window)
        cur_start, cur_end, cur_hits = start, end, [hit]

    window = _window_from_hits(cur_start, cur_end, cur_hits)
    if window.end_s - window.start_s >= cfg.min_window_s:
        windows.append(window)

    if cfg.max_windows:
        windows = windows[: cfg.max_windows]
    return windows


def _window_from_hits(start_s: float, end_s: float, hits: Sequence[CandidateHit]) -> Window:
    return Window(
        start_s=float(start_s),
        end_s=float(end_s),
        first_hit_s=float(min(h.time_s for h in hits)),
        last_hit_s=float(max(h.time_s for h in hits)),
        hit_count=int(len(hits)),
        max_roll_score=float(max(h.best_roll_score for h in hits)),
        max_person_score=float(max(h.best_person_score for h in hits)),
        max_motion_score=float(max(h.motion_score for h in hits)),
        reasons=tuple(sorted({h.reason for h in hits})),
    )


def _format_name(template: str, *, stem: str, window: int, frame: int, t_s: float) -> str:
    try:
        return template.format(stem=stem, window=int(window), frame=int(frame), t_s=float(t_s))
    except Exception as exc:
        raise ValueError(f"Invalid --name-template {template!r}: {exc}") from exc


def _imwrite(path: Path, image_bgr: np.ndarray, *, ext: str, jpg_quality: int) -> None:
    params = []
    if ext in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok = cv2.imwrite(str(path), image_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _extract_windows(cfg: MineConfig, windows: Sequence[Window], *, fps: float) -> int:
    if not windows:
        return 0
    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for extraction: {cfg.video}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    every = 1 if fps <= 0 else max(1, int(round(fps / cfg.extract_fps)))
    saved = 0

    try:
        iterator = tqdm(list(enumerate(windows, start=1)), unit="window") if tqdm is not None else enumerate(windows, start=1)
        for window_idx, window in iterator:
            if fps > 0:
                start_frame = max(0, int(round(window.start_s * fps)))
                end_frame = max(start_frame, int(round(window.end_s * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, window.start_s * 1000.0)
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                end_frame = None

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                current = frame_idx
                frame_idx += 1

                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                t_s = _frame_time_s(frame_idx=current, fps=fps, pos_msec=pos_msec)

                if end_frame is not None and current > end_frame:
                    break
                if end_frame is None and t_s > window.end_s:
                    break
                if every > 1 and current % every != 0:
                    continue

                base = _format_name(
                    cfg.name_template,
                    stem=cfg.video.stem,
                    window=window_idx,
                    frame=current,
                    t_s=t_s,
                )
                out_path = cfg.out_dir / f"{base}.{cfg.ext}"
                _imwrite(out_path, frame, ext=cfg.ext, jpg_quality=cfg.jpg_quality)
                saved += 1

                if cfg.max_images and saved >= cfg.max_images:
                    return saved
    finally:
        cap.release()

    return saved


def _write_csv(path: Path, rows: Sequence[object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def _write_window_csv(path: Path, windows: Sequence[Window]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "window_idx",
            "start_s",
            "end_s",
            "first_hit_s",
            "last_hit_s",
            "hit_count",
            "max_roll_score",
            "max_person_score",
            "max_motion_score",
            "reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, w in enumerate(windows, start=1):
            writer.writerow(
                {
                    "window_idx": idx,
                    "start_s": f"{w.start_s:.3f}",
                    "end_s": f"{w.end_s:.3f}",
                    "first_hit_s": f"{w.first_hit_s:.3f}",
                    "last_hit_s": f"{w.last_hit_s:.3f}",
                    "hit_count": w.hit_count,
                    "max_roll_score": f"{w.max_roll_score:.4f}",
                    "max_person_score": f"{w.max_person_score:.4f}",
                    "max_motion_score": f"{w.max_motion_score:.4f}",
                    "reasons": "|".join(w.reasons),
                }
            )


def main() -> int:
    cfg = _parse_args()
    class_names = load_class_names(str(cfg.metadata))
    roll_id = _name_to_id(class_names, cfg.roll_label)
    person_id = _name_to_id(class_names, cfg.person_label)

    hits, fps, frame_count = _scan_hits(cfg, roll_id=roll_id, person_id=person_id)
    duration_s = (float(frame_count) / fps) if fps > 0 and frame_count > 0 else None
    windows = _merge_windows(hits, cfg, duration_s=duration_s)
    saved = _extract_windows(cfg, windows, fps=fps)

    _write_csv(
        cfg.out_dir / "candidate_hits.csv",
        hits,
        fieldnames=[
            "frame_idx",
            "time_s",
            "roll_count",
            "person_count",
            "best_roll_score",
            "best_person_score",
            "motion_score",
            "reason",
        ],
    )
    _write_window_csv(cfg.out_dir / "windows.csv", windows)
    summary = {
        "video": str(cfg.video),
        "model": str(cfg.model),
        "metadata": str(cfg.metadata),
        "fps": fps,
        "frame_count": frame_count,
        "scan_fps": cfg.scan_fps,
        "extract_fps": cfg.extract_fps,
        "hit_count": len(hits),
        "window_count": len(windows),
        "images_saved": saved,
        "out_dir": str(cfg.out_dir),
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Done. Hits={len(hits)}, windows={len(windows)}, saved={saved}")
    print(f"Frames: {cfg.out_dir}")
    print(f"Windows CSV: {cfg.out_dir / 'windows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

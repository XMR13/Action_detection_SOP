from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from Action_Detection_SOP.roi import RoiPolygon, clamp_rect_to_frame, load_roi_json, resolve_roi_for_frame

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class ExtractConfig:
    video: Path
    out_dir: Path
    every: int
    target_fps: float
    start_s: float
    end_s: float
    max_images: int
    ext: str
    jpg_quality: int
    name_template: str
    roi: Optional[RoiPolygon]
    roi_expand: int
    dedupe_threshold: float
    use_cuda: bool


def _frame_time_s(*, frame_idx: int, fps: float, pos_msec: float) -> float:
    if fps and fps > 0:
        return float(frame_idx) / float(fps)
    if pos_msec and pos_msec > 0:
        return float(pos_msec) / 1000.0
    return 0.0


def _downscale_gray(image_bgr: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def _maybe_crop_roi(frame_bgr: np.ndarray, roi: Optional[RoiPolygon], *, expand_px: int) -> np.ndarray:
    if roi is None:
        return frame_bgr
    x0, y0, x1, y1 = roi.bounding_rect(expand_px=int(expand_px))
    x0, y0, x1, y1 = clamp_rect_to_frame((x0, y0, x1, y1), frame_width=frame_bgr.shape[1], frame_height=frame_bgr.shape[0])
    return frame_bgr[y0:y1, x0:x1]


def _imwrite(path: Path, image_bgr: np.ndarray, *, ext: str, jpg_quality: int) -> None:
    params = []
    if ext in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok = cv2.imwrite(str(path), image_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _parse_args() -> ExtractConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-extract frames (screenshots) from a video for labeling (e.g., CVAT). "
            "Optionally crops to ROI bounding box and skips near-duplicate frames."
        )
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: datasets/cvat_frames/<video_stem>/).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=1.0,
        help="Approx output rate (frames/sec). Used to compute --every based on source FPS.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=0,
        help="Save every Nth frame (overrides --target-fps if >0).",
    )
    parser.add_argument("--start-s", type=float, default=0.0, help="Start timestamp in seconds.")
    parser.add_argument("--end-s", type=float, default=0.0, help="End timestamp in seconds (0 = until end).")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after saving N images (0 = no limit).",
    )
    parser.add_argument("--ext", default="jpg", choices=["jpg", "jpeg", "png"], help="Output image extension.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality (only for jpg/jpeg).")
    parser.add_argument(
        "--name-template",
        default="{stem}__f{frame:06d}__t{t_s:06.2f}s",
        help=(
            "Filename template (without extension). Available fields: "
            "{stem}, {frame}, {t_s}. Example: frame_{frame:06d}__t{t_s:06.2f}s"
        ),
    )

    parser.add_argument("--roi", default=None, help="Optional ROI polygon JSON (from Scripts/calibrate_roi.py).")
    parser.add_argument("--roi-expand", type=int, default=0, help="Expand ROI bounding box crop by N pixels (>=0).")
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.0,
        help=(
            "If >0, skip saving when the mean abs diff to last saved frame (64x64 gray) is below this threshold."
        ),
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use OpenCV CUDA ops for dedupe signature if available (falls back to CPU otherwise).",
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    if args.out_dir is None:
        out_dir = Path("datasets") / "cvat_frames" / video.stem
    else:
        out_dir = Path(args.out_dir)

    if args.start_s < 0:
        raise ValueError("--start-s must be >= 0")
    if args.end_s < 0:
        raise ValueError("--end-s must be >= 0")
    if args.end_s and args.end_s <= args.start_s:
        raise ValueError("--end-s must be > --start-s (or 0)")
    if args.every < 0:
        raise ValueError("--every must be >= 0")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0")
    if args.roi_expand < 0:
        raise ValueError("--roi-expand must be >= 0")
    if not (1 <= int(args.jpg_quality) <= 100):
        raise ValueError("--jpg-quality must be in [1, 100]")
    if args.dedupe_threshold < 0:
        raise ValueError("--dedupe-threshold must be >= 0")

    roi = load_roi_json(Path(args.roi)) if args.roi else None

    # `every` is computed after opening the capture (needs FPS). We store the raw requested value here.
    every_requested = int(args.every)

    return ExtractConfig(
        video=video,
        out_dir=out_dir,
        every=every_requested,
        target_fps=float(args.target_fps),
        start_s=float(args.start_s),
        end_s=float(args.end_s),
        max_images=int(args.max_images),
        ext=str(args.ext).lower(),
        jpg_quality=int(args.jpg_quality),
        name_template=str(args.name_template),
        roi=roi,
        roi_expand=int(args.roi_expand),
        dedupe_threshold=float(args.dedupe_threshold),
        use_cuda=bool(args.use_cuda),
    )


def _compute_every(*, fps: float, every_requested: int, target_fps: float) -> int:
    if every_requested and every_requested > 0:
        return int(every_requested)
    if not fps or fps <= 0:
        return 1
    return max(1, int(round(float(fps) / float(target_fps))))


def _cuda_available() -> bool:
    try:
        return hasattr(cv2, "cuda") and int(cv2.cuda.getCudaEnabledDeviceCount()) > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _downscale_gray_cuda(image_bgr: np.ndarray, size: int = 64) -> np.ndarray:
    gpu = cv2.cuda_GpuMat()  # type: ignore[attr-defined]
    gpu.upload(image_bgr)  # type: ignore[attr-defined]
    gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
    small = cv2.cuda.resize(gray, (size, size), interpolation=cv2.INTER_AREA)  # type: ignore[attr-defined]
    out = small.download()  # type: ignore[attr-defined]
    return out.astype(np.float32)


def _format_name(template: str, *, stem: str, frame: int, t_s: float) -> str:
    try:
        return template.format(stem=stem, frame=int(frame), t_s=float(t_s))
    except Exception as e:
        raise ValueError(f"Invalid --name-template {template!r}: {e}") from e


def main() -> int:
    cfg = _parse_args()

    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames: Optional[int] = frame_count_raw if frame_count_raw > 0 else None

    # Seek to start.
    if cfg.start_s > 0:
        if fps and fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(cfg.start_s * fps)))
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(cfg.start_s) * 1000.0)

    # If we know fps, we can compute the end frame index and a better total for tqdm.
    end_frame_exclusive: Optional[int] = None
    if cfg.end_s > 0 and fps and fps > 0:
        end_frame_exclusive = int(round(cfg.end_s * fps))

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if tqdm is None:
        print("Note: tqdm is not installed; progress bar disabled. (Install via uv if you want a bar.)")

    # Compute sampling.
    every = _compute_every(fps=fps, every_requested=cfg.every, target_fps=cfg.target_fps)

    use_cuda = bool(cfg.use_cuda and cfg.dedupe_threshold > 0 and _cuda_available())
    if cfg.use_cuda and not use_cuda and cfg.dedupe_threshold > 0:
        print("Note: --use-cuda requested but CUDA OpenCV is unavailable; using CPU for dedupe signature.")

    pbar_total: Optional[int] = None
    if total_frames is not None:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        remaining = max(0, total_frames - current_pos)
        if end_frame_exclusive is not None:
            remaining = max(0, min(remaining, max(0, end_frame_exclusive - current_pos)))
        pbar_total = remaining

    pbar = tqdm(total=pbar_total, unit="frame") if tqdm is not None else None

    saved = 0
    read = 0
    last_sig: Optional[np.ndarray] = None
    next_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    roi_active = cfg.roi
    roi_resolved = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            read += 1
            frame_idx = next_frame_idx
            next_frame_idx += 1
            if pbar is not None:
                pbar.update(1)

            if end_frame_exclusive is not None and frame_idx >= end_frame_exclusive:
                break
            if cfg.end_s > 0 and (not fps or fps <= 0):
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pos_msec >= cfg.end_s * 1000.0:
                    break

            if every > 1 and (frame_idx % every) != 0:
                if pbar is not None:
                    pbar.set_postfix(saved=saved)
                continue

            if roi_active is not None and not roi_resolved:
                roi_active = resolve_roi_for_frame(
                    roi_active,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
                roi_resolved = True

            image = _maybe_crop_roi(frame, roi_active, expand_px=cfg.roi_expand)

            if cfg.dedupe_threshold > 0:
                sig = _downscale_gray_cuda(image) if use_cuda else _downscale_gray(image)
                if last_sig is not None:
                    mad = float(np.mean(np.abs(sig - last_sig)))
                    if mad < cfg.dedupe_threshold:
                        if pbar is not None:
                            pbar.set_postfix(saved=saved)
                        continue
                last_sig = sig

            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            t_s = _frame_time_s(frame_idx=frame_idx, fps=fps, pos_msec=pos_msec)

            base = _format_name(cfg.name_template, stem=cfg.video.stem, frame=frame_idx, t_s=t_s)
            name = f"{base}.{cfg.ext}"
            out_path = out_dir / name
            _imwrite(out_path, image, ext=cfg.ext, jpg_quality=cfg.jpg_quality)

            saved += 1
            if pbar is not None:
                pbar.set_postfix(saved=saved)

            if cfg.max_images and saved >= cfg.max_images:
                break

    finally:
        cap.release()
        if pbar is not None:
            pbar.close()

    print(f"Done. Saved {saved} images to: {out_dir} (frames read: {read}, every={every}, fps={fps:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

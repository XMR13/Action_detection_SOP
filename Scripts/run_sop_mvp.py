from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

from Action_Detection_SOP.run_config import apply_run_config, collect_cli_dests, load_run_config
from Action_Detection_SOP.runner_mvp import run_mvp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-A SOP runner (operator session in ROI + helmet compliance).")
    parser.add_argument("--config", default=None, help="Optional JSON config to reduce CLI args (CLI overrides config).")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video", default=None, help="Path to input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")
    src.add_argument("--rtsp", default=None, help="RTSP URL (e.g., rtsp://user:pass@host/...).")
    parser.add_argument(
        "--loop-video",
        action="store_true",
        help="If --video reaches EOF, restart from the beginning (useful to simulate a continuous stream).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle processing to approximate real-time based on timestamps (useful for stream simulation).",
    )
    parser.add_argument(
        "--reconnect",
        action="store_true",
        help="If capture read fails, attempt to reopen the source (useful for flaky RTSP).",
    )
    parser.add_argument(
        "--reconnect-wait-s",
        type=float,
        default=1.0,
        help="Seconds to wait before reconnect attempt (only if --reconnect).",
    )
    parser.add_argument(
        "--reconnect-max-tries",
        type=int,
        default=30,
        help="Max reconnect attempts (only if --reconnect).",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=0.0,
        help="Override capture FPS when OpenCV reports 0/wrong (0=auto).",
    )
    parser.add_argument(
        "--video-fps-out",
        type=float,
        default=0.0,
        help="Override saved video FPS (0=auto from source; falls back to analysis fps).",
    )

    parser.add_argument("--roi", default="configs/roi.json", help="ROI polygon JSON (from Scripts/calibrate_roi.py).")
    parser.add_argument(
        "--sop-profile",
        default=None,
        help="Optional SOP timing profile JSON (admin-tuned). CLI timing flags override values in this profile.",
    )
    parser.add_argument("--model", default="Models/yolov9-s_v2.onnx", help="Path to detector (.onnx/.engine/.pt).")
    parser.add_argument("--metadata", default="Models/metadata.yaml", help="Class metadata yaml (names mapping).")
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument(
        "--onnx-providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument(
        "--require-onnx-provider",
        action="append",
        default=[],
        help='Fail fast if the ONNX Runtime session is not using this provider (repeatable), e.g. "CUDAExecutionProvider".',
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help='Shortcut for --require-onnx-provider CUDAExecutionProvider and forcing --onnx-providers CUDAExecutionProvider.',
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS and only keep top-K detections by score.")

    parser.add_argument("--person-label", action="append", default=["person"], help="Class name for person (repeatable).")
    parser.add_argument("--helmet-label", action="append", default=["helmet"], help="Class name for helmet (repeatable).")
    parser.add_argument(
        "--skip-helmet",
        action="store_true",
        help="Disable helmet check (helmet status becomes UNKNOWN). Useful before you have a helmet-capable model.",
    )
    parser.add_argument(
        "--require-helmet-class",
        action="store_true",
        help="Fail fast if --helmet-label cannot be resolved from --metadata (instead of auto-disabling helmet).",
    )

    parser.add_argument("--analysis-fps", type=float, default=5.0, help="Target analysis FPS (used if source FPS is known).")
    parser.add_argument("--every", type=int, default=0, help="Process every Nth frame (overrides --analysis-fps if >0).")
    parser.add_argument(
        "--start-s",
        type=float,
        default=None,
        help="Session start after sustained person presence (seconds). Overrides --sop-profile if set.",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=None,
        help="Session end after sustained absence (seconds). Overrides --sop-profile if set.",
    )
    parser.add_argument(
        "--min-session-s",
        type=float,
        default=None,
        help="Discard sessions shorter than this (seconds). Overrides --sop-profile if set. 0 = keep all.",
    )
    parser.add_argument(
        "--roi-dwell-s",
        type=float,
        default=None,
        help="ROI dwell DONE after sustained presence (seconds). Overrides --sop-profile if set.",
    )
    parser.add_argument(
        "--roi-dwell-max-gap",
        type=float,
        default=0.4,
        help="Allow up to N seconds missing inside ROI dwell track (>=0).",
    )
    parser.add_argument(
        "--roi-dwell-iou",
        type=float,
        default=0.25,
        help="IoU threshold for matching person tracklets inside ROI (0.05..0.95).",
    )
    parser.add_argument(
        "--roi-dwell-miss",
        type=float,
        default=None,
        help="Max missed seconds before an ROI track is dropped (>=0). Defaults to at least --roi-dwell-max-gap.",
    )
    parser.add_argument(
        "--roi-min-person-height",
        type=int,
        default=0,
        help="If >0, small ROI persons are ignored for dwell tracking.",
    )
    parser.add_argument("--helmet-s", type=float, default=2.0, help="Helmet DONE after sustained association (seconds).")
    parser.add_argument(
        "--helmet-max-gap", type=int, default=1, help="Allow up to N missing frames inside helmet evidence streak (>=0)."
    )
    parser.add_argument(
        "--head-top-frac",
        type=float,
        default=0.35,
        help="Head region height fraction of the person box used to associate helmets (0.05..0.8).",
    )
    parser.add_argument("--min-person-height", type=int, default=0, help="If >0, short/small person sessions become UNKNOWN.")

    parser.add_argument("--roi-upscale", type=float, default=1.0, help="Optional ROI crop upscale factor (>=1.0).")
    parser.add_argument("--roi-expand", type=int, default=0, help="Expand ROI bounding box crop by N pixels (>=0).")
    parser.add_argument(
        "--detect-roi-only",
        action="store_true",
        help="Run detection only on ROI crop (faster, but ignores outside-ROI people).",
    )

    parser.add_argument("--out-dir", default="data", help="Root output directory for sessions/reports.")
    parser.add_argument("--save-video", action="store_true", help="Save per-session annotated MP4 video.")
    parser.add_argument(
        "--save-run-video",
        action="store_true",
        help="Save a full-run annotated MP4 under reports/<date>/run_annotated.mp4.",
    )
    parser.add_argument("--no-thumb", action="store_true", help="Disable per-session thumbnail image.")
    parser.add_argument("--no-evidence", action="store_true", help="Disable evidence clips around DONE events.")
    parser.add_argument("--evidence-pre-s", type=float, default=2.0, help="Seconds of evidence before DONE events.")
    parser.add_argument("--evidence-post-s", type=float, default=2.0, help="Seconds of evidence after DONE events.")
    parser.add_argument("--evidence-max-s", type=float, default=6.0, help="Max total length for each evidence clip.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show console progress (default: on for --video, off for rtsp/webcam).",
    )
    parser.add_argument("--progress-every-s", type=float, default=2.0, help="Progress update interval (seconds).")
    parser.add_argument("--progress-bar-width", type=int, default=30, help="Width of the progress bar.")
    parser.add_argument("--show", action="store_true", help="Show real-time window; press q/ESC to exit.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames (0=no limit).")

    return parser


def main() -> int:
    parser = build_parser()
    cli_dests = collect_cli_dests(parser, sys.argv[1:])
    args = parser.parse_args()

    config_payload: Optional[Dict[str, object]] = None
    config_path = Path(args.config) if args.config else None
    if config_path is not None:
        config_payload = load_run_config(config_path)
        apply_run_config(args=args, payload=config_payload, cli_dests=cli_dests, parser=parser)

    args_raw = dict(vars(args))
    return run_mvp(
        args,
        args_raw=args_raw,
        config_path=config_path,
        config_payload=config_payload,
    )


if __name__ == "__main__":
    raise SystemExit(main())

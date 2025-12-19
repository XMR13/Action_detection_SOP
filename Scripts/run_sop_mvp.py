from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from Action_Detection_SOP.ingest import get_capture_info, open_capture
from Action_Detection_SOP.reporting import today_date_str, write_daily_csv, write_daily_report, write_session_artifacts
from Action_Detection_SOP.roi import RoiPolygon, clamp_rect_to_frame, draw_roi, load_roi_json
from Action_Detection_SOP.sop_engine import HelmetRuleConfig, SessionResult, SessionizationConfig, SopEngine, SopEngineConfig
from yolo_kit import LetterboxConfig, YoloPostConfig, draw_detections, load_class_names, load_pipeline
from yolo_kit.types import Detection


def _name_to_ids(class_names: Dict[int, str], labels: Sequence[str]) -> List[int]:
    wanted = {s.strip().lower() for s in labels if s.strip()}
    ids: List[int] = []
    for cid, name in class_names.items():
        if str(name).strip().lower() in wanted:
            ids.append(int(cid))
    return ids


def _offset_detections(dets: Sequence[Detection], *, dx: float, dy: float, inv_scale: float) -> List[Detection]:
    out: List[Detection] = []
    for d in dets:
        out.append(
            Detection(
                x1=(d.x1 * inv_scale) + dx,
                y1=(d.y1 * inv_scale) + dy,
                x2=(d.x2 * inv_scale) + dx,
                y2=(d.y2 * inv_scale) + dy,
                score=d.score,
                class_id=d.class_id,
            )
        )
    return out


def _filter_by_roi(dets: Sequence[Detection], roi: RoiPolygon) -> List[Detection]:
    kept: List[Detection] = []
    for d in dets:
        cx = (d.x1 + d.x2) * 0.5
        cy = (d.y1 + d.y2) * 0.5
        if roi.contains_point(cx, cy):
            kept.append(d)
    return kept


def _split_classes(dets: Sequence[Detection], *, person_ids: Sequence[int], helmet_ids: Sequence[int]) -> Tuple[List[Detection], List[Detection]]:
    persons: List[Detection] = []
    helmets: List[Detection] = []
    person_set = set(int(x) for x in person_ids)
    helmet_set = set(int(x) for x in helmet_ids)
    for d in dets:
        if d.class_id is None:
            continue
        cid = int(d.class_id)
        if cid in person_set:
            persons.append(d)
        if cid in helmet_set:
            helmets.append(d)
    return persons, helmets


@dataclass(frozen=True)
class RunOutputs:
    date: str
    out_dir: Path
    session_dirs: Tuple[Path, ...]
    daily_report_json: Path
    daily_report_csv: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="MVP-A SOP runner (operator session in ROI + helmet compliance).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", default=None, help="Path to input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")
    src.add_argument("--rtsp", default=None, help="RTSP URL (e.g., rtsp://user:pass@host/...).")

    parser.add_argument("--roi", default="configs/roi.json", help="ROI polygon JSON (from Scripts/calibrate_roi.py).")
    parser.add_argument("--model", default="Models/yolov9-s_v2.onnx", help="Path to detector (.onnx/.engine/.pt).")
    parser.add_argument("--metadata", default="Models/metadata.yaml", help="Class metadata yaml (names mapping).")
    parser.add_argument("--backend", default=None, help="Force backend: onnxruntime / tensorrt / torchscript.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox input size (e.g., 640).")
    parser.add_argument(
        "--onnx-providers",
        default=None,
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider,CPUExecutionProvider".',
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS and only keep top-K detections by score.")

    parser.add_argument("--person-label", action="append", default=["person"], help="Class name for person (repeatable).")
    parser.add_argument("--helmet-label", action="append", default=["helmet"], help="Class name for helmet (repeatable).")

    parser.add_argument("--analysis-fps", type=float, default=5.0, help="Target analysis FPS (used if source FPS is known).")
    parser.add_argument("--every", type=int, default=0, help="Process every Nth frame (overrides --analysis-fps if >0).")
    parser.add_argument("--start-s", type=float, default=2.0, help="Session start after sustained person presence (seconds).")
    parser.add_argument("--end-s", type=float, default=3.0, help="Session end after sustained absence (seconds).")
    parser.add_argument("--helmet-s", type=float, default=2.0, help="Helmet DONE after sustained association (seconds).")
    parser.add_argument("--min-person-height", type=int, default=0, help="If >0, short/small person sessions become UNKNOWN.")

    parser.add_argument("--roi-upscale", type=float, default=1.0, help="Optional ROI crop upscale factor (>=1.0).")
    parser.add_argument("--roi-expand", type=int, default=0, help="Expand ROI bounding box crop by N pixels (>=0).")

    parser.add_argument("--out-dir", default="data", help="Root output directory for sessions/reports.")
    parser.add_argument("--save-video", action="store_true", help="Save per-session annotated MP4 video.")
    parser.add_argument("--show", action="store_true", help="Show real-time window; press q/ESC to exit.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames (0=no limit).")
    args = parser.parse_args()

    roi_path = Path(args.roi)
    roi = load_roi_json(roi_path)

    out_dir = Path(args.out_dir)
    date = today_date_str()

    class_names = load_class_names(args.metadata) if args.metadata else {}
    person_ids = _name_to_ids(class_names, args.person_label)
    helmet_ids = _name_to_ids(class_names, args.helmet_label)
    if not person_ids:
        raise ValueError(f"Could not resolve person class ids from labels: {args.person_label!r}")
    if not helmet_ids:
        raise ValueError(
            f"Could not resolve helmet class ids from labels: {args.helmet_label!r}. "
            "Provide a metadata.yaml that includes a helmet class."
        )

    class_ids = sorted(set(person_ids + helmet_ids))

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    onnx_providers = None
    if args.onnx_providers:
        onnx_providers = [p.strip() for p in str(args.onnx_providers).split(",") if p.strip()]

    pipeline = load_pipeline(
        model_path=args.model,
        backend=args.backend,
        post_cfg=YoloPostConfig(
            conf_threshold=float(args.conf),
            iou_threshold=float(args.iou),
            apply_nms=not bool(args.no_nms),
            class_ids=class_ids,
        ),
        letterbox_cfg=LetterboxConfig(new_shape=(int(args.imgsz), int(args.imgsz))),
        onnx_providers=onnx_providers,
    )

    cap = open_capture(video=args.video, webcam=args.webcam, rtsp=args.rtsp)
    info = get_capture_info(cap)

    if args.roi_upscale < 1.0:
        raise ValueError("--roi-upscale must be >= 1.0")
    if args.roi_expand < 0:
        raise ValueError("--roi-expand must be >= 0")
    if args.start_s <= 0 or args.end_s <= 0 or args.helmet_s <= 0:
        raise ValueError("--start-s/--end-s/--helmet-s must be > 0")

    if args.every and args.every > 0:
        every = int(args.every)
        analysis_fps = info.fps / every if info.fps else float(args.analysis_fps)
    else:
        if info.fps and info.fps > 0:
            every = max(1, int(round(info.fps / float(args.analysis_fps))))
            analysis_fps = info.fps / every
        else:
            every = 1
            analysis_fps = float(args.analysis_fps)

    engine_cfg = SopEngineConfig(
        session=SessionizationConfig(start_seconds=float(args.start_s), end_seconds=float(args.end_s), analysis_fps=analysis_fps),
        helmet=HelmetRuleConfig(
            required_seconds=float(args.helmet_s),
            analysis_fps=analysis_fps,
            min_person_height_px=int(args.min_person_height),
        ),
    )
    engine = SopEngine(engine_cfg)

    frame_idx = 0
    processed = 0
    sessions: List[SessionResult] = []
    session_dirs: List[Path] = []

    writer: Optional[cv2.VideoWriter] = None
    active_session_dir: Optional[Path] = None

    win = "SOP MVP-A"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            if (frame_idx - 1) % every != 0:
                continue

            processed += 1
            if args.max_frames and processed >= int(args.max_frames):
                break

            # Timestamp
            t_s = (frame_idx / info.fps) if info.fps else (processed / analysis_fps)

            # Crop ROI bounding rect (for performance) then map detections back.
            x0, y0, x1, y1 = roi.bounding_rect(expand_px=int(args.roi_expand))
            x0, y0, x1, y1 = clamp_rect_to_frame((x0, y0, x1, y1), frame_width=frame.shape[1], frame_height=frame.shape[0])
            crop = frame[y0:y1, x0:x1]

            inv_scale = 1.0
            if args.roi_upscale != 1.0:
                s = float(args.roi_upscale)
                crop = cv2.resize(crop, (int(round(crop.shape[1] * s)), int(round(crop.shape[0] * s))), interpolation=cv2.INTER_LINEAR)
                inv_scale = 1.0 / s

            dets_local = pipeline(crop)
            dets_global = _offset_detections(dets_local, dx=float(x0), dy=float(y0), inv_scale=inv_scale)
            dets_roi = _filter_by_roi(dets_global, roi)
            persons, helmets = _split_classes(dets_roi, person_ids=person_ids, helmet_ids=helmet_ids)

            result = engine.update(time_s=float(t_s), frame_idx=processed, persons=persons, helmets=helmets)

            # Start per-session video writer lazily once we know the session dir.
            if args.save_video and engine.active_session_id and active_session_dir is None:
                active_session_dir = out_dir / "sessions" / date / f"session_{engine.active_session_id}"
                active_session_dir.mkdir(parents=True, exist_ok=True)
                active_video_path = active_session_dir / "annotated.mp4"
                fps_out = float(analysis_fps) if analysis_fps else 5.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(active_video_path), fourcc, fps_out, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {active_video_path}")

            # Visualization (optional)
            vis = frame
            if args.show or (args.save_video and writer is not None):
                vis = frame.copy()
                vis = draw_roi(vis, roi)
                vis = draw_detections(vis, dets_roi, class_names=class_names, show_score=True)

                sid = engine.active_session_id or "-"
                helmet_status = "-"
                if engine.active_session_id is not None:
                    # best-effort live status
                    helmet_status = "DONE" if any(h.class_id in helmet_ids for h in helmets) else "..."
                cv2.putText(
                    vis,
                    f"session={sid} helmet={helmet_status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            if writer is not None and vis is not None:
                writer.write(vis)

            if args.show:
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if result is not None:
                sessions.append(result)
                session_dir = write_session_artifacts(out_dir=out_dir, date=date, session=result)
                session_dirs.append(session_dir)

                # Close writer for this session
                if writer is not None:
                    writer.release()
                    writer = None
                active_session_dir = None

        # End-of-stream flush
        end_time_s = (frame_idx / info.fps) if info.fps else (processed / analysis_fps)
        tail = engine.flush(time_s=float(end_time_s))
        if tail is not None:
            sessions.append(tail)
            session_dir = write_session_artifacts(out_dir=out_dir, date=date, session=tail)
            session_dirs.append(session_dir)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    daily_json = write_daily_report(out_dir=out_dir, date=date, sessions=sessions)
    daily_csv = write_daily_csv(out_dir=out_dir, date=date, sessions=sessions)

    outputs = RunOutputs(
        date=date,
        out_dir=out_dir,
        session_dirs=tuple(session_dirs),
        daily_report_json=daily_json,
        daily_report_csv=daily_csv,
    )
    print(f"Wrote daily report: {outputs.daily_report_json}")
    print(f"Wrote sessions CSV: {outputs.daily_report_csv}")
    print(f"Sessions: {len(outputs.session_dirs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

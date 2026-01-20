from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    tqdm = None  # type: ignore[assignment]

from Action_Detection_SOP.config import load_sop_profile
from Action_Detection_SOP.evidence import EvidenceClipConfig, EvidenceClipper
from Action_Detection_SOP.evidence_io import write_evidence_clip, write_evidence_manifest
from Action_Detection_SOP.ingest import get_capture_info, open_capture
from Action_Detection_SOP.reporting import (
    today_date_str,
    write_daily_csv,
    write_daily_report,
    write_run_config,
    write_session_artifacts,
    write_session_run_config,
)
from Action_Detection_SOP.roi import RoiPolygon, clamp_rect_to_frame, draw_roi, load_roi_json, resolve_roi_for_frame
from Action_Detection_SOP.sop_engine import (
    HelmetRuleConfig,
    RoiDwellRuleConfig,
    SessionResult,
    SessionizationConfig,
    SopEngine,
    SopEngineConfig,
    helmet_associated_with_person,
)
from yolo_kit import LetterboxConfig, YoloPostConfig, draw_detections, load_class_names, load_pipeline
from yolo_kit.types import Detection

DEFAULT_SESSION_START_S = 2.0
DEFAULT_SESSION_END_S = 3.0
DEFAULT_ROI_DWELL_S = 8.0


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


def _split_classes(
    dets: Sequence[Detection], *, person_ids: Sequence[int], helmet_ids: Sequence[int]
) -> Tuple[List[Detection], List[Detection]]:
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


def _sha256_path(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_metadata(path: Path) -> Dict[str, object]:
    payload: Dict[str, object] = {"path": str(path)}
    if not path.exists():
        payload["exists"] = False
        return payload
    st = path.stat()
    payload.update(
        {
            "exists": True,
            "size_bytes": int(st.st_size),
            "mtime": float(st.st_mtime),
            "sha256": _sha256_path(path),
        }
    )
    return payload


def _sanitize_ort_provider_name(name: str) -> str:
    # PowerShell line continuations and copy/paste can leave stray backticks/quotes.
    return str(name).strip().strip("'\"`")


def _parse_ort_providers(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts: List[str] = []
    for p in str(raw).split(","):
        cleaned = _sanitize_ort_provider_name(p)
        if cleaned:
            parts.append(cleaned)
    return parts or None


def _overlay_style(frame_height: int) -> Tuple[float, int]:
    base_h = 720.0
    scale = 0.8 * (float(frame_height) / base_h)
    scale = max(0.5, min(1.6, scale))
    thickness = max(1, int(round(scale * 2)))
    return scale, thickness


def _format_progress(
    *,
    frame_idx: int,
    processed: int,
    total_frames: Optional[int],
    elapsed_s: float,
    bar_width: int,
) -> str:
    read_fps = (frame_idx / elapsed_s) if elapsed_s > 0 else 0.0
    proc_fps = (processed / elapsed_s) if elapsed_s > 0 else 0.0
    if total_frames and total_frames > 0:
        ratio = min(1.0, frame_idx / float(total_frames))
        filled = int(round(ratio * bar_width))
        bar = "#" * filled + "-" * max(0, bar_width - filled)
        pct = ratio * 100.0
        return f"[{bar}] {pct:5.1f}% frames={frame_idx}/{total_frames} read_fps={read_fps:5.1f} proc_fps={proc_fps:5.1f}"
    return f"frames={frame_idx} read_fps={read_fps:5.1f} proc_fps={proc_fps:5.1f}"


@dataclass(frozen=True)
class RunOutputs:
    date: str
    out_dir: Path
    session_dirs: Tuple[Path, ...]
    daily_report_json: Path
    daily_report_csv: Path


def run_mvp(
    args: argparse.Namespace,
    *,
    args_raw: Dict[str, object],
    config_path: Optional[Path],
    config_payload: Optional[Dict[str, object]],
) -> int:
    def _session_duration_s(session: SessionResult) -> float:
        return max(0.0, float(session.end_time_s) - float(session.start_time_s))

    source_count = int(args.video is not None) + int(args.webcam is not None) + int(args.rtsp is not None)
    if source_count != 1:
        raise ValueError("Exactly one source must be set: --video or --webcam or --rtsp (or via --config).")

    sop_profile = None
    sop_profile_path = Path(args.sop_profile) if args.sop_profile else None
    if sop_profile_path is not None:
        sop_profile = load_sop_profile(sop_profile_path)

    def _resolve_seconds(cli_value: Optional[float], profile_value: Optional[float], default_value: float) -> float:
        if cli_value is not None:
            return float(cli_value)
        if profile_value is not None:
            return float(profile_value)
        return float(default_value)

    start_s = _resolve_seconds(
        args.start_s,
        sop_profile.session_start_seconds if sop_profile else None,
        DEFAULT_SESSION_START_S,
    )
    end_s = _resolve_seconds(
        args.end_s,
        sop_profile.session_end_seconds if sop_profile else None,
        DEFAULT_SESSION_END_S,
    )
    min_session_s = _resolve_seconds(
        args.min_session_s,
        sop_profile.min_session_seconds if sop_profile else None,
        0.0,
    )
    roi_dwell_s = _resolve_seconds(
        args.roi_dwell_s,
        sop_profile.roi_dwell_seconds if sop_profile else None,
        DEFAULT_ROI_DWELL_S,
    )
    args.start_s = start_s
    args.end_s = end_s
    args.min_session_s = min_session_s
    args.roi_dwell_s = roi_dwell_s

    roi_path = Path(args.roi)
    roi_base = load_roi_json(roi_path)
    if roi_base.frame_size is None:
        print(
            "Note: ROI JSON has no frame_size; auto-rescale is disabled. "
            "Re-save ROI using Scripts/calibrate_roi.py to embed calibration resolution."
        )

    out_dir = Path(args.out_dir)
    date = today_date_str()

    class_names = load_class_names(args.metadata) if args.metadata else {}
    person_ids = _name_to_ids(class_names, args.person_label)
    helmet_disabled = bool(args.skip_helmet)
    helmet_ids = [] if helmet_disabled else _name_to_ids(class_names, args.helmet_label)
    if not person_ids:
        raise ValueError(f"Could not resolve person class ids from labels: {args.person_label!r}")
    if not helmet_ids and not helmet_disabled:
        if args.require_helmet_class:
            raise ValueError(
                f"Could not resolve helmet class ids from labels: {args.helmet_label!r}. "
                "Provide a metadata.yaml that includes a helmet class (or pass --skip-helmet)."
            )
        print(
            f"WARNING: Could not resolve helmet class ids from labels: {args.helmet_label!r}. "
            "Helmet check will be disabled (helmet=UNKNOWN)."
        )
        helmet_disabled = True

    class_ids = sorted(set(person_ids + helmet_ids))

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    if args.source_fps < 0:
        raise ValueError("--source-fps must be >= 0")
    if args.video_fps_out < 0:
        raise ValueError("--video-fps-out must be >= 0")
    onnx_providers = None
    require_onnx_providers: List[str] = [
        _sanitize_ort_provider_name(p) for p in (args.require_onnx_provider or []) if _sanitize_ort_provider_name(p)
    ]
    if args.require_cuda:
        require_onnx_providers.append("CUDAExecutionProvider")
        onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        onnx_providers = _parse_ort_providers(args.onnx_providers)

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
    if pipeline.backend_name == "onnxruntime":
        ort_backend = pipeline.backend
        providers_in_use = None
        available_providers = None
        if ort_backend is not None and hasattr(ort_backend, "providers_in_use"):
            providers_in_use = list(getattr(ort_backend, "providers_in_use"))
        if ort_backend is not None and hasattr(ort_backend, "available_providers"):
            available_providers = list(getattr(ort_backend, "available_providers"))

        if providers_in_use is not None:
            print(f"ONNX Runtime session providers: {providers_in_use}")

        if require_onnx_providers:
            missing = []
            for rp in require_onnx_providers:
                if providers_in_use is None or rp not in providers_in_use:
                    missing.append(rp)
            if missing:
                msg = f"Required ONNX Runtime provider(s) not active: {missing}."
                if available_providers is not None:
                    msg += f" Available providers: {available_providers}."
                msg += ' Hint: pass --onnx-providers "CUDAExecutionProvider" (or use --require-cuda).'
                raise RuntimeError(msg)

    cap = open_capture(video=args.video, webcam=args.webcam, rtsp=args.rtsp)
    info = get_capture_info(cap)
    if args.loop_video and not args.video:
        raise ValueError("--loop-video is only valid with --video.")
    if args.reconnect_wait_s < 0:
        raise ValueError("--reconnect-wait-s must be >= 0")
    if args.reconnect_max_tries < 0:
        raise ValueError("--reconnect-max-tries must be >= 0")

    if args.roi_upscale < 1.0:
        raise ValueError("--roi-upscale must be >= 1.0")
    if args.roi_expand < 0:
        raise ValueError("--roi-expand must be >= 0")
    if args.roi_dwell_s <= 0:
        raise ValueError("--roi-dwell-s must be > 0")
    if args.roi_dwell_max_gap < 0:
        raise ValueError("--roi-dwell-max-gap must be >= 0 seconds")
    if not (0.05 <= args.roi_dwell_iou <= 0.95):
        raise ValueError("--roi-dwell-iou must be within [0.05, 0.95]")
    if args.roi_dwell_miss is None:
        args.roi_dwell_miss = float(args.roi_dwell_max_gap)
    if args.roi_dwell_miss < 0:
        raise ValueError("--roi-dwell-miss must be >= 0 seconds")
    if args.roi_dwell_miss + 1e-9 < args.roi_dwell_max_gap:
        raise ValueError("--roi-dwell-miss must be >= --roi-dwell-max-gap (seconds)")
    if args.roi_min_person_height < 0:
        raise ValueError("--roi-min-person-height must be >= 0")
    if args.start_s <= 0 or args.end_s <= 0:
        raise ValueError("--start-s/--end-s must be > 0")
    if args.min_session_s < 0:
        raise ValueError("--min-session-s must be >= 0")
    if not helmet_disabled:
        if args.helmet_s <= 0:
            raise ValueError("--helmet-s must be > 0")
        if args.helmet_max_gap < 0:
            raise ValueError("--helmet-max-gap must be >= 0")
    if not args.no_evidence:
        if args.evidence_pre_s < 0:
            raise ValueError("--evidence-pre-s must be >= 0")
        if args.evidence_post_s < 0:
            raise ValueError("--evidence-post-s must be >= 0")
        if args.evidence_max_s <= 0:
            raise ValueError("--evidence-max-s must be > 0")

    source_fps = float(args.source_fps) if args.source_fps and args.source_fps > 0 else info.fps
    if source_fps is not None and source_fps <= 0:
        source_fps = None

    if args.every and args.every > 0:
        every = int(args.every)
        analysis_fps = source_fps / every if source_fps else float(args.analysis_fps)
    else:
        if source_fps and source_fps > 0:
            every = max(1, int(round(source_fps / float(args.analysis_fps))))
            analysis_fps = source_fps / every
        else:
            every = 1
            analysis_fps = float(args.analysis_fps)

    evidence_enabled = not bool(args.no_evidence)
    evidence_cfg: Optional[EvidenceClipConfig] = None
    evidence_clipper: Optional[EvidenceClipper] = None
    if evidence_enabled:
        evidence_cfg = EvidenceClipConfig(
            pre_seconds=float(args.evidence_pre_s),
            post_seconds=float(args.evidence_post_s),
            max_seconds=float(args.evidence_max_s),
            analysis_fps=float(analysis_fps),
        )
        evidence_clipper = EvidenceClipper(evidence_cfg)

    helmet_cfg = None
    if not helmet_disabled:
        helmet_cfg = HelmetRuleConfig(
            required_seconds=float(args.helmet_s),
            analysis_fps=analysis_fps,
            head_top_fraction=float(args.head_top_frac),
            min_person_height_px=int(args.min_person_height),
            max_gap_frames=int(args.helmet_max_gap),
        )
    roi_gap_frames = max(0, int(round(float(args.roi_dwell_max_gap) * analysis_fps)))
    roi_miss_frames = max(0, int(round(float(args.roi_dwell_miss) * analysis_fps)))
    if roi_miss_frames < roi_gap_frames:
        roi_miss_frames = roi_gap_frames
    roi_dwell_cfg = RoiDwellRuleConfig(
        required_seconds=float(args.roi_dwell_s),
        analysis_fps=analysis_fps,
        max_gap_frames=roi_gap_frames,
        max_track_missed=roi_miss_frames,
        iou_match_threshold=float(args.roi_dwell_iou),
        min_person_height_px=int(args.roi_min_person_height),
    )
    engine_cfg = SopEngineConfig(
        session=SessionizationConfig(start_seconds=float(args.start_s), end_seconds=float(args.end_s), analysis_fps=analysis_fps),
        helmet=helmet_cfg,
        roi_dwell=roi_dwell_cfg,
    )
    engine = SopEngine(engine_cfg)

    frame_idx = 0
    processed = 0
    sessions: List[SessionResult] = []
    session_dirs: List[Path] = []
    roi_for_frame: Optional[RoiPolygon] = None
    save_thumb = not bool(args.no_thumb)

    run_config: Dict[str, object] = {
        "date": date,
        "args": args_raw,
        "source": {
            "video": args.video,
            "webcam": args.webcam,
            "rtsp": args.rtsp,
        },
        "source_fps_raw": float(info.fps) if info.fps else None,
        "source_fps": float(source_fps) if source_fps else None,
        "analysis_fps": float(analysis_fps),
        "every": int(every),
        "sessionization": {
            "start_seconds": float(args.start_s),
            "end_seconds": float(args.end_s),
            "min_session_seconds": float(args.min_session_s),
        },
        "roi": {
            "path": str(roi_path),
            "frame_size": roi_base.frame_size,
            "points": list(roi_base.points),
            "sha256": _sha256_path(roi_path),
        },
        "roi_dwell": {
            "required_seconds": float(args.roi_dwell_s),
            "max_gap_seconds": float(args.roi_dwell_max_gap),
            "max_gap_frames": int(roi_gap_frames),
            "max_track_missed_seconds": float(args.roi_dwell_miss),
            "max_track_missed_frames": int(roi_miss_frames),
            "iou_match_threshold": float(args.roi_dwell_iou),
            "min_person_height_px": int(args.roi_min_person_height),
        },
        "evidence": {
            "enabled": bool(evidence_enabled),
            "pre_seconds": float(args.evidence_pre_s),
            "post_seconds": float(args.evidence_post_s),
            "max_seconds": float(args.evidence_max_s),
            "analysis_fps": float(analysis_fps),
            "events": ["roi_dwell_done", "helmet_done"],
        },
        "model": _file_metadata(Path(args.model)),
        "metadata": _file_metadata(Path(args.metadata)) if args.metadata else {"path": None},
        "postprocess": {
            "conf": float(args.conf),
            "iou": float(args.iou),
            "no_nms": bool(args.no_nms),
        },
        "detect_roi_only": bool(args.detect_roi_only),
        "video_fps_out": float(args.video_fps_out) if args.video_fps_out else None,
    }
    if config_path is None:
        run_config["config"] = {"path": None}
    else:
        run_config["config"] = {
            "path": str(config_path),
            "data": config_payload,
            "file": _file_metadata(config_path),
        }

    if evidence_cfg is not None:
        pre_s, post_s = evidence_cfg.resolved_window()
        run_config["evidence"]["resolved_pre_seconds"] = float(pre_s)
        run_config["evidence"]["resolved_post_seconds"] = float(post_s)

    if sop_profile_path is None:
        run_config["sop_profile"] = {"path": None}
    else:
        run_config["sop_profile"] = {
            "path": str(sop_profile_path),
            "data": asdict(sop_profile) if sop_profile is not None else None,
            "file": _file_metadata(sop_profile_path),
        }

    warnings: List[str] = []
    discarded_sessions: List[Dict[str, object]] = []

    start_wall: Optional[float] = None
    start_t_s: Optional[float] = None
    run_start_dt: Optional[datetime] = None
    reconnect_tries = 0
    reconnect_events = 0
    loop_count = 0
    thumb_written: set[str] = set()
    evidence_session_id: Optional[str] = None
    evidence_clip_counts: Dict[str, int] = {}
    evidence_clips: List[Dict[str, object]] = []

    run_config["stream_sim"] = {
        "loop_video": bool(args.loop_video),
        "realtime": bool(args.realtime),
        "reconnect": bool(args.reconnect),
        "reconnect_wait_s": float(args.reconnect_wait_s),
        "reconnect_max_tries": int(args.reconnect_max_tries),
    }

    writer: Optional[cv2.VideoWriter] = None
    run_writer: Optional[cv2.VideoWriter] = None
    run_video_path: Optional[Path] = None
    active_session_dir: Optional[Path] = None
    last_dets_global: List[Detection] = []
    last_dets_roi: List[Detection] = []
    last_persons_all: List[Detection] = []
    last_helmets_all: List[Detection] = []

    win = "SOP MVP-A"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    total_frames = info.frame_count if info.frame_count and info.frame_count > 0 else None
    progress_enabled = bool(args.progress) if args.progress is not None else bool(args.video)
    progress_every_s = float(args.progress_every_s)
    progress_bar_width = int(args.progress_bar_width)
    if progress_bar_width < 5:
        progress_bar_width = 5
    progress_start = time.monotonic()
    last_progress = progress_start
    printed_progress = False
    pbar = None
    last_pbar_frame = 0
    if progress_enabled and tqdm is not None:
        if total_frames is not None:
            pbar = tqdm(total=total_frames, unit="frame", desc="sop", mininterval=progress_every_s)
        else:
            pbar = tqdm(unit="frame", desc="sop", mininterval=progress_every_s)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video and args.loop_video:
                    cap.release()
                    cap = open_capture(video=args.video)
                    info = get_capture_info(cap)
                    loop_count += 1
                    reconnect_tries = 0
                    continue

                if args.reconnect:
                    reconnect_tries += 1
                    if args.reconnect_max_tries and reconnect_tries > int(args.reconnect_max_tries):
                        raise RuntimeError(f"Reconnect failed after {args.reconnect_max_tries} tries.")
                    cap.release()
                    if args.reconnect_wait_s:
                        time.sleep(float(args.reconnect_wait_s))
                    cap = open_capture(video=args.video, webcam=args.webcam, rtsp=args.rtsp)
                    info = get_capture_info(cap)
                    reconnect_events += 1
                    continue

                break
            reconnect_tries = 0

            frame_idx += 1
            should_process = (frame_idx - 1) % every == 0

            if should_process:
                processed += 1
                if args.max_frames and processed >= int(args.max_frames):
                    break

            if pbar is not None:
                delta = frame_idx - last_pbar_frame
                if delta:
                    pbar.update(delta)
                    last_pbar_frame = frame_idx
            elif progress_enabled and progress_every_s > 0:
                now = time.monotonic()
                if now - last_progress >= progress_every_s:
                    elapsed = now - progress_start
                    msg = _format_progress(
                        frame_idx=frame_idx,
                        processed=processed,
                        total_frames=total_frames,
                        elapsed_s=elapsed,
                        bar_width=progress_bar_width,
                    )
                    sys.stdout.write("\r" + msg)
                    sys.stdout.flush()
                    printed_progress = True
                    last_progress = now

            # Timestamp
            t_s = (frame_idx / source_fps) if source_fps else (processed / analysis_fps)
            if args.realtime:
                if start_wall is None:
                    start_wall = time.monotonic()
                    start_t_s = float(t_s)
                    run_start_dt = datetime.now()
                assert start_t_s is not None
                target_wall = start_wall + (float(t_s) - start_t_s)
                now_wall = time.monotonic()
                if target_wall > now_wall:
                    time.sleep(target_wall - now_wall)
            else:
                if run_start_dt is None:
                    run_start_dt = datetime.now()

            if run_start_dt is not None and "run_start_iso" not in run_config:
                run_config["run_start_iso"] = run_start_dt.isoformat(timespec="seconds")

            if roi_for_frame is None:
                roi_for_frame = resolve_roi_for_frame(
                    roi_base,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
                if roi_base.frame_size is not None:
                    base_w, base_h = roi_base.frame_size
                    stream_size = (int(frame.shape[1]), int(frame.shape[0]))
                    if (base_w, base_h) != stream_size:
                        msg = (
                            "ROI frame_size differs from stream resolution; ROI was auto-rescaled. "
                            f"roi_frame_size={(base_w, base_h)} stream_size={stream_size}"
                        )
                        print(f"WARNING: {msg}")
                        warnings.append(msg)
                run_config["roi_resolved"] = {
                    "frame_size": (int(frame.shape[1]), int(frame.shape[0])),
                    "points": list(roi_for_frame.points),
                }
                if warnings:
                    run_config["warnings"] = list(warnings)
                run_config["loop_count"] = int(loop_count)
                run_config["reconnect_events"] = int(reconnect_events)

            result = None
            if should_process:
                if args.detect_roi_only:
                    # Crop ROI bounding rect (for performance) then map detections back.
                    x0, y0, x1, y1 = roi_for_frame.bounding_rect(expand_px=int(args.roi_expand))
                    x0, y0, x1, y1 = clamp_rect_to_frame(
                        (x0, y0, x1, y1), frame_width=frame.shape[1], frame_height=frame.shape[0]
                    )
                    crop = frame[y0:y1, x0:x1]

                    inv_scale = 1.0
                    if args.roi_upscale != 1.0:
                        s = float(args.roi_upscale)
                        crop = cv2.resize(
                            crop,
                            (int(round(crop.shape[1] * s)), int(round(crop.shape[0] * s))),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        inv_scale = 1.0 / s

                    dets_local = pipeline(crop)
                    dets_global = _offset_detections(dets_local, dx=float(x0), dy=float(y0), inv_scale=inv_scale)
                else:
                    dets_global = pipeline(frame)

                dets_roi = _filter_by_roi(dets_global, roi_for_frame)
                persons_all, helmets_all = _split_classes(dets_global, person_ids=person_ids, helmet_ids=helmet_ids)
                persons_roi = _filter_by_roi(persons_all, roi_for_frame)

                result = engine.update(
                    time_s=float(t_s),
                    frame_idx=processed,
                    persons_in_roi=persons_roi,
                    persons_all=persons_all,
                    helmets_all=helmets_all,
                )
                last_dets_global = list(dets_global)
                last_dets_roi = list(dets_roi)
                last_persons_all = list(persons_all)
                last_helmets_all = list(helmets_all)

            events = engine.pop_events() if should_process else ()
            session_id = engine.active_session_id
            need_thumb = bool(should_process and save_thumb and session_id and session_id not in thumb_written)

            if evidence_enabled and evidence_clipper is not None:
                if evidence_session_id is None and session_id is not None:
                    evidence_session_id = session_id
                    evidence_clipper.reset()
                    evidence_clip_counts = {}
                    evidence_clips = []

            # Start per-session video writer lazily once we know the session dir.
            if (args.save_video or need_thumb) and session_id and active_session_dir is None:
                active_session_dir = out_dir / "sessions" / date / f"session_{session_id}"
                active_session_dir.mkdir(parents=True, exist_ok=True)
                if args.save_video:
                    active_video_path = active_session_dir / "annotated.mp4"
                    if args.video_fps_out and args.video_fps_out > 0:
                        fps_out = float(args.video_fps_out)
                    else:
                        fps_out = float(source_fps) if source_fps else (float(analysis_fps) if analysis_fps else 5.0)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(active_video_path), fourcc, fps_out, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {active_video_path}")

            # Visualization (optional)
            vis = frame
            need_vis = bool(
                args.show
                or args.save_run_video
                or (args.save_video and writer is not None)
                or need_thumb
                or (evidence_enabled and evidence_session_id is not None and should_process)
            )
            if need_vis:
                vis = frame.copy()
                vis = draw_roi(vis, roi_for_frame)
                dets_vis = last_dets_roi if args.detect_roi_only else last_dets_global
                vis = draw_detections(vis, dets_vis, class_names=class_names, show_score=True)

                sid = engine.active_session_id or "-"
                helmet_status = "-"
                roi_status = "-"
                if engine.active_session_id is not None:
                    if engine.cfg.roi_dwell is not None:
                        required_frames = engine.cfg.roi_dwell.required_frames
                        dwell_frames = engine.active_roi_dwell_frames
                        if required_frames > 0 and dwell_frames >= required_frames:
                            roi_status = "OK"
                        else:
                            dwell_s = dwell_frames / analysis_fps if analysis_fps else float(dwell_frames)
                            req_s = required_frames / analysis_fps if analysis_fps else float(required_frames)
                            roi_status = f"{dwell_s:.1f}/{req_s:.1f}s"
                    if helmet_disabled or engine.cfg.helmet is None:
                        helmet_status = "UNKNOWN"
                    else:
                        helmet_status = (
                            "OK"
                            if helmet_associated_with_person(
                                last_persons_all, last_helmets_all, head_top_fraction=engine.cfg.helmet.head_top_fraction
                            )
                            else "..."
                        )
                font_scale, thickness = _overlay_style(int(vis.shape[0]))
                cv2.putText(
                    vis,
                    f"session={sid} roi={roi_status} helmet={helmet_status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                )

            if args.save_run_video and run_writer is None:
                report_dir = out_dir / "reports" / date
                report_dir.mkdir(parents=True, exist_ok=True)
                run_video_path = report_dir / "run_annotated.mp4"
                if args.video_fps_out and args.video_fps_out > 0:
                    fps_out = float(args.video_fps_out)
                else:
                    fps_out = float(source_fps) if source_fps else (float(analysis_fps) if analysis_fps else 5.0)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                run_writer = cv2.VideoWriter(str(run_video_path), fourcc, fps_out, (w, h))
                if not run_writer.isOpened():
                    raise RuntimeError(f"Failed to open run video writer: {run_video_path}")

            if need_thumb and session_id and active_session_dir is not None:
                thumb_path = active_session_dir / "thumbnail.jpg"
                ok = cv2.imwrite(str(thumb_path), vis)
                if not ok:
                    raise RuntimeError(f"Failed to write thumbnail: {thumb_path}")
                thumb_written.add(session_id)

            if evidence_enabled and evidence_clipper is not None and evidence_session_id is not None and should_process:
                completed = evidence_clipper.add_frame(time_s=float(t_s), frame=vis)
                if completed:
                    session_dir = out_dir / "sessions" / date / f"session_{evidence_session_id}"
                    session_dir.mkdir(parents=True, exist_ok=True)
                    for clip_data in completed:
                        idx = evidence_clip_counts.get(clip_data.clip.name, 0) + 1
                        evidence_clip_counts[clip_data.clip.name] = idx
                        meta = write_evidence_clip(
                            clip_data=clip_data,
                            session_dir=session_dir,
                            index=idx,
                        )
                        if meta is not None:
                            evidence_clips.append(meta)
                for ev in events:
                    if ev.session_id != evidence_session_id:
                        continue
                    evidence_clipper.trigger(
                        name=ev.name,
                        time_s=float(ev.time_s),
                        frame_idx=int(ev.frame_idx),
                    )

            if run_writer is not None and vis is not None:
                run_writer.write(vis)

            if writer is not None and vis is not None:
                writer.write(vis)

            if args.show:
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if result is not None:
                if run_start_dt is not None:
                    start_dt = run_start_dt + timedelta(seconds=float(result.start_time_s))
                    end_dt = run_start_dt + timedelta(seconds=float(result.end_time_s))
                    result = replace(
                        result,
                        start_time_iso=start_dt.isoformat(timespec="seconds"),
                        end_time_iso=end_dt.isoformat(timespec="seconds"),
                        start_date=start_dt.date().isoformat(),
                        end_date=end_dt.date().isoformat(),
                    )
                duration_s = _session_duration_s(result)
                if args.min_session_s > 0 and duration_s < float(args.min_session_s):
                    discarded_sessions.append(
                        {
                            "session_id": result.session_id,
                            "duration_s": duration_s,
                            "reason": "min_session_seconds",
                        }
                    )
                    if writer is not None:
                        writer.release()
                        writer = None
                    discard_dir = out_dir / "sessions" / date / f"session_{result.session_id}"
                    if discard_dir.exists():
                        shutil.rmtree(discard_dir, ignore_errors=True)
                    active_session_dir = None
                    thumb_written.discard(result.session_id)
                    if evidence_enabled and evidence_clipper is not None:
                        evidence_clipper.reset()
                        evidence_session_id = None
                        evidence_clip_counts = {}
                        evidence_clips = []
                    continue
                sessions.append(result)
                session_dir = write_session_artifacts(out_dir=out_dir, date=date, session=result)
                run_config["loop_count"] = int(loop_count)
                run_config["reconnect_events"] = int(reconnect_events)
                write_session_run_config(session_dir=session_dir, run_config=run_config)
                if evidence_enabled and evidence_clipper is not None and evidence_session_id == result.session_id:
                    pending = evidence_clipper.flush()
                    for clip_data in pending:
                        idx = evidence_clip_counts.get(clip_data.clip.name, 0) + 1
                        evidence_clip_counts[clip_data.clip.name] = idx
                        meta = write_evidence_clip(
                            clip_data=clip_data,
                            session_dir=session_dir,
                            index=idx,
                        )
                        if meta is not None:
                            evidence_clips.append(meta)
                    write_evidence_manifest(session_dir=session_dir, clips=evidence_clips)
                    evidence_clipper.reset()
                    evidence_session_id = None
                    evidence_clip_counts = {}
                    evidence_clips = []
                session_dirs.append(session_dir)

                # Close writer for this session
                if writer is not None:
                    writer.release()
                    writer = None
                active_session_dir = None

        # End-of-stream flush
        end_time_s = (frame_idx / source_fps) if source_fps else (processed / analysis_fps)
        tail = engine.flush(time_s=float(end_time_s))
        if tail is not None:
            if run_start_dt is not None:
                start_dt = run_start_dt + timedelta(seconds=float(tail.start_time_s))
                end_dt = run_start_dt + timedelta(seconds=float(tail.end_time_s))
                tail = replace(
                    tail,
                    start_time_iso=start_dt.isoformat(timespec="seconds"),
                    end_time_iso=end_dt.isoformat(timespec="seconds"),
                    start_date=start_dt.date().isoformat(),
                    end_date=end_dt.date().isoformat(),
                )
            duration_s = _session_duration_s(tail)
            if args.min_session_s > 0 and duration_s < float(args.min_session_s):
                discarded_sessions.append(
                    {
                        "session_id": tail.session_id,
                        "duration_s": duration_s,
                        "reason": "min_session_seconds",
                    }
                )
                discard_dir = out_dir / "sessions" / date / f"session_{tail.session_id}"
                if discard_dir.exists():
                    shutil.rmtree(discard_dir, ignore_errors=True)
                if evidence_enabled and evidence_clipper is not None:
                    evidence_clipper.reset()
                    evidence_session_id = None
                    evidence_clip_counts = {}
                    evidence_clips = []
            else:
                sessions.append(tail)
                session_dir = write_session_artifacts(out_dir=out_dir, date=date, session=tail)
                run_config["loop_count"] = int(loop_count)
                run_config["reconnect_events"] = int(reconnect_events)
                write_session_run_config(session_dir=session_dir, run_config=run_config)
                if evidence_enabled and evidence_clipper is not None and evidence_session_id == tail.session_id:
                    pending = evidence_clipper.flush()
                    for clip_data in pending:
                        idx = evidence_clip_counts.get(clip_data.clip.name, 0) + 1
                        evidence_clip_counts[clip_data.clip.name] = idx
                        meta = write_evidence_clip(
                            clip_data=clip_data,
                            session_dir=session_dir,
                            index=idx,
                        )
                        if meta is not None:
                            evidence_clips.append(meta)
                    write_evidence_manifest(session_dir=session_dir, clips=evidence_clips)
                    evidence_clipper.reset()
                    evidence_session_id = None
                    evidence_clip_counts = {}
                    evidence_clips = []
                session_dirs.append(session_dir)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if run_writer is not None:
            run_writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if printed_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if pbar is not None:
            pbar.close()

    daily_json = write_daily_report(out_dir=out_dir, date=date, sessions=sessions)
    daily_csv = write_daily_csv(out_dir=out_dir, date=date, sessions=sessions)
    if run_video_path is not None:
        run_config["run_video"] = _file_metadata(run_video_path)
    if discarded_sessions:
        run_config["discarded_sessions"] = list(discarded_sessions)
    run_config_path = write_run_config(out_dir=out_dir, date=date, run_config=run_config)

    outputs = RunOutputs(
        date=date,
        out_dir=out_dir,
        session_dirs=tuple(session_dirs),
        daily_report_json=daily_json,
        daily_report_csv=daily_csv,
    )
    print(f"Wrote daily report: {outputs.daily_report_json}")
    print(f"Wrote sessions CSV: {outputs.daily_report_csv}")
    print(f"Wrote run config: {run_config_path}")
    print(f"Sessions: {len(outputs.session_dirs)}")

    return 0

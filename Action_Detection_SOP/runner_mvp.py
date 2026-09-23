from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    tqdm = None  # type: ignore[assignment]

from Action_Detection_SOP.evidence import EvidenceClipConfig, EvidenceClipper
from Action_Detection_SOP.evidence_io import write_evidence_clip, write_evidence_manifest
from Action_Detection_SOP.helmet_diagnostics_capture import HelmetDiagnosticCapture
from Action_Detection_SOP.ingest import CaptureInfo, get_capture_info, open_capture
from Action_Detection_SOP.reconnect_policy import reconnect_wait_seconds
from Action_Detection_SOP.reporting import (
    SessionReportResult,
    date_for_elapsed_time,
    highest_existing_session_number,
    session_start_datetime,
    today_date_str,
    write_daily_csv,
    write_daily_report,
    write_run_config,
    write_session_artifacts,
    write_session_run_config,
)
from Action_Detection_SOP.roi import RoiPolygon, clamp_rect_to_frame, draw_roi, load_roi_json, resolve_roi_for_frame
from Action_Detection_SOP.runtime_config import (
    PROFILE_OPERATOR_MVP_A,
    PROFILE_ROLL_SOP_V1,
    resolve_run_config,
)
from Action_Detection_SOP.runner_setup import (
    RunConfigPayloadInput,
    build_run_config_payload as _build_run_config_payload,
    build_sop_engine as _build_sop_engine,
    _file_metadata,
    _sha256_path,
    source_label as _source_label,
)
from Action_Detection_SOP.safety_alerts import (
    HelmetAlertConfig,
    HelmetAlertEngine,
    write_helmet_alert_artifacts,
)
from Action_Detection_SOP.sop_engine import (
    SessionResult,
    SopEngine,
    helmet_associated_with_person,
)
from Action_Detection_SOP.roll_sop_engine import RollSopEngine
from Action_Detection_SOP.source_security import redact_source_credentials
from yolo_kit import LetterboxConfig, YoloPostConfig, draw_detections, load_pipeline
from yolo_kit.types import Detection


def _full_frame_roi(*, frame_width: int, frame_height: int) -> RoiPolygon:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width/frame_height must be positive")

    #return all the polygon needed for this
    return RoiPolygon(
        points=(
            (0, 0),
            (int(frame_width) - 1, 0),
            (int(frame_width) - 1, int(frame_height) - 1),
            (0, int(frame_height) - 1),
        ),
        frame_size=(int(frame_width), int(frame_height)),
    )


@dataclass
class StagePerfTracker:
    preprocess_s: List[float]
    inference_s: List[float]
    postprocess_s: List[float]
    total_s: List[float]

    def record(self, *, preprocess_s: float, inference_s: float, postprocess_s: float, total_s: float) -> None:
        self.preprocess_s.append(float(preprocess_s))
        self.inference_s.append(float(inference_s))
        self.postprocess_s.append(float(postprocess_s))
        self.total_s.append(float(total_s))


def _percentile_s(values_s: Sequence[float], q: float) -> float:
    if not values_s:
        return 0.0
    values = sorted(float(v) * 1000.0 for v in values_s)
    if len(values) == 1:
        return float(values[0])
    pos = (float(q) / 100.0) * (len(values) - 1)
    lo = int(pos)
    hi = min(len(values) - 1, lo + 1)
    if lo == hi:
        return float(values[lo])
    t = pos - lo
    return float(values[lo] * (1.0 - t) + values[hi] * t)


def _summary_ms(values_s: Sequence[float]) -> Dict[str, float]:
    if not values_s:
        return {"n": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    total = sum(float(v) for v in values_s)
    count = len(values_s)
    return {
        "n": float(count),
        "mean_ms": (total * 1000.0) / float(count),
        "p50_ms": _percentile_s(values_s, 50.0),
        "p95_ms": _percentile_s(values_s, 95.0),
    }


def _format_stage_summary(label: str, values_s: Sequence[float]) -> str:
    summary = _summary_ms(values_s)
    return (
        f"{label}: n={int(summary['n'])} "
        f"mean={summary['mean_ms']:.3f}ms "
        f"p50={summary['p50_ms']:.3f}ms "
        f"p95={summary['p95_ms']:.3f}ms"
    )


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


def _filter_class_ids(dets: Sequence[Detection], class_ids: Sequence[int]) -> List[Detection]:
    wanted = {int(x) for x in class_ids}
    out: List[Detection] = []
    for d in dets:
        if d.class_id is not None and int(d.class_id) in wanted:
            out.append(d)
    return out


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


def _run_pipeline_timed(pipeline: object, image_bgr: "cv2.Mat") -> Tuple[List[Detection], Tuple[float, float, float, float]]:
    t0 = time.perf_counter()
    prep = pipeline.preprocess(image_bgr)
    t1 = time.perf_counter()
    preds = pipeline._infer_fn(prep.blob)  # type: ignore[attr-defined]
    t2 = time.perf_counter()
    detections = pipeline.post.process(preds, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio)
    t3 = time.perf_counter()
    return detections, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)


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


# this is for the rtsp portion to check
def _rtsp_option_or_none(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return int(value) if int(value) > 0 else None


def _compress_video_with_ffmpeg(src: Path, *, crf: int, preset: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not src.exists() or src.stat().st_size <= 0:
        return False

    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{src.stem}_compressed_",
        suffix=src.suffix or ".mp4",
        dir=src.parent,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0
    if ok:
        tmp_path.replace(src)
        return True
    try:
        tmp_path.unlink()
    except OSError:
        pass
    return False


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
    def _session_duration_s(session: SessionReportResult) -> float:
        return max(0.0, float(session.end_time_s) - float(session.start_time_s))

    # Resolve profile, classes, timing, and ROI inputs.
    runtime = resolve_run_config(args)
    sop_profile_name = runtime.sop_profile.name
    class_names = runtime.classes.class_names
    class_conf_thresholds = runtime.classes.class_conf_thresholds
    person_ids = list(runtime.classes.person_ids)
    helmet_disabled = runtime.classes.helmet_disabled
    helmet_ids = list(runtime.classes.helmet_ids)
    helmet_alert_confidence_floor = float(args.conf)
    helmet_alert_strong_confidence = float(args.conf)
    if bool(args.enable_helmet_alerts) and helmet_ids:
        helmet_alert_confidence_floor = min(
            float(class_conf_thresholds.get(class_id, args.conf)) for class_id in helmet_ids
        )
        # An explicit per-class threshold can be higher than --conf; scores
        # that survive postprocessing must still count as strong evidence.
        helmet_alert_strong_confidence = max(float(args.conf), helmet_alert_confidence_floor)
    roll_ids = list(runtime.classes.roll_ids)
    cleaning_cloth_ids = list(runtime.classes.cleaning_cloth_ids)
    paper_label_ids = list(runtime.classes.paper_label_ids)
    class_ids = list(runtime.classes.active_class_ids)
    for warning in runtime.classes.warnings:
        print(warning)

    args.start_s = runtime.session_timing.start_s
    args.end_s = runtime.session_timing.end_s
    args.min_session_s = runtime.session_timing.min_session_s
    if runtime.operator_rules is not None:
        args.roi_dwell_s = runtime.operator_rules.roi_dwell_s

    roi_path = Path(args.roi)
    roi_base = load_roi_json(roi_path)
    if roi_base.frame_size is None:
        print(
            "Note: ROI JSON has no frame_size; auto-rescale is disabled. "
            "Re-save ROI using Scripts/calibrate_roi.py to embed calibration resolution."
        )

    helmet_alerts_enabled = bool(args.enable_helmet_alerts)
    helmet_alert_roi_path: Optional[Path] = None
    helmet_alert_roi_base: Optional[RoiPolygon] = None
    if helmet_alerts_enabled:
        if not person_ids:
            raise ValueError("--enable-helmet-alerts requires resolved person class ids.")
        if not helmet_ids:
            raise ValueError("--enable-helmet-alerts requires resolved helmet class ids.")
        if bool(args.detect_roi_only):
            raise ValueError("--detect-roi-only cannot be combined with --enable-helmet-alerts.")
        if args.helmet_alert_roi:
            helmet_alert_roi_path = Path(args.helmet_alert_roi)
            helmet_alert_roi_base = load_roi_json(helmet_alert_roi_path)
            if helmet_alert_roi_base.frame_size is None:
                print(
                    "Note: helmet alert ROI JSON has no frame_size; auto-rescale is disabled. "
                    "Re-save ROI using Scripts/calibrate_roi.py to embed calibration resolution."
                )

    out_dir = Path(args.out_dir)
    date = today_date_str()
    is_live_source = args.rtsp is not None or args.webcam is not None
    initial_session_counter = highest_existing_session_number(out_dir=out_dir, date=date)

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    if args.conf < 0.0 or args.conf > 1.0:
        raise ValueError("--conf must be within [0, 1]")
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
            class_conf_thresholds=class_conf_thresholds or None,
            iou_threshold=float(args.iou),
            apply_nms=not bool(args.no_nms),
            class_ids=class_ids,
        ),
        letterbox_cfg=LetterboxConfig(new_shape=(int(args.imgsz), int(args.imgsz))),
        onnx_providers=onnx_providers,
        trt_output_name=args.trt_output_name,
        trt_output_index=int(args.trt_output_index),
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
    elif pipeline.backend_name == "tensorrt":
        trt_backend = pipeline.backend
        if trt_backend is not None and hasattr(trt_backend, "output_names"):
            print(
                "TensorRT outputs:",
                list(getattr(trt_backend, "output_names")),
                "selected:",
                getattr(trt_backend, "primary_output", None),
            )

    if args.loop_video and not args.video:
        raise ValueError("--loop-video is only valid with --video.")
    if args.reconnect_wait_s < 0:
        raise ValueError("--reconnect-wait-s must be >= 0")
    if args.reconnect_wait_max_s < 0:
        raise ValueError("--reconnect-wait-max-s must be >= 0")
    if args.reconnect_backoff < 1.0:
        raise ValueError("--reconnect-backoff must be >= 1.0")
    if args.reconnect_max_tries < 0:
        raise ValueError("--reconnect-max-tries must be >= 0")
    if args.rtsp_open_timeout_ms < 0:
        raise ValueError("--rtsp-open-timeout-ms must be >= 0")
    if args.rtsp_read_timeout_ms < 0:
        raise ValueError("--rtsp-read-timeout-ms must be >= 0")
    if args.rtsp_buffer_size < 0:
        raise ValueError("--rtsp-buffer-size must be >= 0")
    if args.trt_output_index < 0:
        raise ValueError("--trt-output-index must be >= 0")
    if sop_profile_name == PROFILE_ROLL_SOP_V1:
        if args.cleaning_s <= 0:
            raise ValueError("--cleaning-s must be > 0")
        if args.labeling_s <= 0:
            raise ValueError("--labeling-s must be > 0")
        if args.cleaning_max_gap < 0:
            raise ValueError("--cleaning-max-gap must be >= 0")
        if args.labeling_max_gap < 0:
            raise ValueError("--labeling-max-gap must be >= 0")

    if args.roi_upscale < 1.0:
        raise ValueError("--roi-upscale must be >= 1.0")
    if args.roi_expand < 0:
        raise ValueError("--roi-expand must be >= 0")
    if sop_profile_name == PROFILE_OPERATOR_MVP_A:
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
    if sop_profile_name == PROFILE_OPERATOR_MVP_A and not helmet_disabled:
        if args.helmet_s <= 0:
            raise ValueError("--helmet-s must be > 0")
        if args.helmet_max_gap < 0:
            raise ValueError("--helmet-max-gap must be >= 0")
    if helmet_alerts_enabled:
        if args.helmet_alert_s <= 0:
            raise ValueError("--helmet-alert-s must be > 0")
        if args.helmet_alert_recovery_s <= 0:
            raise ValueError("--helmet-alert-recovery-s must be > 0")
        if args.helmet_alert_cooldown_s < 0:
            raise ValueError("--helmet-alert-cooldown-s must be >= 0")
        if args.helmet_alert_min_person_height < 0:
            raise ValueError("--helmet-alert-min-person-height must be >= 0")
        if args.helmet_alert_max_gap < 0:
            raise ValueError("--helmet-alert-max-gap must be >= 0")
    if bool(args.helmet_alert_diagnostics):
        if not helmet_alerts_enabled:
            raise ValueError("--helmet-alert-diagnostics requires --enable-helmet-alerts")
        if int(args.helmet_diagnostics_max_mb) <= 0:
            raise ValueError("--helmet-diagnostics-max-mb must be > 0")
    if not args.no_evidence:
        if args.evidence_pre_s < 0:
            raise ValueError("--evidence-pre-s must be >= 0")
        if args.evidence_post_s < 0:
            raise ValueError("--evidence-post-s must be >= 0")
        if args.evidence_max_s <= 0:
            raise ValueError("--evidence-max-s must be > 0")
    if len(str(args.out_codec)) != 4:
        raise ValueError("--out-codec must be a four-character OpenCV codec such as mp4v")
    if not (0 <= int(args.out_crf) <= 51):
        raise ValueError("--out-crf must be within [0, 51]")

    # Validate command options and prepare capture settings.
    reconnect_tries = 0
    reconnect_events = 0

    rtsp_open_timeout_ms = _rtsp_option_or_none(int(args.rtsp_open_timeout_ms))
    rtsp_read_timeout_ms = _rtsp_option_or_none(int(args.rtsp_read_timeout_ms))
    rtsp_buffer_size = _rtsp_option_or_none(int(args.rtsp_buffer_size))
    rtsp_prefer_ffmpeg = bool(args.rtsp_prefer_ffmpeg)

    def _open_capture_with_retries(*, initial_open: bool) -> Tuple[cv2.VideoCapture, CaptureInfo]:
        nonlocal reconnect_tries, reconnect_events
        while True:
            try:
                new_cap = open_capture(
                    video=args.video,
                    webcam=args.webcam,
                    rtsp=args.rtsp,
                    rtsp_prefer_ffmpeg=rtsp_prefer_ffmpeg,
                    rtsp_open_timeout_ms=rtsp_open_timeout_ms,
                    rtsp_read_timeout_ms=rtsp_read_timeout_ms,
                    rtsp_buffer_size=rtsp_buffer_size,
                )
                new_info = get_capture_info(new_cap)
                if reconnect_tries > 0:
                    print(f"Reconnect succeeded after {reconnect_tries} attempt(s).")
                if not initial_open:
                    reconnect_events += 1
                reconnect_tries = 0
                return new_cap, new_info
            except Exception as exc:
                if not args.reconnect:
                    raise
                reconnect_tries += 1
                if args.reconnect_max_tries and reconnect_tries > int(args.reconnect_max_tries):
                    raise RuntimeError(f"Reconnect failed after {args.reconnect_max_tries} tries.") from exc
                wait_s = reconnect_wait_seconds(
                    attempt=reconnect_tries,
                    base_wait_s=float(args.reconnect_wait_s),
                    backoff=float(args.reconnect_backoff),
                    wait_cap_s=float(args.reconnect_wait_max_s),
                )
                context = "open" if initial_open else "reconnect"
                print(f"Capture {context} failed (attempt {reconnect_tries}): {exc}")
                if wait_s > 0:
                    print(f"Retrying in {wait_s:.1f}s...")
                    time.sleep(wait_s)

    # Open the source and derive the actual analysis cadence.
    cap, info = _open_capture_with_retries(initial_open=True)

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

    # Build the SOP/alert engines and run metadata.
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

    engine_setup = _build_sop_engine(
        args=args,
        sop_profile_name=sop_profile_name,
        helmet_disabled=helmet_disabled,
        analysis_fps=analysis_fps,
        initial_session_counter=initial_session_counter,
    )
    engine = engine_setup.engine
    helmet_alert_engine: Optional[HelmetAlertEngine] = None
    if helmet_alerts_enabled:
        helmet_alert_engine = HelmetAlertEngine(
            HelmetAlertConfig(
                required_seconds=float(args.helmet_alert_s),
                strong_helmet_confidence=helmet_alert_strong_confidence,
                verification_confidence=helmet_alert_confidence_floor,
                analysis_fps=float(analysis_fps),
                recovery_seconds=float(args.helmet_alert_recovery_s),
                absence_seconds=float(args.helmet_alert_recovery_s),
                cooldown_seconds=float(args.helmet_alert_cooldown_s),
                min_person_height_px=int(args.helmet_alert_min_person_height),
                head_top_fraction=float(args.head_top_frac),
                max_gap_frames=int(args.helmet_alert_max_gap),
                safety_area_id=str(args.helmet_alert_safety_area_id),
            ),
            source=_source_label(args),
            camera_id=args.helmet_alert_camera_id,
            diagnostics_enabled=bool(args.helmet_alert_diagnostics),
        )

    frame_idx = 0
    processed = 0
    sessions: List[SessionReportResult] = []
    session_dirs: List[Path] = []
    report_dates: Set[str] = set()
    roi_for_frame: Optional[RoiPolygon] = None
    helmet_alert_roi_for_frame: Optional[RoiPolygon] = None
    helmet_alert_dirs: List[Path] = []
    save_thumb = not bool(args.no_thumb)

    run_config = _build_run_config_payload(
        RunConfigPayloadInput(
            args=args,
            args_raw=args_raw,
            config_path=config_path,
            config_payload=config_payload,
            date=date,
            info=info,
            source_fps=source_fps,
            analysis_fps=analysis_fps,
            every=every,
            roi_path=roi_path,
            roi_base=roi_base,
            evidence_enabled=evidence_enabled,
            evidence_cfg=evidence_cfg,
            runtime=runtime,
            engine_setup=engine_setup,
            rtsp_prefer_ffmpeg=rtsp_prefer_ffmpeg,
            rtsp_open_timeout_ms=rtsp_open_timeout_ms,
            rtsp_read_timeout_ms=rtsp_read_timeout_ms,
            rtsp_buffer_size=rtsp_buffer_size,
        )
    )
    if helmet_alerts_enabled:
        helmet_alert_config: Dict[str, object] = {
            "enabled": True,
            "alert_type": "NO_HELMET",
            "safety_profile": "helmet_alert_v1",
            "scope": "roi" if helmet_alert_roi_base is not None else "full_frame",
            "required_seconds": float(args.helmet_alert_s),
            "helmet_confidence_floor": helmet_alert_confidence_floor,
            "helmet_strong_confidence": helmet_alert_strong_confidence,
            "recovery_seconds": float(args.helmet_alert_recovery_s),
            "absence_seconds": float(args.helmet_alert_recovery_s),
            "cooldown_seconds": float(args.helmet_alert_cooldown_s),
            "min_person_height_px": int(args.helmet_alert_min_person_height),
            "head_top_fraction": float(args.head_top_frac),
            "max_gap_frames": int(args.helmet_alert_max_gap),
            "safety_area_id": str(args.helmet_alert_safety_area_id),
            "camera_id": redact_source_credentials(args.helmet_alert_camera_id),
            "diagnostics": {
                "requested": bool(args.helmet_alert_diagnostics),
                "enabled": False,
                "max_total_mb": int(args.helmet_diagnostics_max_mb),
            },
        }
        if helmet_alert_roi_path is not None and helmet_alert_roi_base is not None:
            helmet_alert_config["roi"] = {
                "path": str(helmet_alert_roi_path),
                "frame_size": helmet_alert_roi_base.frame_size,
                "points": list(helmet_alert_roi_base.points),
                "sha256": _sha256_path(helmet_alert_roi_path),
            }
        run_config["helmet_alerts"] = helmet_alert_config
    else:
        run_config["helmet_alerts"] = {"enabled": False}

    # Initialize per-run state and output writers.
    diagnostic_capture: Optional[HelmetDiagnosticCapture] = None
    warnings: List[str] = []
    discarded_sessions: List[Dict[str, object]] = []

    start_wall: Optional[float] = None
    start_t_s: Optional[float] = None
    run_start_dt: Optional[datetime] = None
    loop_count = 0
    thumb_written: Set[str] = set()
    evidence_session_id: Optional[str] = None
    evidence_clip_counts: Dict[str, int] = {}
    evidence_clips: List[Dict[str, object]] = []

    writer: Optional[cv2.VideoWriter] = None
    run_writer: Optional[cv2.VideoWriter] = None
    run_video_path: Optional[Path] = None
    active_video_path: Optional[Path] = None
    active_session_dir: Optional[Path] = None
    active_session_date: Optional[str] = None
    active_session_start_dt: Optional[datetime] = None
    last_dets_global: List[Detection] = []
    last_dets_roi: List[Detection] = []
    last_persons_all: List[Detection] = []
    last_helmets_all: List[Detection] = []
    last_rolls_roi: List[Detection] = []
    last_cleaning_roi: List[Detection] = []
    last_labels_roi: List[Detection] = []

    def _stamp_session_result(result: SessionReportResult) -> SessionReportResult:
        if active_session_start_dt is None:
            return result
        if is_live_source:
            end_dt = datetime.now()
        elif run_start_dt is not None:
            end_dt = run_start_dt + timedelta(seconds=float(result.end_time_s))
        else:
            end_dt = active_session_start_dt + timedelta(seconds=_session_duration_s(result))
        return replace(
            result,
            start_time_iso=active_session_start_dt.isoformat(timespec="seconds"),
            end_time_iso=end_dt.isoformat(timespec="seconds"),
            start_date=active_session_start_dt.date().isoformat(),
            end_date=end_dt.date().isoformat(),
        )

    # Configure progress display and the optional diagnostic sidecar.
    win = "SOP roll_sop_v1" if sop_profile_name == PROFILE_ROLL_SOP_V1 else "SOP MVP-A"
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
    perf = StagePerfTracker(preprocess_s=[], inference_s=[], postprocess_s=[], total_s=[])
    if progress_enabled and tqdm is not None:
        if total_frames is not None:
            pbar = tqdm(total=total_frames, unit="frame", desc="sop", mininterval=progress_every_s)
        else:
            pbar = tqdm(unit="frame", desc="sop", mininterval=progress_every_s)

    if helmet_alerts_enabled and bool(args.helmet_alert_diagnostics) and helmet_alert_engine is not None:
        helmet_config = run_config.get("helmet_alerts")
        diagnostic_context = {
            "model_sha256": run_config.get("model", {}).get("sha256")
            if isinstance(run_config.get("model"), dict)
            else None,
            "metadata_sha256": run_config.get("metadata", {}).get("sha256")
            if isinstance(run_config.get("metadata"), dict)
            else None,
            "roi_sha256": run_config.get("roi", {}).get("sha256")
            if isinstance(run_config.get("roi"), dict)
            else None,
            "helmet_alerts": (
                {key: value for key, value in helmet_config.items() if key != "diagnostics"}
                if isinstance(helmet_config, dict)
                else None
            ),
        }
        diagnostic_capture = HelmetDiagnosticCapture(
            engine=helmet_alert_engine,
            out_dir=out_dir,
            date=date,
            source=_source_label(args),
            camera_id=args.helmet_alert_camera_id,
            context=diagnostic_context,
            max_total_bytes=int(args.helmet_diagnostics_max_mb) * 1024 * 1024,
        )
        if diagnostic_capture.start() and isinstance(helmet_config, dict):
            diagnostics_config = helmet_config.get("diagnostics")
            if isinstance(diagnostics_config, dict):
                diagnostics_config.update(
                    {
                        "enabled": True,
                        "path": str(diagnostic_capture.path),
                        "run_id": diagnostic_capture.run_id,
                    }
                )

    # Main capture loop: infer, update SOP rules, and persist frame artifacts.
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video and args.loop_video:
                    cap.release()
                    cap = open_capture(
                        video=args.video,
                        rtsp_prefer_ffmpeg=rtsp_prefer_ffmpeg,
                        rtsp_open_timeout_ms=rtsp_open_timeout_ms,
                        rtsp_read_timeout_ms=rtsp_read_timeout_ms,
                        rtsp_buffer_size=rtsp_buffer_size,
                    )
                    info = get_capture_info(cap)
                    loop_count += 1
                    reconnect_tries = 0
                    if diagnostic_capture is not None:
                        diagnostic_capture.start_segment(
                            reason="video_loop",
                            frame_idx=processed,
                            time_s=(frame_idx / source_fps) if source_fps else (processed / analysis_fps),
                        )
                    continue

                if args.reconnect:
                    cap.release()
                    cap, info = _open_capture_with_retries(initial_open=False)
                    if diagnostic_capture is not None:
                        diagnostic_capture.start_segment(
                            reason="capture_reconnected",
                            frame_idx=processed,
                            time_s=(frame_idx / source_fps) if source_fps else (processed / analysis_fps),
                        )
                    continue

                break

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
                if should_process and perf.total_s:
                    elapsed = max(time.monotonic() - progress_start, 1e-6)
                    proc_fps = float(processed) / elapsed
                    total_summary = _summary_ms(perf.total_s)
                    pbar.set_postfix(proc_fps=f"{proc_fps:.1f}", det_ms=f"{total_summary['mean_ms']:.1f}", refresh=False)
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
                    if should_process and perf.total_s:
                        total_summary = _summary_ms(perf.total_s)
                        sys.stdout.write(f" det_ms={total_summary['mean_ms']:.1f}")
                        sys.stdout.flush()

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

            if helmet_alerts_enabled and helmet_alert_roi_for_frame is None:
                if helmet_alert_roi_base is None:
                    helmet_alert_roi_for_frame = _full_frame_roi(
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                else:
                    helmet_alert_roi_for_frame = resolve_roi_for_frame(
                        helmet_alert_roi_base,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                    if helmet_alert_roi_base.frame_size is not None:
                        base_w, base_h = helmet_alert_roi_base.frame_size
                        stream_size = (int(frame.shape[1]), int(frame.shape[0]))
                        if (base_w, base_h) != stream_size:
                            msg = (
                                "Helmet alert ROI frame_size differs from stream resolution; ROI was auto-rescaled. "
                                f"roi_frame_size={(base_w, base_h)} stream_size={stream_size}"
                            )
                            print(f"WARNING: {msg}")
                            warnings.append(msg)
                helmet_alert_config = run_config.get("helmet_alerts")
                if isinstance(helmet_alert_config, dict):
                    helmet_alert_config["roi_resolved"] = {
                        "frame_size": (int(frame.shape[1]), int(frame.shape[0])),
                        "points": list(helmet_alert_roi_for_frame.points),
                    }
                if warnings:
                    run_config["warnings"] = list(warnings)

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

                    dets_local, perf_s = _run_pipeline_timed(pipeline, crop)
                    perf.record(
                        preprocess_s=perf_s[0],
                        inference_s=perf_s[1],
                        postprocess_s=perf_s[2],
                        total_s=perf_s[3],
                    )
                    dets_global = _offset_detections(dets_local, dx=float(x0), dy=float(y0), inv_scale=inv_scale)
                else:
                    dets_global, perf_s = _run_pipeline_timed(pipeline, frame)
                    perf.record(
                        preprocess_s=perf_s[0],
                        inference_s=perf_s[1],
                        postprocess_s=perf_s[2],
                        total_s=perf_s[3],
                    )

                dets_roi = _filter_by_roi(dets_global, roi_for_frame)
                persons_all, helmets_all = _split_classes(dets_global, person_ids=person_ids, helmet_ids=helmet_ids)
                persons_roi = _filter_by_roi(persons_all, roi_for_frame)
                if sop_profile_name == PROFILE_ROLL_SOP_V1:
                    rolls_roi = _filter_class_ids(dets_roi, roll_ids)
                    cleaning_roi = _filter_class_ids(dets_roi, cleaning_cloth_ids)
                    labels_roi = _filter_class_ids(dets_roi, paper_label_ids)
                    assert isinstance(engine, RollSopEngine)
                    result = engine.update(
                        time_s=float(t_s),
                        frame_idx=processed,
                        rolls=rolls_roi,
                        cleaning_cloths=cleaning_roi,
                        labels=labels_roi,
                    )
                    last_rolls_roi = list(rolls_roi)
                    last_cleaning_roi = list(cleaning_roi)
                    last_labels_roi = list(labels_roi)
                else:
                    assert isinstance(engine, SopEngine)
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

                alerts = ()
                if helmet_alert_engine is not None:
                    assert helmet_alert_roi_for_frame is not None
                    alerts = helmet_alert_engine.update(
                        time_s=float(t_s),
                        frame_idx=int(processed),
                        persons=persons_all,
                        helmets=helmets_all,
                        safety_roi=helmet_alert_roi_for_frame,
                        related_session_uid=None,
                        wall_dt=datetime.now() if is_live_source else None,
                    )
                    for alert in alerts:
                        if alert.start_datetime is not None:
                            alert_date = alert.start_datetime.date().isoformat()
                        else:
                            alert_date = date_for_elapsed_time(
                                run_start_dt=run_start_dt,
                                elapsed_s=float(alert.start_time_s),
                                fallback_date=date,
                            )
                        alert_dir = write_helmet_alert_artifacts(
                            out_dir=out_dir,
                            date=alert_date,
                            alert=alert,
                            frame_bgr=frame,
                            safety_roi=helmet_alert_roi_for_frame,
                            run_start_dt=run_start_dt,
                        )
                        helmet_alert_dirs.append(alert_dir)
                    if diagnostic_capture is not None:
                        diagnostic_capture.persist_frame(
                            frame_idx=int(processed),
                            time_s=float(t_s),
                            alert_uids=tuple(alert.alert_uid for alert in alerts),
                        )

            events = engine.pop_events() if should_process else ()
            session_id = engine.active_session_id
            need_thumb = bool(should_process and save_thumb and session_id and session_id not in thumb_written)

            # Reserve the session directory before writing any optional artifact.
            # A failed reservation is safer than mixing a restarted session with
            # an older directory that has the same numeric id.
            if session_id and active_session_dir is None:
                active_session_start_dt = session_start_datetime(
                    run_start_dt=run_start_dt,
                    elapsed_s=float(t_s),
                    live_source=is_live_source,
                    wall_clock_dt=datetime.now() if is_live_source else None,
                )
                active_session_date = active_session_start_dt.date().isoformat()
                active_session_dir = out_dir / "sessions" / active_session_date / f"session_{session_id}"
                active_session_dir.mkdir(parents=True, exist_ok=False)

            if evidence_enabled and evidence_clipper is not None:
                if evidence_session_id is None and session_id is not None:
                    evidence_session_id = session_id
                    evidence_clipper.reset()
                    evidence_clip_counts = {}
                    evidence_clips = []

            if args.save_video and session_id and active_session_dir is not None and writer is None:
                active_video_path = active_session_dir / "annotated.mp4"
                if args.video_fps_out and args.video_fps_out > 0:
                    fps_out = float(args.video_fps_out)
                else:
                    fps_out = float(source_fps) if source_fps else (float(analysis_fps) if analysis_fps else 5.0)
                fourcc = cv2.VideoWriter_fourcc(*str(args.out_codec))
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
                if sop_profile_name == PROFILE_ROLL_SOP_V1:
                    assert isinstance(engine, RollSopEngine)
                    clean_req = engine.cfg.cleaning.required_frames
                    label_req = engine.cfg.labeling.required_frames
                    clean_status = (
                        "OK"
                        if engine.active_cleaning_done
                        else f"{engine.active_cleaning_positive_frames}/{clean_req}f"
                    )
                    label_status = (
                        "OK"
                        if engine.active_labeling_done
                        else f"{engine.active_labeling_positive_frames}/{label_req}f"
                    )
                    overlay_text = f"session={sid} clean={clean_status} label={label_status}"
                else:
                    assert isinstance(engine, SopEngine)
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
                    overlay_text = f"session={sid} roi={roi_status} helmet={helmet_status}"
                font_scale, thickness = _overlay_style(int(vis.shape[0]))
                cv2.putText(
                    vis,
                    overlay_text,
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
                fourcc = cv2.VideoWriter_fourcc(*str(args.out_codec))
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
                    if active_session_dir is None:
                        raise RuntimeError("Evidence clip completed without a reserved session directory")
                    session_dir = active_session_dir
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
                result = _stamp_session_result(result)
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
                    active_video_path = None
                    discard_dir = active_session_dir
                    if discard_dir is not None and discard_dir.exists():
                        shutil.rmtree(discard_dir, ignore_errors=True)
                    active_session_dir = None
                    active_session_date = None
                    active_session_start_dt = None
                    thumb_written.discard(result.session_id)
                    if evidence_enabled and evidence_clipper is not None:
                        evidence_clipper.reset()
                        evidence_session_id = None
                        evidence_clip_counts = {}
                        evidence_clips = []
                    continue
                sessions.append(result)
                session_date = result.start_date or active_session_date or date
                if active_session_dir is None:
                    raise RuntimeError("Finalized session has no reserved session directory")
                session_dir = write_session_artifacts(
                    out_dir=out_dir,
                    date=session_date,
                    session=result,
                    session_dir=active_session_dir,
                )
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
                if sop_profile_name == PROFILE_ROLL_SOP_V1:
                    write_daily_report(
                        out_dir=out_dir,
                        date=session_date,
                        sessions=[result],
                        append=True,
                    )
                    write_daily_csv(
                        out_dir=out_dir,
                        date=session_date,
                        sessions=[result],
                        append=True,
                    )
                    report_dates.add(session_date)

                # Close writer for this session
                if writer is not None:
                    writer.release()
                    writer = None
                    if bool(args.compress_out) and active_video_path is not None:
                        compressed = _compress_video_with_ffmpeg(
                            active_video_path,
                            crf=int(args.out_crf),
                            preset=str(args.out_preset),
                        )
                        if not compressed:
                            print(f"Note: ffmpeg compression unavailable/failed; kept OpenCV output: {active_video_path}")
                    active_video_path = None
                active_session_dir = None
                active_session_date = None
                active_session_start_dt = None

        # End-of-stream flush
        end_time_s = (frame_idx / source_fps) if source_fps else (processed / analysis_fps)
        if sop_profile_name == PROFILE_ROLL_SOP_V1:
            assert isinstance(engine, RollSopEngine)
            tail = engine.flush(time_s=float(end_time_s), frame_idx=int(processed))
        else:
            assert isinstance(engine, SopEngine)
            tail = engine.flush(time_s=float(end_time_s))
        tail_events = engine.pop_events() if tail is not None else ()
        if tail is not None:
            tail = _stamp_session_result(tail)
            duration_s = _session_duration_s(tail)
            if args.min_session_s > 0 and duration_s < float(args.min_session_s):
                discarded_sessions.append(
                    {
                        "session_id": tail.session_id,
                        "duration_s": duration_s,
                        "reason": "min_session_seconds",
                    }
                )
                if writer is not None:
                    writer.release()
                    writer = None
                active_video_path = None
                discard_dir = active_session_dir
                if discard_dir is not None and discard_dir.exists():
                    shutil.rmtree(discard_dir, ignore_errors=True)
                active_session_dir = None
                active_session_date = None
                active_session_start_dt = None
                if evidence_enabled and evidence_clipper is not None:
                    evidence_clipper.reset()
                    evidence_session_id = None
                    evidence_clip_counts = {}
                    evidence_clips = []
            else:
                sessions.append(tail)
                session_date = tail.start_date or active_session_date or date
                if active_session_dir is None:
                    raise RuntimeError("Flushed session has no reserved session directory")
                session_dir = write_session_artifacts(
                    out_dir=out_dir,
                    date=session_date,
                    session=tail,
                    session_dir=active_session_dir,
                )
                run_config["loop_count"] = int(loop_count)
                run_config["reconnect_events"] = int(reconnect_events)
                write_session_run_config(session_dir=session_dir, run_config=run_config)
                if evidence_enabled and evidence_clipper is not None and evidence_session_id == tail.session_id:
                    for ev in tail_events:
                        if ev.session_id != evidence_session_id:
                            continue
                        evidence_clipper.trigger(
                            name=ev.name,
                            time_s=float(ev.time_s),
                            frame_idx=int(ev.frame_idx),
                        )
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
                if sop_profile_name == PROFILE_ROLL_SOP_V1:
                    write_daily_report(
                        out_dir=out_dir,
                        date=session_date,
                        sessions=[tail],
                        append=True,
                    )
                    write_daily_csv(
                        out_dir=out_dir,
                        date=session_date,
                        sessions=[tail],
                        append=True,
                    )
                    report_dates.add(session_date)

    # Close active episodes and release every opened output/capture handle.
    finally:
        if helmet_alert_engine is not None:
            end_time_s = (frame_idx / source_fps) if source_fps else (processed / analysis_fps)
            helmet_alert_engine.flush(time_s=float(end_time_s), frame_idx=int(processed))
            if diagnostic_capture is not None:
                diagnostic_capture.close(frame_idx=int(processed), time_s=float(end_time_s))
        cap.release()
        if writer is not None:
            writer.release()
            if bool(args.compress_out) and active_video_path is not None:
                compressed = _compress_video_with_ffmpeg(
                    active_video_path,
                    crf=int(args.out_crf),
                    preset=str(args.out_preset),
                )
                if not compressed:
                    print(f"Note: ffmpeg compression unavailable/failed; kept OpenCV output: {active_video_path}")
        if run_writer is not None:
            run_writer.release()
            if bool(args.compress_out) and run_video_path is not None:
                compressed = _compress_video_with_ffmpeg(
                    run_video_path,
                    crf=int(args.out_crf),
                    preset=str(args.out_preset),
                )
                if not compressed:
                    print(f"Note: ffmpeg compression unavailable/failed; kept OpenCV output: {run_video_path}")
        if args.show:
            cv2.destroyAllWindows()
        if printed_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if pbar is not None:
            pbar.close()

    # Write final daily reports and the complete run configuration.
    if sop_profile_name == PROFILE_ROLL_SOP_V1 and report_dates:
        report_date = max(report_dates)
        daily_json = out_dir / "reports" / report_date / "daily_report.json"
        daily_csv = out_dir / "reports" / report_date / "sessions.csv"
    else:
        daily_json = write_daily_report(
            out_dir=out_dir,
            date=date,
            sessions=sessions,
            sop_profile="roll_sop_v1" if sop_profile_name == PROFILE_ROLL_SOP_V1 else None,
        )
        daily_csv = write_daily_csv(out_dir=out_dir, date=date, sessions=sessions)
    run_config["performance"] = {
        "preprocess": _summary_ms(perf.preprocess_s),
        "inference": _summary_ms(perf.inference_s),
        "postprocess": _summary_ms(perf.postprocess_s),
        "total": _summary_ms(perf.total_s),
        "samples_recorded": int(len(perf.total_s)),
    }
    if run_video_path is not None:
        run_config["run_video"] = _file_metadata(run_video_path)
    if discarded_sessions:
        run_config["discarded_sessions"] = list(discarded_sessions)
    if isinstance(run_config.get("helmet_alerts"), dict):
        run_config["helmet_alerts"]["alerts_written"] = int(len(helmet_alert_dirs))  # type: ignore[index]
        if diagnostic_capture is not None:
            diagnostic_details = diagnostic_capture.summary()
            run_config["helmet_alerts"]["diagnostics"] = diagnostic_details  # type: ignore[index]
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
    if perf.total_s:
        print(_format_stage_summary("preprocess", perf.preprocess_s))
        print(_format_stage_summary("inference", perf.inference_s))
        print(_format_stage_summary("postprocess", perf.postprocess_s))
        print(_format_stage_summary("total", perf.total_s))
    print(f"Sessions: {len(outputs.session_dirs)}")
    if helmet_alerts_enabled:
        print(f"Helmet alerts: {len(helmet_alert_dirs)}")

    return 0

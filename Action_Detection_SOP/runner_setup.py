from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from Action_Detection_SOP.evidence import EvidenceClipConfig
from Action_Detection_SOP.ingest import CaptureInfo
from Action_Detection_SOP.roi import RoiPolygon
from Action_Detection_SOP.roll_sop_engine import (
    RollEvidenceRuleConfig,
    RollSopEngine,
    RollSopEngineConfig,
)
from Action_Detection_SOP.runtime_config import (
    PROFILE_OPERATOR_MVP_A,
    PROFILE_ROLL_SOP_V1,
    ResolvedRunConfig,
)
from Action_Detection_SOP.session import RollSessionConfig
from Action_Detection_SOP.sop_engine import (
    HelmetRuleConfig,
    RoiDwellRuleConfig,
    SessionizationConfig,
    SopEngine,
    SopEngineConfig,
)
from Action_Detection_SOP.source_security import redact_source_credentials, redact_source_fields


@dataclass(frozen=True)
class EngineSetup:
    engine: Union[SopEngine, RollSopEngine]
    roi_gap_frames: Optional[int] = None
    roi_miss_frames: Optional[int] = None


@dataclass(frozen=True)
class RunConfigPayloadInput:
    args: argparse.Namespace
    args_raw: Dict[str, object]
    config_path: Optional[Path]
    config_payload: Optional[Dict[str, object]]
    date: str
    info: CaptureInfo
    source_fps: Optional[float]
    analysis_fps: float
    every: int
    roi_path: Path
    roi_base: RoiPolygon
    evidence_enabled: bool
    evidence_cfg: Optional[EvidenceClipConfig]
    runtime: ResolvedRunConfig
    engine_setup: EngineSetup
    rtsp_prefer_ffmpeg: bool
    rtsp_open_timeout_ms: Optional[int]
    rtsp_read_timeout_ms: Optional[int]
    rtsp_buffer_size: Optional[int]


def _sha256_path(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> Dict[str, object]:
    payload: Dict[str, object] = {"path": str(path)}
    if not path.exists():
        payload["exists"] = False
        return payload
    stat = path.stat()
    payload.update(
        {
            "exists": True,
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "sha256": _sha256_path(path),
        }
    )
    return payload


def source_label(args: argparse.Namespace) -> str:
    if args.rtsp:
        return redact_source_credentials(str(args.rtsp)) or "rtsp://redacted-source"
    if args.video:
        return redact_source_credentials(str(args.video)) or "source"
    if args.webcam is not None:
        return f"webcam_{int(args.webcam)}"
    return "source"


def build_sop_engine(
    *,
    args: argparse.Namespace,
    sop_profile_name: str,
    helmet_disabled: bool,
    analysis_fps: float,
    initial_session_counter: int = 0,
) -> EngineSetup:
    """
    Build the selected SOP engine and its report-only configuration.
    """
    if sop_profile_name == PROFILE_ROLL_SOP_V1:
        return EngineSetup(
            engine=RollSopEngine(
                RollSopEngineConfig(
                    session=RollSessionConfig(
                        start_seconds=float(args.start_s),
                        end_seconds=float(args.end_s),
                        analysis_fps=analysis_fps,
                    ),
                    cleaning=RollEvidenceRuleConfig(
                        required_seconds=float(args.cleaning_s),
                        analysis_fps=analysis_fps,
                        max_gap_frames=int(args.cleaning_max_gap),
                    ),
                    labeling=RollEvidenceRuleConfig(
                        required_seconds=float(args.labeling_s),
                        analysis_fps=analysis_fps,
                        max_gap_frames=int(args.labeling_max_gap),
                    ),
                ),
                initial_session_counter=initial_session_counter,
            )
        )

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
        session=SessionizationConfig(
            start_seconds=float(args.start_s),
            end_seconds=float(args.end_s),
            analysis_fps=analysis_fps,
        ),
        helmet=helmet_cfg,
        roi_dwell=roi_dwell_cfg,
    )
    return EngineSetup(
        engine=SopEngine(engine_cfg, initial_session_counter=initial_session_counter),
        roi_gap_frames=roi_gap_frames,
        roi_miss_frames=roi_miss_frames,
    )


def build_run_config_payload(payload: RunConfigPayloadInput) -> Dict[str, object]:
    args = payload.args
    runtime = payload.runtime
    sop_profile = runtime.sop_profile
    classes = runtime.classes
    sop_profile_name = sop_profile.name

    evidence_payload: Dict[str, object] = {
        "enabled": bool(payload.evidence_enabled),
        "pre_seconds": float(args.evidence_pre_s),
        "post_seconds": float(args.evidence_post_s),
        "max_seconds": float(args.evidence_max_s),
        "analysis_fps": float(payload.analysis_fps),
        "events": (
            ["roll_entered", "cleaned_done", "labeled_done", "roll_left"]
            if sop_profile_name == PROFILE_ROLL_SOP_V1
            else ["roi_dwell_done", "helmet_done"]
        ),
    }
    if payload.evidence_cfg is not None:
        pre_s, post_s = payload.evidence_cfg.resolved_window()
        evidence_payload["resolved_pre_seconds"] = float(pre_s)
        evidence_payload["resolved_post_seconds"] = float(post_s)

    run_config: Dict[str, object] = {
        "date": payload.date,
        "args": redact_source_fields(payload.args_raw),
        "source": {
            "video": redact_source_credentials(args.video),
            "webcam": args.webcam,
            "rtsp": redact_source_credentials(args.rtsp),
        },
        "source_fps_raw": float(payload.info.fps) if payload.info.fps else None,
        "source_fps": float(payload.source_fps) if payload.source_fps else None,
        "analysis_fps": float(payload.analysis_fps),
        "every": int(payload.every),
        "sessionization": {
            "sop_profile": sop_profile_name,
            "start_seconds": float(args.start_s),
            "end_seconds": float(args.end_s),
            "min_session_seconds": float(args.min_session_s),
        },
        "roi": {
            "path": str(payload.roi_path),
            "frame_size": payload.roi_base.frame_size,
            "points": list(payload.roi_base.points),
            "sha256": _sha256_path(payload.roi_path),
        },
        "evidence": evidence_payload,
        "model": _file_metadata(Path(args.model)),
        "metadata": _file_metadata(Path(args.metadata)) if args.metadata else {"path": None},
        "postprocess": {
            "conf": float(args.conf),
            "label_conf": {
                str(classes.class_names.get(class_id, class_id)): float(threshold)
                for class_id, threshold in sorted(classes.class_conf_thresholds.items())
            },
            "iou": float(args.iou),
            "no_nms": bool(args.no_nms),
        },
        "detect_roi_only": bool(args.detect_roi_only),
        "video_fps_out": float(args.video_fps_out) if args.video_fps_out else None,
        "video_output": {
            "codec": str(args.out_codec),
            "compress_out": bool(args.compress_out),
            "ffmpeg_crf": int(args.out_crf),
            "ffmpeg_preset": str(args.out_preset),
        },
        "stream_sim": {
            "loop_video": bool(args.loop_video),
            "realtime": bool(args.realtime),
            "reconnect": bool(args.reconnect),
            "reconnect_wait_s": float(args.reconnect_wait_s),
            "reconnect_wait_max_s": float(args.reconnect_wait_max_s),
            "reconnect_backoff": float(args.reconnect_backoff),
            "reconnect_max_tries": int(args.reconnect_max_tries),
            "rtsp_prefer_ffmpeg": bool(payload.rtsp_prefer_ffmpeg),
            "rtsp_open_timeout_ms": (
                int(payload.rtsp_open_timeout_ms) if payload.rtsp_open_timeout_ms is not None else None
            ),
            "rtsp_read_timeout_ms": (
                int(payload.rtsp_read_timeout_ms) if payload.rtsp_read_timeout_ms is not None else None
            ),
            "rtsp_buffer_size": int(payload.rtsp_buffer_size) if payload.rtsp_buffer_size is not None else None,
        },
    }

    if sop_profile_name == PROFILE_OPERATOR_MVP_A:
        run_config["roi_dwell"] = {
            "required_seconds": float(args.roi_dwell_s),
            "max_gap_seconds": float(args.roi_dwell_max_gap),
            "max_gap_frames": int(payload.engine_setup.roi_gap_frames or 0),
            "max_track_missed_seconds": float(args.roi_dwell_miss),
            "max_track_missed_frames": int(payload.engine_setup.roi_miss_frames or 0),
            "iou_match_threshold": float(args.roi_dwell_iou),
            "min_person_height_px": int(args.roi_min_person_height),
        }

    if payload.config_path is None:
        run_config["config"] = {"path": None}
    else:
        run_config["config"] = {
            "path": str(payload.config_path),
            "data": (
                redact_source_fields(payload.config_payload)
                if payload.config_payload is not None
                else None
            ),
            "file": _file_metadata(payload.config_path),
        }

    if sop_profile.path is None:
        run_config["sop_profile"] = {"name": sop_profile_name, "path": None}
    else:
        run_config["sop_profile"] = {
            "name": sop_profile_name,
            "path": str(sop_profile.path),
            "data": asdict(sop_profile.profile) if sop_profile.profile is not None else None,
            "file": _file_metadata(sop_profile.path),
        }

    if sop_profile_name == PROFILE_ROLL_SOP_V1:
        run_config["roll_sop_v1"] = {
            "roll_labels": list(args.roll_label),
            "roll_class_ids": list(classes.roll_ids),
            "cleaning_cloth_labels": list(args.cleaning_cloth_label),
            "cleaning_cloth_class_ids": list(classes.cleaning_cloth_ids),
            "paper_label_labels": list(args.paper_label),
            "paper_label_class_ids": list(classes.paper_label_ids),
            "cleaning_required_seconds": float(args.cleaning_s),
            "cleaning_max_gap_frames": int(args.cleaning_max_gap),
            "labeling_required_seconds": float(args.labeling_s),
            "labeling_max_gap_frames": int(args.labeling_max_gap),
        }

    return run_config

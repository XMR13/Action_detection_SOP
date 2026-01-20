from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_run_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid run config JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Run config must be a JSON object")
    return payload


def collect_cli_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    dests: set[str] = set()
    for opt, action in parser._option_string_actions.items():
        for arg in argv:
            if arg == opt or arg.startswith(f"{opt}="):
                dests.add(action.dest)
                break
    return dests


def _coerce_str_list(value: object, key: str) -> List[str]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{key} must not be an empty string")
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        cleaned = [item.strip() for item in value]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError(f"{key} must not contain empty strings")
        return cleaned
    raise ValueError(f"{key} must be a string or list of strings")


def apply_run_config(
    *,
    args: argparse.Namespace,
    payload: Dict[str, object],
    cli_dests: set[str],
    parser: argparse.ArgumentParser,
) -> None:
    allowed = {action.dest for action in parser._actions if action.dest != "help"}
    if "config" in payload:
        raise ValueError("run config must not include the 'config' key")
    if "source" in payload and any(k in payload for k in ("video", "webcam", "rtsp")):
        raise ValueError("Use either 'source' block or top-level video/webcam/rtsp keys, not both.")
    unknown = sorted(k for k in payload.keys() if k not in allowed and k != "source")
    if unknown:
        raise ValueError(f"Unknown run config keys: {unknown}")

    source = payload.get("source")
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("run config 'source' must be an object")
        source_unknown = sorted(k for k in source.keys() if k not in ("video", "webcam", "rtsp"))
        if source_unknown:
            raise ValueError(f"Unknown run config source keys: {source_unknown}")
        non_empty = [k for k in ("video", "webcam", "rtsp") if source.get(k) not in (None, "")]
        if len(non_empty) > 1:
            raise ValueError("run config 'source' must set only one of video/webcam/rtsp")
        for key in ("video", "webcam", "rtsp"):
            if key in source and key not in cli_dests:
                value = source[key]
                if value in (None, ""):
                    continue
                if key == "webcam":
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError("source.webcam must be an integer index")
                    setattr(args, key, int(value))
                else:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"source.{key} must be a non-empty string")
                    setattr(args, key, value)

    str_keys = {
        "video",
        "rtsp",
        "roi",
        "sop_profile",
        "model",
        "metadata",
        "backend",
        "out_dir",
        "onnx_providers",
    }
    int_keys = {
        "webcam",
        "reconnect_max_tries",
        "imgsz",
        "every",
        "roi_min_person_height",
        "helmet_max_gap",
        "min_person_height",
        "roi_expand",
        "max_frames",
        "progress_bar_width",
    }
    float_keys = {
        "reconnect_wait_s",
        "source_fps",
        "video_fps_out",
        "conf",
        "iou",
        "analysis_fps",
        "start_s",
        "end_s",
        "min_session_s",
        "roi_dwell_s",
        "roi_dwell_max_gap",
        "roi_dwell_iou",
        "roi_dwell_miss",
        "helmet_s",
        "head_top_frac",
        "roi_upscale",
        "evidence_pre_s",
        "evidence_post_s",
        "evidence_max_s",
        "progress_every_s",
    }
    bool_keys = {
        "loop_video",
        "realtime",
        "reconnect",
        "require_cuda",
        "no_nms",
        "skip_helmet",
        "require_helmet_class",
        "detect_roi_only",
        "save_video",
        "save_run_video",
        "no_thumb",
        "no_evidence",
        "show",
        "progress",
    }

    for key, value in payload.items():
        if key == "source":
            continue
        if key in cli_dests:
            continue
        if value is None:
            continue
        if key not in allowed:
            continue
        if key in ("person_label", "helmet_label", "require_onnx_provider"):
            value = _coerce_str_list(value, key)
            setattr(args, key, value)
            continue
        if key == "onnx_providers":
            if isinstance(value, list):
                value = _coerce_str_list(value, key)
                setattr(args, key, ",".join(value))
            elif isinstance(value, str) and value.strip():
                setattr(args, key, value)
            else:
                raise ValueError("onnx_providers must be a non-empty string or list of strings")
            continue
        if key in str_keys:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            setattr(args, key, value)
            continue
        if key in bool_keys:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            setattr(args, key, value)
            continue
        if key in int_keys:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be an integer")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{key} must be an integer")
            setattr(args, key, int(value))
            continue
        if key in float_keys:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            setattr(args, key, float(value))
            continue
        raise ValueError(f"Unsupported run config key: {key}")

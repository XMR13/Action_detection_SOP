from __future__ import annotations

import json
import csv
import io
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import re
from datetime import datetime, date as Date
import hashlib
import uuid

import cv2

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .auth import BasicAuthConfig, BasicAuthMiddleware, issue_session_token
from .review_store import get_review, get_reviews_by_uid, init_db, upsert_review
from .session_index import SessionArtifact, SessionIndex
from .settings import WebMvpSettings
from .sop_status import (
    ReviewOverrideError,
    evaluate_sop_status,
    effective_review_for_session,
    normalize_session_checklist_payload,
    validate_review_overrides,
)
from ..shifts import assign_shift_for_interval, parse_iso_datetime

API_CONTRACT_VERSION = "2026-03-03.v1"
_MAX_REVIEW_NOTE_LEN = 4000


def _display_status(value: str) -> str:
    # Render "NOT_DONE" as "NOT DONE" to match UI labels.
    return value.replace("_", " ").upper() if value else "-"

def _parse_iso_ts(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v).timestamp()
    except Exception:
        return 0.0


def _shift_fields_from_isos(*, start_iso: Any, end_iso: Any) -> Dict[str, Any]:
    start_dt = parse_iso_datetime(start_iso) or parse_iso_datetime(end_iso)
    if start_dt is None:
        return {}
    end_dt = parse_iso_datetime(end_iso) or start_dt
    assignment = assign_shift_for_interval(start_dt=start_dt, end_dt=end_dt)
    if assignment is None:
        return {}
    return assignment.to_iso_fields()


def _normalize_shift_filter(value: Any) -> str:
    """
    Normalze filter data
    """
    raw = str(value or "").strip().upper().replace(" ", "").replace("_", "")
    #basics if staement for normalziing the if statements
    if raw in {"S1", "SHIFT1", "1"}:
        return "S1"
    if raw in {"S2", "SHIFT2", "2"}:
        return "S2"
    if raw in {"S3", "SHIFT3", "3"}:
        return "S3"
    return "ALL"


def _sessions_root_signature_ns(data_dir: Path) -> int:
    """
    Cheap change detector for ingestion into sessions folders under `data_dir`.

    The runner MVP commonly writes to `<data_dir>/<run_name>/sessions/<date>/...`,
    while the "web-first" layout is `<data_dir>/sessions/<date>/...`.

    We watch:
    - `<data_dir>/sessions`
    - `<data_dir>/*/sessions` (one-level nested runner outputs)

    For each sessions root, we only look at:
    - the root mtime
    - mtimes of the immediate date subdirectories

    Creating a new session directory updates the parent date directory mtime,
    so this detects new uploads without reading any JSON.
    """

    def mtime_ns(p: Path) -> int:
        try:
            return int(p.stat().st_mtime_ns)
        except FileNotFoundError:
            return 0
        except OSError:
            return 0

    def iter_session_roots() -> List[Path]:
        roots: List[Path] = []
        direct = data_dir / "sessions"
        if direct.exists() and direct.is_dir():
            roots.append(direct)
        try:
            for child in data_dir.iterdir():
                if not child.is_dir():
                    continue
                if child.name.startswith(".") or child.name in {"_web_cache"}:
                    continue
                nested = child / "sessions"
                if nested.exists() and nested.is_dir():
                    roots.append(nested)
        except OSError:
            # If the directory is temporarily unavailable, keep signature stable.
            pass
        # Dedupe by resolved path.
        unique: Dict[str, Path] = {}
        for r in roots:
            try:
                unique[str(r.resolve())] = r
            except OSError:
                unique[str(r)] = r
        return list(unique.values())

    sig = 0
    for root in iter_session_roots():
        sig = max(sig, mtime_ns(root))
        try:
            for child in root.iterdir():
                if child.is_dir():
                    sig = max(sig, mtime_ns(child))
        except OSError:
            continue
    return sig


def _utc_iso_from_timestamp(ts: float) -> Optional[str]:
    if ts <= 0:
        return None
    try:
        return datetime.utcfromtimestamp(ts).isoformat(timespec="seconds") + "Z"
    except Exception:
        return None


def _dir_stats(path: Path) -> Dict[str, Any]:
    files = 0
    total_bytes = 0
    newest_mtime = 0.0
    exists = path.exists()
    is_dir = path.is_dir()
    if exists and is_dir:
        try:
            for child in path.rglob("*"):
                try:
                    if not child.is_file():
                        continue
                    st = child.stat()
                except OSError:
                    continue
                files += 1
                total_bytes += int(st.st_size)
                newest_mtime = max(newest_mtime, float(st.st_mtime))
        except OSError:
            pass
    return {
        "path": str(path),
        "exists": bool(exists),
        "files": int(files),
        "bytes": int(total_bytes),
        "last_modified_utc": _utc_iso_from_timestamp(newest_mtime),
    }


def _spool_bucket_stats(path: Path, *, now_ts: Optional[float] = None) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    stats = _dir_stats(path)
    oldest_pending_ts = 0.0
    newest_pending_ts = 0.0
    if path.exists() and path.is_dir():
        try:
            for child in path.glob("*.json"):
                try:
                    st = child.stat()
                except OSError:
                    continue
                ts = float(st.st_mtime)
                oldest_pending_ts = ts if oldest_pending_ts <= 0 else min(oldest_pending_ts, ts)
                newest_pending_ts = max(newest_pending_ts, ts)
        except OSError:
            pass
    stats["oldest_item_utc"] = _utc_iso_from_timestamp(oldest_pending_ts)
    stats["newest_item_utc"] = _utc_iso_from_timestamp(newest_pending_ts)
    stats["oldest_item_age_s"] = max(0.0, now - oldest_pending_ts) if oldest_pending_ts > 0 else None
    return stats


def _spool_pending_retry_stats(path: Path, *, now_ts: Optional[float] = None) -> Dict[str, Any]:
    """Queue retry for attempting to upload the files and folders"""
    now = float(now_ts if now_ts is not None else time.time())
    ready_now_files = 0
    retry_due_files = 0
    retry_scheduled_files = 0
    invalid_payload_files = 0
    max_attempts_seen = 0
    oldest_task_created_ts = 0.0
    next_retry_ts = 0.0


    if path.exists() and path.is_dir():
        try:
            #ts stand for timestamp
            for child in path.glob("*.json"):
                fallback_created_ts = 0.0
                try:
                    st = child.stat()
                    fallback_created_ts = float(st.st_mtime)
                except OSError:
                    fallback_created_ts = 0.0
                try:
                    payload_raw = child.read_text(encoding="utf-8")
                    payload = json.loads(payload_raw)
                except Exception:
                    invalid_payload_files += 1
                    continue
                if not isinstance(payload, dict):
                    invalid_payload_files += 1
                    continue

                attempts_raw = payload.get("attempts", 0)
                try:
                    attempts = max(0, int(attempts_raw))
                except Exception:
                    attempts = 0
                max_attempts_seen = max(max_attempts_seen, attempts)

                created_ts = _parse_iso_ts(payload.get("created_at_utc")) or fallback_created_ts
                if created_ts > 0:
                    oldest_task_created_ts = created_ts if oldest_task_created_ts <= 0 else min(oldest_task_created_ts, created_ts)

                next_retry_raw = payload.get("next_retry_ts", 0.0)
                try:
                    due_ts = float(next_retry_raw)
                except Exception:
                    due_ts = 0.0
                if due_ts <= 0:
                    ready_now_files += 1
                    continue
                if due_ts <= now:
                    ready_now_files += 1
                    if attempts > 0:
                        retry_due_files += 1
                    next_retry_ts = due_ts if next_retry_ts <= 0 else min(next_retry_ts, due_ts)
                    continue
                retry_scheduled_files += 1
                next_retry_ts = due_ts if next_retry_ts <= 0 else min(next_retry_ts, due_ts)
        except OSError:
            pass

    oldest_task_age_s: Optional[float] = None
    if oldest_task_created_ts > 0:
        oldest_task_age_s = max(0.0, now - oldest_task_created_ts)
    next_retry_in_s: Optional[float] = None
    if next_retry_ts > 0:
        next_retry_in_s = max(0.0, next_retry_ts - now)

    return {
        "ready_now_files": int(ready_now_files),
        "retry_due_files": int(retry_due_files),
        "retry_scheduled_files": int(retry_scheduled_files),
        "invalid_payload_files": int(invalid_payload_files),
        "max_attempts_seen": int(max_attempts_seen),
        "oldest_task_created_utc": _utc_iso_from_timestamp(oldest_task_created_ts),
        "oldest_task_age_s": oldest_task_age_s,
        "next_retry_utc": _utc_iso_from_timestamp(next_retry_ts),
        "next_retry_in_s": next_retry_in_s,
    }


def _spool_state_file_stats(spool_root: Path, *, now_ts: Optional[float] = None) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    state_path = spool_root / "state.json"
    out: Dict[str, Any] = {
        "path": str(state_path),
        "exists": state_path.exists(),
        "parse_error": None,
        "generated_at_utc": None,
        "generated_age_s": None,
        "is_stale": None,
        "watch_mode": None,
        "cycle": None,
        "last_success_utc": None,
        "last_dead_utc": None,
    }
    if not state_path.exists() or not state_path.is_file():
        return out
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["parse_error"] = str(exc)
        return out
    if not isinstance(payload, dict):
        out["parse_error"] = "state payload must be object"
        return out

    generated_at = payload.get("generated_at_utc")
    generated_ts = _parse_iso_ts(generated_at)
    generated_age_s: Optional[float] = None
    is_stale: Optional[bool] = None
    if generated_ts > 0:
        generated_age_s = max(0.0, now - generated_ts)
        is_stale = generated_age_s > 300.0

    out["generated_at_utc"] = generated_at if isinstance(generated_at, str) else None
    out["generated_age_s"] = generated_age_s
    out["is_stale"] = is_stale
    out["watch_mode"] = bool(payload.get("watch_mode")) if "watch_mode" in payload else None
    out["cycle"] = int(payload.get("cycle")) if isinstance(payload.get("cycle"), (int, float)) else None
    out["last_success_utc"] = payload.get("last_success_utc") if isinstance(payload.get("last_success_utc"), str) else None
    out["last_dead_utc"] = payload.get("last_dead_utc") if isinstance(payload.get("last_dead_utc"), str) else None
    return out


def _safe_file_size(path: Path) -> int:
    try:
        if path.exists() and path.is_file():
            return int(path.stat().st_size)
    except OSError:
        return 0
    return 0


def _disk_health(
    disk: Any,
    *,
    warning_used_pct: float,
    critical_used_pct: float,
) -> Dict[str, Any]:
    total = max(0, int(disk.total))
    used = max(0, int(disk.used))
    free = max(0, int(disk.free))
    used_pct = (used / total * 100.0) if total > 0 else 0.0
    free_pct = (free / total * 100.0) if total > 0 else 0.0
    warning = float(warning_used_pct)
    critical = float(critical_used_pct)
    if critical < warning:
        critical = warning

    #the issues as a list of string
    issues: List[str] = []
    status = "ok"
    if used_pct >= critical:
        status = "critical"
        issues.append("penggunaan disk mencapai kritikal")
    elif used_pct >= warning:
        status = "warning"
        issues.append("WARNING: Penggunaan disk mencapai sudah banyak")

    return {
        "status": status,
        "issues": issues,
        "used_pct": round(float(used_pct), 3),
        "free_pct": round(float(free_pct), 3),
        "warning_used_pct": round(float(warning), 3),
        "critical_used_pct": round(float(critical), 3),
    }


def _session_storage_breakdown(sessions: List[SessionArtifact]) -> Dict[str, Any]:
    checklist_bytes = 0
    run_config_bytes = 0
    thumbnail_bytes = 0
    evidence_json_bytes = 0
    evidence_clip_bytes = 0
    annotated_video_bytes = 0
    evidence_clip_files = 0
    annotated_video_files = 0

    for session in sessions:
        checklist_bytes += _safe_file_size(session.paths.checklist_json)
        run_config_bytes += _safe_file_size(session.paths.run_config_json)
        thumbnail_bytes += _safe_file_size(session.paths.thumbnail_jpg)
        evidence_json_bytes += _safe_file_size(session.paths.evidence_json)

        annotated_path = session.paths.session_dir / "annotated.mp4"
        annotated_size = _safe_file_size(annotated_path)
        if annotated_size > 0:
            annotated_video_files += 1
            annotated_video_bytes += annotated_size

        evidence_dir = session.paths.session_dir / "evidence"
        if evidence_dir.exists() and evidence_dir.is_dir():
            try:
                for clip in evidence_dir.glob("*.mp4"):
                    clip_size = _safe_file_size(clip)
                    if clip_size <= 0:
                        continue
                    evidence_clip_files += 1
                    evidence_clip_bytes += clip_size
            except OSError:
                continue

    categories = {
        "checklists": {"files": len(sessions), "bytes": int(checklist_bytes)},
        "run_configs": {"files": sum(1 for s in sessions if s.paths.run_config_json.exists()), "bytes": int(run_config_bytes)},
        "thumbnails": {"files": sum(1 for s in sessions if s.paths.thumbnail_jpg.exists()), "bytes": int(thumbnail_bytes)},
        "evidence_manifests": {"files": sum(1 for s in sessions if s.paths.evidence_json.exists()), "bytes": int(evidence_json_bytes)},
        "evidence_clips": {"files": int(evidence_clip_files), "bytes": int(evidence_clip_bytes)},
        "annotated_videos": {"files": int(annotated_video_files), "bytes": int(annotated_video_bytes)},
    }
    total_bytes = sum(int(item["bytes"]) for item in categories.values())
    total_files = sum(int(item["files"]) for item in categories.values())
    return {
        "categories": categories,
        "total_files": int(total_files),
        "total_bytes": int(total_bytes),
    }


def _clip_count(session: SessionArtifact) -> int:
    clips = session.evidence.get("clips")
    if isinstance(clips, list):
        return len(clips)
    return 0


def _first_clip_rel_file(session: SessionArtifact) -> Optional[str]:
    clips = session.evidence.get("clips")
    if not isinstance(clips, list) or not clips:
        return None
    clip0 = clips[0]
    if not isinstance(clip0, dict):
        return None
    rel_file = clip0.get("file")
    if not isinstance(rel_file, str) or not rel_file:
        return None
    # Evidence writer may emit Windows-style separators; normalize for URLs.
    return rel_file.replace("\\", "/")


def _matches_evidence_filter(*, clip_count: int, has_thumbnail: bool, evidence_filter: str) -> bool:
    mode = str(evidence_filter or "ANY").upper()
    if mode == "CLIP_THUMB":
        return clip_count > 0 and has_thumbnail
    if mode == "CLIP_ONLY":
        return clip_count > 0 and not has_thumbnail
    if mode == "THUMB_ONLY":
        return clip_count <= 0 and has_thumbnail
    return True


def _has_review_evidence(session: SessionArtifact) -> bool:
    return _clip_count(session) > 0 or session.paths.thumbnail_jpg.exists()


def _effective_review_for_web(*, session: SessionArtifact, review: Any, settings: WebMvpSettings) -> Any:
    return effective_review_for_session(
        session=session,
        review=review,
        auto_approve_done_enabled=settings.auto_approve_done_enabled,
        auto_approve_min_duration_s=settings.auto_approve_min_duration_s,
        has_evidence=_has_review_evidence(session),
    )


def _safe_session_dir(data_dir: Path, session: SessionArtifact, rel_path: str) -> Path:
    base = session.paths.session_dir.resolve()
    target = (session.paths.session_dir / rel_path).resolve()
    if base == target:
        return target
    try:
        target.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid media path") from e
    return target


def _safe_cache_file_name(rel_path: str, src_path: Path) -> str:
    src_stat = src_path.stat()
    digest_src = f"{rel_path}|{src_path.name}|{src_stat.st_size}|{src_stat.st_mtime_ns}"
    digest = hashlib.sha1(digest_src.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
    return f"{digest}.mp4"


def _try_transcode_with_ffmpeg(*, src_path: Path, dst_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    return dst_path.exists() and dst_path.stat().st_size > 0


def _try_transcode_with_opencv(*, src_path: Path, dst_path: Path) -> bool:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return False
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 10.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, frame0 = cap.read()
            if not ok or frame0 is None:
                return False
            height, width = frame0.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for fourcc_tag in ("avc1", "H264", "X264"):
            if dst_path.exists():
                dst_path.unlink(missing_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                continue
            wrote = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    writer.write(frame)
                    wrote += 1
            finally:
                writer.release()
            if wrote > 0 and dst_path.exists() and dst_path.stat().st_size > 0:
                return True
        return False
    finally:
        cap.release()


def _ensure_browser_playback_path(*, settings: WebMvpSettings, session_uid: str, rel_path: str, src_path: Path) -> Path:
    # Cache transcodes by source path + source mtime/size so stale results are auto-rotated.
    cache_dir = settings.data_dir / "_web_cache" / "transcoded" / session_uid
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_name = _safe_cache_file_name(rel_path=rel_path, src_path=src_path)
    cached_path = cache_dir / cached_name
    if cached_path.exists() and cached_path.stat().st_size > 0:
        return cached_path

    tmp = cached_path.with_suffix(".tmp.mp4")
    tmp.unlink(missing_ok=True)

    ok = _try_transcode_with_ffmpeg(src_path=src_path, dst_path=tmp)
    if not ok:
        ok = _try_transcode_with_opencv(src_path=src_path, dst_path=tmp)
    if ok and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(cached_path)
        return cached_path

    tmp.unlink(missing_ok=True)
    # Fallback to original when transcode tools/codecs are unavailable.
    return src_path


def _safe_rel_path(rel_path: str) -> Path:
    rel_path = rel_path.replace("\\", "/")
    p = Path(rel_path)
    if p.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid rel_path (absolute)")
    if not rel_path.strip():
        raise HTTPException(status_code=400, detail="Invalid rel_path (empty)")
    parts = [part for part in p.parts if part not in (".", "")]
    if any(part == ".." for part in parts):
        raise HTTPException(status_code=400, detail="Invalid rel_path (parent traversal)")
    return Path(*parts)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class ReviewUpsertIn(BaseModel):
    review_status: Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"] = Field(...)
    review_note: str = Field(default="")
    overrides: Dict[str, Any] = Field(default_factory=dict)


class SessionUpsertIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_uid: str = Field(...)
    session_id: str = Field(...)
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD; used for storage layout.")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD; used if start_date is missing.")


def _storage_date(payload: SessionUpsertIn) -> str:
    if payload.start_date and isinstance(payload.start_date, str):
        return payload.start_date
    if payload.end_date and isinstance(payload.end_date, str):
        return payload.end_date
    raise HTTPException(status_code=400, detail="Missing start_date/end_date (YYYY-MM-DD)")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _validate_date_ymd(date: str) -> None:
    if not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Invalid date (expected YYYY-MM-DD)")
    if _parse_date_ymd(date) is None:
        raise HTTPException(status_code=400, detail="Invalid date (calendar day out of range)")


def _parse_date_ymd(date_raw: str) -> Optional[Date]:
    if not isinstance(date_raw, str):
        return None
    if not _DATE_RE.match(date_raw):
        return None
    try:
        return datetime.strptime(date_raw, "%Y-%m-%d").date()
    except Exception:
        return None


def _validate_token(value: str, *, label: str) -> None:
    if not value or not _SAFE_TOKEN_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")


def _validate_session_uid(session_uid: str) -> None:
    # We generate uuid4 hex today; accept safe tokens so we can evolve later.
    _validate_token(session_uid, label="session_uid")


def _validate_session_id(session_id: str) -> None:
    _validate_token(session_id, label="session_id")


def _validate_review_note(review_note: str) -> None:
    if len(review_note) > _MAX_REVIEW_NOTE_LEN:
        raise HTTPException(status_code=400, detail=f"review_note too long (max {_MAX_REVIEW_NOTE_LEN})")


def _validate_review_overrides(*, checklist: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, str]:
    try:
        return validate_review_overrides(checklist=checklist, raw=raw)
    except ReviewOverrideError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _resolve_date_window(
    *,
    date: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[Optional[Date], Optional[Date], Optional[str], Optional[str]]:
    if date and (date_from or date_to):
        raise HTTPException(status_code=400, detail="Use either `date` or `date_from/date_to`, not both")
    if date:
        _validate_date_ymd(date)
        d = _parse_date_ymd(date)
        if d is None:
            raise HTTPException(status_code=400, detail="Invalid date (expected YYYY-MM-DD)")
        return d, d, date, date
    if date_from:
        _validate_date_ymd(date_from)
    if date_to:
        _validate_date_ymd(date_to)
    from_d = _parse_date_ymd(date_from) if date_from else None
    to_d = _parse_date_ymd(date_to) if date_to else None
    if date_from and from_d is None:
        raise HTTPException(status_code=400, detail="Invalid `date_from` (expected YYYY-MM-DD)")
    if date_to and to_d is None:
        raise HTTPException(status_code=400, detail="Invalid `date_to` (expected YYYY-MM-DD)")
    if from_d and to_d and from_d > to_d:
        raise HTTPException(status_code=400, detail="Invalid date range (`date_from` > `date_to`)")
    return from_d, to_d, date_from, date_to


def _filter_sessions_by_date_window(
    sessions: List[SessionArtifact],
    *,
    date: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[List[SessionArtifact], Optional[str], Optional[str]]:
    lower, upper, lower_raw, upper_raw = _resolve_date_window(date=date, date_from=date_from, date_to=date_to)
    if lower or upper:
        filtered: List[SessionArtifact] = []
        for session in sessions:
            session_date = _parse_date_ymd(session.date)
            if session_date is None:
                # Ignore invalid folder date when explicit date filtering is requested.
                continue
            if lower and session_date < lower:
                continue
            if upper and session_date > upper:
                continue
            filtered.append(session)
        sessions = filtered
    return sessions, lower_raw, upper_raw


def create_app(settings: WebMvpSettings) -> FastAPI:
    init_db(settings.db_path)
    index = SessionIndex(data_dir=settings.data_dir)
    refresh_lock = threading.Lock()

    app = FastAPI(title="SOP Review MVP", version="0.1.0")
    auth_cfg = BasicAuthConfig(username=settings.admin_username, password=settings.admin_password)
    app.add_middleware(
        BasicAuthMiddleware,
        cfg=auth_cfg,
    )

    ui_dir = settings.ui_dir
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    @app.on_event("startup")
    def _startup() -> None:
        with refresh_lock:
            index.refresh()

        # Optional server-side auto-rescan. Clients don't need to click Rescan
        # to see new uploads, and all clients observe updates via /api/config.
        auto_s = float(getattr(settings, "auto_rescan_seconds", 0.0) or 0.0)
        if auto_s <= 0:
            return

        stop_evt = threading.Event()

        def loop() -> None:
            last_sig = _sessions_root_signature_ns(settings.data_dir)
            # Sleep first so startup doesn't immediately re-scan twice.
            while not stop_evt.wait(timeout=auto_s):
                sig = _sessions_root_signature_ns(settings.data_dir)
                if sig <= last_sig:
                    continue
                try:
                    with refresh_lock:
                        index.refresh()
                    last_sig = sig
                except Exception:
                    # Keep loop alive; next poll will attempt again.
                    continue

        t = threading.Thread(target=loop, name="web_mvp_auto_rescan", daemon=True)
        app.state._auto_rescan_stop = stop_evt
        app.state._auto_rescan_thread = t
        t.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        stop_evt = getattr(app.state, "_auto_rescan_stop", None)
        t = getattr(app.state, "_auto_rescan_thread", None)
        if isinstance(stop_evt, threading.Event):
            stop_evt.set()
        if isinstance(t, threading.Thread):
            t.join(timeout=2.0)

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/ui/index.html")

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {
            "status": "ok",
            "api_contract_version": API_CONTRACT_VERSION,
        }

    class LoginIn(BaseModel):
        username: str = Field(...)
        password: str = Field(...)

    @app.post("/api/auth/login")
    def login(payload: LoginIn) -> JSONResponse:
        if not settings.admin_password:
            raise HTTPException(status_code=400, detail="Auth disabled (no admin password configured)")
        if payload.username != settings.admin_username or payload.password != settings.admin_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = issue_session_token(cfg=auth_cfg, username=settings.admin_username)
        resp = JSONResponse({"status": "ok"})
        resp.set_cookie(
            key=auth_cfg.cookie_name,
            value=token,
            max_age=int(auth_cfg.cookie_max_age_s),
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp

    @app.post("/api/auth/logout")
    def logout() -> JSONResponse:
        resp = JSONResponse({"status": "ok"})
        resp.delete_cookie(key=auth_cfg.cookie_name, path="/")
        return resp

    @app.post("/api/admin/rescan")
    def rescan() -> Dict[str, str]:
        with refresh_lock:
            index.refresh()
        return {
            "status": "ok",
            "last_scan_utc": index.last_scan_utc or "",
            "session_count": str(len(index.list())),
        }

    @app.get("/api/admin/storage")
    def storage() -> Dict[str, Any]:
        sessions = index.list()
        disk = shutil.disk_usage(settings.data_dir)
        disk_health = _disk_health(
            disk,
            warning_used_pct=settings.disk_warning_used_pct,
            critical_used_pct=settings.disk_critical_used_pct,
        )
        clip_count = sum(_clip_count(s) for s in sessions)
        thumb_count = sum(1 for s in sessions if s.paths.thumbnail_jpg.exists())
        annotated_count = sum(1 for s in sessions if (s.paths.session_dir / "annotated.mp4").exists())
        return {
            "status": "ok",
            "data_dir": str(settings.data_dir),
            "db_path": str(settings.db_path),
            "last_scan_utc": index.last_scan_utc,
            "session_count": len(sessions),
            "clip_count": clip_count,
            "thumbnail_count": thumb_count,
            "annotated_count": annotated_count,
            "disk_total_bytes": int(disk.total),
            "disk_used_bytes": int(disk.used),
            "disk_free_bytes": int(disk.free),
            "disk_health": disk_health,
        }

    @app.get("/api/admin/ops")
    def ops() -> Dict[str, Any]:
        sessions = index.list()
        disk = shutil.disk_usage(settings.data_dir)
        disk_health = _disk_health(
            disk=disk,
            warning_used_pct=settings.disk_warning_used_pct,
            critical_used_pct=settings.disk_critical_used_pct,
        )
        now_ts = time.time()
        spool_root = settings.data_dir / "uploader_spool"
        cache_root = settings.data_dir / "_web_cache"
        reports_root = settings.data_dir / "reports"
        session_storage = _session_storage_breakdown(sessions)
        spool_pending = _spool_bucket_stats(spool_root / "pending", now_ts=now_ts)
        spool_done = _spool_bucket_stats(spool_root / "done", now_ts=now_ts)
        spool_dead = _spool_bucket_stats(spool_root / "dead", now_ts=now_ts)
        spool_pending_retry = _spool_pending_retry_stats(spool_root / "pending", now_ts=now_ts)
        spool_state_file = _spool_state_file_stats(spool_root, now_ts=now_ts)
        spool_total_bytes = int(spool_pending["bytes"]) + int(spool_done["bytes"]) + int(spool_dead["bytes"])
        spool_issues: List[str] = []
        dead_files = int(spool_dead["files"])
        pending_files = int(spool_pending["files"])
        if dead_files > 0:
            spool_issues.append("dead_tasks_present")
        if int(spool_pending_retry["invalid_payload_files"]) > 0:
            spool_issues.append("invalid_pending_payload")
        if bool(spool_state_file.get("parse_error")):
            spool_issues.append("state_file_parse_error")
        if spool_state_file.get("is_stale") is True:
            spool_issues.append("state_file_stale")
        if int(spool_pending_retry["retry_due_files"]) > 0:
            spool_issues.append("retry_due_backlog")
        if pending_files > 0 and dead_files <= 0:
            spool_issues.append("pending_backlog")

        spool_health_status = "ok"
        if dead_files > 0:
            spool_health_status = "error"
        elif len(spool_issues) > 0:
            spool_health_status = "warning"

        db_exists = settings.db_path.exists()
        db_size = 0
        db_mtime_utc: Optional[str] = None
        if db_exists:
            try:
                st = settings.db_path.stat()
                db_size = int(st.st_size)
                db_mtime_utc = _utc_iso_from_timestamp(float(st.st_mtime))
            except OSError:
                db_size = 0
                db_mtime_utc = None

        reports = _dir_stats(reports_root)
        cache = _dir_stats(cache_root)
        managed_total_files = (
            int(session_storage["total_files"])
            + int(reports["files"])
            + int(cache["files"])
            + (1 if db_exists else 0)
            + int(spool_pending["files"])
            + int(spool_done["files"])
            + int(spool_dead["files"])
        )
        managed_total_bytes = (
            int(session_storage["total_bytes"])
            + int(reports["bytes"])
            + int(cache["bytes"])
            + int(db_size)
            + int(spool_total_bytes)
        )
        managed_storage_pct_of_disk = (
            (float(managed_total_bytes) / float(disk.total) * 100.0)
            if int(disk.total) > 0
            else 0.0
        )

        return {
            "status": "ok",
            "last_scan_utc": index.last_scan_utc,
            "session_count": len(sessions),
            "disk": {
                "path": str(settings.data_dir),
                "total_bytes": int(disk.total),
                "used_bytes": int(disk.used),
                "free_bytes": int(disk.free),
                "health": disk_health,
            },
            "database": {
                "path": str(settings.db_path),
                "exists": db_exists,
                "bytes": db_size,
                "last_modified_utc": db_mtime_utc,
            },
            "managed_storage": {
                "total_files": int(managed_total_files),
                "total_bytes": int(managed_total_bytes),
                "percent_of_disk": round(float(managed_storage_pct_of_disk), 3),
                "sessions": session_storage,
                "reports": reports,
                "cache": cache,
                "database": {"files": 1 if db_exists else 0, "bytes": int(db_size)},
                "uploader_spool": {"files": int(spool_pending["files"]) + int(spool_done["files"]) + int(spool_dead["files"]), "bytes": int(spool_total_bytes)},
            },
            "reports": reports,
            "cache": cache,
            "uploader_spool": {
                "path": str(spool_root),
                "exists": spool_root.exists(),
                "pending": spool_pending,
                "done": spool_done,
                "dead": spool_dead,
                "pending_retry": spool_pending_retry,
                "state_file": spool_state_file,
                "health": {"status": spool_health_status, "issues": spool_issues},
            },
            "settings": {
                "auto_rescan_seconds": float(getattr(settings, "auto_rescan_seconds", 0.0) or 0.0),
                "auto_approve_done_enabled": settings.auto_approve_done_enabled,
                "auto_approve_min_duration_s": settings.auto_approve_min_duration_s,
                "disk_warning_used_pct": settings.disk_warning_used_pct,
                "disk_critical_used_pct": settings.disk_critical_used_pct,
            },
        }

    @app.post("/api/admin/storage/test")
    def storage_test() -> Dict[str, Any]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        marker = settings.data_dir / f".sop_storage_test_{uuid.uuid4().hex}.tmp"
        try:
            marker.write_bytes(b"ok\n")
            marker.unlink(missing_ok=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Storage test failed: {e}") from e
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> Dict[str, Any]:
        return {
            "data_dir": str(settings.data_dir),
            "db_path": str(settings.db_path),
            "last_scan_utc": index.last_scan_utc,
            "session_count": len(index.list()),
            "auto_rescan_seconds": float(getattr(settings, "auto_rescan_seconds", 0.0) or 0.0),
            "auto_approve_done_enabled": settings.auto_approve_done_enabled,
            "auto_approve_min_duration_s": settings.auto_approve_min_duration_s,
            "api_contract_version": API_CONTRACT_VERSION,
        }

    @app.get("/api/stats")
    def stats(
        *,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        sessions = index.list()
        sessions, applied_date_from, applied_date_to = _filter_sessions_by_date_window(
            sessions,
            date=date,
            date_from=date_from,
            date_to=date_to,
        )
        reviews = get_reviews_by_uid(settings.db_path, (s.session_uid for s in sessions))
        pending = 0
        approved = 0
        rejected = 0
        decided = 0
        human_reviewed = 0
        auto_approved = 0
        manual_overrides = 0
        manual_helmet_overrides = 0

        helmet_sessions = 0
        helmet_decided = 0
        machine_done = 0
        machine_not_done = 0
        machine_unknown = 0
        final_done = 0
        final_not_done = 0
        final_unknown = 0

        machine_sop_done = 0
        machine_sop_not_done = 0
        machine_sop_unknown = 0
        final_sop_done = 0
        final_sop_not_done = 0
        final_sop_unknown = 0

        for s in sessions:
            r = reviews.get(s.session_uid)
            sop_status = evaluate_sop_status(session=s, review=r)
            eff = _effective_review_for_web(session=s, review=r, settings=settings)
            status = eff.status
            if status == "QUALIFIED":
                approved += 1
                decided += 1
            elif status == "NOT_QUALIFIED":
                rejected += 1
                decided += 1
            else:
                pending += 1

            if r is not None:
                human_reviewed += 1
                if r.overrides:
                    manual_overrides += 1
            elif eff.source == "AUTO" and status == "QUALIFIED":
                auto_approved += 1

            if sop_status.profile == "operator_mvp_a":
                helmet_sessions += 1
                if status in {"QUALIFIED", "NOT_QUALIFIED"}:
                    helmet_decided += 1

                machine_helmet = sop_status.machine_helmet
                if machine_helmet == "DONE":
                    machine_done += 1
                elif machine_helmet == "NOT_DONE":
                    machine_not_done += 1
                else:
                    machine_unknown += 1

                final_helmet = sop_status.final_helmet
                override = r.overrides.get("helmet") if r is not None else None
                if isinstance(override, str) and override:
                    if final_helmet != machine_helmet:
                        manual_helmet_overrides += 1

                if final_helmet == "DONE":
                    final_done += 1
                elif final_helmet == "NOT_DONE":
                    final_not_done += 1
                else:
                    final_unknown += 1

            machine_sop = sop_status.machine_sop
            if machine_sop == "DONE":
                machine_sop_done += 1
            elif machine_sop == "NOT_DONE":
                machine_sop_not_done += 1
            else:
                machine_sop_unknown += 1

            final_sop = sop_status.final_sop
            if final_sop == "DONE":
                final_sop_done += 1
            elif final_sop == "NOT_DONE":
                final_sop_not_done += 1
            else:
                final_sop_unknown += 1

        total = len(sessions)
        review_completion_pct = (float(decided) * 100.0 / float(total)) if total > 0 else 0.0
        final_unknown_pct = (float(final_unknown) * 100.0 / float(helmet_sessions)) if helmet_sessions > 0 else 0.0
        reviewed_final_done_pct = (float(final_done) * 100.0 / float(helmet_decided)) if helmet_decided > 0 else 0.0
        final_sop_unknown_pct = (float(final_sop_unknown) * 100.0 / float(total)) if total > 0 else 0.0
        reviewed_final_sop_done_pct = (float(final_sop_done) * 100.0 / float(decided)) if decided > 0 else 0.0
        return {
            "total_sessions": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            # Keep `unknown` for compatibility with existing UI cards.
            "unknown": final_unknown,
            "reviewed": decided,
            "human_reviewed": human_reviewed,
            "auto_approved": auto_approved,
            "review_completion_pct": review_completion_pct,
            "manual_overrides": manual_overrides,
            "manual_helmet_overrides": manual_helmet_overrides,
            "helmet_session_count": helmet_sessions,
            "machine_helmet_done": machine_done,
            "machine_helmet_not_done": machine_not_done,
            "machine_helmet_unknown": machine_unknown,
            "final_helmet_done": final_done,
            "final_helmet_not_done": final_not_done,
            "final_helmet_unknown": final_unknown,
            "final_unknown_pct": final_unknown_pct,
            "reviewed_final_done_pct": reviewed_final_done_pct,
            "machine_sop_done": machine_sop_done,
            "machine_sop_not_done": machine_sop_not_done,
            "machine_sop_unknown": machine_sop_unknown,
            "final_sop_done": final_sop_done,
            "final_sop_not_done": final_sop_not_done,
            "final_sop_unknown": final_sop_unknown,
            "final_sop_unknown_pct": final_sop_unknown_pct,
            "reviewed_final_sop_done_pct": reviewed_final_sop_done_pct,
            "date_from": applied_date_from,
            "date_to": applied_date_to,
        }

    def _filtered_session_rows(
        *,
        date: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        review_status: Optional[Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"]],
        evidence: Literal["ANY", "CLIP_THUMB", "CLIP_ONLY", "THUMB_ONLY"],
        shift: str,
        sort: Literal["NEWEST", "OLDEST", "MACHINE_UNKNOWN_FIRST", "PENDING_FIRST"],
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str], str]:
        sessions = index.list()
        sessions, applied_date_from, applied_date_to = _filter_sessions_by_date_window(
            sessions,
            date=date,
            date_from=date_from,
            date_to=date_to,
        )
        reviews = get_reviews_by_uid(settings.db_path, (s.session_uid for s in sessions))
        shift_filter = _normalize_shift_filter(shift)
        out: List[Dict[str, Any]] = []
        for s in sessions:
            r = reviews.get(s.session_uid)
            eff = _effective_review_for_web(session=s, review=r, settings=settings)
            rs = eff.status
            if review_status and rs != review_status:
                continue

            has_thumbnail = s.paths.thumbnail_jpg.exists()
            clip_count = _clip_count(s)
            if not _matches_evidence_filter(
                clip_count=clip_count,
                has_thumbnail=has_thumbnail,
                evidence_filter=evidence,
            ):
                continue

            start_iso = s.checklist.get("start_time_iso")
            end_iso = s.checklist.get("end_time_iso")
            start_s = float(s.checklist.get("start_time_s") or 0.0)
            end_s = float(s.checklist.get("end_time_s") or 0.0)
            duration_s = max(0.0, end_s - start_s)
            start_ts = _parse_iso_ts(start_iso) or _parse_iso_ts(end_iso)
            shift_fields = {}
            if (
                not isinstance(s.checklist.get("shift_id"), str)
                or not s.checklist.get("shift_id")
                or not isinstance(s.checklist.get("shift_date"), str)
                or not s.checklist.get("shift_date")
            ):
                shift_fields = _shift_fields_from_isos(start_iso=start_iso, end_iso=end_iso)
            resolved_shift_id = str(s.checklist.get("shift_id") or shift_fields.get("shift_id") or "").strip().upper()
            if shift_filter != "ALL" and resolved_shift_id != shift_filter:
                continue

            sop_status = evaluate_sop_status(session=s, review=r)

            out.append(
                {
                    "session_uid": s.session_uid,
                    "date": s.date,
                    "session_id": s.session_id,
                    "start_time_iso": start_iso,
                    "end_time_iso": end_iso,
                    "shift_id": str(s.checklist.get("shift_id") or shift_fields.get("shift_id") or ""),
                    "shift_name": str(s.checklist.get("shift_name") or shift_fields.get("shift_name") or ""),
                    "shift_date": str(s.checklist.get("shift_date") or shift_fields.get("shift_date") or ""),
                    "duration_s": duration_s,
                    "machine_helmet": sop_status.machine_helmet,
                    "machine_sop": sop_status.machine_sop,
                    "machine_roi_dwell": sop_status.machine_roi_dwell,
                    "review_status": rs,
                    "review_source": eff.source,
                    "final_helmet": sop_status.final_helmet,
                    "final_sop": sop_status.final_sop,
                    "sop": sop_status.summary,
                    "has_thumbnail": has_thumbnail,
                    "thumbnail_url": f"/media/{s.session_uid}/thumbnail.jpg" if has_thumbnail else None,
                    "clip_count": clip_count,
                    "first_clip_url": (
                        f"/media/{s.session_uid}/{_first_clip_rel_file(s)}" if _first_clip_rel_file(s) else None
                    ),
                    "_sort_start_ts": start_ts,
                }
            )

        if sort == "OLDEST":
            out.sort(key=lambda x: (float(x.get("_sort_start_ts") or 0.0), str(x.get("session_uid") or "")))
        elif sort == "MACHINE_UNKNOWN_FIRST":
            out.sort(
                key=lambda x: (
                    0 if str(x.get("machine_sop") or "").upper() == "UNKNOWN" else 1,
                    -float(x.get("_sort_start_ts") or 0.0),
                    str(x.get("session_uid") or ""),
                )
            )
        elif sort == "PENDING_FIRST":
            out.sort(
                key=lambda x: (
                    0 if str(x.get("review_status") or "").upper() == "PENDING" else 1,
                    -float(x.get("_sort_start_ts") or 0.0),
                    str(x.get("session_uid") or ""),
                )
            )
        else:
            out.sort(key=lambda x: (-float(x.get("_sort_start_ts") or 0.0), str(x.get("session_uid") or "")))

        return out, applied_date_from, applied_date_to, shift_filter

    @app.get("/api/sessions")
    def list_sessions(
        *,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        review_status: Optional[Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"]] = None,
        evidence: Literal["ANY", "CLIP_THUMB", "CLIP_ONLY", "THUMB_ONLY"] = Query(default="ANY"),
        shift: str = Query(default="ALL"),
        sort: Literal["NEWEST", "OLDEST", "MACHINE_UNKNOWN_FIRST", "PENDING_FIRST"] = Query(default="NEWEST"),
        page: int = Query(default=1, ge=1),
        page_size: Optional[int] = Query(default=None, ge=1, le=2000),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> Dict[str, Any]:
        out, applied_date_from, applied_date_to, shift_filter = _filtered_session_rows(
            date=date,
            date_from=date_from,
            date_to=date_to,
            review_status=review_status,
            evidence=evidence,
            shift=shift,
            sort=sort,
        )

        effective_page_size = int(page_size) if page_size is not None else int(limit)
        total = len(out)
        start_idx = (int(page) - 1) * effective_page_size
        end_idx = start_idx + effective_page_size
        page_rows = out[start_idx:end_idx]
        total_pages = ((total + effective_page_size - 1) // effective_page_size) if total > 0 else 0
        has_prev = total > 0 and page > 1
        has_next = end_idx < total

        for row in page_rows:
            row.pop("_sort_start_ts", None)
        return {
            "sessions": page_rows,
            "last_scan_utc": index.last_scan_utc,
            "date_from": applied_date_from,
            "date_to": applied_date_to,
            "evidence": evidence,
            "shift": shift_filter,
            "total": total,
            "page": int(page),
            "page_size": effective_page_size,
            "total_pages": total_pages,
            "has_prev": has_prev,
            "has_next": has_next,
        }

    #export csv capabilites
    @app.get("/api/sessions/export.csv")
    def export_sessions_csv(
        *,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        review_status: Optional[Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"]] = None,
        evidence: Literal["ANY", "CLIP_THUMB", "CLIP_ONLY", "THUMB_ONLY"] = Query(default="ANY"),
        shift: str = Query(default="ALL"),
        sort: Literal["NEWEST", "OLDEST", "MACHINE_UNKNOWN_FIRST", "PENDING_FIRST"] = Query(default="NEWEST"),
    ) -> Response:

        """
        Function for expoerting csv from sessions data
        that is filtered to match the data from review queue page.
        """
        rows, _, _, _ = _filtered_session_rows(
            date=date,
            date_from=date_from,
            date_to=date_to,
            review_status=review_status,
            evidence=evidence,
            shift=shift,
            sort=sort,
        )

        #write headers for the output
        headers = [
            "session_uid",
            "session_id",
            "date",
            "shift",
            "start_time",
            "end_time",
            "duration_s",
            "machine_sop",
            "review_status",
            "review_source",
            "final_sop",
            "machine_cleaned",
            "machine_labeled",
            "machine_overall_status",
            "final_cleaned",
            "final_labeled",
            "final_overall_status",
            "clip_count",
            "has_thumbnail",
        ]

        #create a temporary object files
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            sop = row.get("sop") if isinstance(row.get("sop"), dict) else {}
            machine = sop.get("machine") if isinstance(sop.get("machine"), dict) else {}
            final = sop.get("final") if isinstance(sop.get("final"), dict) else {}
            shift_label = str(row.get("shift_name") or row.get("shift_id") or "")
            #write in csv
            writer.writerow(
                {
                    "session_uid": row.get("session_uid"),
                    "session_id": row.get("session_id"),
                    "date": row.get("date"),
                    "shift": shift_label,
                    "start_time": row.get("start_time_iso"),
                    "end_time": row.get("end_time_iso"),
                    "duration_s": round(float(row.get("duration_s") or 0.0), 3),
                    "machine_sop": row.get("machine_sop"),
                    "review_status": row.get("review_status"),
                    "review_source": row.get("review_source"),
                    "final_sop": row.get("final_sop"),
                    "machine_cleaned": machine.get("cleaned"),
                    "machine_labeled": machine.get("labeled"),
                    "machine_overall_status": machine.get("overall_status"),
                    "final_cleaned": final.get("cleaned"),
                    "final_labeled": final.get("labeled"),
                    "final_overall_status": final.get("overall_status"),
                    "clip_count": row.get("clip_count"),
                    "has_thumbnail": row.get("has_thumbnail"),
                }
            )

        #return it as a object response
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="sop_review_queue.csv"'},
        )

    @app.get("/api/sessions/{session_uid}")
    def get_session(session_uid: str) -> Dict[str, Any]:
        _validate_session_uid(session_uid)
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        r = get_review(settings.db_path, session_uid)
        eff = _effective_review_for_web(session=s, review=r, settings=settings)
        sop_status = evaluate_sop_status(session=s, review=r)

        start_iso = s.checklist.get("start_time_iso")
        end_iso = s.checklist.get("end_time_iso")
        shift_fields = {}
        if (
            not isinstance(s.checklist.get("shift_id"), str)
            or not s.checklist.get("shift_id")
            or not isinstance(s.checklist.get("shift_date"), str)
            or not s.checklist.get("shift_date")
        ):
            shift_fields = _shift_fields_from_isos(start_iso=start_iso, end_iso=end_iso)

        clips: List[Dict[str, Any]] = []
        if isinstance(s.evidence.get("clips"), list):
            for clip in s.evidence["clips"]:
                if not isinstance(clip, dict):
                    continue
                rel_file = clip.get("file")
                if not isinstance(rel_file, str) or not rel_file:
                    continue
                rel_file = rel_file.replace("\\", "/")
                clips.append(
                    {
                        **clip,
                        "url": f"/media/{s.session_uid}/{rel_file}",
                        "playback_url": f"/media-playback/{s.session_uid}/{rel_file}",
                    }
                )

        artifacts: List[Dict[str, str]] = []
        for rel in ("checklist.json", "run_config.json", "thumbnail.jpg", "evidence.json", "annotated.mp4"):
            p = s.paths.session_dir / rel
            if p.exists():
                artifacts.append({"name": rel, "url": f"/media/{s.session_uid}/{rel}"})
        if (s.paths.session_dir / "evidence").exists():
            for mp4 in sorted((s.paths.session_dir / "evidence").glob("*.mp4")):
                rel = str(mp4.relative_to(s.paths.session_dir))
                artifacts.append({"name": rel, "url": f"/media/{s.session_uid}/{rel}"})

        annotated_path = s.paths.session_dir / "annotated.mp4"
        has_annotated = annotated_path.exists()
        return {
            "session_uid": s.session_uid,
            "date": s.date,
            "session_id": s.session_id,
            "checklist": s.checklist,
            "shift_id": str(s.checklist.get("shift_id") or shift_fields.get("shift_id") or ""),
            "shift_name": str(s.checklist.get("shift_name") or shift_fields.get("shift_name") or ""),
            "shift_date": str(s.checklist.get("shift_date") or shift_fields.get("shift_date") or ""),
            "machine_helmet": sop_status.machine_helmet,
            "machine_sop": sop_status.machine_sop,
            "machine_roi_dwell": sop_status.machine_roi_dwell,
            "final_helmet": sop_status.final_helmet,
            "final_sop": sop_status.final_sop,
            "sop": sop_status.summary,
            "review_status": eff.status,
            "review_source": eff.source,
            "auto_review_reason": eff.auto_reason,
            "review": None if r is None else r.__dict__,
            "clips": clips,
            "has_thumbnail": s.paths.thumbnail_jpg.exists(),
            "thumbnail_url": f"/media/{s.session_uid}/thumbnail.jpg" if s.paths.thumbnail_jpg.exists() else None,
            "has_annotated": has_annotated,
            "annotated_url": f"/media/{s.session_uid}/annotated.mp4" if has_annotated else None,
            "annotated_playback_url": f"/media-playback/{s.session_uid}/annotated.mp4" if has_annotated else None,
            "artifacts": artifacts,
        }

    @app.put("/api/sessions/{session_uid}")
    def put_session(session_uid: str, payload: SessionUpsertIn) -> Dict[str, Any]:
        if payload.session_uid != session_uid:
            raise HTTPException(status_code=400, detail="session_uid mismatch")

        date = _storage_date(payload)
        _validate_session_uid(session_uid)
        _validate_session_id(payload.session_id)
        if payload.start_date:
            _validate_date_ymd(payload.start_date)
        if payload.end_date:
            _validate_date_ymd(payload.end_date)
        _validate_date_ymd(date)

        # In ingestion mode, store by UID to avoid collisions across devices/runs.
        session_dir = settings.data_dir / "sessions" / date / session_uid
        session_dir.mkdir(parents=True, exist_ok=True)

        checklist_path = session_dir / "checklist.json"
        try:
            checklist_payload = normalize_session_checklist_payload(payload.model_dump(mode="json"))
        except ReviewOverrideError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        _atomic_write_json(checklist_path, checklist_payload)

        index.refresh()
        return {
            "status": "ok",
            "session_uid": session_uid,
            "date": date,
            "session_id": payload.session_id,
        }

    @app.put("/api/sessions/{session_uid}/review")
    def put_review(session_uid: str, payload: ReviewUpsertIn) -> Dict[str, Any]:
        _validate_session_uid(session_uid)
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        _validate_review_note(payload.review_note)
        validated_overrides = _validate_review_overrides(checklist=s.checklist, raw=payload.overrides)
        rec = upsert_review(
            db_path=settings.db_path,
            session_uid=session_uid,
            review_status=payload.review_status,
            review_note=payload.review_note,
            overrides=validated_overrides,
        )
        return {"review": rec.__dict__}

    @app.post("/api/sessions/{session_uid}/artifacts")
    async def post_artifact(
        session_uid: str,
        request: Request,
        rel_path: str = Query(..., description="Relative path under the session dir, e.g. evidence/helmet_done_01.mp4"),
    ) -> Dict[str, Any]:
        _validate_session_uid(session_uid)
        s = index.get(session_uid)
        if s is None:
            # Allow for eventual consistency (client may upload fast after upsert).
            index.refresh()
            s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found (upsert first)")

        rel = _safe_rel_path(rel_path)
        base = s.paths.session_dir.resolve()
        target = (s.paths.session_dir / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid rel_path") from e

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as f:
            async for chunk in request.stream():
                if chunk:
                    f.write(chunk)
        tmp.replace(target)

        # Refresh index so evidence/thumbnail manifests show immediately.
        index.refresh()

        rel_url = str(rel).replace("\\", "/")
        return {
            "status": "ok",
            "rel_path": rel_url,
            "url": f"/media/{session_uid}/{rel_url}",
        }

    @app.get("/media/{session_uid}/{rel_path:path}", include_in_schema=False)
    def media(session_uid: str, rel_path: str) -> FileResponse:
        _validate_session_uid(session_uid)
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        target = _safe_session_dir(settings.data_dir, s, rel_path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(target))

    @app.get("/media-playback/{session_uid}/{rel_path:path}", include_in_schema=False)
    def media_playback(session_uid: str, rel_path: str) -> FileResponse:
        _validate_session_uid(session_uid)
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        target = _safe_session_dir(settings.data_dir, s, rel_path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        normalized_rel = rel_path.replace("\\", "/")
        if target.suffix.lower() != ".mp4":
            return FileResponse(str(target))

        playback = _ensure_browser_playback_path(
            settings=settings,
            session_uid=session_uid,
            rel_path=normalized_rel,
            src_path=target,
        )
        return FileResponse(str(playback))

    return app

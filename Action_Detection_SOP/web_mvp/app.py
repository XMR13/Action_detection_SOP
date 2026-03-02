from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import BasicAuthConfig, BasicAuthMiddleware
from .review_store import ReviewRecord, get_review, get_reviews_by_uid, init_db, upsert_review
from .session_index import SessionArtifact, SessionIndex
from .settings import WebMvpSettings


def _display_status(value: str) -> str:
    # Render "NOT_DONE" as "NOT DONE" to match UI labels.
    return value.replace("_", " ").upper() if value else "-"


def _machine_helmet_status(session: SessionArtifact) -> str:
    helmet = session.checklist.get("helmet")
    return str(helmet) if isinstance(helmet, str) and helmet else "UNKNOWN"


def _machine_roi_status(session: SessionArtifact) -> str:
    roi = session.checklist.get("roi_dwell")
    return str(roi) if isinstance(roi, str) and roi else "UNKNOWN"


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


class ReviewUpsertIn(BaseModel):
    review_status: Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"] = Field(...)
    review_note: str = Field(default="")
    overrides: Dict[str, Any] = Field(default_factory=dict)


def create_app(settings: WebMvpSettings) -> FastAPI:
    init_db(settings.db_path)
    index = SessionIndex(data_dir=settings.data_dir)

    app = FastAPI(title="SOP Review MVP", version="0.1.0")
    app.add_middleware(
        BasicAuthMiddleware,
        cfg=BasicAuthConfig(password=settings.admin_password),
    )

    ui_dir = settings.ui_dir
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    @app.on_event("startup")
    def _startup() -> None:
        index.refresh()

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/ui/index.html")

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/admin/rescan")
    def rescan() -> Dict[str, str]:
        index.refresh()
        return {
            "status": "ok",
            "last_scan_utc": index.last_scan_utc or "",
            "session_count": str(len(index.list())),
        }

    @app.get("/api/config")
    def config() -> Dict[str, Any]:
        return {
            "data_dir": str(settings.data_dir),
            "db_path": str(settings.db_path),
            "last_scan_utc": index.last_scan_utc,
            "session_count": len(index.list()),
        }

    @app.get("/api/stats")
    def stats() -> Dict[str, Any]:
        sessions = index.list()
        reviews = get_reviews_by_uid(settings.db_path, (s.session_uid for s in sessions))
        pending = 0
        approved = 0
        rejected = 0
        unknown = 0
        for s in sessions:
            r = reviews.get(s.session_uid)
            status = "PENDING" if r is None else r.review_status
            if status == "QUALIFIED":
                approved += 1
            elif status == "NOT_QUALIFIED":
                rejected += 1
            else:
                pending += 1
                unknown += 1
        return {
            "total_sessions": len(sessions),
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "unknown": unknown,
        }

    @app.get("/api/sessions")
    def list_sessions(
        *,
        date: Optional[str] = None,
        review_status: Optional[Literal["QUALIFIED", "NOT_QUALIFIED", "PENDING"]] = None,
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> Dict[str, Any]:
        sessions = index.list()
        if date:
            sessions = [s for s in sessions if s.date == date]
        reviews = get_reviews_by_uid(settings.db_path, (s.session_uid for s in sessions))
        out: List[Dict[str, Any]] = []
        for s in sessions:
            r = reviews.get(s.session_uid)
            rs = "PENDING" if r is None else r.review_status
            if review_status and rs != review_status:
                continue

            start_iso = s.checklist.get("start_time_iso")
            end_iso = s.checklist.get("end_time_iso")
            start_s = float(s.checklist.get("start_time_s") or 0.0)
            end_s = float(s.checklist.get("end_time_s") or 0.0)
            duration_s = max(0.0, end_s - start_s)

            machine = _machine_helmet_status(s)
            final = machine
            if r is not None:
                override = r.overrides.get("helmet")
                if isinstance(override, str) and override:
                    final = override

            out.append(
                {
                    "session_uid": s.session_uid,
                    "date": s.date,
                    "session_id": s.session_id,
                    "start_time_iso": start_iso,
                    "end_time_iso": end_iso,
                    "duration_s": duration_s,
                    "machine_helmet": machine,
                    "machine_roi_dwell": _machine_roi_status(s),
                    "review_status": rs,
                    "final_helmet": final,
                    "has_thumbnail": s.paths.thumbnail_jpg.exists(),
                    "thumbnail_url": f"/media/{s.session_uid}/thumbnail.jpg" if s.paths.thumbnail_jpg.exists() else None,
                    "clip_count": _clip_count(s),
                    "first_clip_url": (
                        f"/media/{s.session_uid}/{_first_clip_rel_file(s)}" if _first_clip_rel_file(s) else None
                    ),
                }
            )
            if len(out) >= limit:
                break
        return {"sessions": out, "last_scan_utc": index.last_scan_utc}

    @app.get("/api/sessions/{session_uid}")
    def get_session(session_uid: str) -> Dict[str, Any]:
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        r = get_review(settings.db_path, session_uid)
        machine_helmet = _machine_helmet_status(s)
        final_helmet = machine_helmet
        if r is not None:
            override = r.overrides.get("helmet")
            if isinstance(override, str) and override:
                final_helmet = override

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
                    }
                )

        artifacts: List[Dict[str, str]] = []
        for rel in ("checklist.json", "run_config.json", "thumbnail.jpg", "evidence.json"):
            p = s.paths.session_dir / rel
            if p.exists():
                artifacts.append({"name": rel, "url": f"/media/{s.session_uid}/{rel}"})
        if (s.paths.session_dir / "evidence").exists():
            for mp4 in sorted((s.paths.session_dir / "evidence").glob("*.mp4")):
                rel = str(mp4.relative_to(s.paths.session_dir))
                artifacts.append({"name": rel, "url": f"/media/{s.session_uid}/{rel}"})

        return {
            "session_uid": s.session_uid,
            "date": s.date,
            "session_id": s.session_id,
            "checklist": s.checklist,
            "machine_helmet": machine_helmet,
            "machine_roi_dwell": _machine_roi_status(s),
            "final_helmet": final_helmet,
            "review": None if r is None else r.__dict__,
            "clips": clips,
            "has_thumbnail": s.paths.thumbnail_jpg.exists(),
            "thumbnail_url": f"/media/{s.session_uid}/thumbnail.jpg" if s.paths.thumbnail_jpg.exists() else None,
            "artifacts": artifacts,
        }

    @app.put("/api/sessions/{session_uid}/review")
    def put_review(session_uid: str, payload: ReviewUpsertIn) -> Dict[str, Any]:
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        rec = upsert_review(
            db_path=settings.db_path,
            session_uid=session_uid,
            review_status=payload.review_status,
            review_note=payload.review_note,
            overrides=payload.overrides,
        )
        return {"review": rec.__dict__}

    @app.get("/media/{session_uid}/{rel_path:path}", include_in_schema=False)
    def media(session_uid: str, rel_path: str) -> FileResponse:
        s = index.get(session_uid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        target = _safe_session_dir(settings.data_dir, s, rel_path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(target))

    return app

# Action Detection SOP (Roll Wrapping)

On-prem computer vision pipeline for SOP compliance in roll wrapping, optimized for Jetson deployment.

## Scope (current MVP)

- Sessionization by person-in-ROI.
- SOP checks: `operator_present`, `roi_dwell`, `helmet`.
- Filesystem-first outputs (`data/sessions`, `data/reports`) + FastAPI review website.
- Edge-to-web sync via uploader with persistent offline spool, retry backoff, and watch mode.
- Stable `session_uid` in session artifacts for idempotent review and upload flows.

## Current Status

- Done: MVP runner, evidence clips, review website viewer/ingestion modes, uploader sync, config-file runs, golden-set runner.
- Partial: supervised PPE dataset reproducibility, official PPE deployment model naming, roll-session pipeline integration.
- Next: ops hardening (`M5.5`) for monitoring, retention, backup/restore, and handover docs.
- Deferred: reviewer identity + audit trail (`M5.4`) until the deployment is no longer single-user.

## Requirements

- Python `3.10`
- Dependency manager: `uv`
- Deployment target: Jetson Orin NX (JetPack `6.x+`)

## Quick Start

1. Prepare ROI (once per camera):
   - `python3 -m Scripts.calibrate_roi --video path/to/video.mp4 --out configs/roi.json`
2. Run SOP MVP:
   - `python3 -m Scripts.run_sop_mvp --video path/to/video.mp4 --roi configs/roi.json --model Models/your_ppe_model.onnx --metadata configs/metadata_PPE.yaml`
3. Optional config-file mode:
   - `cp configs/run_sop_mvp.example.json configs/run_sop_mvp.json`
   - `python3 -m Scripts.run_sop_mvp --config configs/run_sop_mvp.json`

## Output Paths

- Session artifacts: `data/sessions/YYYY-MM-DD/session_<id>/`
- Daily report: `data/reports/YYYY-MM-DD/`
- Evidence clips: `data/sessions/YYYY-MM-DD/session_<id>/evidence/*.mp4`
- Uploader spool: `data/uploader_spool/{pending,done,dead}/`

Each `checklist.json` includes a generated `session_uid`, which is the primary key used by the web API and uploader.

## Web Review MVP

- Set credentials:
  - `SOP_ADMIN_USERNAME=admin`
  - `SOP_ADMIN_PASSWORD=your_password`
- Run web server:
  - `uv run python -m Scripts.run_web_mvp --host 0.0.0.0 --port 8000 --data-dir data`
- Open:
  - `http://<WEB_SERVER_IP>:8000/`

Current web behavior:
- `/` redirects to `/ui/index.html`
- viewer mode indexes local disk artifacts
- ingestion mode accepts uploader/API pushes into UID-based session folders
- auth is single-admin for now; per-reviewer audit history is not implemented yet
- ops visibility is available via `/api/admin/ops`

## Jetson Sync (Uploader)

- One-shot upload:
  - `python3 -m Scripts.sop_uploader --server http://<WEB_SERVER_IP>:8000 --data-dir data`
- Continuous sync (recommended):
  - `python3 -m Scripts.sop_uploader --server http://<WEB_SERVER_IP>:8000 --data-dir data --watch --poll-s 5 --retry-wait-s 2 --retry-backoff 2 --retry-wait-max-s 120 --max-attempts 0`
- Spool folders:
  - `data/uploader_spool/pending`
  - `data/uploader_spool/done`
  - `data/uploader_spool/dead`

Uploader notes:
- upserts `checklist.json` first, then uploads artifacts
- retries transient failures with exponential backoff
- leaves exhausted tasks in `dead/` for operator follow-up

## Operations

- Ops summary endpoint:
  - `GET /api/admin/ops`
- Retention cleanup dry-run:
  - `python3 -m Scripts.cleanup_retention --data-dir data`
- Retention cleanup apply:
  - `python3 -m Scripts.cleanup_retention --data-dir data --apply`

## Key References

- API contract: `docs/web_mvp_api_contract.md`
- Operations runbook: `docs/operations_runbook.md`
- Project roadmap: `plan.md`

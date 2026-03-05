# Action Detection SOP (Roll Wrapping)

On-prem computer vision pipeline for SOP compliance in roll wrapping, optimized for Jetson deployment.

## Scope (current MVP)

- Sessionization by person-in-ROI.
- SOP checks: `operator_present`, `roi_dwell`, `helmet`.
- Filesystem-first outputs (`data/sessions`, `data/reports`) + review website MVP.
- Edge-to-web sync via uploader with persistent offline spool + retry backoff.

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

## Web Review MVP

- Set credentials:
  - `SOP_ADMIN_USERNAME=admin`
  - `SOP_ADMIN_PASSWORD=your_password`
- Run web server:
  - `uv run python -m Scripts.run_web_mvp --host 0.0.0.0 --port 8000 --data-dir data`
- Open:
  - `http://<AI_BOX_IP>:8000/`

## Jetson Sync (Uploader)

- One-shot upload:
  - `python3 -m Scripts.sop_uploader --server http://<AI_BOX_IP>:8000 --data-dir data`
- Continuous sync (recommended):
  - `python3 -m Scripts.sop_uploader --server http://<AI_BOX_IP>:8000 --data-dir data --watch --poll-s 5 --retry-wait-s 2 --retry-backoff 2 --retry-wait-max-s 120 --max-attempts 0`
- Spool folders:
  - `data/uploader_spool/pending`
  - `data/uploader_spool/done`
  - `data/uploader_spool/dead`

## Key References

- API contract: `docs/web_mvp_api_contract.md`
- Project roadmap: `plan.md`

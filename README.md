# Action Detection SOP (Roll Wrapping)

On-prem computer vision pipeline for roll-wrapping SOP review, built for Jetson deployment and a filesystem-first review workflow.

## Current State

- Working MVP: person-in-ROI sessions, helmet checks, evidence clips, daily reports, FastAPI review website, and Jetson-to-web uploader.
- Next profile: `roll_sop_v1`, a roll-centric SOP flow for the moved CCTV viewpoint.
- Current `roll_sop_v1` target checks: `cleaned`, `labeled`, and `overall_status`.
- Separate safety flow: helmet alerts are independent website alert records, not part of roll `overall_status`.

## Requirements

- Python `3.10`
- Dependency manager: `uv`
- Deployment target: Jetson Orin NX / JetPack `6.x+`

### Jetson runtime environment

Laptop development uses `uv`, but the Jetson runtime must keep the CUDA,
TensorRT, NumPy, OpenCV, and CUDA Python packages supplied for its JetPack
installation. Do not run `uv sync` or install this project's desktop
`pyproject.toml` on the Jetson.

After cloning the repository on the Jetson, create a Python 3.10 virtual
environment that can see the JetPack system packages:

```bash
bash Scripts/setup_jetson_env.sh
source .venv/bin/activate
```

The setup script uses `python3 -m venv --system-site-packages`, does not install
packages, and runs `Scripts/check_jetson_runtime.py`. The check requires Python
3.10, NumPy, OpenCV, TensorRT, CUDA Python runtime bindings, and `trtexec` to be
visible inside the environment. `ffmpeg` is reported as optional unless checked
with `--require-ffmpeg`.

To recheck an existing Jetson environment without recreating it:

```bash
source .venv/bin/activate
python -m Scripts.check_jetson_runtime
```

`requirements-jetson.txt` intentionally contains no NVIDIA/CV packages. Add
only application-level dependencies that have been shown to be missing on the
target; do not use it to replace packages managed by JetPack.

The model and machine-specific files are git-ignored and must be transferred
separately. For the current roll SOP bring-up, provide the candidate ONNX model,
matching `configs/metadata_roll_sop_v1.yaml`, recorded validation footage, and
later the production ROI/runtime config. Build the `.engine` on the target
Jetson rather than transferring an engine built on the laptop.

## Important Files

- Runner: `Scripts/run_sop_mvp.py`
- Web review server: `Scripts/run_web_mvp.py`
- Uploader: `Scripts/sop_uploader.py`
- Frame extraction: `Scripts/extract_frames_for_labeling.py`
- Auto pre-annotation: `Scripts/auto_annotate_person_coco.py`
- Current PPE metadata: `configs/metadata_PPE.yaml`
- Next roll SOP metadata: `configs/metadata_roll_sop_v1.yaml`
- Roadmap: `plan.md`
- Domain terms: `CONTEXT.md`

## Current MVP Run

Prepare ROI once per camera:

```bash
python3 -m Scripts.calibrate_roi \
  --video path/to/video.mp4 \
  --out configs/roi.json
```

Run the existing SOP MVP:

```bash
python3 -m Scripts.run_sop_mvp \
  --video path/to/video.mp4 \
  --roi configs/roi.json \
  --model Models/yolo10s-PPE.onnx \
  --metadata configs/metadata_PPE.yaml
```

Config-file mode is preferred for repeatable runs:

```bash
python3 -m Scripts.run_sop_mvp --config configs/run_sop_mvp.json
```

## Labeling Workflow

Extract screenshots from long videos at a coarse rate first:

```bash
python3 -m Scripts.extract_frames_for_labeling \
  --video path/to/video_01.mp4 \
  --out-dir datasets/roll_sop_v1/frames/video_01 \
  --target-fps 0.2 \
  --dedupe-threshold 2.0
```

Pre-annotate `person` and `helmet` with the current PPE model:

```bash
python3 -m Scripts.auto_annotate_person_coco \
  --images-dir datasets/roll_sop_v1/frames/video_01 \
  --out datasets/roll_sop_v1/frames/video_01/auto_coco.json \
  --model Models/yolo10s-PPE.onnx \
  --metadata configs/metadata_PPE.yaml \
  --label person \
  --label helmet
```

In CVAT, manually add the remaining classes:

- `roll`
- `cleaning_cloth`
- `label`

The next training contract is:

```text
datasets/roll_sop_v1/
Models/roll_sop_v1.onnx
configs/metadata_roll_sop_v1.yaml
```

## Output Paths

- Session artifacts: `data/sessions/YYYY-MM-DD/session_<id>/`
- Daily reports: `data/reports/YYYY-MM-DD/`
- Evidence clips: `data/sessions/YYYY-MM-DD/session_<id>/evidence/*.mp4`
- Uploader spool: `data/uploader_spool/{pending,done,dead}/`
- Helmet alerts: `data/alerts/YYYY-MM-DD/<alert_uid>/`

Each session `checklist.json` includes `session_uid`, used by the web API and uploader for idempotent sync.

## Web Review

Run the review website:

```bash
SOP_ADMIN_USERNAME=admin \
SOP_ADMIN_PASSWORD=your_password \
uv run python -m Scripts.run_web_mvp --host 0.0.0.0 --port 8000 --data-dir data
```

Open:

```text
http://<WEB_SERVER_IP>:8000/
```

## Jetson Sync

One-shot upload:

```bash
python3 -m Scripts.sop_uploader \
  --server http://<WEB_SERVER_IP>:8000 \
  --data-dir data
```

Continuous sync:

```bash
python3 -m Scripts.sop_uploader \
  --server http://<WEB_SERVER_IP>:8000 \
  --data-dir data \
  --watch \
  --poll-s 5 \
  --retry-wait-s 2 \
  --retry-backoff 2 \
  --retry-wait-max-s 120 \
  --max-attempts 0
```

## Operations

For a same-host live pilot, install the RTSP worker and review website as
systemd services using [deploy/systemd/README.md](deploy/systemd/README.md).
The installer keeps RTSP and website credentials outside the unit files and
does not modify Python dependencies or lockfiles.

Ops summary endpoint:

```text
GET /api/admin/ops
```

The ops payload includes disk health fields:

- `disk.health.status`: `ok`, `warning`, or `critical`
- default warning threshold: `disk.health.used_pct >= 75`
- default critical threshold: `disk.health.used_pct >= 85`
- tune with `--disk-warning-used-pct` / `--disk-critical-used-pct`

Retention cleanup:
This is for running the retention_cleanup script :
```bash
python3 -m Scripts.cleanup_retention --config configs/retention.yaml
python3 -m Scripts.cleanup_retention --config configs/retention.yaml --apply
```

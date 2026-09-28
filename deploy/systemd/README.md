# Systemd services — same-host live pilot

This deployment slice runs the TensorRT RTSP worker and FastAPI review website
on one machine with one shared data directory:

```text
RTSP camera -> action-sop-rtsp.service -> shared data -> action-sop-web.service -> browser
```

This is the fastest two-service pilot. The planned separate web-server topology
also needs `Scripts.sop_uploader` on the Jetson; these two units alone do not
transfer artifacts between machines.

## Install

The Jetson environment and TensorRT engine must already be working. From the
repository root:

```bash
sudo bash Scripts/install_systemd_services.sh \
  --service-user "$(id -un)"
```

The installer:

- renders units into `/etc/systemd/system/`
- creates protected configuration under `/etc/action-sop/`
- preserves existing environment files on reinstall
- does not enable or start either service until explicitly requested

No Python packages, `pyproject.toml`, or `uv.lock` are changed.

## Configure

Edit the RTSP source, friendly camera label, engine, metadata, and ROI:

```bash
sudoedit /etc/action-sop/rtsp.env
```

Set `SOP_HELMET_ALERT_CAMERA_ID` to the operator-facing camera name (for
example, `Camera 16 RW3`). Keep the RTSP URL in `SOP_RTSP_URL`; it is not a
display label.

Edit the website credentials and bind settings:

```bash
sudoedit /etc/action-sop/web.env
```

Both services read the same data root from:

```bash
sudoedit /etc/action-sop/common.env
```

Keep the RTSP URL and web password out of shell history and unit files. The
installer creates the environment files as root-owned and service-group-readable
with mode `0640`.

After configuration, rerun the installer with `--start`. Existing environment
files are preserved; the command validates placeholder credentials, enables
both units for boot, starts the web service, then starts the RTSP worker:

```bash
sudo bash Scripts/install_systemd_services.sh \
  --service-user "$(id -un)" \
  --start
```

## Start and verify

Validate the rendered unit syntax on the target machine:

```bash
sudo systemd-analyze verify \
  /etc/systemd/system/action-sop-rtsp.service \
  /etc/systemd/system/action-sop-web.service
```

Start the web service first, then the RTSP worker:

```bash
sudo systemctl start action-sop-web.service
curl http://127.0.0.1:8000/api/health

sudo systemctl start action-sop-rtsp.service
```

Inspect status and live logs:

```bash
sudo systemctl status action-sop-web.service action-sop-rtsp.service
sudo journalctl \
  -u action-sop-web.service \
  -u action-sop-rtsp.service \
  -f
```

Open `http://<JETSON_IP>:8000/` from an approved LAN workstation. Restrict port
8000 with the host/network firewall; binding to `0.0.0.0` does not provide TLS.

## Recovery and performance defaults

The RTSP worker has two recovery layers:

1. `run_sop_mvp` retries capture failures forever with bounded backoff.
2. systemd restarts the Python process after an unhandled failure or clean exit.

Initial live settings are deliberately conservative:

- TensorRT, batch size 1
- 640-pixel model input
- 5 analyzed frames per second
- RTSP capture buffer hint of one frame
- FFmpeg capture preference
- 10-second open timeout and 5-second read timeout
- no full-run video recording

The web service uses one Uvicorn process and SQLite, automatically qualifies a
`roll_sop_v1` session when both `cleaned` and `labeled` are `DONE`, and rescans
the shared data directory every five seconds. Manual review remains available
to correct an AI result. Do not add multiple Uvicorn workers around the same
SQLite file without measuring and testing write behavior.

Tune `/etc/action-sop/rtsp.env` only after collecting live GPU, CPU, memory,
temperature, frame-cadence, and disk-growth evidence.

## Optional blue-roll exclusion

Blue outer covering identifies rolls whose wrapping SOP was completed upstairs.
After validating the rule against recorded blue and non-blue rolls from this
camera, set `SOP_BLUE_ROLL_ARGS="--exclude-blue-rolls"` in
`/etc/action-sop/rtsp.env`. The default blue fraction threshold is `0.30`; add
`--blue-roll-min-fraction VALUE` only when replay evidence supports a different
value. Rerender the RTSP unit with the installer and restart the worker. The
rule is off when `SOP_BLUE_ROLL_ARGS` is empty. Its exclusion counts are stored
in the run configuration; old review records are unaffected.

## Short roll-session duration exclusion

The RTSP service uses the runner's existing minimum session duration check.
Set `SOP_MIN_SESSION_ARGS="--min-session-s 30"` in
`/etc/action-sop/rtsp.env` to discard a completed roll session when its
`end_time_s - start_time_s` duration is below 30 seconds. This matches the
website's **Durasi** field; a session at exactly 30 seconds is retained. The
RTSP environment example enables this 30-second threshold by default. Set the
variable to an empty string to keep sessions of all durations. Rerender the
RTSP unit with the installer and restart the worker after changing it.

## Backup and restore acceptance

For the live deployment, record the source commit and the configured data and
database paths from `/etc/action-sop/common.env` and
`/etc/action-sop/web.env`. The repository's example `data/` paths may differ
from the installed paths. Keep the protected environment files out of shared
reports.

At an agreed maintenance window, stop the RTSP worker, uploader (if deployed),
and web service before copying the SQLite database or session artifacts. Back
up the configured data root (including sessions, reports, and uploader spool),
any database outside that root, the deployed model/engine and metadata,
ROI/config, service units, and protected environment files. Store the copy on
a separate restricted disk and record its checksum. Restart services and
verify health and new session ingestion.

Prove the restore in an isolated location before relying on the backup:
restore to test paths, verify SQLite integrity, session and alert counts,
review decisions, a sample checklist/media file, and report downloads. Do not
replace production files during this test. A production recovery requires all
writers stopped before replacement and a post-restore rescan and ingestion
check.

## Optional helmet diagnostics

Set `SOP_HELMET_DIAGNOSTICS_ARGS` in `/etc/action-sop/rtsp.env` to
`"--helmet-alert-diagnostics --helmet-diagnostics-max-mb 512"` after confirming
the available disk space and required capture duration. The cap covers all
helmet diagnostic JSONL files under `SOP_DATA_DIR`; an empty value disables
capture. Existing environment files are preserved on reinstall, so set this
value explicitly on an already configured Jetson.

After pulling the updated repository, rerun the installer for the RTSP
component with the same service user, Python, and data-directory options used
for the installed service. Add `--start` to render the unit, reload systemd,
and restart the worker. Logs are written under
`<SOP_DATA_DIR>/diagnostics/helmet/YYYY-MM-DD/`. The diagnostics analyzer can
read the copied JSONL files after capture.

On diagnostic event frames, each tracked person observation may contain an
optional `shadow` object. It records the person's recent verified-helmet count
and consecutive unverified frames against the configured alert duration.
An `alert_emitted` event lists `candidate_track_ids` when those people can be
matched to diagnostic tracks in the same frame; use those IDs to interpret
the shadow state in scenes with multiple people. The analyzer prints linked
shadow values at each emitted alert when these fields are present.
`sustained_unverified` is a counterfactual for review, not an emitted alert;
the live helmet rule, 10-second default, and alert artifacts are unchanged.
The recent window is capped at 150 analyzed frames. Missing person frames
reset the shadow streak, and temporary track IDs may change when association
is lost. Older schema-v2 captures remain readable without this object.

## Stop or restart

```bash
sudo systemctl restart action-sop-rtsp.service
sudo systemctl restart action-sop-web.service

sudo systemctl stop action-sop-rtsp.service action-sop-web.service
```

`systemctl stop` does not trigger the configured automatic restart.

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

Edit the RTSP source, engine, metadata, and ROI:

```bash
sudoedit /etc/action-sop/rtsp.env
```

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

## Stop or restart

```bash
sudo systemctl restart action-sop-rtsp.service
sudo systemctl restart action-sop-web.service

sudo systemctl stop action-sop-rtsp.service action-sop-web.service
```

`systemctl stop` does not trigger the configured automatic restart.

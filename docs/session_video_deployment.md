# Single session video — AI-box handoff

Prepared 12 October 2026. Feature branch: `codex/stitched-session-evidence`.
Patch base: `401099cb3fb8f282c0cddaae7caac42d2e4b96e0`.

## What will change

New sessions save and upload one `evidence/session.webm`, with event positions
in `evidence.json`. The website plays this source file directly, without a
combined-cache copy. It shows brief step captions; the four named buttons seek
inside that same video. They are not four separate clips.

This video contains existing event windows with overlapping frames removed.
Unsaved intervals are skipped. It is not continuous full-session recording.
Historical clips remain readable. Models, ROI, RTSP credentials, SOP scoring,
helmet alerts, dependencies, and existing retention day counts are not changed.

The preview was accepted by the user. Local checks passed: 260 Python tests,
22 existing skips, all three Node suites, and Firefox playback/captions/seeking
at 375/768/1280 widths. Target acceptance remains open.

## Install from GitHub

The feature is published on `codex/stitched-session-evidence`; it is not merged
into master. Update the AI-box checkout **and** the website checkout if these
are separate machines. Use the existing deployment interpreter/environment;
do not run dependency synchronization or package installation for this change.

From each repository root, inspect and preserve existing local edits:

```bash
git status --short --branch
```

For the first installation, fetch and create a local tracking branch:

```bash
git fetch origin
git switch --track -c codex/stitched-session-evidence origin/codex/stitched-session-evidence
```

If this local branch already exists, switch to it and update it:

```bash
git switch codex/stitched-session-evidence
git pull --ff-only origin codex/stitched-session-evidence
```

The commit excludes the separate dashboard hint/index-version edit, local
AGENTS.md, CONTEXT.md, Progress files, raw footage, and model/config assets.
Keep your existing environment and assets in place.

The previously prepared ZIP is an optional offline snapshot of the original
handoff. GitHub is the current deployment path requested by the user.

## AI-box probe before restarting capture

Use the same Python 3.10 executable/environment as the deployed RTSP worker;
`python3` below is an example. This probe uses temporary synthetic frames and
the actual writer/worker. It does not open the camera or load a model:

```bash
python3 -m Scripts.check_session_video
```

Require `"status": "passed"`, `"files": 1`, and `"verified_frames": 3`. If VP8
encoding is unavailable, keep the current live worker running and resolve the
capture-host encoder choice first. No dependency changes are included here.

During an idle/maintenance window, run a bounded one-repeat performance pilot:

```bash
python3 -m Scripts.benchmark_session_evidence \
  --out-dir /tmp/session-evidence-target-pilot --repeats 1
```

Read `/tmp/session-evidence-target-pilot/metrics.json`. This is a CPU benchmark
and competes for resources if run alongside production inference. The local
1080p median was 14.6 seconds total preparation for a burst of four windows,
with median producer submission below 1 ms. The single WebM was 6.11 MB versus
4.29 MB for the old MP4 defaults. Encoding format/quality settings are different.
Do not assume the workstation timings hold on Jetson. The codec probe only
proves availability; it does not establish acceptable throughput or quality.

## Activate and accept

After the probe and target pilot are acceptable, update the web host first:

```bash
sudo systemctl restart action-sop-web.service
```

Then restart capture on the AI box:

```bash
sudo systemctl restart action-sop-rtsp.service
sudo journalctl -u action-sop-rtsp.service -n 100 --no-pager
```

These names come from the repository service templates. Use the actual names if
your installation differs. In a separate-server installation, the existing
uploader must continue running against the same data root; its artifact scan
recognizes the final video automatically.

Complete one real roll and verify:

1. The new session's `evidence/` contains one `session.webm`, without four new MP4
   event files. Its `evidence.json` has a `video` entry with event positions.
2. The uploaded session also contains one primary evidence video. Its API
   `combined_video.playback_url` points to `/media/.../evidence/session.webm`,
   not a generation endpoint. Playback creates no combined-video cache.
3. Press play once. Check captions and every step button, pause/seek, and
   readable cloth/label detail. `Perekaman selesai` means recording ended,
   rather than an observed roll exit.
4. Check fresh session/helmet artifacts and stream dimensions, not just service
   liveness. Compare inference throughput, time until the session is reviewable,
   upload retries, encoder backlog/errors, and RAM with your current baseline.
5. Test a second roll while the previous video's encoding/upload completes.
   Existing Python tests do not establish detector accuracy or live acceptance.

Source-video encoding runs on one worker with a 128-pending-frame budget. If it
cannot keep up, it fails explicitly instead of publishing incomplete evidence.

## Rollback

If target acceptance fails, restore the previous capture runner from the known
base, preserving any subsequent local edits, then restart the RTSP service:

```bash
git restore --source=401099cb3fb8f282c0cddaae7caac42d2e4b96e0 \
  -- Action_Detection_SOP/runner_mvp.py
sudo systemctl restart action-sop-rtsp.service
```

Keep the updated web reader so any already-created single-video sessions remain
readable. Do not delete session metadata or footage as part of rollback.

Detailed workstation results: [session_evidence_benchmark.md](session_evidence_benchmark.md).

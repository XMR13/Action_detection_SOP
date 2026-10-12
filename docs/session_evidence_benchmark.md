# Session evidence benchmark — 12 October 2026

New sessions now produce **one source evidence video before upload**, with event
positions for Indonesian step captions. The web server serves it directly; it
does not create a combined cache copy. This report measures the final encoder
and compares it with the previous writer.

## Method

- Local workstation: AMD Ryzen 5 4600H, Python 3.10.12, OpenCV 4.11.0.
- Deterministic simulated CCTV: texture, moving roll, a small label, and source
  timestamps; no model inference or GPU workload. This is not target Jetson data.
- Four event windows at 5 FPS, 21 frames per window. Two overlap. Old output:
  84 frames across four files. Combined output: 73 unique frames in one file,
  14.6 seconds of playback; unsaved intervals are excluded.
- Three repetitions per approach/resolution; table values are medians.
- Previous output uses the existing mp4v writer. The four-WebM control isolates
  the effect of combining frames from changing the codec. Quality settings are
  defaults and are **not matched**; file-size differences do not establish
  equivalent visual quality.
- Final preparation includes encoder work, decoded-frame verification, and
  publication. Previous preparation excludes verification and any web-browser
  transcoding, which would add work where required.
- The final background-worker test submits all four windows as one burst, then
  waits for readiness. Live sessions deliver windows over time, allowing work
  to overlap capture. Submission time measures capture-thread dispatch, not CPU
  inference latency or CPU resource contention.
- Loopback upload uses the real streaming uploader against an in-memory HTTP
  sink, three repetitions. It excludes receiving-disk writes, authentication,
  API indexing, and LAN bandwidth. The 10 Mbps column is a payload-only estimate,
  not a network measurement. Timings vary with workstation load.

## Results

| Resolution | Approach | Frames | Video MB | Preparation s | Max window submit/write ms | Decode s | Loopback upload ms | Payload upload at 10 Mbps s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640x360 | Previous: four MP4 | 84 | 0.549 | 0.229 | 61.452 | 0.090 | 11.85 | 0.439 |
| 640x360 | Control: four WebM | 84 | 0.872 | 2.059 | 525.223 | 0.152 | 12.40 | 0.697 |
| 1920x1080 | Previous: four MP4 | 84 | 4.286 | 1.783 | 455.244 | 0.469 | 17.66 | 3.429 |
| 1920x1080 | Control: four WebM | 84 | 6.976 | 16.557 | 4710.395 | 1.102 | 28.01 | 5.580 |
| 640x360 | Final: one queued WebM | 73 | 0.763 | 2.017 | 0.790 | 0.113 | 3.55 | 0.610 |
| 1920x1080 | Final: one queued WebM | 73 | 6.108 | 14.555 | 0.958 | 0.880 | 9.93 | 4.887 |

## Findings

- One final file replaces four source files. There is no new duplicate playback
  video. Removing overlap retained 73/84 frames, about 13% fewer frames.
- Against the equal-codec four-WebM control, the single video is approximately
  12% smaller at both resolutions. One request reduced loopback upload overhead.
- Against the old mp4v output, the chosen browser-ready VP8/WebM defaults make
  the single file about 39% larger at 640×360 and 43% larger at 1920×1080.
  Combining files alone cannot guarantee a smaller payload.
- VP8 encoding is substantially slower than the old MPEG4 writer on this host.
  The queued implementation keeps median maximum submission below 1 ms, while
  total preparation is roughly 2 seconds at 640×360 and 14.6 seconds at 1080p.
  Encoding uses CPU in the background, and may still contend with other work.
- This is event footage, not an entire session recording. End-to-end readiness
  also includes window collection, publication, uploader polling, retries, and
  transfer; these component measurements do not measure that whole process.

## Implementation checks and target gate

- Encoding has one worker and a 128-frame pending budget. A full queue or failed
  encode is an explicit error. It does not silently publish partial evidence.
- Video and event manifest are finalized before the checklist makes a session
  discoverable by the uploader. Temporary files are excluded from artifact
  discovery; normal shutdown drains the worker.
- The direct-source video has been checked in Firefox with captions, seeks,
  play/pause, responsive widths, and failed-video fallback. Old multi-clip
  sessions remain readable and are not converted or deleted automatically.
- Before deploying, run a small Jetson probe for VP8 support, then the benchmark
  with one repeat, and a completed real roll. Measure live inference throughput,
  encoder backlog, peak RAM, first-review readiness, and actual LAN upload.
  Review cloth/label visibility. These local timings do not establish target
  acceptance or measured detector accuracy.
- If target CPU, RAM, preparation time, or file size is unacceptable, evaluate
  encoder/quality settings from measured footage before enabling this pipeline.
  Retain one source artifact in that decision; do not restore a duplicate cache.

Reproduce locally:

```bash
.venv/bin/python -m Scripts.benchmark_session_evidence \
  --out-dir /tmp/session-evidence-benchmark --repeats 3 --loopback-upload
```

The raw measurements are in [session_evidence_benchmark.json](session_evidence_benchmark.json).
Generated movies remain outside the repository. No dependency or lockfile change
was needed.

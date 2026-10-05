# SOP status decisions

This document describes the implemented status rules for `roll_sop_v1` and the
older `operator_mvp_a` profile. It describes code behavior, not a claim about
the effective settings on a particular Jetson. Check that run's
`run_config.json` before diagnosing a website row.

## What `UNKNOWN` means

`UNKNOWN` means the machine cannot decide that a required step is `DONE` or
`NOT_DONE` from the evidence it counted. It is a result, not a reason code:
it does not, by itself, prove that the session was short, that an action was
skipped, or that the model failed. A human should review the video when the
business decision matters.

For a roll session, the website shows `cleaned`, `labeled`, and an overall SOP
result. The machine result, the result after any human overrides, and the
review status (`PENDING`, `QUALIFIED`, `NOT_QUALIFIED`, `OUT_OF_SCOPE`) are separate fields.
`PENDING` and the dashboard's “Perlu ditinjau” wording are workflow states,
not synonyms for machine `UNKNOWN`.

## Operator-facing final verdict

The website combines the effective review decision and final SOP result into
one operator-facing verdict:

| Effective review | Final SOP | Operator verdict |
| --- | --- | --- |
| `QUALIFIED` | `DONE` | Sesuai SOP |
| `NOT_QUALIFIED` | `NOT_DONE` | Tidak sesuai SOP |
| `OUT_OF_SCOPE` | Any result | Di luar cakupan SOP |
| `PENDING` | Any result | Perlu ditinjau |
| Any other combination | Any result | Perlu ditinjau |

The final row catches older records whose review decision and SOP result do
not agree. The website does not silently rewrite those records. New roll
review saves reject a decision that conflicts with the final SOP result; the
reviewer must correct the step or overall override before saving. Returning
to the queue without saving leaves the current decision unchanged. Machine
SOP values and the review record remain available in the API and export for
diagnosis.

## Resolving rolls outside checking scope

A retained roll session can be outside the checking SOP even when it lasts
30 seconds or longer. A reviewer can select **Di luar cakupan SOP** in the
existing session-detail decision toolbar and save a required reason:

| Stored reason | Operator label | Use |
| --- | --- | --- |
| `PASSING_THROUGH` | Hanya melintas | Roll passes through without requiring checking here |
| `ALREADY_WRAPPED` | Sudah dibungkus | Roll was already wrapped before this checking area |
| `OTHER` | Lainnya | Another exclusion, explained in the review note |

This is a manual `roll_sop_v1` review outcome. Duration or uncertain AI evidence
alone is not grounds for exclusion. If a required step was missed by the AI,
the reviewer corrects the step/overall result and saves a scored decision.
If evidence is insufficient, leave the session pending by returning to the queue
without saving a decision.

An exclusion takes precedence over automatic approval, resolves the pending
review, and keeps the session, evidence, AI results, and any existing SOP
overrides accessible. The queue and dashboard show a separate exclusion count;
the existing queue filter can select excluded sessions. The session CSV includes
`scope_reason` and `review_note`. A subsequent scored decision must still agree
with the final SOP result and clears the exclusion reason.

### Counts and percentages

- Total detected sessions and raw machine-result counts include exclusions.
- Pending verdict counts and final SOP counters exclude out-of-scope sessions.
- Compliance is `verdict_done / (verdict_done + verdict_not_done) * 100`.
  Pending, conflicting older reviews, and excluded sessions are outside this
  denominator. With no eligible decisions, the API returns `null` and the UI
  displays **—**.
- Review completion counts matching compliant/noncompliant verdicts plus
  exclusions, divided by all detected sessions. Conflicting old reviews remain
  unresolved even if their raw review status contains an older decision.
- The final SOP unknown percentage uses in-scope sessions as its denominator.

### Deployment and rollback

Before deploying, inspect the actual website unit and database path using
`systemctl cat action-sop-web.service` and `/api/admin/storage`. Back up the
review SQLite database using SQLite's online backup API and retain the previous
source revision. For example, run this with the actual database and a new backup
path (do not copy a live SQLite file without its transaction state):

```bash
python3 -c 'import sqlite3,sys; src=sqlite3.connect(sys.argv[1]); dst=sqlite3.connect(sys.argv[2]); src.backup(dst); dst.close(); src.close()' /actual/reviews.sqlite3 /actual/reviews-before-scope.sqlite3
```

Deploy the reviewed source revision, restart `action-sop-web.service`, and hard
reload the UI. Startup adds a nullable `scope_reason` column to `reviews`; it
preserves existing rows and is safe to run again. Verify a passing roll over
30 seconds, the required reason, a failed save followed by a retry, persisted
exclusion after reload, queue filtering, and the compliance denominator.

Older source does not understand `OUT_OF_SCOPE`. A source-only rollback after
new exclusions have been saved will misclassify those reviews. Pause review
writes and retain a fresh backup before rollback; either deploy a version that
understands exclusions or restore the pre-change database along with its source.
Restoring that backup loses review decisions made afterward, so reconcile those
records before resuming operations. Runner artifacts are separate from reviews.

## Roll step evidence thresholds

The runner defaults are:

| Step | CLI setting | Default | Frames at exactly 5 analyzed FPS | Allowed missing analyzed frames within a streak |
| --- | --- | ---: | ---: | ---: |
| Cleaning | `--cleaning-s` | 0.4 s | 2 positive frames | 3 (`--cleaning-max-gap`) |
| Labeling | `--labeling-s` | 1.0 s | 5 positive frames | 1 (`--labeling-max-gap`) |

For each step, the engine calculates `required_frames = max(1,
round(required_seconds * actual_analysis_fps))`. The *actual* analysis FPS can
differ from the requested `--analysis-fps` (default 5) because the runner
selects frames from the source. `--every` can also override that selection.
Use `analysis_fps` in `run_config.json`, not an assumed camera or video FPS.

The engine counts a positive frame only when it receives the appropriate
cloth or label detection overlapping the selected roll. It needs that many
positive analyzed frames in one evidence streak to mark the step `DONE`.
The gap setting lets a streak survive a small number of missing frames; missing
frames do **not** count as positive evidence. For example, at exactly 5
analyzed FPS, two positive cleaning frames can complete cleaning even with up
to three missing analyzed frames between them. These values are evidence
thresholds, not minimum elapsed session durations.

At session end, each roll step follows this order:

1. If the evidence streak reached its required positive-frame count: `DONE`.
2. Otherwise, if the session contains fewer analyzed frames than the step's
   required-frame count: `UNKNOWN` (`short_session_is_unknown=True`).
3. Otherwise, if the step has at least one positive frame but never completed
   a qualifying streak: `UNKNOWN` (partial evidence).
4. Otherwise: `NOT_DONE` (zero positive frames in a session long enough to
   decide under the current rule).

`total_frames` counts analyzed frames while the roll session is active. It is
not the raw camera frame count. For example, at exactly 5 analyzed FPS, a
session with 4 analyzed frames is below the 5-frame labeling threshold. If
labeling has not already completed, `labeled=UNKNOWN`; cleaning may already
be `DONE` because its threshold is only 2 positive frames. A longer session
with just one positive labeling frame is also `UNKNOWN`. A longer session
with **zero** positive labeling frames is `NOT_DONE`, even if the real action
was missed by the camera or detector. The status alone cannot distinguish
that failure from a truly skipped action.

The roll runner's fallback session timing starts after 1 second of sustained
roll detection and ends after 2 seconds of sustained absence. The session's
frame count starts when the session opens and continues while it stays active.
At roughly 5 analyzed FPS, a normally closed session therefore usually has
enough analyzed frames to pass the 5-frame *length* check, even if the roll
was visible only briefly after opening. An end-of-stream flush, different
timing settings, or a different analysis cadence can change that. Do not
interpret an existing website `UNKNOWN` as “session under 1 second” without
checking its `total_frames` and effective run settings.

The roll engine currently adds `insufficient_cleaning_evidence` or
`insufficient_labeling_evidence` to `notes` when a step is `UNKNOWN` and has
positive frames. It does not currently add a dedicated short-session note for
roll steps. Compare `total_frames` with the calculated threshold to identify
that case. A missing note does not prove the evidence was adequate.

## Overall roll result

| Step combination | `overall_status` |
| --- | --- |
| Cleaning and labeling are both `DONE` | `SESUAI SOP` |
| Either step is `NOT_DONE` | `TIDAK SESUAI SOP` |
| No step is `NOT_DONE`, and at least one is `UNKNOWN` | `UNKNOWN` |

The website also normalizes the overall result to `DONE`, `NOT_DONE`, or
`UNKNOWN` for some views and filters. A reviewer can override step or overall
values; inspect both machine and final values before explaining a row.

## Minimum session duration is a separate gate

`--min-session-s` checks elapsed time (`end_time_s - start_time_s`) after the
session ends. If it is greater than zero and the duration is **less** than the
configured value, the runner discards the session before saving or uploading
its checklist. This does not turn the session into a website `UNKNOWN` row.
At exactly the configured duration, the session is retained.

The runner's fallback is 0 seconds (gate off). The RTSP environment example
sets `SOP_MIN_SESSION_ARGS="--min-session-s 30"`, which requests a 30-second
gate. That example does not establish what a particular running service uses.
Read `sessionization.min_session_seconds` in its `run_config.json` and verify
the running service arguments when diagnosing live behavior.

## Older operator profile

`operator_mvp_a` checks operator presence, ROI dwell, and helmet association.
ROI dwell or helmet can be `UNKNOWN` when the check is disabled, the session
has fewer analyzed frames than the relevant threshold, or the person is too
small under a configured minimum-height rule. That engine records notes such
as `session_too_short_for_helmet_decision` and
`person_too_small_for_reliable_helmet`. The roll cleaning and labeling rules
above should not be applied to this older profile.

## Diagnose a website `UNKNOWN` row

1. Confirm `sop_profile` and compare the machine and final fields on the
   session detail page. Check whether a review override changed the result.
2. Open the session's `checklist.json`. For a roll session, inspect `cleaned`,
   `labeled`, `overall_status`, `total_frames`,
   `cleaning_positive_frames`, `labeling_positive_frames`, and `notes`.
3. Open that session's `run_config.json`. Inspect `analysis_fps`,
   `roll_sop_v1.cleaning_required_seconds`,
   `roll_sop_v1.labeling_required_seconds`, the two max-gap settings, and
   `sessionization.min_session_seconds`. Calculate the frame thresholds from
   the recorded values.
4. Review the evidence video and detection overlay, if available. A
   positive-frame count indicates detected evidence under the overlap rule;
   it does not prove the full physical SOP action happened. Zero detections
   do not prove it did not.

Implementation references: `Action_Detection_SOP/roll_sop_engine.py`,
`Action_Detection_SOP/sop_engine.py`, `Action_Detection_SOP/runner_mvp.py`,
`Action_Detection_SOP/web_mvp/sop_status.py`, and `Scripts/run_sop_mvp.py`.

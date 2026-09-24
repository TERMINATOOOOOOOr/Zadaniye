# TrafficEye — traffic event detection from a fixed road camera

Solution for the WIUT Hackathon 2026, Computer Vision track, elimination task.
Given an `.mp4` from a fixed CCTV camera the system returns every traffic event as a time segment
`[start_sec, end_sec, label]` (Part A) and, frame by frame and causally, the probability that an
accident starts within the next 5 seconds (Part B).

Team: **TrafficEye** (see [Team](#team)). Website: **https://trafficeye-production.up.railway.app**.

## Quick start

```bash
pip install -r requirements.txt            # or: docker build -t team .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

Weights are shipped in `weights/` (YOLO11n + YOLO11s COCO checkpoints, 25 MB in total). Nothing has to be
downloaded at run time; `weights/download.sh` only restores the files if they are missing.
Python 3.10+; tested on Python 3.11 with CUDA 12.8 (RTX 4050) and on CPU.

## What is where

```
solution.py            the interface the harness imports: detect_events(), RiskEstimator
run_submission.py      organizers' harness (unchanged)
evaluate.py            organizers' metric (unchanged)
requirements.txt       runtime dependencies; Dockerfile for the same two commands in a container
weights/               yolo11n.pt, yolo11s.pt (+ download.sh, README with licences)
configs/scene.json     scene geometry of the camera (road, lanes, crossings, stop lines, signal ROI)
src/                   the pipeline: video reading, detection+tracking, scene geometry, rules per class,
                       segment post-processing, causal risk estimator, EDA and rendering
tools/                 annotator.html (labelling), scene_editor.html (scene geometry), inspect_video.py,
                       run_samples.py (batch run + website materials)
tests/                 format, overlap, time-budget, determinism and causality checks (pytest)
site/                  the team website (FastAPI); see site/README.md. railway.json at the root deploys it
predictions_samples.json   our output on the sample videos
```

## Approach

**Detector → tracker → scene geometry → rules → segments.** Nothing is trained: the only learned
component is a COCO-pretrained YOLO11 detector; every event class is a rule on object trajectories
and on the hand-made scene geometry of this camera (allowed by the task).

1. **Frames.** Every 3rd frame (8.3 fps at 25 fps source), long side resized to 960 px. Skipped frames
   are only `grab()`bed, never converted.
2. **Detection + tracking.** YOLO11s (COCO classes person, bicycle, car, motorcycle, bus, truck, traffic
   light, some animals/objects) with the built-in ByteTrack of Ultralytics. Detection confidence is set
   low (0.15) on purpose: ByteTrack does its own two-stage association with high/low score boxes.
3. **Trajectories.** For every track we compute smoothed centre and ground-contact point, speed,
   acceleration and heading. All thresholds are relative to the object's own box height, so the same rule
   works near and far from the camera.
4. **Scene.** `configs/scene.json` holds the road polygon, lanes with their direction vectors and
   allowed manoeuvres, crossings, stop lines with approach direction, solid lines and the traffic-signal
   ROI. It is drawn once in `tools/scene_editor.html`. If the file is absent, a direction field and a road
   mask are estimated from the vehicle tracks of the video itself (used by the website demo on arbitrary
   footage); rules that need crossings/stop lines/lines are then disabled.
5. **Rules per class** (`src/rules/`):
   - `stopped_vehicle`: speed below 12 % of box height for ≥ 10 s on the carriageway, not in a queue
     (no other stopped vehicle within 3.5 box heights, not at a regular stop location, not in front of a
     stop line while the signal is red).
   - `congestion`: per direction of flow, ≥ 4 vehicles of which ≥ 75 % are crawling, for ≥ 45 s; when
     the signal is visible, only queues that survive a green phase.
   - `wrong_way`: heading opposes the lane direction (or the dominant flow of the cell) by > 120° for
     ≥ 1.5 s with a real displacement against the flow.
   - `jaywalking` / `failure_to_yield`: pedestrian ground point inside the road polygon and outside every
     crossing; a moving vehicle inside a crossing while a pedestrian is on it.
   - `red_light` / `stop_line`: the traffic light colour is read from its ROI (HSV); a vehicle whose
     ground point crosses a stop line in the approach direction on red; a vehicle that stops beyond the
     stop line on red without entering the intersection.
   - `illegal_u_turn` / `illegal_turn`: heading change ≥ 150° (U-turn) or 60–125° from a lane whose
     allowed manoeuvres exclude it.
   - `solid_line_crossing`: the trajectory crosses a solid-line polyline while moving.
   - `accident` / `near_miss`: pairs of tracks with a fast approach; contact (normalised centre distance
     < 0.45 or box IoU > 0.25) followed by an abrupt stop of both objects = accident; close approach
     without contact plus hard braking or a swerve = near miss. Approaching a queued/stopped vehicle at a
     regular stop location is explicitly excluded (the classic false positive).
   - `road_obstacle`: animals/objects on the road; `fire_smoke`: not predicted (no reliable detector).
6. **Segments.** Per-frame flags become intervals; intervals of one class are merged when closer than a
   class-specific gap, blips shorter than a class-specific minimum are dropped, segments of the same class
   never overlap (`src/segments.py`).

**Part B (`src/risk.py`).** A separate causal estimator: YOLO11n at 640 px every 4th frame, ByteTrack,
1.5 s of history per track. Risk = time-to-collision between pairs on a collision course (lateral miss
distance below the box widths, approach along the line of centres), corrected by whether the follower is
already braking hard enough to stop in time; plus hard braking and pedestrians on the carriageway. Regular
stop locations (queues) are learned online from already-seen frames and lower the risk of "caught up with
the queue". The score is squared so that the alarm threshold 0.5 is reached only with strong evidence,
and decays by 0.8 per update without a signal. `step()` never reads the video file and never uses Part A
output.

**Learned vs rule-based.** Learned: YOLO11 detector (COCO weights, not fine-tuned). Rule-based:
everything else.

## Datasets and licences

| Asset | Use | Licence |
|---|---|---|
| YOLO11n / YOLO11s COCO checkpoints (Ultralytics assets v8.3.0) | detector, as-is | AGPL-3.0 |
| Ultralytics 8.4.152 (YOLO + ByteTrack) | inference and tracking | AGPL-3.0 |
| Sample videos from the organisers | our dev labels and EDA only; not redistributed | organisers' |

No external training data was used; no model was fine-tuned.

## Determinism and seeds

`src/config.seed_everything()` fixes Python, NumPy and Torch seeds (seed 0) and sets cuDNN to
deterministic mode. ByteTrack is deterministic for a fixed frame order. Two consecutive runs on the same
machine give identical `predictions.json` (checked by `tests/test_format_and_determinism.py`).
Timestamps are rounded to 3 decimals by the harness.

## Time budget

Measured with the organizers' harness (Part A + Part B, including all video decoding) on an RTX 4050
laptop GPU (i7-13650HX), fp16 inference, nothing else running:

| Footage | Part A | Part B | Total | Limit |
|---|---|---|---|---|
| C3896.MP4, 4K 30 fps 147 Mbit/s, 340 s | 99 s (0.29×) | 242 s (0.71×) | 341 s = 1.00× | 3× |
| C3897.MP4, 318 s | 93 s (0.29×) | 215 s (0.68×) | 308 s = 0.97× | 3× |
| C3902.MP4, 318 s | 99 s (0.31×) | 215 s (0.68×) | 314 s = 0.99× | 3× |
| C3905.MP4, 128 s | 39 s (0.30×) | 72 s (0.56×) | 112 s = 0.87× | 3× |
| 1080p 30 fps, 12 min crash compilation | | | 0.40× | 3× |

Part B is dominated by the harness decoding every 4K frame itself (about 0.5× of the duration on this CPU);
our own work per frame is one resize and, every 4th frame, a YOLO11n pass. Part A reads the video through an
ffmpeg pipe that subsamples and rescales inside the decoder (NVDEC when the ffmpeg build supports it). Frame stride and input size are environment variables if the budget ever gets
tight on slower hardware:

| Variable | Default | Meaning |
|---|---|---|
| `WIUT_STRIDE_A` / `WIUT_IMGSZ_A` | 3 / 960 | Part A frame stride and long side |
| `WIUT_STRIDE_B` / `WIUT_IMGSZ_B` | 4 / 640 | Part B frame stride and long side |
| `WIUT_MODEL_A` / `WIUT_MODEL_B` | yolo11s.pt / yolo11n.pt | detector checkpoints in `weights/` |
| `WIUT_DEVICE` | auto | `cuda:0` or `cpu` |
| `WIUT_SCENE` | configs/scene.json | scene geometry file |
| `WIUT_CACHE` | (off) | directory for cached detections (development only) |

## Development workflow

```bash
python tools/inspect_video.py data/samples/C3896.MP4        # tracks, flow field, candidates per rule
python -m src.eda data/samples --out site/static/eda         # EDA figures + eda.json for the website
tools/annotator.html                                         # label the samples → data/labels/my_labels.json
tools/scene_editor.html                                      # draw the scene → configs/scene.json
python tools/run_samples.py data/samples --team NAME --out predictions_samples.json --site --labels data/labels/my_labels.json
python -m pytest tests -q
```

## Results on the sample videos

The four sample clips are one signalised T-junction at rush hour (4K, 30 fps, 2–6 min each). Our output
on them is `predictions_samples.json`; the annotated clips and timelines are on the website.

| Clip | Length | Events we report |
|---|---|---|
| C3896.MP4 | 340 s | jaywalking 19, failure_to_yield 5 |
| C3897.MP4 | 318 s | jaywalking 21, failure_to_yield 7, stopped_vehicle 1, accident 1 |
| C3902.MP4 | 318 s | jaywalking 19, failure_to_yield 3, stopped_vehicle 2 |
| C3905.MP4 | 128 s | jaywalking 7, failure_to_yield 2, stopped_vehicle 1, congestion 1 |

We did not have labels for the samples; instead every rule's candidates were reviewed on contact sheets
(6 frames per candidate) and thresholds were tightened until the reviewed candidates were plausible.
What that review showed:

- Pedestrians crossing outside the zebras are constant at this junction: `jaywalking` is our most frequent
  class and most reviewed candidates were real.
- `failure_to_yield` fires when a moving car passes through a zebra within 1.5 car heights of a pedestrian
  on it; the reviewed cases were cars pushing through the crossing on green while people were still on it.
- The `accident` in C3897 (315–318 s) is a false positive (an SUV passing pedestrians on the side zebra)
  that survived the tightened rule; we left it in rather than tune against a single case.
- `red_light` and `stop_line` are switched off: the lamp colour could not be read reliably from this
  camera and the traffic-derived state stayed "red" through most of rush hour (see the report).
- The risk curve raises an alarm (≥ 0.5) on about 1–2 % of frames, 2–5 alarm runs per clip, all on
  fast approaches to a slow or stationary road user near the camera.

## Team

| Member | Role | Did what |
|---|---|---|
| Abdulaziz Qosimov (captain) | pipeline, tracking, risk estimator, packaging | detector + ByteTrack integration, ffmpeg reader and time budget, scene geometry, queue-aware rules, Part B, Docker, tests |
| Member 2 | event rules, dev review | manual review of rule candidates on the samples, threshold tuning, annotation conventions |
| Member 3 | EDA, website, report | EDA figures, team website and live demo (Railway), technical report |

# TrafficEye — traffic event detection from a fixed road camera
[![CI](https://github.com/TERMINATOOOOOOOr/Zadaniye/actions/workflows/ci.yml/badge.svg)](https://github.com/TERMINATOOOOOOOr/Zadaniye/actions/workflows/ci.yml)

Solution for the WIUT Hackathon 2026, Computer Vision track, elimination task.
Given an `.mp4` from a fixed CCTV camera the system returns every traffic event as a time segment
`[start_sec, end_sec, label]` (Part A) and, frame by frame and causally, the probability that an
accident starts within the next 5 seconds (Part B).

Team: **Air MAX** (see [Team](#team)). Website: **https://trafficeye-production.up.railway.app**. Repository: **https://github.com/TERMINATOOOOOOOr/Zadaniye**.

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
configs/scene.json     scene geometry of the camera: road, intersection, direction zones, crossings, stop line,
                       solid lines, islands, traffic-signal ROI (drawn on a frame of C3896)
configs/scene_ref.npz  reference background of that camera (SIFT keypoints, traffic-light anchors) used to align
                       the scene to every clip; rebuilt by tools/make_scene_ref.py
src/                   the pipeline:
  video.py               ffmpeg reader: frame stride, resize and the full-resolution lamp crop in one decode
  tracking.py            YOLO11 + ByteTrack; tracks with smoothed kinematics
  align.py               per-clip scene alignment (median background → SIFT → RANSAC similarity)
  scene.py, flow.py      scene geometry; direction field and stop map learned from the clip's own tracks
  signal.py              traffic-light state: lamp reader, pedestrian phases, traffic phases, fusion
  static_objects.py      background model for objects the detector does not know (road_obstacle)
  context.py, rules/     shared track/scene queries and one rule module per event class
  segments.py            per-frame flags → merged, non-overlapping segments
  risk.py                Part B causal risk estimator
  render.py, eda.py      annotated videos and EDA figures for the website
tools/                 annotator.html (labelling), scene_editor.html (scene geometry), inspect_video.py,
                       ingest_samples.py (copy/probe the samples), run_samples.py (batch run + website materials),
                       make_scene_ref.py (rebuild configs/scene_ref.npz), ci_smoke.py (synthetic clip and the
                       determinism compare used by CI), results_table.py (README tables from predictions_samples.json)
tests/                 pytest (53 tests): output format, determinism and causality, scene alignment, lamp reader and
                       signal fusion, static objects, rules on synthetic tracks, the CI smoke run, website pages
.github/workflows/ci.yml   CI: pytest, validation of predictions_samples.json, harness run twice on a synthetic clip
site/                  the team website (FastAPI); see site/README.md. railway.json at the root deploys it
predictions_samples.json   our output on the sample videos, with the harness log (time budget) per video
```

## Approach

**Detector → tracker → aligned scene → rules → segments.** Nothing is trained: the only learned
component is a COCO-pretrained YOLO11 detector; every event class is a rule on object trajectories
and on the hand-made scene geometry of this camera (allowed by the task).

1. **Frames.** Every 3rd frame (10 fps for the 30 fps samples), long side resized to 1280 px. The video is
   read through an ffmpeg pipe that subsamples and rescales inside the decoder; the same filter graph cuts a
   full-resolution crop around the traffic light and appends it to the frame (`fps → split → scale + crop →
   overlay`, one rawvideo stream), so the lamp is read at 4K without a second decode. NVDEC is used when the
   ffmpeg build and the file allow it; the samples are 4K 4:2:2 10-bit, which NVDEC does not decode, so on
   this footage the decoding is software on all cores.
2. **Scene alignment per clip** (`src/align.py`). The scene was drawn on C3896, but the camera is not
   perfectly fixed between clips. For every clip we take 9 frames spread over its length (`ffmpeg -ss`,
   960 px), build a median background (moving objects vanish), extract SIFT features and match them against
   the reference background of C3896 (`configs/scene_ref.npz`); a similarity transform (shift, scale,
   rotation) is estimated by RANSAC and accepted only with ≥ 40 inliers, a scale of 0.8–1.25 and a rotation
   ≤ 5°. Otherwise the footage is treated as another camera and the scene is switched off. Measured on the
   samples (transform of the scene into the clip, 4K pixels): C3896 identity (2714 inliers); C3897 shift
   (0.7, −0.3) (1328 inliers); C3902 shift (−122.0, +70.8), scale 1.014, rotation −0.75° (137 inliers);
   C3905 shift (−33.7, +41.9), scale 1.014, rotation −1.08° (120 inliers). The residuals between the three
   traffic lights of the scene and the detector's static "traffic light" tracks drop from 87/95/125 px to
   7/3/7 px on C3902 and from 21/15/48 px to 8/36/8 px on C3905; footage from other cameras is rejected with
   16–22 inliers. Cost 1.5–3 s per clip. The light residuals are an independent check written to the log,
   not part of the decision.
3. **Detection + tracking.** YOLO11s (COCO classes person, bicycle, car, motorcycle, bus, truck, traffic
   light, some animals/objects) with the built-in ByteTrack of Ultralytics. Detection confidence is set
   low (0.15) on purpose: ByteTrack does its own two-stage association with high/low score boxes. 1280 px
   instead of 960 px raises the number of tracked pedestrians on C3905 from 269 to 430 (median person
   height 133 → 108 px: the far zebra becomes visible) for about 0.1× of the duration in Part A.
4. **Trajectories.** For every track we compute smoothed centre and ground-contact point, speed,
   acceleration and heading. All thresholds are relative to the object's own box height, so the same rule
   works near and far from the camera.
5. **Scene** (`configs/scene.json`). Road polygon, intersection polygon, three pedestrian crossings, the stop
   line of the far approach with its approach direction, five direction zones (`far_approach`, `far_exit`,
   `near_carriageway`, `side_street_slip`, `side_street_main`; each with a unit direction and the manoeuvres
   it allows), four solid lane dividers on the far approach (`far_L1`–`far_L4`, verified continuous by a
   paint-profile scan of an empty frame), the traffic islands and the traffic-signal ROI. Drawn once in
   `tools/scene_editor.html`. If the file is absent or the alignment rejects the clip, a direction field and
   a road mask are estimated from the vehicle tracks of the video itself (used by the website demo on
   arbitrary footage); rules that need crossings, stop lines or solid lines are then disabled.
6. **Traffic-signal state** (`src/signal.py`). Three independent estimates are fused:
   - *Lamp reader.* The lamp is cut from the 4K frame in the same ffmpeg pass. The red/amber/green sections
     are located from the detector's "traffic light" track (or the ROI) and classified by the colour energy
     of each section **relative to the other sections** of the same lamp, not by absolute HSV thresholds;
     with occlusion detection (a bus passing in front of the lamp) and hysteresis. Works in daylight with a
     faint lamp (C3896) and at dusk (C3905).
   - *Pedestrian phases.* Red while pedestrians walk on the zebra the stop line protects; green while
     vehicles cross the line at speed and nobody is on the zebra.
   - *Traffic phases.* A standing queue at the line = red, vehicles crossing = green. The weakest source: a
     queue stands on green too.
   - *Fusion.* The lamp wins when its phases are plausible (both colours present, phases 10–150 s, amber
     ≤ 6 s) and either periodic (≥ 2 interior red and ≥ 2 green phases within ±25 % of their medians) or in
     agreement with the pedestrian phases ≥ 60 % of the time; otherwise pedestrians, then traffic. On all
     four samples the lamp was accepted: a 75 s cycle (red ≈ 36–39 s, green ≈ 38 s, amber ≈ 3 s), e.g.
     C3896 red 0–27, green 27–66, red 66–103, green 103–141 …; agreement with pedestrians 0.95 / 0.72 /
     1.00 / 0.90 (C3897 is low because people there cross on their own red).
7. **Rules per class** (`src/rules/`):
   - `stopped_vehicle`: speed below 12 % of box height for ≥ 10 s on the carriageway, and not a queue: no
     other stopped vehicle nearby (checked at three moments of the stop), not at a regular stop location,
     not above the stop line while the lamp is red (any depth of the queue), not within 2.5 box heights of a
     zebra (waiting for pedestrians or for its own signal); a bus at a stop only after 60 s.
   - `congestion`: per direction of flow, ≥ 4 vehicles of which ≥ 75 % are crawling, for ≥ 45 s; when
     the signal is visible, only queues that survive a green phase.
   - `wrong_way`: heading opposes the direction zone of the scene (inside the zones) or the dominant flow of
     the cell (outside them) by > 120° for ≥ 2.5 s with a displacement of ≥ 2 box heights against the flow.
     Silent inside the intersection polygon, in the unreferenced strip left between opposing zones, and for
     observations whose box is clipped at the frame edge.
   - `jaywalking` / `failure_to_yield`: pedestrian ground point inside the road polygon and outside every
     crossing; a moving vehicle inside a crossing while a pedestrian is on it.
   - `red_light` / `stop_line` (**on**; red is taken **only from the lamp** — at this junction pedestrians
     cross on their own red and a queue stands on green too, so both other sources give false red).
     `red_light` = crossing the stop line in the approach direction at speed after the red has been on for
     ≥ 1.5 s and ≥ 1 s before green, with the vehicle entering the intersection within 4 s; the event ends
     when it leaves the intersection (≤ 8 s). `stop_line` = a vehicle that arrived during that red stands
     0.35–2.5 box heights beyond the line for ≥ 2 s without entering the intersection; the event lasts until
     green.
   - `illegal_u_turn` / `illegal_turn`: heading change ≥ 150° (U-turn) or 60–125° from a zone whose
     allowed manoeuvres exclude it. No arrows are painted on this junction, so every zone allows every
     manoeuvre and `illegal_turn` stays silent by design.
   - `solid_line_crossing`: a settled lane change across one of the four solid dividers: 1 s on one side,
     1 s on the other, ≥ 0.15 box heights from the line on both sides, ≥ 80 % of the frames agreeing.
     Box jitter on the line (a standing queue, partial occlusion, merged boxes) does not pass. Precise,
     deliberately low recall.
   - `accident` / `near_miss`: pairs of tracks with a fast approach (> 1.5 box heights/s); contact
     (normalised centre distance < 0.35 or box IoU > 0.35) followed by an abrupt stop of both objects that
     lasts ≥ 4 s = accident; closest approach of 0.5–0.6 box heights without contact, at speed, with hard
     braking **and** a swerve = near miss. Approaching a queued/stopped vehicle at a regular stop location
     is explicitly excluded (the classic false positive); a car–pedestrian contact counts only if the car
     was fast. If the pair breaks up within 1 s after the contact (boxes merge, a track ends), the accident
     is kept only when a surviving track itself comes to rest within 2.5 s and stays at rest for 4 s; a pair
     that simply vanishes at the frame edge is not an accident.
     A second, independent accident rule works on impact dynamics instead of pair geometry (`src/rules/impact.py`): on the raw, unsmoothed detector positions a vehicle keeps a speed of at least 1.5 box heights/s for 0.6 s, loses at least 60 % of it within 0.35 s and stays slow for 0.6 s (a braking car needs 1–3 s for the same drop), another road user is within 2 box heights at that moment, and somebody rests at that spot for at least 3 s afterwards; a track that ends at speed and is replaced by a track born stationary counts the same way. Regular stop locations are excluded, a pedestrian partner must fall or stop. The two rules are united.
   - `road_obstacle`: two sources. COCO "obstacle" classes (animals, suitcase, chair, ball) with confidence
     ≥ 0.35 on the road, and a background model (`src/static_objects.py`) for things the detector does not
     know: a slowly adapting background at 480 px; a foreground blob that persists ≥ 5 s, covers 0.02–3 % of
     the frame, is compact and has its own edges and contrast is a candidate; it is rejected if covered by a
     tracked vehicle/person/rider, off the road, at a regular stop hotspot, with a stationary tracked
     neighbour within one box height for ≥ 50 % of its life (pedestrian shadows, queue members), without
     ≥ 2 vehicles passing by, or living ≥ 60 % of its time during red (a queue the detector did not track).
     Blobs whose life starts within the first 20 s are ignored: the background is initialised from the first
     frame, so anything present at t = 0 leaves a ghost when it moves away. Cost: fractions of a millisecond
     per frame. `fire_smoke`: not predicted (no reliable detector).
8. **Segments.** Per-frame flags become intervals; intervals of one class are merged when closer than a
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
everything else (alignment, signal reading and fusion, background model, the 14 rules, segments, Part B).

## Datasets and licences

| Asset | Use | Licence |
|---|---|---|
| YOLO11n / YOLO11s COCO checkpoints (Ultralytics assets v8.3.0) | detector, as-is | AGPL-3.0 |
| Ultralytics 8.4.152 (YOLO + ByteTrack) | inference and tracking | AGPL-3.0 |
| Sample videos from the organisers | our dev labels and EDA only; not redistributed | organisers' |

No external training data was used; no model was fine-tuned. `configs/scene_ref.npz` holds SIFT features and
a 960 px grey median background of one sample clip, not the footage.

## Determinism and seeds

`src/config.seed_everything()` fixes Python, NumPy and Torch seeds (seed 0) and sets cuDNN to
deterministic mode. ByteTrack is deterministic for a fixed frame order. The scene alignment calls
`cv2.setRNGSeed(0)` before RANSAC and sorts the keypoints, so the same clip gives the same transform. The
background model is integer arithmetic on a fixed frame sequence, the lamp reader and the signal fusion are
pure functions of the frames. Two consecutive runs on the same machine give identical `predictions.json`
(checked by `tests/test_format_and_determinism.py`; CI additionally runs the organizers' harness twice on a
synthetic clip and compares the outputs). Timestamps are rounded to 3 decimals by the harness.

## Time budget

Measured with the organizers' harness (Part A + Part B, including all video decoding) on an RTX 4050
laptop GPU (i7-13650HX), fp16 inference — the final run that produced `predictions_samples.json`
(`python tools/results_table.py --table budget` regenerates the rows from its `log` section):

| Footage | Length | Part A | Part B | Total | Limit |
|---|---|---|---|---|---|
| C3896.MP4 | 340 s | 178 s (0.52×) | 310 s (0.91×) | 488 s = 1.43× | 3× |
| C3897.MP4 | 318 s | 154 s (0.49×) | 275 s (0.86×) | 429 s = 1.35× | 3× |
| C3902.MP4 | 318 s | 174 s (0.55×) | 278 s (0.88×) | 452 s = 1.42× | 3× |
| C3905.MP4 | 128 s | 68 s (0.53×) | 88 s (0.69×) | 156 s = 1.22× | 3× |

The samples are 4K 30 fps at 147 Mbit/s in 4:2:2 10-bit, which NVDEC does not decode, so all decoding is
software on the CPU cores. Part B is dominated by the harness decoding every 4K frame itself; our own work
per frame is one resize and, every 4th frame, a YOLO11n pass. Part A now includes the scene alignment
(1.5–3 s), the lamp reader and the background model, and runs the detector at 1280 px. The final run was made
while the machine was also busy with other work; on an idle machine the first submission (960 px, no
alignment) measured 0.87–1.00× and the Part B code has not changed since, so the numbers above are an upper
bound for this hardware. On C3905 the official harness gave 1.13× at 960 px versus 1.25× at 1280 px under
the same load. Frame stride and input size are environment variables if the budget ever gets tight on
slower hardware: `WIUT_IMGSZ_A=960` saves about 0.1× of the duration (Part A 68 s → 55 s on C3905) at the
price of the far pedestrians, `WIUT_STRIDE_A=4` saves more.

| Variable | Default | Meaning |
|---|---|---|
| `WIUT_STRIDE_A` / `WIUT_IMGSZ_A` | 3 / 1280 | Part A frame stride and long side |
| `WIUT_STRIDE_B` / `WIUT_IMGSZ_B` | 4 / 640 | Part B frame stride and long side |
| `WIUT_MODEL_A` / `WIUT_MODEL_B` | yolo11s.pt / yolo11n.pt | detector checkpoints in `weights/` |
| `WIUT_CONF` / `WIUT_IOU` | 0.15 / 0.5 | detector confidence and NMS IoU for Part A |
| `WIUT_DEVICE` | auto | `cuda:0` or `cpu` |
| `WIUT_SCENE` | configs/scene.json | scene geometry file (the reference `configs/scene_ref.npz` sits next to it) |
| `WIUT_CACHE` | (off) | directory for cached detections, lamp features and static objects (development only) |

## Development workflow

```bash
python tools/inspect_video.py data/samples/C3896.MP4        # tracks, flow field, candidates per rule
python -m src.eda data/samples --out site/static/eda         # EDA figures + eda.json for the website
tools/annotator.html                                         # label the samples → data/labels/my_labels.json
tools/scene_editor.html                                      # draw the scene → configs/scene.json
python tools/make_scene_ref.py --video data/samples/C3896.MP4  # rebuild configs/scene_ref.npz after redrawing
python tools/run_samples.py data/samples --team "Air MAX" --out predictions_samples.json --site
python tools/results_table.py                                # README rows (events per clip, time budget)
python -m pytest tests -q
```

## Results on the sample videos

The four sample clips are one signalised T-junction at rush hour (4K, 30 fps, 2–6 min each). Our output
on them is `predictions_samples.json`; the annotated clips, timelines, the dashboard and the reviewed
contact sheets are on the website. The table is printed by `python tools/results_table.py`:

| Clip | Length | Events we report |
|---|---|---|
| C3896.MP4 | 340 s | red_light 2, wrong_way 1, jaywalking 13, failure_to_yield 5 |
| C3897.MP4 | 318 s | jaywalking 25, failure_to_yield 7 |
| C3902.MP4 | 318 s | stopped_vehicle 1, jaywalking 26, failure_to_yield 1, solid_line_crossing 2 |
| C3905.MP4 | 128 s | jaywalking 6, failure_to_yield 2, stop_line 1 |

We did not have labels for the samples; instead every rule's candidates were reviewed on contact sheets
(6 frames per candidate) and thresholds were tightened until the reviewed candidates were plausible. The
sheet reviews were done on the 960 px detections. The final run uses 1280 px, whose track set differs, so
on the 1280 run every event of a rare class (accident, road_obstacle, wrong_way, solid_line_crossing,
stopped_vehicle) was re-checked on frames with the same rules; that re-check led to the two fixes
(accident, road_obstacle) described below. Per class, what was verified and what was not:

- `jaywalking` (70 segments): pedestrians crossing outside the zebras are constant at this junction; most
  reviewed candidates were real. The crossing polygons are the tightest part of the scene, which is why the
  scene is now aligned per clip: before the alignment C3902 carried a 54 s "jaywalking" segment over the far
  zebra because the polygons were 120 px off.
- `failure_to_yield` (15): fires when a moving car passes through a zebra within 1.5 car heights of a
  pedestrian on it; the reviewed cases were cars pushing through the crossing while people were still on it.
- `red_light` (C3896 73.1–77.9 s and 79.3–82.5 s) and `stop_line` (C3905 79.2–114.8 s) were verified on
  contact sheets and both survived the 1280 run unchanged: in C3896 cars cross the far zebra while
  pedestrians are walking on it, the lamp red since 66 s (plausible true); in C3905 a row of cars stands on
  the zebra beyond the stop line for the whole red phase and pedestrians walk around them (true). Before the
  "red only from the lamp" rule C3897 produced 5 false `red_light` events from pedestrian/queue-derived red
  while the frames show green.
- `accident` (0) / `near_miss` (0): the pair rule is a contact-then-stop signature. On the 960 sheets its
  candidates were an SUV passing pedestrians (C3897 315 s) and queue arrivals (C3905); on the 1280 run it
  first produced five more — the articulated bus stopping beside the head of the queue (C3896 124 s), two
  cars passing at the bottom-right corner of the frame with clipped boxes (C3897 217 s) and three pairs of
  35–100 px boxes at the top-left corner (C3902) — all through one branch: "the pair disappeared right
  after the contact counts as came to rest". That branch was removed: when a pair breaks up within 1 s
  after the contact, the accident is kept only if a surviving track itself comes to rest within 2.5 s and
  stays at rest for 4 s. With that, no accident is reported on the samples; a real collision, where both
  objects stop and stay, still passes. `near_miss` needs speed, hard braking and a swerve together and
  never fired.
  Checked on real crashes: three compilations of Seattle traffic-camera crashes (36 minutes, roughly 40 collisions) and one Tashkent CCTV compilation from 2017. The impact rule fired 5 times: four are real collisions (a rear-end at E Marginal Way, two SUVs at Westlake, a T-bone at Rainier & Henderson, a T-bone in Tashkent) and one is a montage cut between two clips; on our four sample clips it fires 0 times. Recall on the compilations is low, about one collision in eight: many crashes there end in continued motion, or the clip is cut right after the impact, so "somebody rests at the spot for 3 s" is never observed. The old pair rule found one of those collisions. Precision is what we optimised for.
- `wrong_way` (C3896 120.9–124.8 s) is a false positive: the articulated bus driving in the right
  direction along the median towards the stop line; the bottom of its tall box falls into the zone of the
  opposite carriageway. We left it in the output rather than tune against a single case; a lane-level zone
  for the median side would fix it.
- `stopped_vehicle`: 9 events in the first submission, of which 6 were queues or cars waiting at a zebra
  (reviewed on sheets); the tightened rule leaves 0 on the 960 run and 1 on the final run: C3902
  130.2–144.4 s, a white sedan standing in the kerb lane next to the bus stop for 14 s while traffic passes
  it — plausible true on the frames we looked at. On the 960 run the kerb stops near the bus stop fell
  below 10 s or had a stopped neighbour.
- `road_obstacle` (0): the background model found candidates on every clip and rejected all of them on the
  960 run (a pedestrian's shadow on the side zebra in C3897, six pieces of untracked queue during red in
  C3902). The 1280 run first kept one in C3896 (0.3–47.9 s), a blob at the left edge of the frame present
  from the first frame: the background is initialised from the first frame, so whatever is there at t = 0
  leaves a ghost when it moves away. Blobs whose life starts within the first 20 s are now rejected
  (warm-up), and no road_obstacle is reported on the samples.
- `solid_line_crossing`: the settled-lane-change rule gave two candidates on the 960 run, both on sheets:
  C3896 107.6–109.0 s (a black car merging right across `far_L4`, plausible true) and C3897 130.1–131.6 s
  (a lane change behind the mast pole, unclear). Neither appears in the 1280 run, whose track set differs;
  the rule instead reports two crossings in C3902 (1.8–3.8 s across `far_L4` in the moving queue, and
  299.6–301.1 s across `far_L2` near the left edge), which we could not confirm or reject on frames.
- `congestion`, `illegal_u_turn`, `illegal_turn`, `fire_smoke`: not predicted on the final run. The single
  congestion of the first submission (C3905) was a queue during the red phase; with the lamp-based signal
  the rule keeps only queues that survive a green phase, so it is gone. `illegal_turn` is silent by design
  (no arrows on the pavement), `fire_smoke` has no detector.
- The risk curve raises an alarm (≥ 0.5) on 0.1–1.5 % of the frames, 1–8 alarm runs per clip (most in
  C3902), all on fast approaches to a slow or stationary road user near the camera (see the dashboard).

## Team

| Member | Role | Did what |
|---|---|---|
| Xusan Mirzabayev | pipeline, tracking, event rules, risk estimator | detector + ByteTrack integration, ffmpeg reader and time budget, rule engine for all 14 classes and segment post-processing, Part B, Docker, tests |
| Avazbek Akbaraliyev (captain) | scene geometry, dev review, report | scene drawing, manual review of candidates on the samples, threshold tuning, failure cases, organizer communication |
| Munis Tursunov | website, deployment, EDA | team website and live demo, Railway hosting, EDA figures, results pages, report editing |

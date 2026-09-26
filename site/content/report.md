# Technical report

## What we built

A detector-plus-rules system for one fixed road camera. A COCO-pretrained YOLO11s finds road users,
ByteTrack turns detections into trajectories, a hand-drawn scene (road, intersection, direction zones,
crossings, solid lines, stop line, signal ROI) is aligned to each clip and gives the trajectories meaning,
and one small rule per event class turns them into time segments. The traffic-light state is read from
the lamp itself and cross-checked against pedestrian and traffic behaviour. A background model finds
static objects the detector does not know. A separate causal estimator (YOLO11n, every 4th frame) scores
the risk of an accident from time-to-collision, braking and queue context. Everything is deterministic,
offline, and runs at 1.2–1.4× of the video duration on the 4K samples with a laptop GPU while the machine
was busy with other work (0.9–1.0× idle on the first submission; limit 3×).

## The camera

Four 4K clips (30 fps, 147 Mbit/s, 4:2:2 10-bit, 2–6 minutes) of a signalised T-junction on a wide avenue:
two carriageways, a side street with two zebra crossings, a raised island with the traffic light, a bus
stop on the far side. Rush-hour density: 500–700 vehicle tracks and 270–700 pedestrian tracks per clip,
up to 80 objects in a frame. Pedestrians cross diagonally outside the zebras all the time; the far
carriageway queues through more than one signal cycle at dusk. The 10-bit 4:2:2 encoding rules out NVDEC,
so every 4K frame is decoded in software.

## Scene alignment

**Why.** The scene was drawn on one frame of C3896. Reviewing the first submission on contact sheets showed
that C3902 is framed about 120 px to the left and 70 px lower: everybody on the far zebra was reported as
jaywalking for 54 s because the crossing polygon missed the stripes. The camera is "fixed" in the sense of
the task, not to the pixel.

**How** (`src/align.py`). Nine frames spread over the clip are read with `ffmpeg -ss` at 960 px and
reduced to a median background, which removes every moving object. SIFT features of that background are
matched to the reference background of C3896 (`configs/scene_ref.npz`, built once by
`tools/make_scene_ref.py`) and a similarity transform is estimated by RANSAC with a fixed seed. It is
accepted only with ≥ 40 inliers, a scale of 0.8–1.25 and a rotation ≤ 5°; otherwise the clip is treated as
another camera and every scene-based rule stays silent. The three traffic lights of the scene are compared
with the static "traffic light" tracks of the detector as an independent check that goes into the log only.

| Clip | Transform of the scene into the clip (4K px) | Inliers | Light residuals before → after |
|---|---|---|---|
| C3896 | identity | 2714 | 0.7 / 0.9 / 0.7 px |
| C3897 | shift (0.7, −0.3) | 1328 | 1.6 / 1.3 / 1.3 → 0.9 / 1.2 / 1.3 px |
| C3902 | shift (−122.0, +70.8), scale 1.014, rotation −0.75° | 137 | 87 / 95 / 125 → 7.3 / 3.2 / 7.4 px |
| C3905 | shift (−33.7, +41.9), scale 1.014, rotation −1.08° | 120 | 21 / 15 / 48 → 7.9 / 36 / 7.8 px |
| other cameras (our dev clips) | rejected | 16–22 | — |

Cost 1.5–3 s per clip. The remaining 36 px on the middle light of C3905 is the light itself: at dusk the
box the detector puts on that lamp sits higher than in daylight; the road markings align to a few pixels.

## Reading the traffic light

The first version read the lamp colour with absolute HSV thresholds in its ROI and failed: daylight glare
gave "amber/unknown" for whole clips, dusk gave "red" almost constantly, and the fallback (a standing queue
= red) stayed red through most of the rush hour. We therefore switched `red_light` and `stop_line` off for
the first submission. The final version fuses three sources (`src/signal.py`):

| Source | Looks at | Weakness |
|---|---|---|
| Lamp reader | the lamp cut from the 4K frame in the same ffmpeg pass; sections red/amber/green located from the "traffic light" track of the detector (or the ROI); each section scored by its colour energy **relative to the other sections** of the same lamp; occlusion detection when a bus passes in front; hysteresis of 0.4 s, gaps bridged up to 4 s | needs the lamp in the frame |
| Pedestrian phases | red while ≥ 2 pedestrians (or one for ≥ 3 s) walk on the zebra the stop line protects; green while vehicles cross the line at speed with nobody on the zebra | people cross on their own red here |
| Traffic phases | a queue of ≥ 2 vehicles standing at the line for ≥ 3 s = red; crossing without a queue = green | a queue stands on green too |

**Fusion.** The lamp wins when its phases are plausible — both colours present, red/green phases of
10–150 s, amber ≤ 6 s — and either periodic (at least two interior red and two green phases within ±25 % of
their medians) or agreeing with the pedestrian phases ≥ 60 % of the time where both are known. Otherwise
pedestrians, then traffic. The fused track carries a `reliable` flag (at least one red and one green phase
of plausible length); without it the signal rules stay silent.

| Clip | Decision | Phases read from the lamp |
|---|---|---|
| C3896 (day, faint lamp) | lamp, periodic, agreement with pedestrians 0.95 | red 0–27, green 27–66, red 66–103, green 103–141 … (10 phases) |
| C3897 (day) | lamp, periodic, agreement 0.72 (jaywalkers) | red 0–21, green 21–59, red 59–95, green 95–134, red 134–170, green 170–209, red 209–245, green 245–284, red 285–318 |
| C3902 (dusk) | lamp, periodic, agreement 1.00 | red 0–37, green 38–75, amber 76–78, red 78–118 … |
| C3905 (dusk) | lamp, agreement 0.90 over 105 s | red 0–35, green 35–73, amber 73–76, red 76–115, green 115–128 |

A 75 s cycle with red ≈ 36–39 s, green ≈ 38 s and amber ≈ 3 s, the same at noon and at dusk.

## red_light and stop_line

Both rules are conservative because a wrongly predicted class costs more than a missed one under
macro-F1, and both take red **only from the lamp**: pedestrians at this junction cross on their own red
and a queue stands on green, so both other sources produce false red. Before that restriction C3897 had
5 `red_light` events whose frames show a green lamp.

- `red_light`: a vehicle crosses the stop line in the approach direction at speed after the red has been on for ≥ 1.5 s and ≥ 1 s before green, and enters the intersection within 4 s; the event ends when it leaves the intersection (≤ 8 s). Verified on contact sheets: C3896 73.1–77.9 s and 79.3–82.5 s — cars cross the far zebra while pedestrians are walking on it, the lamp red since 66 s. Plausible true.
- `stop_line`: a vehicle that arrived during the red stands 0.35–2.5 box heights beyond the line for ≥ 2 s without entering the intersection; the event lasts until green. Verified: C3905 79.2–114.8 s — a row of cars stands on the zebra beyond the stop line for the whole red phase and pedestrians walk around them. True.

## Static objects (road_obstacle)

The COCO detector knows a few "obstacles" (animals, suitcase, chair, ball) but not a tyre, a box or a
fallen load. `src/static_objects.py` looks for them with a background model that runs in the same frame
pass as the detector at 480 px, at a cost of fractions of a millisecond per frame: a slowly adapting
background (one grey level per frame, only where the frame agrees with the background), a map of "since
when does this pixel differ", and a candidate whenever a compact blob with its own edges and contrast has
persisted for ≥ 5 s and covers 0.02–3 % of the frame. A candidate is rejected if it is covered by a
tracked vehicle, person or rider, lies off the road or at a regular stop hotspot, has a stationary tracked
neighbour within one box height for ≥ 50 % of its life, sees fewer than 2 vehicles pass by, or lives ≥ 60 %
of its time during red.

On the four samples the model produced candidates on every clip and, on the development run, rejected all
of them. Two typical ones are on the Analysis page: C3897 58–78 s, the shadow of a pedestrian standing on
the side zebra (rejected by the stationary-neighbour filter), and C3902 80–124 s, six blobs that are parts
of cars in a dense queue during red which the detector missed (rejected by the red-life filter). The 1280 run
first kept one object, C3896 0.3–47.9 s: a 32×64 px blob at the left edge of the frame that is there from
the first frame. The background is initialised from the first frame, so whatever stands at t = 0 and later
moves away leaves a ghost; blobs whose life starts within the first 20 s are now rejected (warm-up), and
no road_obstacle is reported on the four samples.

## Lanes and solid lines

`configs/scene.json` now carries five direction zones — `far_approach` (5 lanes towards the stop line,
direction 25°), `far_exit`, `near_carriageway` (from the right edge through the zebra to the upper left
corner), `side_street_slip`, `side_street_main` — built from the flow of the tracks of C3896, C3897 and
C3905, with an unreferenced strip left between opposing zones. No arrows are painted on the pavement, so
every zone allows straight, left and right and `illegal_turn` stays silent by design. Four solid dividers
between the approach lanes (`far_L1`–`far_L4`) were traced on an empty frame of C3897 and verified
continuous by a paint-profile scan.

- `solid_line_crossing` requires a settled lane change: 1 s on one side, 1 s on the other, ≥ 0.15 box heights from the line in both windows, ≥ 80 % of the frames agreeing. Box jitter on the line (a standing queue, a partial occlusion, merged boxes) does not pass. On the development run it produced two candidates, both on contact sheets: C3896 107.6–109.0 s, a black car merging right across `far_L4` (plausible true), and C3897 130.1–131.6 s, a lane change behind the mast pole (unclear). Precise, low recall.
- `wrong_way` uses the zone direction inside the zones and the flow field outside; it ignores observations whose box is clipped at the frame edge, stays silent inside the intersection and needs ≥ 2.5 s and ≥ 2 box heights of displacement against the reference.

## Detector input: 1280 vs 960

Part A now runs the detector at 1280 px on the long side instead of 960 px. The ablation was done on C3905
(dusk, the hardest clip), detections and tracking only, same cache format:

| Configuration | Tracks | Vehicles | Persons | Person observations | Median person height |
|---|---|---|---|---|---|
| YOLO11s 960, stride 3 (first submission) | 556 | 255 | 269 | 19.2k | 133 px |
| YOLO11s 1280, stride 3 (final) | 763 | 302 | 430 | 30.6k | 108 px |
| YOLO11s 960, stride 2 | 674 | 292 | 329 | 29.6k | 130 px |
| YOLO11n 960, stride 3 | 580 | 315 | 249 | 17.1k | 143 px |

The larger input finds the small, far pedestrians (the median person shrinks because new small ones are
added); a denser stride keeps more tracks alive but does not see them; the small model loses them. With the
official harness on C3905 the price is 1.13× → 1.25× of the duration (Part A 55 → 68 s, Part B unchanged),
measured with another process using the CPU.

## Time budget

Final run with the organizers' harness (Part A + Part B, all decoding included) on an RTX 4050 laptop GPU
(i7-13650HX), fp16, while other work was running on the machine:

| Footage | Length | Part A | Part B | Total | Limit |
|---|---|---|---|---|---|
| C3896.MP4 | 340 s | 178 s (0.52×) | 310 s (0.91×) | 488 s = 1.43× | 3× |
| C3897.MP4 | 318 s | 154 s (0.49×) | 275 s (0.86×) | 429 s = 1.35× | 3× |
| C3902.MP4 | 318 s | 174 s (0.55×) | 278 s (0.88×) | 452 s = 1.42× | 3× |
| C3905.MP4 | 128 s | 68 s (0.53×) | 88 s (0.69×) | 156 s = 1.22× | 3× |

Part B is dominated by the harness decoding every 4K frame in software; our own work per frame is one
resize and, every 4th frame, a YOLO11n pass. Part A includes the alignment (1.5–3 s), the lamp reader and
the background model. The first submission on an idle machine measured 0.87–1.00× at 960 px; the Part B
code has not changed since, so the difference is machine load. `WIUT_IMGSZ_A=960` saves about 0.1× of the
duration if a machine is slow, `WIUT_STRIDE_A=4` more.

## What worked

- **Relative thresholds.** Speeds and distances in box heights per second made one set of thresholds work from the near lanes (300 px boxes) to the far ones (60 px), without any metric calibration.
- **Queue awareness.** "A car joins a queue" looks like a rear-end collision, a stopped vehicle and congestion at once. A stationary neighbour, a regular stop location, and now "above the stop line while the lamp is red" removed the bulk of false alarms in Part A and in the risk estimator.
- **Relative colour instead of absolute colour.** Comparing the three sections of the lamp with each other reads a faint lamp at noon and a bright one at dusk with the same rule; the periodicity test then tells a real signal cycle from noise.
- **Aligning instead of tolerating.** A 120 px shift cannot be absorbed by a tolerance on the polygons; a similarity transform from the background removes it for good and rejects footage from other cameras.
- **One decode.** Reading 4K through an ffmpeg pipe that subsamples, rescales and cuts the lamp crop in one filter graph cut Part A from 1.0× to about 0.3× on an idle machine.
- **Vectorised risk.** The pairwise time-to-collision over up to 80 objects is a few numpy matrix operations; the Python version cost 90 ms per frame in this scene, the vectorised one about 3 ms.
- **Re-checking the rare classes on the final detections.** Switching to 1280 px changed the track set; looking at every accident, obstacle, wrong-way, solid-line and stopped candidate of the new run on frames caught two rule errors (the frame-edge "came to rest" branch, the first-frame ghosts) that the 960 review could not show.
- **The harness first.** The `run_submission.py` / `evaluate.py` of the organizers were used unchanged from the first hour; CI runs them twice on a synthetic clip and compares the outputs bit for bit.

## Failure cases and what we learned

Every candidate of the first submission was reviewed on 6-frame contact sheets before thresholds were
tightened; the sheets with verdicts are on the Analysis page. The lessons, one per family:

| Case | Verdict | What it taught us |
|---|---|---|
| C3902 jaywalking 140 s: everybody is on a zebra | false positive | the scene must be aligned per clip; done, the 54 s segment is gone |
| C3896 near_miss 45 s: a moped and a pedestrian at walking pace next to a queue | false positive | near_miss needs speed, hard braking **and** a swerve |
| C3897 accident 315 s: an SUV passing pedestrians on the side zebra | false positive | box overlap between a car and pedestrians on a crossing is normal; contact for car–pedestrian pairs should be measured on ground points and needs a fast car |
| C3905 accident (two sheets): cars closing the gap in a dusk queue; the long box of a semi-trailer overlapping its neighbours in a turn | false positive | catching up with a queue is excluded by the stop hotspot; box IoU is a poor contact test for long vehicles seen at an angle |
| C3896 stopped_vehicle 255 s, C3897 293 s, C3905 1.5 s: cars in the signal queue, a car at the head of the queue, a car and a moped waiting at the zebra | false positive | a queue is any stop above the stop line during red; a stop within 2.5 box heights of a zebra is waiting; check the neighbour at three moments, not one |
| C3897 stopped_vehicle 51 s, 230 s: a man or a moped rider standing at the island tip | false positive | riders and pedestrians must not feed vehicle rules; these are jaywalking |
| C3896 illegal_u_turn 56 s, C3902 2 s: no vehicle turns around | false positive | heading from a track that jumps inside a queue, or from the first two seconds of a track, is not a manoeuvre; the class is absent from the output |
| C3896 wrong_way 7 s, 92 s: cars turning out of the side street | false positive | a turn across the lane direction is not wrong-way driving; the rule now needs displacement against the flow |
| C3897 road_obstacle 58 s: the shadow of a standing pedestrian | false positive | context beats appearance: a stationary tracked neighbour for most of the life of the blob |
| C3902 road_obstacle 80–124 s: parts of untracked cars in a queue during red | false positive | a blob that lives mainly during red is a queue |
| C3896 red_light 73 s, 79 s; C3905 stop_line 79 s | true positive | red only from the lamp; the earlier pedestrian/queue-derived red gave 5 false red_light events in C3897 |
| C3896 solid_line_crossing 107 s | plausible true | the settled-lane-change test keeps box jitter out |
| C3897 solid_line_crossing 130 s: behind the mast pole | unclear | lines behind the pole are the weak spot; stills cannot separate a lane change from a box jump |

The final run uses 1280 px and its track set differs from the reviewed 960 px run, so every event of a
rare class on the 1280 run (accident, road_obstacle, wrong_way, solid_line_crossing, stopped_vehicle) was
re-checked on frames with the same rules. That check found two systematic errors, and both were fixed:

- **Accidents at the frame edge.** Five candidates — the articulated bus stopping beside the head of the queue (C3896 124 s), two cars passing at the bottom-right corner with clipped boxes (C3897 217 s), three pairs of 35–100 px boxes at the top-left corner of C3902 — all came from one branch of the rule: a pair that disappeared right after the contact counted as "came to rest". The branch is gone; when a pair breaks up within 1 s after the contact, the accident is kept only if a surviving track itself comes to rest within 2.5 s and stays at rest for 4 s. No accident is reported on the samples now; a real collision, where both objects stop and stay, still passes.
- **Ghosts of the first frame.** The background model starts from the first frame, so a blob that is there at t = 0 and later moves away becomes a "static object" (C3896 0.3–47.9 s at the left frame edge). Blobs whose life starts within the first 20 s are rejected now; no road_obstacle is reported on the samples.

What remains on the 1280 run and was left as it is: C3896 wrong_way 120.9–124.8 s (the articulated bus
driving in the right direction along the median, the bottom of its tall box in the zone of the opposite
carriageway — a false positive we keep rather than tune against a single case), C3902 stopped_vehicle
130.2–144.4 s (a white sedan standing in the kerb lane by the bus stop for 14 s while traffic passes —
plausible true) and C3902 solid_line_crossing 1.8–3.8 s and 299.6–301.1 s (the moving queue in the first
seconds and a large box near the left edge — unclear on frames).

## Real crashes as a sanity check

Our clips contain no collision, so the accident rule had never seen a positive. We took public footage of real ones: three compilations of crashes caught by Seattle traffic cameras (36 minutes, roughly 40 collisions, fixed cameras, dissolves between clips) and a Tashkent CCTV compilation from 2017. The pair rule (contact, then both objects at rest) found one collision out of all of them: on a real impact the box centres stay 0.5–0.9 box heights apart and the tracker breaks the tracks at the moment of contact, so "the same pair touched and then rested" almost never holds.

That led to the second rule, which looks at impact dynamics on the raw detector positions instead of pair geometry: a sustained speed of at least 1.5 box heights/s that drops by at least 60 % within 0.35 s and stays low (braking takes 1–3 s for the same drop), another road user within 2 box heights, and somebody at rest at that spot for at least 3 s afterwards. A track that ends at speed and is replaced by one born stationary counts the same way.

| Firing | Footage | Verdict |
|---|---|---|
| 677 s, Rainier & Henderson | Seattle #13 | true: a white SUV T-bones a dark SUV, both stop together |
| 338 s, E Marginal Way & S Hudson | Seattle #13 | true: a taxi rear-ends a grey car |
| 98 s, Westlake Ave | Seattle #9 | true: two SUVs collide at the crossing |
| 61 s, Tashkent, 2017 | Tashkent CCTV | true: a white car T-bones a blue one |
| 509 s, Alaskan Way | Seattle #9 | false: a dissolve between two clips, not a collision |

Five firings, four real; zero firings on our four sample clips (checked on the 1280 px detections). Recall on the compilations stays low, about one collision in eight: many crashes there end in continued motion, or the clip is cut right after the impact, so the "rest" condition is never observed. Part B raises its alarm at the moment of impact on several of these clips (score 1.0 at Rainier and E Marginal Way) rather than five seconds before; anticipation from a fixed camera without a learned model remains the open problem.

## Limitations, honestly

- **No labels.** The samples came without annotations; precision was reviewed by eye on contact sheets, recall was never measured. We know what we report is mostly plausible; we do not know what we miss.
- **Classes never predicted on the samples**: `near_miss` (the evasive-action test is strict on purpose), `illegal_u_turn` (heading-based candidates were all id switches), `illegal_turn` (no arrows on this junction), `congestion` (the one candidate of the first submission was a queue during red; with the lamp-based signal only queues that survive a green phase count), `accident` and `road_obstacle` (every candidate on these clips was a false positive, see above), `fire_smoke` (no detector). On a hidden test from the same camera these classes are at risk of zero recall.
- **Boundaries.** Our segments follow the annotation conventions literally, but "queue clears" or "all objects stop moving" can differ from a human annotator by a second or two — visible at IoU 0.7.
- **The accident rules** report nothing on the samples. On public crash footage the impact rule finds real collisions with four true firings out of five, but only about one collision in eight (see the sanity check above): crashes where the vehicles keep moving, or that the footage cuts away from, are missed. A re-scoring model trained on public crash datasets would be the next step.
- **The scene is one camera.** Direction zones, solid lines and the stop line exist only for the far approach and the side street; the near carriageway has no stop line in view, so no `red_light` can be reported there.
- **Timing** was measured on a busy laptop; the hardware of the organizers may differ, and the environment variables exist for that reason.

## What we would do next

- Label the four sample clips fully (the browser annotator is in `tools/`) and replace hand-tuned thresholds with a small gradient-boosted classifier over the same trajectory features.
- A clip classifier for `accident` / `near_miss` trained on public dashcam/CCTV datasets (DoTA, CCD) to re-score the rule candidates.
- Ground-point contact for car–pedestrian pairs and long vehicles instead of box IoU.
- A lane-level zone for the median side of the far approach (the bus case) and an occlusion mask for the mast pole.
- A full contact-sheet review of the 1280-run events we could only check on frames (the C3902 solid lines and the kerb stop).

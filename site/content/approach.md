# Problem and approach

The camera is fixed, so the road never moves: only the road users do. We therefore treat every event
class as a statement about **trajectories** on a **known scene**, and we keep the only learned component
(the object detector) exactly as it comes: no training, no labels needed, nothing that could overfit
four sample videos.

## Pipeline

1. **Frames.** Every 3rd frame (10 fps for the 30 fps samples), resized to 1280 px on the long side. The video is read through an ffmpeg pipe that subsamples and rescales inside the decoder; the same pass cuts a full-resolution crop around the traffic light and glues it under the frame, so the lamp is read at 4K without a second decode. The samples are 4K 4:2:2 10-bit, which NVDEC cannot decode, so on this footage the decoding is software on all cores.
2. **Scene alignment.** The scene was drawn on one frame of C3896, but the camera is not perfectly fixed between clips: C3902 is framed about 120 px to the left and 70 px lower. Before anything else we estimate a similarity transform per clip: a median background from 9 frames (`ffmpeg -ss`, 960 px), SIFT features matched against the reference background of C3896 (`configs/scene_ref.npz`), RANSAC. The transform is accepted only with ≥ 40 inliers, a scale of 0.8–1.25 and a rotation ≤ 5°; otherwise the footage is treated as another camera and the scene is switched off. Cost: 1.5–3 s per clip.
3. **Detection + tracking.** YOLO11s (COCO weights) finds persons, bicycles, cars, motorcycles, buses, trucks, traffic lights and a few animals/objects. ByteTrack (built into Ultralytics) links them into tracks with stable ids. The detection threshold is deliberately low (0.15): ByteTrack itself separates confident boxes from weak ones and uses the weak ones only to bridge gaps. 1280 px instead of 960 px costs about 0.1× of the video duration and raises the number of tracked pedestrians on C3905 from 269 to 430 (the far zebra is 60–100 px tall in 4K).
4. **Trajectories.** Every track gets a smoothed centre, a ground-contact point (bottom of the box), speed, acceleration and heading. All thresholds are expressed in *box heights per second*, so a rule tuned near the camera also holds far away.
5. **Scene geometry.** A small JSON drawn once in our scene editor: road polygon, intersection polygon, three pedestrian crossings, the stop line of the far approach with its approach direction, five direction zones (far approach, far exit, near carriageway, two parts of the side street) with the manoeuvres they allow, four solid lane dividers on the far approach, the traffic islands and the traffic-light ROI. The task explicitly allows hard-coding this. Without the file (live demo on arbitrary footage) the system estimates a direction field and a road mask from the video's own vehicle tracks and disables the rules that need crossings, stop lines or solid lines.
6. **Traffic-signal state.** Three independent estimates are fused (see below). The rules that depend on the signal (`red_light`, `stop_line`, queue exclusions) take red only from the lamp.
7. **Rules per class.** One small module per class turns tracks + scene into per-frame flags, following the start/end conventions of the annotation guide. Static objects on the road come from a background model that runs in the same pass as the detector.
8. **Segments.** Flags become intervals; intervals of one class are merged across short gaps, blips shorter than a class-specific minimum are dropped, same-class segments never overlap.

## Reading the traffic light

The first version read the lamp colour with absolute HSV thresholds and failed: daylight glare gave
"unknown" for whole clips, dusk gave "red" almost constantly. The final version fuses three sources:

| Source | What it looks at | Strength |
|---|---|---|
| Lamp reader | the lamp cut from the 4K frame; the three sections are located from the detector's "traffic light" track (or the ROI) and classified by the colour energy of each section **relative to the other sections** of the same lamp, with occlusion detection (a bus passing in front) and hysteresis | primary |
| Pedestrian phases | red while pedestrians walk on the zebra the stop line protects; green while vehicles cross the line at speed and nobody is on the zebra | check and fallback |
| Traffic phases | a standing queue at the line = red, vehicles crossing = green | weakest: a queue stands on green too |

The lamp wins when its phases are plausible (both colours present, phases of 10–150 s, amber ≤ 6 s) and
either periodic (at least two interior red and two green phases within ±25 % of their medians) or in
agreement with the pedestrian phases at least 60 % of the time. On all four samples the lamp was
accepted: a 75 s cycle with red ≈ 36–39 s, green ≈ 38 s, amber ≈ 3 s; agreement with pedestrians
0.72–1.00 (the 0.72 is C3897, where people cross on their own red).

## Learned vs rule-based

| Component | Kind | Notes |
|---|---|---|
| YOLO11s / YOLO11n detector | learned (COCO, not fine-tuned) | open weights, AGPL-3.0 |
| ByteTrack association | algorithmic | deterministic for a fixed frame order |
| Scene geometry | hand-made | `configs/scene.json`, drawn with our scene editor |
| Scene alignment | algorithmic | SIFT + RANSAC against `configs/scene_ref.npz`, seeded |
| Signal reading and fusion, background model, all 14 event rules, segment post-processing | rule-based | `src/signal.py`, `src/static_objects.py`, `src/rules/`, `src/segments.py` |
| Risk estimator (Part B) | rule-based | time-to-collision, braking, pedestrians on the road |

## The rules in one line each

- `stopped_vehicle`: speed < 12 % of box height for ≥ 10 s on the carriageway; not a queue (another stopped vehicle nearby, checked at three moments of the stop; a regular stop location; any stop above the stop line while the lamp is red); not within 2.5 box heights of a zebra (waiting for pedestrians or for its own signal); a bus at a stop only after 60 s.
- `congestion`: per direction of flow, ≥ 4 vehicles of which ≥ 75 % crawl for ≥ 45 s; with a visible signal, only queues that survive a green phase.
- `wrong_way`: heading opposes the direction zone of the scene (or, outside the zones, the dominant flow of the cell) by more than 120° for ≥ 2.5 s with a displacement of ≥ 2 box heights against the flow. Silent inside the intersection polygon, in the unreferenced strip between opposing zones, and for boxes clipped at the frame edge.
- `jaywalking`: pedestrian ground point on the road and outside every crossing.
- `failure_to_yield`: a moving vehicle inside a crossing while a pedestrian is on it.
- `red_light`: a vehicle crosses the stop line in the approach direction at speed while the lamp has been red for ≥ 1.5 s and stays red for ≥ 1 s more, and enters the intersection within 4 s; the event ends when it leaves the intersection (≤ 8 s).
- `stop_line`: a vehicle that arrived during the red stands 0.35–2.5 box heights beyond the line for ≥ 2 s without entering the intersection; the event lasts until green.
- `illegal_u_turn` / `illegal_turn`: heading change ≥ 150°, or 60–125° from a zone that does not allow it. No arrows are painted on this junction, so every zone allows every manoeuvre and `illegal_turn` stays silent by design.
- `solid_line_crossing`: a settled lane change across one of the four solid dividers: 1 s on one side, 1 s on the other, ≥ 0.15 box heights from the line on both sides, ≥ 80 % of the frames agreeing. Precise and deliberately low on recall.
- `accident`: fast approach (> 1.5 box heights/s), contact (normalised centre distance < 0.35 or box IoU > 0.35), then an abrupt stop of both objects that lasts ≥ 4 s. Approaching a queued vehicle at a regular stop location is excluded on purpose; a car–pedestrian contact counts only if the car was fast. If the pair breaks up within 1 s after the contact, the accident is kept only when a surviving track itself comes to rest within 2.5 s and stays at rest for 4 s; a pair that vanishes at the frame edge is not an accident.
- `near_miss`: closest approach of 0.5–0.6 box heights without contact, at speed, with hard braking **and** a swerve. Either alone is too common in dense traffic.
- `road_obstacle`: COCO "obstacle" classes (animals, suitcase, chair, ball) on the road, plus objects from the background model: a foreground blob that persists ≥ 5 s, is compact, has its own edges and contrast, covers 0.02–3 % of the frame, is not covered by a tracked vehicle or person, is on the road, not at a regular stop location, has no stationary tracked neighbour for most of its life, sees ≥ 2 vehicles pass by, does not live mainly during red and does not start within the first 20 s (the background is initialised from the first frame, so anything present at t = 0 leaves a ghost when it moves away). `fire_smoke`: not predicted (no reliable detector).

## Part B: accident anticipation

A separate causal estimator sees frames one by one: YOLO11n at 640 px every 4th frame, ByteTrack, 1.5 s of
history per track. For every pair of road users on a collision course (lateral miss distance smaller than
the box widths, approach along the line of centres) it computes the time to collision and checks whether
the follower is already braking hard enough to stop in time. Hard braking and pedestrians on the
carriageway add risk. Regular stop locations are learned online from frames already seen, so "caught up
with the queue" does not raise an alarm. The score is squared (alarms need strong evidence) and decays
without a signal. `step()` never opens the video and never uses Part A output.

## Why this and not a video model

Two and a half days, no labels, a hidden test from the same camera. A detector-plus-rules system is
transparent (every event can be explained by a trajectory and a line on the scene), tunable on our own
review of candidates in minutes, deterministic, and fast enough for 4K within the budget. A fine-tuned
video classifier would need data we do not have and would be hard to debug at IoU 0.7 boundaries.

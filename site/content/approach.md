# Problem and approach

The camera is fixed, so the road never moves: only the road users do. We therefore treat every event
class as a statement about **trajectories** on a **known scene**, and we keep the only learned component
(the object detector) exactly as it comes: no training, no labels needed, nothing that could overfit
four sample videos.

## Pipeline

1. **Frames.** Every 3rd frame (8.3 fps from a 25 fps source), resized to 960 px on the long side.
   Skipped frames are decoded but never converted, which keeps a 4K video well inside the 3× time budget.
2. **Detection + tracking.** YOLO11s (COCO weights) finds persons, bicycles, cars, motorcycles, buses,
   trucks, traffic lights and a few animals/objects. ByteTrack (built into Ultralytics) links them into
   tracks with stable ids. The detection threshold is deliberately low (0.15): ByteTrack itself separates
   confident boxes from weak ones and uses the weak ones only to bridge gaps.
3. **Trajectories.** Every track gets a smoothed centre, a ground-contact point (bottom of the box),
   speed, acceleration and heading. All thresholds are expressed in *box heights per second*, so a rule
   tuned near the camera also holds far away.
4. **Scene geometry.** A small JSON drawn once on a frame: road polygon, lanes with direction vectors and
   allowed manoeuvres, pedestrian crossings, stop lines with approach direction, solid lines, the
   traffic-signal ROI. The task explicitly allows hard-coding this. Without the file (live demo on
   arbitrary footage) the system estimates a direction field and a road mask from the video's own vehicle
   tracks and disables the rules that need crossings or stop lines.
5. **Rules per class.** One small module per class turns tracks + scene into per-frame flags, following
   the start/end conventions of the annotation guide.
6. **Segments.** Flags become intervals; intervals of one class are merged across short gaps, blips
   shorter than a class-specific minimum are dropped, same-class segments never overlap.

## Learned vs rule-based

| Component | Kind | Notes |
|---|---|---|
| YOLO11s / YOLO11n detector | learned (COCO, not fine-tuned) | open weights, AGPL-3.0 |
| ByteTrack association | algorithmic | deterministic for a fixed frame order |
| Scene geometry | hand-made | `configs/scene.json`, drawn with our scene editor |
| Signal colour, all 14 event rules, segment post-processing | rule-based | `src/rules/`, `src/segments.py` |
| Risk estimator (Part B) | rule-based | time-to-collision, braking, pedestrians on the road |

## The rules in one line each

- `stopped_vehicle`: speed < 12 % of box height for ≥ 10 s on the carriageway, not in a queue, not at a
  regular stop location, not before a stop line on red.
- `congestion`: per direction of flow, ≥ 4 vehicles of which ≥ 75 % crawl for ≥ 45 s; with a visible
  signal, only queues that survive a green phase.
- `wrong_way`: heading opposes the lane (or the dominant flow) by more than 120° for ≥ 1.5 s with real
  displacement against the flow.
- `jaywalking`: pedestrian ground point on the road and outside every crossing.
- `failure_to_yield`: a moving vehicle inside a crossing while a pedestrian is on it.
- `red_light` / `stop_line`: the light colour is read from its ROI; crossing a stop line in the approach
  direction on red; stopping beyond the stop line on red without entering the intersection.
- `illegal_u_turn` / `illegal_turn`: heading change ≥ 150°, or 60–125° from a lane that does not allow it.
- `solid_line_crossing`: the ground point crosses a solid-line polyline while moving.
- `accident`: fast approach, contact (normalised distance < 0.45 or IoU > 0.25), then both objects stop
  abruptly. Approaching a queued vehicle at a regular stop location is excluded on purpose.
- `near_miss`: close approach without contact plus hard braking or a swerve; both road users moving.
- `road_obstacle`: animals or objects on the road. `fire_smoke`: not predicted (no reliable detector).

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
dev labels in minutes, deterministic, and fast enough for 4K within the budget. A fine-tuned video
classifier would need data we do not have and would be hard to debug at IoU 0.7 boundaries.

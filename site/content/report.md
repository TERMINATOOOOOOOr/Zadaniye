# Technical report

_Draft written before the sample videos became available; the numbers on our own labels will be added after the final run._

## What we built

A detector-plus-rules system for a fixed road camera. A COCO-pretrained YOLO11s finds road users,
ByteTrack turns detections into trajectories, a hand-drawn scene (lanes, crossings, stop lines, signal
ROI) gives the trajectories meaning, and one small rule per event class turns them into time segments.
A separate causal estimator (YOLO11n, every 4th frame) scores the risk of an accident from time-to-collision,
braking and pedestrians on the carriageway. Everything is deterministic, offline and runs at about 0.9× of
the video duration on 4K footage with a laptop GPU (limit: 3×).

## What worked

- **Relative thresholds.** Expressing speeds and distances in box heights per second made one set of
  thresholds work across the whole field of view; nothing had to be calibrated in metres.
- **Queue awareness.** The single biggest source of false alarms was "a car joins a queue at a red light":
  it looks like a rear-end collision, a stopped vehicle and congestion at once. Treating a stationary
  neighbour or a regular stop location as "queue" removed almost all of it on real intersection footage.
- **Causality by construction.** The risk estimator has no access to the file and no access to Part A;
  regular stop locations are learned only from frames already seen.
- **Time budget.** fp16 inference, a frame stride of 3 for Part A and 4 for Part B, and `grab()` for the
  skipped frames keep 4K within a third of the budget.
- **Engineering.** The organizers' harness and metric were used unchanged from the first hour; a Docker
  image reproduces the two official commands offline; two consecutive runs give identical output.

## What did not work (or is not solved)

- `fire_smoke` is not predicted: we found no reliable open-weights detector and no examples to tune a
  colour heuristic on. A missed class costs one zero in the macro average; a noisy one would cost the same
  and add false positives elsewhere.
- Classes that need the scene (`red_light`, `stop_line`, `jaywalking`, `failure_to_yield`,
  `solid_line_crossing`, `illegal_turn`) are only as good as the scene file. Without `camera.md` we drew
  the scene ourselves from the sample frames.
- Crash compilations from other cameras produced spurious U-turns and near misses at clip cuts; this
  does not happen on a single fixed camera, but it shows the rules trust track continuity.
- Boundaries at IoU 0.7 remain the hardest part: the annotation conventions (e.g. "all involved objects
  stop moving") are implemented literally, but the exact frame an annotator chose can still differ by a
  second or two.

## What we would do next

- Label more footage from the same camera and replace the hand-tuned thresholds by a small
  gradient-boosted classifier over the same trajectory features.
- A short clip classifier for `accident` / `near_miss` trained on public dashcam/CCTV crash datasets
  (DoTA, CCD) to re-score the rule candidates.
- Read the traffic signal from vehicle behaviour when the light itself is not visible.
- An operator dashboard: events per hour, per lane, per class, with the annotated clips.

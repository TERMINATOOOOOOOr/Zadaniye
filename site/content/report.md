# Technical report

## What we built

A detector-plus-rules system for one fixed road camera. A COCO-pretrained YOLO11s finds road users,
ByteTrack turns detections into trajectories, a hand-drawn scene (road, crossings, traffic islands,
intersection zone, stop line, signal ROI) gives the trajectories meaning, and one small rule per event
class turns them into time segments. A separate causal estimator (YOLO11n, every 4th frame) scores the
risk of an accident from time-to-collision, braking and queue context. Everything is deterministic,
offline, and runs at about 0.85× of the video duration on the 4K samples with a laptop GPU (limit: 3×).

## The camera

Four 4K clips (30 fps, 147 Mbit/s, 2–6 minutes) of a signalised T-junction on a wide avenue: two
carriageways, a side street with two zebra crossings, a raised island with the traffic light, a bus stop
on the far side. Rush-hour density: 500–700 vehicle tracks and 270–700 pedestrian tracks per clip, up to
80 objects in a frame. Pedestrians cross diagonally outside the zebras all the time; the far carriageway
queues through more than one signal cycle at dusk.

## What worked

- **Relative thresholds.** Speeds and distances in box heights per second made one set of thresholds
  work from the near lanes (300 px boxes) to the far ones (60 px), without any metric calibration.
- **Queue awareness.** "A car joins a queue" looks like a rear-end collision, a stopped vehicle and
  congestion at once. Treating a stationary neighbour or a regular stop location as a queue removed the
  bulk of false alarms in Part A and in the risk estimator.
- **Video decoding.** Reading 4K through an ffmpeg pipe that subsamples and rescales inside the decoder
  cut Part A from 1.0× to 0.26× of the video duration; NVDEC is used when the ffmpeg build supports it.
- **Vectorised risk.** The pairwise time-to-collision over up to 80 objects is a few numpy matrix
  operations; the Python version cost 90 ms per frame in this scene, the vectorised one about 3 ms.
- **The harness first.** The organizers' `run_submission.py` / `evaluate.py` were used unchanged from the
  first hour; a Docker image reproduces the two official commands offline with the GPU.

## What did not work

- **Reading the traffic light.** The lamp's HSV colour in its ROI was unreliable on this camera
  (daylight glare gave "amber/unknown" for whole clips, dusk gave "red" almost constantly). A second
  estimate from traffic behaviour (a standing queue at the stop line = red) also stayed "red" most of the
  time at rush hour. With no trustworthy signal state we **do not predict `red_light` and `stop_line`**:
  a wrongly predicted class costs more than a missed one under macro-F1.
- **`near_miss` / `accident` in dense traffic.** Merging queues and pedestrians stepping in front of slow
  cars produced 10+ candidates per clip. We now require a fast approach, deep box overlap, an abrupt stop
  of both objects and no queue around — and accept that we will miss gentle contacts.
- **`fire_smoke`, `road_obstacle`, `illegal_turn`, `solid_line_crossing`** are effectively not
  predicted: no reliable detector for the first two, no lane-level markings drawn for the last two.
- **Boundaries.** Our segments follow the annotation conventions literally, but "queue clears" or "all
  objects stop moving" can differ from a human annotator by a second or two — visible at IoU 0.7.

## What we would do next

- Label the four sample clips fully (the browser annotator is in `tools/`) and replace hand-tuned
  thresholds with a small gradient-boosted classifier over the same trajectory features.
- A clip classifier for `accident` / `near_miss` trained on public dashcam/CCTV datasets (DoTA, CCD) to
  re-score the rule candidates.
- Read the signal from the *stopping pattern* of cross traffic rather than one carriageway, then switch
  `red_light` and `stop_line` back on.
- Lane polygons with allowed manoeuvres for `illegal_turn`, `solid_line_crossing` and a precise
  `wrong_way`.

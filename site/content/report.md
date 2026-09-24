> Placeholder: fill in after the final run on the dev videos and the organisers' samples.

## What worked

- **Detector + tracker fit the CPU budget.** Every 2nd frame, YOLO11n/s and ByteTrack give ~1× video duration against a 3× budget.
- **Rules for the "geometric" classes** (`wrong_way`, `stopped_vehicle`, `jaywalking`, `congestion`) are stable once the scene markings are set explicitly.
- **Segment post-processing** lifts F1 noticeably at tIoU 0.5–0.7: without gap merging, events were split into 2–3 pieces.
- **Part B via TTC** gives an early signal 2–4 s before contact on the dev videos with real accidents (numbers after the run).

## What did not work

- **`accident` vs `near_miss`** — the boundary is blurry with TTC alone; a clip classifier on top of the candidates is needed.
- **Traffic-light phase** is inferred from flow behaviour rather than the signal itself — on empty junctions `red_light` and `stop_line` are missed.
- **`illegal_turn` / `solid_line_crossing`** are sensitive to lane-marking quality; config mistakes immediately produce false positives.
- **Night and rain** — the detector loses small objects, the tracker breaks ids, the rules flicker.

## Honest numbers

| Metric | Value |
|---|---|
| Score_A (mean F1 @ tIoU 0.3/0.5/0.7) | — after the run |
| Score_B (AP, early warning) | — after the run |
| Time per 1 min of video (CPU) | — after the run |

## What comes next

1. A clip classifier for the `accident` / `near_miss` pair (a small 3D-CNN or averaged detector embeddings).
2. Automatic lane and stop-line estimation from accumulated trajectories — drop the manual config.
3. A traffic-light state detector when the signal is in frame.
4. Detector fine-tuning on night and rain frames.
5. Calibrating the Part B risk on time-to-event, not only on TTC.

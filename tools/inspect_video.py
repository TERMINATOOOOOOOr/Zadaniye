"""Диагностика одного видео: сколько треков, что видит поле направлений, какие кандидаты дало каждое правило.

    python tools/inspect_video.py data/dev/car-detection.mp4 [--cache out/cache]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.pipeline import build_context, run_rules  # noqa: E402
from src.segments import finalize_events  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cache", default="out/cache")
    args = ap.parse_args()
    st = Settings()
    st.cache_dir = args.cache
    ctx = build_context(args.video, st)
    m = ctx.meta
    print(f"video: {m['width']}x{m['height']} {m['fps']:.2f} fps, {m['duration']:.1f} s, detect_sec={m.get('detect_sec', 'cache')}")
    kinds = Counter(t.kind for t in ctx.tracks.values())
    print(f"tracks: {len(ctx.tracks)} → {dict(kinds)}")
    lens = [t.t1 - t.t0 for t in ctx.tracks.values()]
    if lens:
        print(f"track length s: median {np.median(lens):.1f}, max {max(lens):.1f}; obs per track median {np.median([len(t.t) for t in ctx.tracks.values()]):.0f}")
    fa = ctx.flow.to_arrays()
    strong = (fa["consensus"] > 0.75) & (fa["count"] >= 5)
    print(f"flow: cell={fa['cell']} px, road cells={int(fa['road_mask'].sum())}, cells with clear direction={int(strong.sum())}")
    if ctx.signal is not None:
        print(f"signal: {Counter(ctx.signal.state)}")
    if ctx.scene is None:
        print("scene: нет (configs/scene.json отсутствует) — правила на переходах/стоп-линиях/сплошных выключены")
    else:
        s = ctx.scene
        print(f"scene: lanes={len(s.lanes)} crosswalks={len(s.crosswalks)} stop_lines={len(s.stop_lines)} solid={len(s.solid_lines)} signal={s.signal_visible}")
    vehicles = ctx.vehicles()
    if vehicles:
        rel = [float(np.median(t.speed) / t.size) for t in vehicles]
        print(f"vehicle median rel speed (heights/s): median {np.median(rel):.2f}, p10 {np.percentile(rel, 10):.2f}, p90 {np.percentile(rel, 90):.2f}")
    cands = run_rules(ctx)
    print("candidates per rule (before merge/min-len):")
    for k, v in cands.items():
        if v:
            print(f"  {k:20s} {len(v):3d}  " + ", ".join(f"[{s:.1f}-{e:.1f}]" for s, e in sorted(v)[:8]))
        else:
            print(f"  {k:20s}   0")
    events = finalize_events(cands, ctx.duration)
    print(f"final events: {len(events)}")
    for e in events:
        print("  ", e)


if __name__ == "__main__":
    main()

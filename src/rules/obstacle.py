"""road_obstacle: посторонний объект на проезжей части; fire_smoke: отключено, пока нет надёжного детектора.

Два источника: COCO-классы детектора (животные, чемодан, стул, мяч) и статические объекты по модели фона
(`src/static_objects.py`, ctx.static_objects) — для предметов, которых детектор не знает.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..segments import mask_to_intervals

MIN_CONF = 0.35


def run_obstacle(ctx: Context) -> list[tuple[float, float]]:
    out = []
    for tr in ctx.tracks.values():
        if tr.kind != "obstacle" or float(np.median(tr.conf)) < MIN_CONF:
            continue
        mask = np.array([ctx.on_road(tr.cx[i], tr.by[i]) for i in range(len(tr.t))])
        out.extend(mask_to_intervals(tr.t, mask))
    # статические объекты уже отфильтрованы в StaticObjectDetector.finish() (дорога, треки, очереди)
    for so in getattr(ctx, "static_objects", None) or []:
        if so.t1 > so.t0:
            out.append((float(so.t0), float(so.t1)))
    return out


def run_fire_smoke(ctx: Context) -> list[tuple[float, float]]:
    return []

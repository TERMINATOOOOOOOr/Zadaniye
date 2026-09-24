"""road_obstacle: посторонний объект (животное, предмет) на проезжей части; fire_smoke: отключено, пока нет надёжного детектора."""
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
    return out


def run_fire_smoke(ctx: Context) -> list[tuple[float, float]]:
    return []

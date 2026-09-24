"""solid_line_crossing: траектория машины пересекает сплошную линию разметки из сцены."""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..scene import seg_intersect

MOVE_REL = 0.3
SETTLE_SEC = 1.0    # «полностью в новой полосе» ≈ через секунду после пересечения


def run(ctx: Context) -> list[tuple[float, float]]:
    if ctx.scene is None or not ctx.scene.solid_lines:
        return []
    out = []
    for tr in ctx.vehicles():
        last_cross = -10.0
        for i in range(1, len(tr.t)):
            if tr.speed[i] < MOVE_REL * tr.size:
                continue
            p0 = (tr.cx[i - 1], tr.by[i - 1]); p1 = (tr.cx[i], tr.by[i])
            for _, pts in ctx.scene.solid_lines:
                hit = False
                for k in range(len(pts) - 1):
                    if seg_intersect(p0, p1, pts[k], pts[k + 1]):
                        hit = True
                        break
                if hit and tr.t[i] - last_cross > SETTLE_SEC:
                    t0 = float(tr.t[i - 1])
                    out.append((t0, min(t0 + SETTLE_SEC, float(tr.t[-1]))))
                    last_cross = float(tr.t[i])
    return out

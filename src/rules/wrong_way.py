"""wrong_way: машина едет против направления своей полосы (или встречного потока)."""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..segments import mask_to_intervals, merge_intervals

MOVE_REL_SPEED = 0.5    # анализируем только движущиеся: > 50 % высоты бокса в секунду
ANGLE_DEG = 120.0       # угол между скоростью и эталоном больше этого = против движения
MIN_CONSENSUS = 0.75    # для поля направлений: ячейка должна быть «однонаправленной»
MIN_RUN_SEC = 1.5
MIN_DISPLACEMENT = 1.0  # суммарное смещение против потока не меньше одной высоты бокса


def run(ctx: Context) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cos_thr = np.cos(np.radians(ANGLE_DEG))
    use_lanes = ctx.scene is not None and ctx.scene.has_lanes()
    for tr in ctx.vehicles():
        n = len(tr.t)
        if n < 4:
            continue
        thr = MOVE_REL_SPEED * tr.size
        own = _own_contribution(ctx, tr, thr) if not use_lanes else {}
        mask = np.zeros(n, dtype=bool)
        against = np.zeros(n)
        for i in range(n):
            if tr.speed[i] < thr:
                continue
            if ctx.scene is not None and ctx.scene.in_intersection(tr.cx[i], tr.by[i]):
                continue
            if use_lanes:
                d, cons = ctx.ref_direction(tr.cx[i], tr.by[i])
            else:
                d, cons = _flow_direction_without(ctx, tr.cx[i], tr.by[i], own)
            if d is None or cons < MIN_CONSENSUS:
                continue
            c = (tr.vx[i] * d[0] + tr.vy[i] * d[1]) / tr.speed[i]
            if c < cos_thr:
                mask[i] = True
                against[i] = -c * tr.speed[i]
        for s, e in merge_intervals(mask_to_intervals(tr.t, mask), gap=1.0):
            if e - s < MIN_RUN_SEC:
                continue
            i0, i1 = tr.index_at(s), tr.index_at(e)
            if i1 <= i0:
                continue
            seg_t, seg_a = tr.t[i0:i1 + 1], against[i0:i1 + 1]
            disp = float(np.sum(0.5 * (seg_a[1:] + seg_a[:-1]) * np.diff(seg_t)))
            if disp >= MIN_DISPLACEMENT * tr.size:
                out.append((s, e))
    return out


def _own_contribution(ctx: Context, tr, thr: float) -> dict[tuple[int, int], list[float]]:
    """Вклад самого трека в ячейки поля направлений, чтобы нарушитель не голосовал за своё направление."""
    own: dict[tuple[int, int], list[float]] = {}
    for i in range(len(tr.t)):
        if tr.speed[i] <= thr:
            continue
        cell = ctx.flow._cell(tr.cx[i], tr.by[i])
        acc = own.setdefault(cell, [0.0, 0.0, 0])
        acc[0] += tr.vx[i] / tr.speed[i]
        acc[1] += tr.vy[i] / tr.speed[i]
        acc[2] += 1
    return own


def _flow_direction_without(ctx: Context, x: float, y: float, own: dict):
    fl = ctx.flow
    r, c = fl._cell(x, y)
    sx, sy, n = fl.sum_vx[r, c], fl.sum_vy[r, c], int(fl.count[r, c])
    o = own.get((r, c))
    if o is not None:
        sx, sy, n = sx - o[0], sy - o[1], n - o[2]
    if n < 5:
        return None, 0.0
    mx, my = sx / n, sy / n
    cons = float(np.hypot(mx, my))
    if cons < 1e-6:
        return None, 0.0
    return np.array([mx / cons, my / cons]), cons

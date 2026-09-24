"""jaywalking: пешеход на проезжей части вне перехода; failure_to_yield: машина проезжает переход, на котором пешеход.

Оба правила требуют размеченных переходов (scene.crosswalks): без них любой переход выглядел бы как нарушение.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..segments import mask_to_intervals, merge_intervals

PED_MARGIN = 0.4         # пешеход считается «на переходе», если точка стояния ближе 0.4 его роста к полигону
VEH_MOVE_REL = 0.5       # машина «проезжает», если её скорость > 50 % высоты бокса в секунду
PED_LOOKBACK = 0.7       # пешеход был на переходе не раньше чем за 0.7 с до въезда машины (или во время)


def run_jaywalking(ctx: Context) -> list[tuple[float, float]]:
    if ctx.scene is None or not ctx.scene.has_crosswalks() or ctx.scene.road is None:
        return []
    out = []
    for tr in ctx.persons():
        n = len(tr.t)
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            x, y = tr.cx[i], tr.by[i]
            if not ctx.scene.in_road(x, y):
                continue
            if _near_crosswalk(ctx, x, y, PED_MARGIN * tr.h[i]):
                continue
            mask[i] = True
        out.extend(mask_to_intervals(tr.t, mask))
    return out


def run_failure_to_yield(ctx: Context) -> list[tuple[float, float]]:
    if ctx.scene is None or not ctx.scene.has_crosswalks():
        return []
    # интервалы присутствия пешеходов на каждом переходе
    ped_on: dict[str, list[tuple[float, float]]] = {name: [] for name, _ in ctx.scene.crosswalks}
    for tr in ctx.persons():
        for name, poly in ctx.scene.crosswalks:
            mask = np.array([_near_poly(poly, tr.cx[i], tr.by[i], PED_MARGIN * tr.h[i]) for i in range(len(tr.t))])
            ped_on[name].extend(mask_to_intervals(tr.t, mask))
    ped_on = {k: merge_intervals(v, 0.5) for k, v in ped_on.items()}
    out = []
    for tr in ctx.vehicles():
        thr = VEH_MOVE_REL * tr.size
        for name, poly in ctx.scene.crosswalks:
            if not ped_on[name]:
                continue
            inside = np.array([ctx.scene.crosswalk_at(tr.cx[i], tr.by[i]) == name for i in range(len(tr.t))])
            for s, e in mask_to_intervals(tr.t, inside):
                i0, i1 = tr.index_at(s), tr.index_at(e)
                if i1 <= i0 or float(np.mean(tr.speed[i0:i1 + 1] > thr)) < 0.6:
                    continue   # стоял на переходе, а не проезжал
                if any(ps <= e and pe >= s - PED_LOOKBACK for ps, pe in ped_on[name]):
                    out.append((s, e))
    return out


def _near_crosswalk(ctx: Context, x: float, y: float, margin: float) -> bool:
    return any(_near_poly(poly, x, y, margin) for _, poly in ctx.scene.crosswalks)


def _near_poly(poly: np.ndarray, x: float, y: float, margin: float) -> bool:
    import cv2
    d = cv2.pointPolygonTest(poly, (float(x), float(y)), True)
    return d >= -margin

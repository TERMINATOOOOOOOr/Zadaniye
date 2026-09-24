"""jaywalking: пешеход на проезжей части вне перехода; failure_to_yield: машина проезжает переход рядом с пешеходом на нём.

Оба правила требуют размеченных переходов (scene.crosswalks): без них любой переход выглядел бы как нарушение.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..context import Context
from ..segments import mask_to_intervals, merge_intervals

PED_MARGIN = 0.5         # пешеход «на переходе», если точка стояния ближе 0.5 его роста к полигону перехода
ROAD_DEPTH = 0.6         # «на проезжей части»: глубже 0.6 роста от края дороги (край тротуара не считаем)
MIN_H_FRAC = 0.03        # пешеходы ниже 3 % высоты кадра слишком далеко — не судим
VEH_MOVE_REL = 0.8       # машина «проезжает», если её скорость > 80 % высоты бокса в секунду
PED_LOOKBACK = 0.7       # пешеход был на переходе не раньше чем за 0.7 с до въезда машины (или во время)
YIELD_DIST_REL = 1.5     # пешеход ближе 1.5 высот машины от неё — конфликт; дальше — просто оба на широком переходе


def run_jaywalking(ctx: Context) -> list[tuple[float, float]]:
    if ctx.scene is None or not ctx.scene.has_crosswalks() or ctx.scene.road is None:
        return []
    min_h = MIN_H_FRAC * ctx.meta["height"]
    road = ctx.scene.road
    out = []
    for tr in ctx.persons():
        if tr.size < min_h:
            continue
        n = len(tr.t)
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            x, y = tr.cx[i], tr.by[i]
            if cv2.pointPolygonTest(road, (float(x), float(y)), True) < ROAD_DEPTH * tr.h[i]:
                continue
            if _near_crosswalk(ctx, x, y, PED_MARGIN * tr.h[i]):
                continue
            mask[i] = True
        out.extend(mask_to_intervals(tr.t, mask))
    return out


def run_failure_to_yield(ctx: Context) -> list[tuple[float, float]]:
    if ctx.scene is None or not ctx.scene.has_crosswalks():
        return []
    min_h = MIN_H_FRAC * ctx.meta["height"]
    persons = [p for p in ctx.persons() if p.size >= min_h]
    out = []
    for tr in ctx.vehicles():
        thr = VEH_MOVE_REL * tr.size
        for name, poly in ctx.scene.crosswalks:
            inside = np.array([ctx.scene.crosswalk_at(tr.cx[i], tr.by[i]) == name for i in range(len(tr.t))])
            for s, e in mask_to_intervals(tr.t, inside):
                i0, i1 = tr.index_at(s), tr.index_at(e)
                if i1 <= i0 or float(np.mean(tr.speed[i0:i1 + 1] > thr)) < 0.6:
                    continue   # стоял на переходе, а не проезжал
                if _pedestrian_conflict(tr, i0, i1, poly, persons):
                    out.append((s, e))
    return out


def _pedestrian_conflict(tr, i0: int, i1: int, poly, persons) -> bool:
    """Есть ли пешеход на этом переходе рядом с машиной (по расстоянию, а не просто где-то на зебре)."""
    for i in range(i0, i1 + 1):
        t = tr.t[i]
        vx, vy, r = tr.cx[i], tr.by[i], YIELD_DIST_REL * tr.size
        for p in persons:
            if not (p.t0 - PED_LOOKBACK <= t <= p.t1):
                continue
            j = p.index_at(t)
            if np.hypot(p.cx[j] - vx, p.by[j] - vy) > r:
                continue
            if cv2.pointPolygonTest(poly, (float(p.cx[j]), float(p.by[j])), True) >= 0:
                return True
    return False


def _near_crosswalk(ctx: Context, x: float, y: float, margin: float) -> bool:
    return any(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin for _, poly in ctx.scene.crosswalks)

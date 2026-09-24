"""red_light: пересёк стоп-линию на красный; stop_line: остановился за стоп-линией на красный, не въезжая на перекрёсток.

Нужны стоп-линии в сцене и видимый светофор.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..scene import seg_intersect, side

MOVE_REL = 0.5
STOP_REL = 0.15
PAST_LINE_MIN = 0.8      # «за линией»: нижняя точка бокса (бампер) глубже 0.8 высоты за стоп-линией
PAST_LINE_MAX = 2.5      # но не глубже 2.5 высот (иначе это уже проезд перекрёстка)
MAX_EVENT_SEC = 8.0


SIGNAL_RELIABLE = False   # состояние светофора на этой камере читается ненадёжно (блики днём, очередь стоит и на зелёный);
                          # пока это так, red_light и stop_line не выдаём: ложный класс дороже пропущенного


def run_red_light(ctx: Context) -> list[tuple[float, float]]:
    if not SIGNAL_RELIABLE:
        return []
    if ctx.scene is None or not ctx.scene.has_stop_lines() or ctx.signal is None or not len(ctx.signal.t):
        return []
    out = []
    for tr in ctx.vehicles():
        for i in range(1, len(tr.t)):
            p0 = (tr.cx[i - 1], tr.by[i - 1]); p1 = (tr.cx[i], tr.by[i])
            for sl in ctx.scene.stop_lines:
                if not seg_intersect(p0, p1, sl.p1, sl.p2):
                    continue
                v = np.array([tr.vx[i], tr.vy[i]])
                if float(np.dot(v, sl.approach)) <= 0 or tr.speed[i] < MOVE_REL * tr.size:
                    continue
                if ctx.signal.at(tr.t[i]) != "red":
                    continue
                t_end = _leave_time(ctx, tr, i)
                out.append((float(tr.t[i]), t_end))
                break
    return out


def run_stop_line(ctx: Context) -> list[tuple[float, float]]:
    if not SIGNAL_RELIABLE:
        return []
    if ctx.scene is None or not ctx.scene.has_stop_lines() or ctx.signal is None or not len(ctx.signal.t):
        return []
    out = []
    reds = ctx.signal.intervals("red")
    for tr in ctx.vehicles():
        for sl in ctx.scene.stop_lines:
            mid = (sl.p1 + sl.p2) / 2
            n = len(tr.t)
            past = np.array([float(np.dot(np.array([tr.cx[i], tr.by[i]]) - mid, sl.approach)) for i in range(n)])
            stopped_past = (tr.speed < STOP_REL * tr.size) & (past > PAST_LINE_MIN * tr.size) & (past < PAST_LINE_MAX * tr.size)
            # не въехал на перекрёсток пока красный
            if ctx.scene.intersection is not None:
                inside = np.array([ctx.scene.in_intersection(tr.cx[i], tr.by[i]) for i in range(n)])
                stopped_past &= ~inside
            for i in np.flatnonzero(stopped_past):
                t = float(tr.t[i])
                for rs, re in reds:
                    if rs <= t <= re:
                        out.append((t, re))
                        break
    return out


def _leave_time(ctx: Context, tr, i: int) -> float:
    """Конец события: машина покинула перекрёсток (если размечен) или кадр, не позже MAX_EVENT_SEC."""
    t0 = float(tr.t[i])
    if ctx.scene.intersection is not None:
        entered = False
        for j in range(i, len(tr.t)):
            inside = ctx.scene.in_intersection(tr.cx[j], tr.by[j])
            if inside:
                entered = True
            elif entered:
                return min(float(tr.t[j]), t0 + MAX_EVENT_SEC)
    return min(float(tr.t[-1]), t0 + MAX_EVENT_SEC)

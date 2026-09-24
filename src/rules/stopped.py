"""stopped_vehicle: машина стоит на проезжей части ≥ 10 с и это не очередь у светофора."""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..segments import mask_to_intervals

STOP_REL_SPEED = 0.12      # скорость < 12 % высоты бокса в секунду = стоит
QUEUE_RADIUS = 2.5         # сосед на расстоянии < 2.5 высоты бокса, тоже стоящий, = очередь
MIN_STOP_SEC = 10.0
BUS_LONG_SEC = 60.0        # автобус на остановке стоит штатно; событие только если дольше минуты
HOTSPOT_LONG_SEC = 90.0    # в точке штатных остановок (светофор) событие только если стоит дольше цикла


def run(ctx: Context) -> list[tuple[float, float]]:
    vehicles = ctx.vehicles()
    stopped_ivs: dict[int, list[tuple[float, float]]] = {}
    for tr in vehicles:
        if tr.t1 - tr.t0 < MIN_STOP_SEC * 0.8:
            continue
        thr = STOP_REL_SPEED * tr.size
        on_road = np.array([ctx.on_road(tr.cx[i], tr.by[i]) for i in range(len(tr.t))])
        mask = (tr.speed < thr) & on_road
        ivs = [iv for iv in mask_to_intervals(tr.t, mask) if iv[1] - iv[0] >= MIN_STOP_SEC * 0.8]
        if ivs:
            stopped_ivs[tr.id] = ivs
    out: list[tuple[float, float]] = []
    for tid, ivs in stopped_ivs.items():
        tr = ctx.tracks[tid]
        for s, e in ivs:
            if tr.cls == "bus" and e - s < BUS_LONG_SEC:
                continue
            if _in_queue(ctx, tr, s, e, stopped_ivs) or _signal_queue(ctx, tr, s, e):
                continue
            i = tr.index_at((s + e) / 2)
            if e - s < HOTSPOT_LONG_SEC and (ctx.is_stop_hotspot(tr.cx[i], tr.by[i])
                                             or ctx.in_queue(tr.cx[i], tr.by[i], s + 2.0, tr.size, exclude=tr.id)):
                continue
            out.append((s, e))
    return out


def _in_queue(ctx: Context, tr, s: float, e: float, stopped_ivs) -> bool:
    """Очередь: в середине остановки рядом стоит ещё хотя бы одна машина (цепочка)."""
    tm = (s + e) / 2
    i = tr.index_at(tm)
    x, y = tr.cx[i], tr.by[i]
    r = QUEUE_RADIUS * tr.size
    for oid, ivs in stopped_ivs.items():
        if oid == tr.id:
            continue
        if not any(os <= tm <= oe for os, oe in ivs):
            continue
        o = ctx.tracks[oid]
        j = o.index_at(tm)
        if np.hypot(o.cx[j] - x, o.by[j] - y) < r:
            return True
    return False


def _signal_queue(ctx: Context, tr, s: float, e: float) -> bool:
    """Стоит перед стоп-линией на красный: не нарушение. Работает, если размечены стоп-линии и виден светофор."""
    if ctx.scene is None or not ctx.scene.has_stop_lines() or ctx.signal is None:
        return False
    i = tr.index_at((s + e) / 2)
    x, y = tr.cx[i], tr.by[i]
    for sl in ctx.scene.stop_lines:
        # точка «перед» линией по направлению подъезда и недалеко от неё
        mid = (sl.p1 + sl.p2) / 2
        along = float(np.dot(np.array([x, y]) - mid, sl.approach))
        dist_line = abs(_dist_to_segment(x, y, sl.p1, sl.p2))
        if -8 * tr.size < along < 0.5 * tr.size and dist_line < 8 * tr.size:
            if ctx.signal.at(s + 1.0) == "red":
                return True
    return False


def _dist_to_segment(x, y, a, b) -> float:
    ab = b - a
    ap = np.array([x, y]) - a
    denom = float(np.dot(ab, ab)) or 1e-9
    u = float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
    proj = a + u * ab
    return float(np.hypot(x - proj[0], y - proj[1]))

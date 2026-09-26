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
CROSSWALK_WAIT_REL = 2.5   # стоит ближе 2.5 высот к зебре — пропускает пешеходов / ждёт свой сигнал, не событие
RED_OVERLAP = 0.6          # стоянка выше стоп-линии, из которой ≥ 60 % пришлось на красный, — очередь
APPROACH_DEPTH_REL = 40.0  # зона подъезда к стоп-линии: до 40 высот бокса выше линии


def run(ctx: Context, debug: list | None = None) -> list[tuple[float, float]]:
    """debug — если передан список, в него попадают (id трека, начало, конец) каждого события."""
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
                                             or any(ctx.in_queue(tr.cx[i], tr.by[i], tq, tr.size, exclude=tr.id)
                                                    for tq in (s + 2.0, (s + e) / 2, max(s + 2.0, e - 2.0)))):
                continue
            if _waiting_at_crosswalk(ctx, tr, i):
                continue
            out.append((s, e))
            if debug is not None:
                debug.append((tr.id, s, e))
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
    """Стоит выше стоп-линии, пока горит красный (или почти весь интервал стоянки красный): очередь, не нарушение.
    Глубина очереди на этой камере — десятки высот бокса, поэтому берётся вся зона подъезда."""
    if ctx.scene is None or not ctx.scene.has_stop_lines() or ctx.signal is None or not len(ctx.signal.t):
        return False
    from ..signal import stop_line_offset
    i = tr.index_at((s + e) / 2)
    x, y = tr.cx[i], tr.by[i]
    for sl in ctx.scene.stop_lines:
        off = stop_line_offset(sl, x, y)           # < 0 — выше линии (ещё не пересёк)
        if not (-APPROACH_DEPTH_REL * tr.size < off < 0.5 * tr.size):
            continue
        # в створе линии с запасом: очередь занимает и полосы, которые линия не покрывает точно
        ab = sl.p2 - sl.p1
        u = float(np.dot(np.array([x, y]) - sl.p1, ab) / (float(np.dot(ab, ab)) or 1e-9))
        if not (-0.6 <= u <= 1.6):
            continue
        ts = np.arange(s, e, 0.5)
        red = float(np.mean([ctx.signal.at(float(t)) == "red" for t in ts])) if len(ts) else 0.0
        if red >= RED_OVERLAP or ctx.signal.at(s + 1.0) == "red":
            return True
    return False


def _waiting_at_crosswalk(ctx: Context, tr, i: int) -> bool:
    """Стоит вплотную к зебре (ближе CROSSWALK_WAIT_REL высот): пропускает пешеходов или ждёт свой сигнал."""
    if ctx.scene is None or not ctx.scene.has_crosswalks():
        return False
    import cv2
    x, y = float(tr.cx[i]), float(tr.by[i])
    for _, poly in ctx.scene.crosswalks:
        if cv2.pointPolygonTest(poly, (x, y), True) >= -CROSSWALK_WAIT_REL * tr.size:
            return True
    return False


def _dist_to_segment(x, y, a, b) -> float:
    ab = b - a
    ap = np.array([x, y]) - a
    denom = float(np.dot(ab, ab)) or 1e-9
    u = float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
    proj = a + u * ab
    return float(np.hypot(x - proj[0], y - proj[1]))

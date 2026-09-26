"""red_light: пересёк стоп-линию на красный; stop_line: остановился за стоп-линией на красный, не въезжая на перекрёсток.

Нужны стоп-линии в сцене и надёжное состояние светофора (ctx.signal.reliable — см. signal.fuse_signals).
Оба правила консервативны: ложный класс дороже пропущенного, поэтому красный берётся только от лампы
(не от пешеходов и не от очереди), а спорные моменты (только что включился красный, вот-вот зелёный) пропускаются.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..scene import seg_intersect
from ..segments import mask_to_intervals
from ..signal import along_line, stop_line_offset

MOVE_REL = 0.5           # «на скорости»: быстрее половины высоты бокса в секунду
STOP_REL = 0.15          # «стоит»: медленнее 15 % высоты бокса в секунду
RED_SETTLED_SEC = 1.5    # красный горит не меньше этого (пропускаем переход жёлтый→красный)
RED_TAIL_SEC = 1.0       # и до зелёного ещё не меньше этого
SEEN_BEFORE_SEC = 1.0    # трек наблюдался до пересечения (не родился на самой линии)
ENTER_WITHIN_SEC = 4.0   # после пересечения на скорости машина въезжает на перекрёсток
MAX_EVENT_SEC = 8.0
PAST_LINE_MIN = 0.35     # stop_line: точка стояния глубже 0.35 высоты за линией (бампер заметно за линией)
PAST_LINE_MAX = 2.5      # но не глубже 2.5 высот (иначе это уже проезд перекрёстка)
STOP_MIN_SEC = 2.0       # стоит за линией не меньше этого
RED_SOURCES = ("lamp",)   # красный только от лампы: пешеходы на этой камере ходят и на свой красный (джейвокеры),
                          # а очередь стоит и на зелёный — оба источника дают ложный «красный»


def _ready(ctx: Context) -> bool:
    return ctx.scene is not None and ctx.scene.has_stop_lines() and ctx.signal_reliable() and len(ctx.signal.t) > 0


def _red_run(ctx: Context, t: float) -> tuple[float, float] | None:
    """Интервал красного, в который попадает t, если красный в этот момент от надёжного источника."""
    if ctx.signal.at(t) != "red" or ctx.signal.source_at(t) not in RED_SOURCES:
        return None
    for s, a, b in ctx.signal.runs():
        if s == "red" and a <= t <= b:
            return a, b
    return None


def _in_span(sl, x: float, y: float) -> bool:
    return -0.05 <= along_line(sl, x, y) <= 1.05


def run_red_light(ctx: Context) -> list[tuple[float, float]]:
    if not _ready(ctx):
        return []
    out = []
    duration = ctx.duration
    for tr in ctx.vehicles():
        move = MOVE_REL * tr.size
        for i in range(1, len(tr.t)):
            t = float(tr.t[i])
            if t - tr.t0 < SEEN_BEFORE_SEC or tr.speed[i] < move:
                continue
            p0 = (tr.cx[i - 1], tr.by[i - 1]); p1 = (tr.cx[i], tr.by[i])
            hit = None
            for sl in ctx.scene.stop_lines:
                if seg_intersect(p0, p1, sl.p1, sl.p2) and float(np.dot(np.array([tr.vx[i], tr.vy[i]]), sl.approach)) > 0:
                    hit = sl
                    break
            if hit is None:
                continue
            run = _red_run(ctx, t)
            if run is None:
                continue
            rs, re = run
            if t - rs < RED_SETTLED_SEC or (re < duration - 0.5 and re - t < RED_TAIL_SEC):
                continue
            if ctx.scene.intersection is not None and not _enters_intersection(ctx, tr, i):
                continue
            out.append((t, _leave_time(ctx, tr, i)))
            break
    return out


def run_stop_line(ctx: Context) -> list[tuple[float, float]]:
    if not _ready(ctx):
        return []
    out = []
    for tr in ctx.vehicles():
        n = len(tr.t)
        if n < 3:
            continue
        for sl in ctx.scene.stop_lines:
            off = np.array([stop_line_offset(sl, tr.cx[i], tr.by[i]) for i in range(n)])
            span = np.array([_in_span(sl, tr.cx[i], tr.by[i]) for i in range(n)])
            stopped_past = span & (tr.speed < STOP_REL * tr.size) & (off > PAST_LINE_MIN * tr.size) & (off < PAST_LINE_MAX * tr.size)
            if ctx.scene.intersection is not None:
                inside = np.array([ctx.scene.in_intersection(tr.cx[i], tr.by[i]) for i in range(n)])
                stopped_past &= ~inside
            if not stopped_past.any():
                continue
            for s, e in mask_to_intervals(tr.t, stopped_past):
                if e - s < STOP_MIN_SEC:
                    continue
                run = _red_run(ctx, s + STOP_MIN_SEC / 2)
                if run is None:
                    continue
                rs, re = run
                # машина заехала за линию при этом красном (или на жёлтом перед ним), а не стояла там с начала видео
                before = (tr.t < s) & (off < 0)
                if not before.any() or float(tr.t[np.flatnonzero(before)[-1]]) < rs - 5.0:
                    continue
                end = re if e >= re - 3.0 else e
                out.append((s, min(end, re)))
    return out


def _enters_intersection(ctx: Context, tr, i: int) -> bool:
    t0 = float(tr.t[i])
    for j in range(i, len(tr.t)):
        if tr.t[j] - t0 > ENTER_WITHIN_SEC:
            break
        if ctx.scene.in_intersection(tr.cx[j], tr.by[j]):
            return True
    return False


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

"""solid_line_crossing: траектория машины пересекает сплошную линию разметки из сцены.

Пересечение засчитывается, только если машина действительно «переселилась»: за SETTLE_SEC до него опорная
точка (центр бокса по x, низ бокса) устойчиво лежала по одну сторону линии, а через SETTLE_SEC после — по другую,
причём в обоих окнах на расстоянии не меньше SIDE_MARGIN_REL высот бокса. Дрожание бокса на самой линии
(стоящая очередь, частичное перекрытие соседом, слияние боксов) так не проходит.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..context import Context
from ..scene import seg_intersect

MOVE_REL = 0.3            # пересечение считаем только в движении: скорость выше 30 % высоты бокса в секунду
SETTLE_SEC = 1.0          # окна «до» и «после» пересечения
SETTLE_GAP_SEC = 0.15     # ближайшие к пересечению кадры не смотрим: точка ещё на линии
SIDE_MARGIN_REL = 0.15    # средняя дистанция до линии в каждом окне не меньше 0.15 высоты бокса
SIDE_AGREE = 0.8          # доля кадров окна, лежащих на стороне окна
MIN_WINDOW_OBS = 4        # в каждом окне не меньше четырёх наблюдений (около 0.4 с при шаге 3 кадра)
EVENT_BEFORE_SEC = 1.0    # событие: секунда до пересечения и секунда после
EVENT_AFTER_SEC = 1.0


@dataclass
class Crossing:
    """Подтверждённое пересечение: трек, имя линии, момент и опорная точка в координатах кадра."""

    track_id: int
    line: str
    t: float
    x: float
    y: float
    size: float


def run(ctx: Context) -> list[tuple[float, float]]:
    out = []
    for c in crossings(ctx):
        tr = ctx.tracks[c.track_id]
        out.append((max(tr.t0, c.t - EVENT_BEFORE_SEC), min(tr.t1, c.t + EVENT_AFTER_SEC)))
    return out


def crossings(ctx: Context) -> list[Crossing]:
    """Все подтверждённые пересечения сплошных (для правила, диагностики и контактных листов)."""
    if ctx.scene is None or not ctx.scene.solid_lines:
        return []
    out: list[Crossing] = []
    for tr in ctx.vehicles():
        n = len(tr.t)
        if n < 2 * MIN_WINDOW_OBS:
            continue
        for name, pts in ctx.scene.solid_lines:
            if len(pts) < 2:
                continue
            dist = signed_distance(pts, tr.cx, tr.by)
            last = -10.0
            for i in range(1, n):
                if tr.speed[i] < MOVE_REL * tr.size or tr.t[i] - last < SETTLE_SEC:
                    continue
                if not _hits(pts, (tr.cx[i - 1], tr.by[i - 1]), (tr.cx[i], tr.by[i])):
                    continue
                t_cross = 0.5 * (float(tr.t[i - 1]) + float(tr.t[i]))
                if _settled(tr.t, dist, t_cross, tr.size):
                    out.append(Crossing(tr.id, name, t_cross, float(tr.cx[i]), float(tr.by[i]), float(tr.size)))
                    last = float(tr.t[i])
    return out


def _hits(pts: np.ndarray, p0, p1) -> bool:
    return any(seg_intersect(p0, p1, pts[k], pts[k + 1]) for k in range(len(pts) - 1))


def signed_distance(pts: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Знаковое расстояние точек до полилинии: перпендикуляр к прямой ближайшего звена (без обрезки по концам).
    Знак одинаков для всех звеньев полилинии, нарисованной в одном направлении; для двухточечной линии это
    просто сторона прямой."""
    p = np.asarray(pts, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    best = np.full(len(xs), np.inf)
    signed = np.zeros(len(xs))
    for k in range(len(p) - 1):
        a, b = p[k], p[k + 1]
        d = b - a
        L2 = float(d @ d)
        if L2 < 1e-9:
            continue
        u = ((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / L2
        uc = np.clip(u, 0.0, 1.0)
        near = np.hypot(xs - (a[0] + uc * d[0]), ys - (a[1] + uc * d[1]))
        perp = (d[0] * (ys - a[1]) - d[1] * (xs - a[0])) / np.sqrt(L2)
        take = near < best
        best[take] = near[take]
        signed[take] = perp[take]
    return signed


def _settled(t: np.ndarray, dist: np.ndarray, t_cross: float, size: float) -> bool:
    before = (t >= t_cross - SETTLE_GAP_SEC - SETTLE_SEC) & (t <= t_cross - SETTLE_GAP_SEC)
    after = (t >= t_cross + SETTLE_GAP_SEC) & (t <= t_cross + SETTLE_GAP_SEC + SETTLE_SEC)
    if before.sum() < MIN_WINDOW_OBS or after.sum() < MIN_WINDOW_OBS:
        return False
    db, da = dist[before], dist[after]
    mb, ma = float(np.mean(db)), float(np.mean(da))
    margin = SIDE_MARGIN_REL * size
    if abs(mb) < margin or abs(ma) < margin or np.sign(mb) == np.sign(ma):
        return False
    agree_b = float(np.mean(np.sign(db) == np.sign(mb)))
    agree_a = float(np.mean(np.sign(da) == np.sign(ma)))
    return agree_b >= SIDE_AGREE and agree_a >= SIDE_AGREE

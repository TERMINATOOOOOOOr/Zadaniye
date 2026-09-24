"""congestion: поток одного направления стоит или ползёт по всем полосам (минимальная длина — в segments.MIN_LEN)."""
from __future__ import annotations

import math

import numpy as np

from ..context import Context
from ..segments import mask_to_intervals

BIN = 0.5               # шаг по времени, с
SLOW_REL_SPEED = 0.3    # «ползёт»: < 30 % высоты бокса в секунду
MIN_VEHICLES = 4        # минимум машин в направлении, чтобы говорить о потоке
SLOW_FRACTION = 0.75    # доля медленных


def run(ctx: Context) -> list[tuple[float, float]]:
    vehicles = ctx.vehicles()
    if not vehicles or ctx.duration <= 0:
        return []
    ts = np.arange(0.0, ctx.duration, BIN)
    # направление группы: сектор доминирующего потока в точке машины (4 сектора по 90°)
    groups: dict[int, np.ndarray] = {}
    for tr in vehicles:
        thr = SLOW_REL_SPEED * tr.size
        for i in range(len(tr.t)):
            x, y = tr.cx[i], tr.by[i]
            if not ctx.on_road(x, y):
                continue
            d, cons = ctx.ref_direction(x, y)
            if d is None:
                if tr.speed[i] < 1e-6:
                    continue
                d = np.array([tr.vx[i], tr.vy[i]]) / tr.speed[i]
            sector = int(((math.degrees(math.atan2(d[1], d[0])) + 360 + 45) % 360) // 90)
            b = int(tr.t[i] // BIN)
            if b >= len(ts):
                continue
            arr = groups.setdefault(sector, np.zeros((len(ts), 2)))
            arr[b, 0] += 1
            arr[b, 1] += 1 if tr.speed[i] < thr else 0
    out: list[tuple[float, float]] = []
    for sector, arr in groups.items():
        n, slow = arr[:, 0], arr[:, 1]
        mask = (n >= MIN_VEHICLES) & (slow >= SLOW_FRACTION * np.maximum(n, 1))
        # если виден светофор — очередь на красный не пробка, оставляем только участки, пережившие зелёный
        out.extend(mask_to_intervals(ts, mask, pad=BIN / 2))
    if ctx.signal is not None and len(ctx.signal.t):
        greens = ctx.signal.intervals("green")
        if greens:
            out = [iv for iv in out if any(gs < iv[1] and ge > iv[0] for gs, ge in greens)]
    return out

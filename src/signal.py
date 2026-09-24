"""Состояние светофора по цвету пикселей в размеченной области (ROI) кадра."""
from __future__ import annotations

import cv2
import numpy as np

from .context import SignalTrack


def classify_roi(frame_bgr: np.ndarray, roi: tuple[int, int, int, int], scale: float = 1.0) -> str:
    """'red' | 'green' | 'amber' | 'unknown' по доле ярких насыщенных пикселей нужного тона."""
    x, y, w, h = (int(round(v * scale)) for v in roi)
    crop = frame_bgr[max(0, y):y + max(1, h), max(0, x):x + max(1, w)]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat > 100) & (val > 140)
    red = bright & ((hue < 10) | (hue > 165))
    green = bright & (hue > 40) & (hue < 95)
    amber = bright & (hue >= 10) & (hue <= 35)
    counts = {"red": int(red.sum()), "green": int(green.sum()), "amber": int(amber.sum())}
    best = max(counts, key=counts.get)
    total = crop.shape[0] * crop.shape[1]
    if counts[best] < max(6, 0.01 * total):
        return "unknown"
    return best


class SignalReader:
    """Накапливает состояния по кадрам и сглаживает: одиночные выбросы не меняют состояние."""

    def __init__(self, roi: tuple[int, int, int, int]) -> None:
        self.roi = roi
        self.ts: list[float] = []
        self.raw: list[str] = []

    def push(self, frame_bgr: np.ndarray, t: float, scale: float) -> None:
        self.ts.append(t)
        self.raw.append(classify_roi(frame_bgr, self.roi, scale))

    def finish(self, hold: int = 3) -> SignalTrack:
        states: list[str] = []
        cur = "unknown"
        run_state, run_len = None, 0
        for s in self.raw:
            if s == run_state:
                run_len += 1
            else:
                run_state, run_len = s, 1
            if run_len >= hold and s != "unknown":
                cur = s
            states.append(cur)
        return SignalTrack(t=np.array(self.ts), state=states)


def signal_from_traffic(ctx, bin_sec: float = 0.5, queue_min: int = 2, queue_hold_sec: float = 3.0,
                        queue_depth_rel: float = 6.0, stop_rel: float = 0.15) -> SignalTrack | None:
    """Состояние светофора по поведению машин у стоп-линии: очередь стоит ≥ queue_hold_sec — «красный»,
    машины пересекают линию и очереди нет — «зелёный», иначе «неизвестно». Не зависит от цвета лампы в кадре."""
    if ctx.scene is None or not ctx.scene.has_stop_lines() or ctx.duration <= 0:
        return None
    from .scene import seg_intersect
    ts = np.arange(0.0, ctx.duration + bin_sec, bin_sec)
    queued_ids: list[set] = [set() for _ in ts]   # какие машины стоят у линии в каждом бине
    crossing = np.zeros(len(ts))
    for sl in ctx.scene.stop_lines:
        mid = (sl.p1 + sl.p2) / 2
        for tr in ctx.vehicles():
            for i in range(len(tr.t)):
                b = int(tr.t[i] // bin_sec)
                if b >= len(ts):
                    continue
                d = np.array([tr.cx[i], tr.by[i]]) - mid
                along = float(np.dot(d, sl.approach))
                lateral = abs(float(d[0] * sl.approach[1] - d[1] * sl.approach[0]))
                half = float(np.hypot(*(sl.p2 - sl.p1))) / 2 + tr.size
                if -queue_depth_rel * tr.size < along < 0.3 * tr.size and lateral < half and tr.speed[i] < stop_rel * tr.size:
                    queued_ids[b].add(tr.id)
                if i > 0 and seg_intersect((tr.cx[i - 1], tr.by[i - 1]), (tr.cx[i], tr.by[i]), sl.p1, sl.p2) \
                        and float(np.dot(np.array([tr.vx[i], tr.vy[i]]), sl.approach)) > 0:
                    crossing[b] += 1
    queued = np.array([len(q) for q in queued_ids], dtype=float)
    hold = max(1, int(round(queue_hold_sec / bin_sec)))
    states: list[str] = []
    run = 0
    for b in range(len(ts)):
        run = run + 1 if queued[b] >= queue_min else 0
        if run >= hold:
            states.append("red")
        elif crossing[max(0, b - 2):b + 1].sum() >= 1 and queued[b] < queue_min:
            states.append("green")
        else:
            states.append("unknown")
    return SignalTrack(t=ts, state=states)

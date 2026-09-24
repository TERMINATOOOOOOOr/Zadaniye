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

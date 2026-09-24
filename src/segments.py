"""Сборка событий из покадровых флагов и интервалов: слияние, отсев коротких, запрет пересечений внутри класса."""
from __future__ import annotations

import numpy as np

Interval = tuple[float, float]

# Минимальная длина события по классам (секунды): короче — считаем шумом.
MIN_LEN = {
    "accident": 1.0, "near_miss": 0.6, "red_light": 0.8, "wrong_way": 1.5, "illegal_u_turn": 1.5,
    "stopped_vehicle": 10.0, "jaywalking": 1.0, "failure_to_yield": 0.6, "illegal_turn": 1.0,
    "solid_line_crossing": 0.5, "stop_line": 2.0, "congestion": 45.0, "road_obstacle": 3.0, "fire_smoke": 2.0,
}
# Разрыв между интервалами одного класса, который сшиваем (секунды).
MERGE_GAP = {
    "accident": 2.0, "near_miss": 1.0, "red_light": 1.0, "wrong_way": 1.5, "illegal_u_turn": 1.0,
    "stopped_vehicle": 2.0, "jaywalking": 1.0, "failure_to_yield": 0.8, "illegal_turn": 1.0,
    "solid_line_crossing": 0.5, "stop_line": 2.0, "congestion": 3.0, "road_obstacle": 3.0, "fire_smoke": 3.0,
}


def mask_to_intervals(t: np.ndarray, mask: np.ndarray, pad: float = 0.0) -> list[Interval]:
    """Непрерывные участки True в маске (по временной оси трека) → интервалы [t_start, t_end]."""
    out: list[Interval] = []
    if len(t) == 0:
        return out
    start = None
    for i in range(len(t)):
        if mask[i] and start is None:
            start = float(t[i])
        if not mask[i] and start is not None:
            out.append((start - pad, float(t[i - 1]) + pad))
            start = None
    if start is not None:
        out.append((start - pad, float(t[-1]) + pad))
    return out


def merge_intervals(intervals: list[Interval], gap: float = 0.0) -> list[Interval]:
    """Объединение пересекающихся/близких интервалов (одного класса)."""
    if not intervals:
        return []
    iv = sorted((float(s), float(e)) for s, e in intervals)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def finalize_events(candidates: dict[str, list[Interval]], duration: float) -> list[list]:
    """Кандидаты по классам → валидный список событий для evaluate.py."""
    events: list[list] = []
    for label, ivs in candidates.items():
        merged = merge_intervals(ivs, MERGE_GAP.get(label, 1.0))
        for s, e in merged:
            if not (np.isfinite(s) and np.isfinite(e)):
                continue
            s = max(0.0, s)
            e = min(duration, e) if duration > 0 else e
            if e - s < MIN_LEN.get(label, 0.5):
                continue
            if s < e:
                events.append([round(s, 3), round(e, 3), label])
    events.sort(key=lambda x: (x[0], x[2]))
    return events


def union_length(intervals: list[Interval]) -> float:
    return sum(e - s for s, e in merge_intervals(intervals))

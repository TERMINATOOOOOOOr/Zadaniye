"""Сводка для операторской панели (/dashboard) из results.json.

Всё считается здесь, на сервере, чистыми функциями без состояния; шаблон
``dashboard.html`` только рисует готовые прямоугольники inline-SVG. Так панель
работает без JavaScript, а расчёты проверяются тестами.
"""
from __future__ import annotations

import math
from typing import Any

import config

ALARM_THRESHOLD = 0.5  # порог тревоги по кривой риска (как в отчёте и на странице Results)

# геометрия SVG (единицы viewBox; ширина растягивается на 100 % контейнера)
CHART_W = 720
CHART_H = 220
CHART_PAD = {"l": 34, "r": 10, "t": 12, "b": 26}
STRIP_W = 720
STRIP_LANE_H = 14
STRIP_AXIS_H = 20
STRIP_GUTTER = 120


def _label(event: Any) -> str | None:
    return str(event[2]) if len(event) >= 3 else None


def class_order(labels: set[str]) -> list[str]:
    """Классы в официальном порядке, неизвестные — в конце по алфавиту."""
    known = [c for c in config.CLASSES if c in labels]
    return known + sorted(labels - set(config.CLASSES))


def events_per_minute(events: list[list[Any]], duration: float) -> dict[str, Any]:
    """Число событий по минутам и классам; событие относится к минуте своего начала."""
    minutes = max(1, int(math.ceil(max(float(duration or 0.0), 1e-6) / 60.0)))
    labels = {lab for e in events if (lab := _label(e))}
    classes = class_order(labels)
    idx = {c: i for i, c in enumerate(classes)}
    bins = [[0] * len(classes) for _ in range(minutes)]
    for e in events:
        lab = _label(e)
        if lab is None:
            continue
        m = int(float(e[0]) // 60)
        if m >= minutes:  # событие за пределами заявленной длительности — в последнюю минуту
            m = minutes - 1
        bins[max(0, m)][idx[lab]] += 1
    totals = [sum(row) for row in bins]
    return {"classes": classes, "bins": bins, "minutes": minutes, "max": max(totals) if totals else 0}


def class_totals(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Таблица «класс × видео»: число событий и суммарная длительность (с)."""
    labels: set[str] = set()
    for v in videos:
        labels |= {lab for e in (v.get("events") or []) if (lab := _label(e))}
    classes = class_order(labels)
    names = [str(v.get("name") or f"video {i + 1}") for i, v in enumerate(videos)]
    rows = []
    for c in classes:
        counts, secs = [], []
        for v in videos:
            evs = [e for e in (v.get("events") or []) if _label(e) == c]
            counts.append(len(evs))
            secs.append(sum(max(0.0, float(e[1]) - float(e[0])) for e in evs))
        rows.append({"cls": c, "counts": counts, "total": sum(counts), "seconds": sum(secs)})
    col_totals = [sum(r["counts"][i] for r in rows) for i in range(len(videos))]
    return {"videos": names, "rows": rows, "col_totals": col_totals, "grand_total": sum(col_totals)}


def alarm_runs(risk: list[list[float]], threshold: float = ALARM_THRESHOLD) -> dict[str, Any]:
    """Серии подряд идущих отсчётов риска ≥ threshold.

    Длина серии = (t_last − t_first) + шаг дискретизации кривой (медианный),
    чтобы серия из одного отсчёта имела ненулевую длину.
    """
    pts = [(float(p[0]), float(p[1])) for p in risk if len(p) >= 2]
    pts.sort(key=lambda p: p[0])
    if len(pts) >= 2:
        gaps = sorted(b[0] - a[0] for a, b in zip(pts, pts[1:]))
        dt = gaps[len(gaps) // 2]
    else:
        dt = 0.0
    runs: list[dict[str, float]] = []
    start = last = None
    peak = 0.0
    for t, s in pts:
        if s >= threshold:
            if start is None:
                start, peak = t, s
            last = t
            peak = max(peak, s)
        elif start is not None:
            runs.append({"start": start, "end": last + dt, "length": last - start + dt, "peak": peak})
            start = None
    if start is not None:
        runs.append({"start": start, "end": last + dt, "length": last - start + dt, "peak": peak})
    n_alarm = sum(1 for _, s in pts if s >= threshold)
    return {
        "threshold": threshold,
        "runs": runs,
        "count": len(runs),
        "total_sec": sum(r["length"] for r in runs),
        "longest_sec": max((r["length"] for r in runs), default=0.0),
        "share": (n_alarm / len(pts)) if pts else 0.0,
        "samples": len(pts),
        "dt": dt,
    }


def nice_step(rng: float, target: int) -> float:
    raw = rng / max(1, target)
    if raw <= 0:
        return 1.0
    pow10 = 10 ** math.floor(math.log10(raw))
    f = raw / pow10
    m = 1 if f < 1.5 else 2 if f < 3.5 else 5 if f < 7.5 else 10
    return m * pow10


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:04.1f}"


def bar_chart(per_minute: dict[str, Any]) -> dict[str, Any]:
    """Геометрия столбчатой диаграммы «события в минуту» (stacked) для inline-SVG."""
    p = CHART_PAD
    inner_w = CHART_W - p["l"] - p["r"]
    inner_h = CHART_H - p["t"] - p["b"]
    n = per_minute["minutes"]
    ymax = max(1, per_minute["max"])
    ystep = nice_step(ymax, 4)
    ymax = int(math.ceil(ymax / ystep) * ystep)
    slot = inner_w / n
    bw = max(2.0, slot * 0.7)
    rects = []
    for m, row in enumerate(per_minute["bins"]):
        y0 = p["t"] + inner_h
        x = p["l"] + m * slot + (slot - bw) / 2
        for c, cnt in zip(per_minute["classes"], row):
            if cnt <= 0:
                continue
            h = cnt / ymax * inner_h
            y0 -= h
            rects.append({"x": round(x, 2), "y": round(y0, 2), "w": round(bw, 2), "h": round(h, 2), "cls": c,
                          "title": f"minute {m}: {c} × {cnt}"})
    yticks = []
    v = 0
    while v <= ymax:
        yticks.append({"v": v, "y": round(p["t"] + inner_h - v / ymax * inner_h, 2)})
        v += ystep
    xstep = max(1, int(math.ceil(n / 12)))
    xticks = [{"label": f"{m}′", "x": round(p["l"] + (m + 0.5) * slot, 2)} for m in range(0, n, xstep)]
    return {"w": CHART_W, "h": CHART_H, "pad": p, "rects": rects, "yticks": yticks, "xticks": xticks, "ymax": ymax,
            "x0": p["l"], "x1": CHART_W - p["r"], "y0": p["t"], "y1": p["t"] + inner_h}


def timeline_strip(events: list[list[Any]], duration: float) -> dict[str, Any]:
    """Геометрия полосы событий: одна дорожка на класс, отрезки в цветах классов."""
    labels = {lab for e in events if (lab := _label(e))}
    classes = class_order(labels)
    lanes = {c: i for i, c in enumerate(classes)}
    dur = max(float(duration or 0.0), 1e-6)
    x0, x1 = STRIP_GUTTER, STRIP_W - 8
    segs = []
    for e in events:
        lab = _label(e)
        if lab is None:
            continue
        s, t = max(0.0, float(e[0])), min(dur, float(e[1]))
        xa = x0 + s / dur * (x1 - x0)
        xb = max(xa + 2.0, x0 + t / dur * (x1 - x0))
        segs.append({"x": round(xa, 2), "w": round(xb - xa, 2), "y": lanes[lab] * STRIP_LANE_H + 3,
                     "h": STRIP_LANE_H - 6, "cls": lab, "title": f"{lab} {fmt_time(s)}–{fmt_time(t)} ({t - s:.1f} s)"})
    step = nice_step(dur, 8)
    ticks = []
    v = 0.0
    while v <= dur + 1e-9:
        ticks.append({"x": round(x0 + v / dur * (x1 - x0), 2), "label": fmt_time(v)})
        v += step
    h = max(1, len(classes)) * STRIP_LANE_H + STRIP_AXIS_H
    return {"w": STRIP_W, "h": h, "lanes": classes, "lane_h": STRIP_LANE_H, "segs": segs, "ticks": ticks,
            "x0": x0, "x1": x1, "axis_y": max(1, len(classes)) * STRIP_LANE_H}


def build(data: dict[str, Any]) -> dict[str, Any]:
    """Всё, что нужно шаблону панели, из содержимого results.json."""
    videos = list(data.get("videos") or [])
    out = []
    for i, v in enumerate(videos):
        events = [e for e in (v.get("events") or []) if len(e) >= 3]
        risk = v.get("risk") or []
        duration = float(v.get("duration") or 0.0)
        if not duration:
            ends = [float(e[1]) for e in events] + [float(p[0]) for p in risk[-1:]]
            duration = max(ends) if ends else 0.0
        per_minute = events_per_minute(events, duration)
        out.append({
            "name": str(v.get("name") or f"video {i + 1}"),
            "duration": duration,
            "n_events": len(events),
            "per_minute": per_minute,
            "chart": bar_chart(per_minute),
            "strip": timeline_strip(events, duration),
            "alarms": alarm_runs(risk),
        })
    return {"videos": out, "totals": class_totals(videos), "threshold": ALARM_THRESHOLD}

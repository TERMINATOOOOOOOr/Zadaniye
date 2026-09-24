"""accident и near_miss по парам треков: сближение, контакт, резкая остановка / уклонение.

Осторожные пороги: ложная авария стоит дороже пропущенной, а «догнал очередь и встал» очень похоже
на столкновение в проекции камеры. Отличаем по резкости торможения и глубине перекрытия боксов.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..tracking import Track

APPROACH_REL = 1.5       # скорость сближения > 1.5 средней высоты боксов в секунду
CONTACT_ND = 0.35        # нормированная дистанция центров (на среднюю высоту) при контакте
CONTACT_IOU = 0.35
REST_REL = 0.15          # «остановился»: < 15 % высоты в секунду
REST_WITHIN = 2.5        # остановка не позже 2.5 с после контакта
REST_HOLD = 4.0          # и стоят не меньше 4 с (очередь у светофора трогается раньше, авария — нет)
JERK_REL = 2.0           # резкость: скорость упала на 2 высоты/с за секунду
NEAR_ND = 0.6            # near miss: сблизились ближе 0.6 высоты, но без контакта
NEAR_MIN_ND = 0.5
EVASION_ACC_REL = 3.0    # уклонение: замедление > 3 высот/с² или поворот > 40°
EVASION_TURN_DEG = 40.0
NEAR_MIN_SPEED_REL = 2.0 # near miss только на скорости: медленный подъезд к пешеходу у перехода — норма
MIN_OVERLAP = 0.6        # минимальное совместное время пары, с
ACCIDENT_MIN_SEC = 3.0   # длительность аварии по разметке организаторов обычно 3–6 с
NEAR_MISS_MIN_SEC = 2.0
MAX_PAIR_DIST_REL = 6.0  # пары дальше 6 высот друг от друга даже не рассматриваем


def run(ctx: Context) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    tracks = [t for t in ctx.tracks.values() if t.kind in ("vehicle", "person", "rider") and len(t.t) >= 4]
    accidents: list[tuple[float, float]] = []
    near: list[tuple[float, float]] = []
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            ta, tb = tracks[a], tracks[b]
            if ta.kind != "vehicle" and tb.kind != "vehicle":
                continue
            s, e = max(ta.t0, tb.t0), min(ta.t1, tb.t1)
            if e - s < MIN_OVERLAP:
                continue
            res = _analyse_pair(ta, tb, s, e, lambda x, y, t, size: ctx.in_queue(x, y, t, size, exclude={ta.id, tb.id}))
            if res is None:
                continue
            kind, iv = res
            (accidents if kind == "accident" else near).append(iv)
    return accidents, near


def _resample(tr: Track, ts: np.ndarray) -> dict:
    return {k: np.interp(ts, tr.t, getattr(tr, k)) for k in ("cx", "cy", "w", "h", "speed", "vx", "vy", "accel")}


def _analyse_pair(ta: Track, tb: Track, s: float, e: float, is_hotspot=None):
    ts = np.arange(s, e, 0.08)
    if len(ts) < 6:
        return None
    A, B = _resample(ta, ts), _resample(tb, ts)
    size = (A["h"] + B["h"]) / 2.0
    dist = np.hypot(A["cx"] - B["cx"], A["cy"] - B["cy"])
    nd = dist / np.maximum(size, 1.0)
    if nd.min() > MAX_PAIR_DIST_REL:
        return None
    iou = _iou_series(A, B)
    closing = -np.gradient(dist, ts)               # >0 — сближаются
    approach = closing > APPROACH_REL * size
    i_min = int(np.argmin(nd))
    lo = max(0, i_min - int(2.0 / 0.08))
    # «догнал очередь»: тот, к кому подъехали, УЖЕ стоял до подъезда и стоял в очереди/точке штатных остановок
    if is_hotspot is not None:
        for X in (A, B):
            slow_now = X["speed"][i_min] < 0.5 * X["h"][i_min]
            if slow_now and (is_hotspot(X["cx"][lo], X["cy"][lo] + X["h"][lo] / 2, float(ts[lo]), X["h"][lo])
                             or is_hotspot(X["cx"][i_min], X["cy"][i_min] + X["h"][i_min] / 2, float(ts[i_min]), X["h"][i_min])):
                return None
    # был ли быстрый подход за 2 с до минимума
    fast_before = bool(approach[lo:i_min + 1].any())
    moving_before = bool((A["speed"][lo:i_min + 1] > 0.8 * A["h"][lo:i_min + 1]).any() or
                         (B["speed"][lo:i_min + 1] > 0.8 * B["h"][lo:i_min + 1]).any())
    if not (fast_before and moving_before):
        return None
    contact = (nd < CONTACT_ND) | (iou > CONTACT_IOU)
    if contact.any():
        i_c = int(np.argmax(contact))
        t_c = float(ts[i_c])
        rest = _rest_time(A, B, ts, i_c)
        jerk = _jerk(A, B, ts, i_c)
        # наезд на пешехода засчитываем только при быстрой машине: медленный подъезд к людям у зебры — норма
        if ta.kind != "vehicle" or tb.kind != "vehicle":
            V = A if ta.kind == "vehicle" else B
            if (V["speed"][lo:i_c + 1] < 2.0 * V["h"][lo:i_c + 1]).all():
                return None
        if rest is not None and jerk:
            # конец: объекты остановились и постояли (по конвенции «все остановились»); не короче ACCIDENT_MIN_SEC
            return "accident", (t_c, max(rest + 1.0, t_c + ACCIDENT_MIN_SEC))
        return None
    # без контакта: сближение и уклонение; для пары машин обе должны были двигаться
    a_fast = (A["speed"][lo:i_min + 1] > NEAR_MIN_SPEED_REL * A["h"][lo:i_min + 1]).any()
    b_fast = (B["speed"][lo:i_min + 1] > NEAR_MIN_SPEED_REL * B["h"][lo:i_min + 1]).any()
    vehicles_fast = (a_fast or ta.kind != "vehicle") and (b_fast or tb.kind != "vehicle") and (a_fast or b_fast)
    if nd[i_min] < NEAR_ND and nd[i_min] >= NEAR_MIN_ND and vehicles_fast:
        w0 = max(0, i_min - int(1.0 / 0.08)); w1 = min(len(ts) - 1, i_min + int(1.0 / 0.08))
        dec = min(A["accel"][w0:w1 + 1].min() / max(A["h"][i_min], 1), B["accel"][w0:w1 + 1].min() / max(B["h"][i_min], 1))
        turn = max(_turn_deg(A, w0, w1), _turn_deg(B, w0, w1))
        if dec < -EVASION_ACC_REL and turn > EVASION_TURN_DEG / 2:   # уклонение = торможение И виляние; одно без другого в плотном потоке слишком часто
            onset = float(ts[w0 + int(np.argmin(np.minimum(A["accel"][w0:w1 + 1], B["accel"][w0:w1 + 1])))]) if dec < -EVASION_ACC_REL else float(ts[w0])
            clear = ts[i_min:][nd[i_min:] > 1.5]
            t_end = float(clear[0]) if len(clear) else float(ts[-1])
            start = min(onset, float(ts[i_min]))
            return "near_miss", (start, max(t_end, start + NEAR_MISS_MIN_SEC))
    return None


def _rest_time(A, B, ts, i_c):
    """Момент, когда оба объекта остановились после контакта и стоят REST_HOLD секунд; None если не остановились."""
    step = ts[1] - ts[0]
    hold = int(REST_HOLD / step)
    limit = min(len(ts), i_c + int(REST_WITHIN / step) + hold + 1)
    for i in range(i_c, limit - hold):
        a_rest = (A["speed"][i:i + hold] < REST_REL * A["h"][i:i + hold]).all()
        b_rest = (B["speed"][i:i + hold] < REST_REL * B["h"][i:i + hold]).all()
        if a_rest and b_rest:
            return float(ts[i])
    # объект пропал (слился с другим боксом) почти сразу после контакта — тоже признак
    if len(ts) - i_c < int(1.0 / step):
        return float(ts[-1])
    return None


def _jerk(A, B, ts, i_c) -> bool:
    step = ts[1] - ts[0]
    w = int(1.0 / step)
    lo, hi = max(0, i_c - w), min(len(ts) - 1, i_c + w)
    for X in (A, B):
        drop = (X["speed"][lo:i_c + 1].max() - X["speed"][i_c:hi + 1].min()) / max(X["h"][i_c], 1)
        if drop > JERK_REL:
            return True
    return False


def _turn_deg(X, w0, w1) -> float:
    v0 = np.array([X["vx"][w0], X["vy"][w0]]); v1 = np.array([X["vx"][w1], X["vy"][w1]])
    n0, n1 = np.hypot(*v0), np.hypot(*v1)
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1, 1))))


def _iou_series(A, B) -> np.ndarray:
    ax1, ay1 = A["cx"] - A["w"] / 2, A["cy"] - A["h"] / 2
    ax2, ay2 = A["cx"] + A["w"] / 2, A["cy"] + A["h"] / 2
    bx1, by1 = B["cx"] - B["w"] / 2, B["cy"] - B["h"] / 2
    bx2, by2 = B["cx"] + B["w"] / 2, B["cy"] + B["h"] / 2
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = A["w"] * A["h"] + B["w"] * B["h"] - inter
    return inter / np.maximum(union, 1e-6)

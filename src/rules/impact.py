"""accident по «точке удара»: второе правило, не зависящее от целостности треков.

Парное правило (collision.py) требует, чтобы одна и та же пара треков сблизилась, «коснулась» и встала. На настоящих
ДТП с уличных камер это почти никогда не выполняется: центры боксов двух столкнувшихся машин остаются на 0,5–0,9
высоты друг от друга, а трекер на ударе рвёт треки и раздаёт новые id. Здесь удар ищется иначе.

Главный признак — резкость. При торможении скорость падает за 1–3 с, при ударе — за 0,1–0,3 с. Сглаженные
скорости трека (окно 0,4 с) эту разницу стирают, поэтому резкость считается по сырым наблюдениям детектора:
скорость по центрам боксов на шаге STEP_SEC до и после момента, падение ≥ IMPACT_DROP за ≤ ABRUPT_SEC при
скорости до удара ≥ IMPACT_FAST высот/с.

Второй вариант того же события — трек быстрого участника обрывается внутри кадра, а рядом сразу рождается
трек, стоящий с первого наблюдения (трекер перезапустил объект после удара).

Дальше проверяются контекст и последствия: рядом в момент удара есть второй участник (ближе IMPACT_NEAR высот),
и после удара у этой точки кто-то стоит не меньше REST_MIN с (id не важен). Пешеход как второй участник
должен упасть/остановиться, иначе это обычный пропуск пешехода. Точки штатных остановок (StopMap) исключаются.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..tracking import Track

IMPACT_FAST = 1.5        # скорость до удара: > 1.5 высоты бокса в секунду
IMPACT_DROP = 0.6        # потеря ≥ 60 % скорости
ABRUPT_SEC = 0.35        # за ≤ 0.35 с (сырые наблюдения, без сглаживания)
HOLD_SEC = 0.6           # устойчиво быстро до и устойчиво медленно после — не меньше 0.6 с
LOW_FRAC = 0.4           # после удара сырая скорость не выше 40 % от уровня до удара
STEP_SEC = 0.2           # скорость по сырым центрам считается на шаге 0.2 с
IMPACT_NEAR = 2.0        # второй участник ближе 2 средних высот в момент удара (±0.5 с)
REST_WINDOW = (0.3, 6.0) # покой ищем в этом окне после удара
REST_MIN = 3.0           # стоит не меньше 3 с
REST_REL = 0.15          # «стоит»: < 15 % высоты в секунду
REST_RADIUS = 2.0        # у точки удара: ближе 2 высот
BORDER_REL = 1.0         # обрыв трека ближе высоты бокса к краю кадра — уход из кадра, не удар
SUCCESSOR_SEC = 0.8      # после обрыва в течение этого времени рядом рождается стоящий трек
SUCCESSOR_NEAR = 1.5
ACCIDENT_MIN_SEC = 3.0
MAX_EVENT_SEC = 12.0


def _raw_speed(tr: Track, step: float = STEP_SEC) -> np.ndarray:
    """Скорость по сырым центрам боксов: смещение за ~step секунд назад, px/с (без сглаживания)."""
    n = len(tr.obs)
    cx = np.array([(o.x1 + o.x2) / 2 for o in tr.obs]); cy = np.array([(o.y1 + o.y2) / 2 for o in tr.obs])
    t = tr.t
    v = np.zeros(n)
    for i in range(n):
        j = int(np.searchsorted(t, t[i] - step))
        j = min(j, i - 1) if i > 0 else 0
        dt = t[i] - t[j]
        if dt > 1e-3:
            v[i] = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j])) / dt
    return v


def abrupt_stops(tr: Track) -> list[int]:
    """Индексы наблюдений, где сырая скорость устойчиво высокая (≥ IMPACT_FAST высот/с не меньше HOLD_SEC подряд)
    падает за ≤ ABRUPT_SEC до устойчиво низкой (≤ LOW_FRAC от уровня до удара не меньше HOLD_SEC), и сглаженная
    скорость трека подтверждает остановку. Минимум по окну «до» и максимум по окну «после» не пропускают
    одиночные выбросы от дрожания бокса."""
    n = len(tr.t)
    if n < 6:
        return []
    h = max(tr.size, 1.0)
    v = _raw_speed(tr)
    t = tr.t
    out: list[int] = []
    last = -10.0
    for i in range(2, n):
        a1 = int(np.searchsorted(t, t[i] - ABRUPT_SEC, side="right")) - 1   # последнее наблюдение до перехода
        if a1 < 1:
            continue
        a0 = int(np.searchsorted(t, t[a1] - HOLD_SEC))
        if a1 - a0 < 2:
            continue
        before = v[a0:a1 + 1]
        if float(before.min()) < IMPACT_FAST * h:
            continue
        b1 = int(np.searchsorted(t, t[i] + HOLD_SEC, side="right"))
        if b1 - i < 2 or t[b1 - 1] - t[i] < HOLD_SEC * 0.6:
            continue
        after = v[i:b1]
        level = float(np.median(before))
        if float(after.max()) > LOW_FRAC * level:
            continue
        # сглаженная скорость: была высокой, стала низкой
        s0 = tr.speed[max(0, a0):a1 + 1].max(); s1 = tr.speed[i:b1].min()
        if s0 < 1.0 * h or s1 > 0.4 * h:
            continue
        if t[i] - last > 2.0:
            out.append(i)
            last = float(t[i])
    return out


def _partner_near(ctx: Context, tr: Track, i: int, radius: float) -> Track | None:
    t = float(tr.t[i]); x, y = float(tr.cx[i]), float(tr.by[i])
    best, best_d = None, None
    for o in ctx.tracks.values():
        if o.id == tr.id or o.kind not in ("vehicle", "person", "rider") or not (o.t0 - 0.5 <= t <= o.t1 + 0.5):
            continue
        j = o.index_at(t)
        d = float(np.hypot(o.cx[j] - x, o.by[j] - y))
        if d < radius and (best_d is None or d < best_d):
            best, best_d = o, d
    return best


def _rest_near(ctx: Context, x: float, y: float, t0: float, t1: float, radius: float, min_sec: float) -> tuple[float, float] | None:
    """Кто-нибудь стоит у точки (x, y) внутри [t0, t1] не меньше min_sec подряд → (начало, конец) покоя."""
    best = None
    for o in ctx.tracks.values():
        if o.kind not in ("vehicle", "person", "rider") or o.t1 < t0 or o.t0 > t1:
            continue
        lo, hi = int(np.searchsorted(o.t, t0)), int(np.searchsorted(o.t, t1, side="right"))
        if hi - lo < 3:
            continue
        near = np.hypot(o.cx[lo:hi] - x, o.by[lo:hi] - y) < radius
        still = o.speed[lo:hi] < REST_REL * max(o.size, 1.0)
        ok = near & still
        start = None
        for k in range(len(ok) + 1):
            cur = ok[k] if k < len(ok) else False
            if cur and start is None:
                start = k
            if not cur and start is not None:
                a, b = float(o.t[lo + start]), float(o.t[lo + k - 1])
                if b - a >= min_sec and (best is None or a < best[0]):
                    best = (a, b)
                start = None
    return best


def _successor_born_still(ctx: Context, tr: Track) -> bool:
    """После обрыва трека рядом рождается трек, стоящий с первого наблюдения (перезапуск объекта после удара)."""
    t, x, y = tr.t1, float(tr.cx[-1]), float(tr.by[-1])
    h = max(tr.size, 1.0)
    for o in ctx.tracks.values():
        if o.id == tr.id or o.kind != "vehicle" or not (t - 0.2 <= o.t0 <= t + SUCCESSOR_SEC) or len(o.t) < 3:
            continue
        if np.hypot(o.cx[0] - x, o.by[0] - y) < SUCCESSOR_NEAR * h and float(o.speed[:3].max()) < 0.5 * max(o.size, 1.0):
            return True
    return False


def candidates(ctx: Context, tr: Track) -> list[tuple[int, str]]:
    out = [(i, "drop") for i in abrupt_stops(tr)]
    h = max(tr.size, 1.0)
    n = len(tr.t)
    v = _raw_speed(tr)
    k0 = int(np.searchsorted(tr.t, tr.t[-1] - HOLD_SEC))
    if n >= 4 and n - k0 >= 3 and float(v[k0:].min()) >= IMPACT_FAST * h and _successor_born_still(ctx, tr):
        out.append((n - 1, "end"))
    return out


def run(ctx: Context, queue_checks: bool = True, debug: list | None = None) -> list[tuple[float, float]]:
    W, H = ctx.meta["width"], ctx.meta["height"]
    out: list[tuple[float, float]] = []
    for tr in ctx.vehicles():
        h = max(tr.size, 1.0)
        for i, kind in candidates(ctx, tr):
            t = float(tr.t[i]); x, y = float(tr.cx[i]), float(tr.by[i])
            if x < BORDER_REL * h or x > W - BORDER_REL * h or y < BORDER_REL * h or y > H - BORDER_REL * h:
                continue
            if not ctx.on_road(x, y):
                continue
            partner = _partner_near(ctx, tr, i, IMPACT_NEAR * h)
            if partner is None:
                continue
            if partner.kind != "vehicle":
                j = partner.index_at(t)
                fell = partner.t1 - t < 1.5 or bool((partner.speed[j:] < REST_REL * max(partner.size, 1.0)).any())
                if not fell:
                    continue
            if queue_checks and ctx.is_stop_hotspot(x, y):
                continue
            rest = _rest_near(ctx, x, y, t + REST_WINDOW[0], t + REST_WINDOW[1], REST_RADIUS * h, REST_MIN)
            if rest is None:
                continue
            end = max(t + ACCIDENT_MIN_SEC, min(rest[1] + 1.0, t + MAX_EVENT_SEC))
            out.append((t, end))
            if debug is not None:
                debug.append((tr.id, kind, round(t, 1), int(x), int(y), partner.id, partner.kind))
    return out

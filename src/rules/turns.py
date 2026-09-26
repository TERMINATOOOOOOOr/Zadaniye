"""illegal_u_turn: разворот на 180° там, где запрещено; illegal_turn: поворот из неположенной полосы / в запрещённом направлении.

Полоса манёвра — «полоса происхождения»: последняя зона сцены, в которой машина была до начала вращения
(поворот начинается уже на перекрёстке, где зон нет), не дальше LOOKBACK_SEC назад. Углы считаются в
координатах кадра, где перспектива сжимает и растягивает повороты, поэтому диапазон поворота широкий.
"""
from __future__ import annotations

import numpy as np

from ..context import Context
from ..scene import Lane
from ..tracking import Track

MOVE_REL = 0.4
U_TURN_DEG = 160.0
U_TURN_MIN_DISP = 2.5    # смещение между началом и концом разворота не меньше 2.5 высот бокса
INTERSECTION_MARGIN = 2.0  # разворот, начатый ближе 2 высот к зоне перекрёстка, считаем манёвром на перекрёстке
TURN_MIN_DEG, TURN_MAX_DEG = 50.0, 150.0
WINDOW_SEC = 12.0
RATE_DEG_PER_S = 8.0     # начало/конец манёвра: угловая скорость выше этого
GAP_SEC = 0.5            # пауза в повороте короче этого не разрывает манёвр
LOOKBACK_SEC = 4.0       # полосу происхождения ищем не дальше этого назад от начала вращения
U_TURN_ILLEGAL_DEFAULT = True   # если сцена не говорит, что разворот разрешён, считаем запрещённым


def _headings(tr) -> np.ndarray:
    return np.degrees(np.unwrap(np.arctan2(tr.vy, tr.vx)))


def _maneuvers(tr):
    """Участки поворота: (i_start, i_end, суммарный угол в градусах, знак). Конец = последний кадр с вращением."""
    n = len(tr.t)
    if n < 6:
        return []
    moving = tr.speed > MOVE_REL * tr.size
    hd = _headings(tr)
    rate = np.abs(np.gradient(hd, tr.t))
    active = moving & (rate > RATE_DEG_PER_S)
    out = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = last = i
        while j + 1 < n and moving[j + 1] and (active[j + 1] or tr.t[j + 1] - tr.t[last] < GAP_SEC):
            j += 1
            if active[j]:
                last = j
            if tr.t[last] - tr.t[i] > WINDOW_SEC:
                break
        total = float(hd[last] - hd[i])
        out.append((i, last, abs(total), np.sign(total)))
        i = last + 1
    return out


def origin_lane(ctx: Context, tr: Track, i: int) -> Lane | None:
    """Зона сцены, из которой машина пришла к кадру i: сам кадр или последний кадр в зоне не дальше LOOKBACK_SEC назад."""
    if ctx.scene is None or not ctx.scene.has_lanes():
        return None
    for k in range(i, -1, -1):
        if tr.t[i] - tr.t[k] > LOOKBACK_SEC:
            break
        ln = ctx.scene.lane_at(tr.cx[k], tr.by[k])
        if ln is not None:
            return ln
    return None


def u_turns(ctx: Context) -> list[tuple[Track, float, float]]:
    """Запрещённые развороты: (трек, начало, конец)."""
    out = []
    for tr in ctx.vehicles():
        for i, j, deg, _ in _maneuvers(tr):
            if deg < U_TURN_DEG:
                continue
            if not ctx.on_road(tr.cx[i], tr.by[i]):
                continue
            if ctx.scene is not None and ctx.scene.intersection is not None:
                import cv2
                if cv2.pointPolygonTest(ctx.scene.intersection, (float(tr.cx[i]), float(tr.by[i])), True) > -INTERSECTION_MARGIN * tr.size:
                    continue   # разворот на перекрёстке или рядом — обычный манёвр, судить без разметки полос нельзя
            if np.hypot(tr.cx[j] - tr.cx[i], tr.by[j] - tr.by[i]) < U_TURN_MIN_DISP * tr.size:
                continue
            allowed = False
            if ctx.scene is not None and ctx.scene.has_lanes():
                ln = origin_lane(ctx, tr, i)
                allowed = bool(ln and "uturn" in ln.allowed)
            elif not U_TURN_ILLEGAL_DEFAULT:
                allowed = True
            if not allowed:
                out.append((tr, float(tr.t[i]), float(tr.t[j])))
    return out


def run_u_turn(ctx: Context) -> list[tuple[float, float]]:
    return [(s, e) for _, s, e in u_turns(ctx)]


def illegal_turns(ctx: Context) -> list[tuple[Track, float, float, str, str]]:
    """Повороты в запрещённом направлении: (трек, начало, конец, направление, полоса происхождения)."""
    if ctx.scene is None or not ctx.scene.has_lanes():
        return []
    out = []
    for tr in ctx.vehicles():
        for i, j, deg, sign in _maneuvers(tr):
            if not (TURN_MIN_DEG <= deg <= TURN_MAX_DEG):
                continue
            ln = origin_lane(ctx, tr, i)
            if ln is None:
                continue
            # в координатах изображения (y вниз) положительный угол = поворот по часовой = направо
            turn = "right" if sign > 0 else "left"
            if turn not in ln.allowed:
                out.append((tr, float(tr.t[i]), float(tr.t[j]), turn, ln.name))
    return out


def run_illegal_turn(ctx: Context) -> list[tuple[float, float]]:
    return [(s, e) for _, s, e, _, _ in illegal_turns(ctx)]

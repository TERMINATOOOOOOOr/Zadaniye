"""Состояние светофора по видео: три независимых источника и их слияние.

1. `LampReader` — цвет лампы по вырезу полного разрешения вокруг ROI из сцены. Секции (красная/жёлтая/зелёная)
   оцениваются по энергии своего тона относительно других секций той же лампы, а не по абсолютным порогам:
   так читается и тусклая лампа днём, и яркая в сумерках. Положение секций калибруется по самому видео
   (где во времени вспыхивает красное и где зелёное) или берётся из трека детектора класса «traffic light».
2. `signal_from_pedestrians` — пока пешеходы идут по зебре, которую защищает стоп-линия, машинам горит красный;
   пока машины пересекают линию на скорости и на зебре никого нет — зелёный.
3. `signal_from_traffic` — очередь у стоп-линии стоит / машины едут (самый слабый источник: очередь стоит и на зелёный).

`fuse_signals` выбирает лампу, если её фазы правдоподобны и согласуются с пешеходами, иначе пешеходов, иначе
трафик, и ставит флаг `reliable`, без которого правила red_light/stop_line молчат.
"""
from __future__ import annotations

import numpy as np

from .context import SignalTrack
from .video import Crop, normalize_crop

LAMP_MARGIN = 2.0        # вырез вокруг ROI лампы: во столько раз больше самой ROI (сдвиг камеры до ~половины ROI)
LAMP_CELL = 4            # клетка сетки признаков, px исходного кадра
FEATURES_VERSION = 2     # меняется вместе с lamp_features(): ключ кэша

HUE_RED = ((0, 8), (168, 180))
HUE_AMBER = (8, 35)
HUE_GREEN = (40, 100)

DOMINANCE = 3.0          # энергия тона в своей секции ≥ DOMINANCE × энергия того же тона в дальней секции
ENERGY_FLOOR = 0.004     # и не меньше этого абсолютно (доля насыщенности×яркости; днём горящий красный ≈ 0.015)
HOLD_SEC = 0.4           # новое состояние принимается, когда держится столько секунд
MAX_GAP_SEC = 4.0        # состояние тянется через «неизвестно» (автобус закрыл лампу) не дольше этого
PHASE_MIN, PHASE_MAX = 10.0, 150.0   # правдоподобная длина фазы красного/зелёного
AMBER_MAX = 6.0          # жёлтый дольше этого — не жёлтый
AGREE_MIN = 0.6          # согласие лампы с пешеходными фазами там, где известны обе (пешеходы ходят и на красный)
PERIOD_TOL = 0.25        # лампа «периодична»: длины внутренних фаз красного и зелёного в пределах ±25 % от медианы
FUSE_STEP = 0.2


# ---------------------------------------------------------------------------
# Лампа: признаки по кадру
# ---------------------------------------------------------------------------
def lamp_crop_rect(roi: tuple[int, int, int, int], width: int, height: int, margin: float = LAMP_MARGIN) -> Crop:
    """Вырез вокруг ROI лампы с запасом (в пикселях исходного кадра, чётный, в границах кадра)."""
    x, y, w, h = roi
    cw, ch = w * margin, h * margin
    return normalize_crop((int(round(x + w / 2 - cw / 2)), int(round(y + h / 2 - ch / 2)), int(round(cw)), int(round(ch))), width, height)


def lamp_features(crop_bgr: np.ndarray, cell: int = LAMP_CELL) -> np.ndarray:
    """Сетка (rows, cols, 4): энергия красного, жёлтого и зелёного тона (насыщенность×яркость в своей полосе тона)
    и средняя яркость по клеткам cell×cell px. Полное разрешение сохраняется в тоне, память — на сетке."""
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    weight = (hsv[..., 1].astype(np.float32) / 255.0) * (hsv[..., 2].astype(np.float32) / 255.0)
    red = ((hue < HUE_RED[0][1]) | (hue >= HUE_RED[1][0])).astype(np.float32) * weight
    amber = ((hue >= HUE_AMBER[0]) & (hue <= HUE_AMBER[1])).astype(np.float32) * weight
    green = ((hue >= HUE_GREEN[0]) & (hue <= HUE_GREEN[1])).astype(np.float32) * weight
    bright = hsv[..., 2].astype(np.float32) / 255.0
    h, w = hue.shape
    cols, rows = max(1, int(np.ceil(w / cell))), max(1, int(np.ceil(h / cell)))
    stack = np.dstack([red, amber, green, bright])
    return cv2.resize(stack, (cols, rows), interpolation=cv2.INTER_AREA)


def _p95_peak(m: np.ndarray, floor: float) -> tuple[int, int] | None:
    """Клетка с максимумом карты, если это явный пик (≥ 3× медианы карты и ≥ floor)."""
    if m.size == 0:
        return None
    r, c = np.unravel_index(int(np.argmax(m)), m.shape)
    peak = float(m[r, c])
    if peak < floor or peak < 3.0 * float(np.median(m)) + floor:
        return None
    return int(r), int(c)


class LampReader:
    """Копит сетки признаков по кадрам; в finish() калибрует положение секций и читает состояние во времени."""

    def __init__(self, roi: tuple[int, int, int, int], crop: Crop, cell: int = LAMP_CELL) -> None:
        self.roi = tuple(int(v) for v in roi)
        self.crop = tuple(int(v) for v in crop)
        self.cell = cell
        self.ts: list[float] = []
        self.feats: list[np.ndarray] = []
        self.debug: dict = {}

    # --- накопление ---
    def push(self, crop_bgr: np.ndarray, t: float) -> None:
        self.ts.append(float(t))
        self.feats.append(lamp_features(crop_bgr, self.cell).astype(np.float16))

    def load(self, t: np.ndarray, feats: np.ndarray) -> None:
        self.ts = [float(v) for v in t]
        self.feats = [f for f in feats]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.feats:
            return np.zeros(0), np.zeros((0, 1, 1, 4), dtype=np.float16)
        return np.asarray(self.ts), np.stack(self.feats)

    # --- калибровка секций ---
    def _box_from_tracks(self, tracks) -> tuple[float, float, float, float] | None:
        """Бокс лампы (x0, y0, x1, y1 в координатах выреза) по треку «traffic light», ближайшему к ROI."""
        if not tracks:
            return None
        rx, ry = self.roi[0] + self.roi[2] / 2, self.roi[1] + self.roi[3] / 2
        cx0, cy0, cw, ch = self.crop
        best, best_d = None, None
        for tr in (tracks.values() if isinstance(tracks, dict) else tracks):
            if tr.cls != "traffic light" or len(tr.t) < 30:
                continue
            mx, my = float(np.median(tr.cx)), float(np.median(tr.cy))
            if not (cx0 <= mx < cx0 + cw and cy0 <= my < cy0 + ch):
                continue
            d = float(np.hypot(mx - rx, my - ry))
            if best_d is None or d < best_d:
                best, best_d = tr, d
        if best is None:
            return None
        w, h = float(np.median(best.w)), float(np.median(best.h))
        mx, my = float(np.median(best.cx)) - cx0, float(np.median(best.cy)) - cy0
        return (max(0.0, mx - w / 2), max(0.0, my - h / 2), min(cw, mx + w / 2), min(ch, my + h / 2))

    def _sections(self, feats: np.ndarray, box: tuple[float, float, float, float]) -> tuple[list[tuple[int, int, int, int]], str]:
        """Три секции (r0, r1, c0, c1) в клетках сетки: по временным пикам красного и зелёного внутри бокса,
        иначе — бокс на три равные части."""
        cell = self.cell
        c0, r0 = int(box[0] // cell), int(box[1] // cell)
        c1, r1 = int(np.ceil(box[2] / cell)), int(np.ceil(box[3] / cell))
        c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)
        sub = feats[:, r0:r1, c0:c1, :].astype(np.float32)
        red_map = np.percentile(sub[..., 0], 95, axis=0)
        green_map = np.percentile(sub[..., 2], 95, axis=0)
        pr, pg = _p95_peak(red_map, ENERGY_FLOOR), _p95_peak(green_map, ENERGY_FLOOR)
        if pr is not None and pg is not None and pg[0] - pr[0] >= 2 and abs(pg[1] - pr[1]) <= max(2, (pg[0] - pr[0]) // 2):
            yr, yg = pr[0] + r0 + 0.5, pg[0] + r0 + 0.5
            xc = (pr[1] + pg[1]) / 2 + c0 + 0.5
            pitch = (yg - yr) / 2                      # шаг между линзами
            half = max(1.0, pitch / 2)
            centres = (yr, (yr + yg) / 2, yg)
            secs = []
            for yc in centres:
                secs.append((max(0, int(np.floor(yc - half))), max(1, int(np.ceil(yc + half))),
                             max(0, int(np.floor(xc - half))), max(1, int(np.ceil(xc + half)))))
            return secs, "colour"
        third = (r1 - r0) / 3
        secs = [(int(r0 + k * third), max(int(r0 + k * third) + 1, int(r0 + (k + 1) * third)), c0, c1) for k in range(3)]
        return secs, "thirds"

    # --- чтение ---
    def finish(self, tracks=None) -> SignalTrack:
        ts, feats = self.arrays()
        n = len(ts)
        if n == 0:
            return SignalTrack(origin="lamp")
        box = self._box_from_tracks(tracks)
        box_src = "track"
        if box is None:
            cx0, cy0, cw, ch = self.crop
            box = (max(0.0, self.roi[0] - cx0), max(0.0, self.roi[1] - cy0),
                   min(cw, self.roi[0] + self.roi[2] - cx0), min(ch, self.roi[1] + self.roi[3] - cy0))
            box_src = "roi"
        secs, sec_src = self._sections(feats, box)
        f32 = feats.astype(np.float32)
        # энергия каждого тона в каждой секции: (n, 3 секции, 3 тона)
        e = np.zeros((n, 3, 3), dtype=np.float32)
        for k, (r0, r1, c0, c1) in enumerate(secs):
            e[:, k, :] = f32[:, r0:r1, c0:c1, :3].mean(axis=(1, 2))
        # своя секция против дальней: красный — верх против низа, зелёный — низ против верха,
        # жёлтый — середина против большей из крайних (жёлтый «засвечивает» соседей)
        red_score = e[:, 0, 0] / (e[:, 2, 0] + ENERGY_FLOOR / 3)
        green_score = e[:, 2, 2] / (e[:, 0, 2] + ENERGY_FLOOR / 3)
        amber_score = e[:, 1, 1] / (np.maximum(e[:, 0, 1], e[:, 2, 1]) + ENERGY_FLOOR / 3)
        red_on = (red_score >= DOMINANCE) & (e[:, 0, 0] >= ENERGY_FLOOR)
        green_on = (green_score >= DOMINANCE) & (e[:, 2, 2] >= ENERGY_FLOOR)
        amber_on = (amber_score >= DOMINANCE) & (e[:, 1, 1] >= ENERGY_FLOOR)
        occluded = self._occlusion(f32)
        raw: list[str] = []
        for i in range(n):
            if occluded[i] or (red_on[i] and green_on[i]):
                raw.append("unknown")
            elif red_on[i]:
                raw.append("red")          # красный+жёлтый перед зелёным — всё ещё красный
            elif green_on[i]:
                raw.append("green")
            elif amber_on[i]:
                raw.append("amber")
            else:
                raw.append("unknown")
        states = smooth_states(raw, ts, HOLD_SEC, MAX_GAP_SEC)
        self.debug = {"box": box, "box_src": box_src, "sections": secs, "sections_src": sec_src, "raw": raw,
                      "energy": e, "occluded": occluded}
        return SignalTrack(t=ts, state=states, source=["lamp"] * n, origin="lamp")

    @staticmethod
    def _occlusion(f32: np.ndarray) -> np.ndarray:
        """Резкая смена яркости или общей цветности всего выреза относительно скользящей медианы — что-то
        проехало перед лампой (или прямо за ней); такие кадры не читаем."""
        from scipy.ndimage import median_filter
        n = f32.shape[0]
        bright = f32[..., 3].mean(axis=(1, 2))
        colour = f32[..., :3].sum(axis=3).mean(axis=(1, 2))
        win = min(n, 41) | 1
        base_b = median_filter(bright, size=win, mode="nearest")
        base_c = median_filter(colour, size=win, mode="nearest")
        db, dc = np.abs(bright - base_b), np.abs(colour - base_c)
        thr_b = max(0.06, 5.0 * float(np.median(db)))
        thr_c = max(0.02, 5.0 * float(np.median(dc)))
        return (db > thr_b) | (dc > thr_c)


def smooth_states(raw: list[str], ts: np.ndarray, hold_sec: float = HOLD_SEC, max_gap_sec: float = MAX_GAP_SEC) -> list[str]:
    """Гистерезис: новое состояние принимается, продержавшись hold_sec; «неизвестно» не сбрасывает состояние,
    пока разрыв короче max_gap_sec."""
    out: list[str] = []
    cur = "unknown"
    run_state, run_start, last_known = None, 0.0, None
    for i, s in enumerate(raw):
        t = float(ts[i])
        if s == "unknown":
            if cur != "unknown" and last_known is not None and t - last_known > max_gap_sec:
                cur = "unknown"
            out.append(cur)
            continue
        if s != run_state:
            run_state, run_start = s, t
        if s != cur and (t - run_start >= hold_sec or i == 0):
            cur = s
        if cur == s:
            last_known = t
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Геометрия стоп-линии
# ---------------------------------------------------------------------------
def stop_line_offset(sl, x: float, y: float) -> float:
    """Расстояние от точки до прямой стоп-линии по нормали, положительное — за линией по направлению подъезда.
    (Сама линия в кадре не перпендикулярна направлению подъезда из-за перспективы, проекция на approach врёт.)"""
    from .scene import side
    length = float(np.hypot(*(sl.p2 - sl.p1))) or 1e-9
    mid = (sl.p1 + sl.p2) / 2
    sign = 1.0 if side(mid[0] + sl.approach[0], mid[1] + sl.approach[1], sl.p1, sl.p2) >= 0 else -1.0
    return sign * side(x, y, sl.p1, sl.p2) / length


def along_line(sl, x: float, y: float) -> float:
    """Положение точки вдоль стоп-линии: 0 — у p1, 1 — у p2 (для проверки, что машина в створе линии)."""
    ab = sl.p2 - sl.p1
    denom = float(np.dot(ab, ab)) or 1e-9
    return float(np.dot(np.array([x, y]) - sl.p1, ab) / denom)


def bind_crosswalk(scene, sl) -> tuple[str, np.ndarray] | None:
    """Зебра, которую защищает стоп-линия: ближайший полигон перехода впереди по направлению подъезда."""
    mid = (sl.p1 + sl.p2) / 2
    best, best_d = None, None
    for name, poly in scene.crosswalks:
        c = poly.mean(axis=0)
        d = c - mid
        if float(np.dot(d, sl.approach)) <= 0:
            continue
        dist = float(np.hypot(*d))
        if best_d is None or dist < best_d:
            best, best_d = (name, poly), dist
    return best


# ---------------------------------------------------------------------------
# Пешеходные фазы
# ---------------------------------------------------------------------------
PED_MIN_H_FRAC = 0.03    # пешеходы ниже 3 % высоты кадра не считаем (далеко, шум детектора)
PED_MANY = 2             # столько пешеходов на зебре — машинам точно красный
PED_ONE_SEC = 3.0        # или один, но идёт по ней не меньше стольких секунд
PED_PAD_SEC = 1.5        # красный расширяем на столько в обе стороны (люди идут группами с просветами)
CROSS_MOVE_REL = 0.5     # «пересёк на скорости»: быстрее половины высоты бокса в секунду
GREEN_WINDOW_SEC = 3.0   # зелёный держится столько после последнего проезда линии на скорости


def signal_from_pedestrians(ctx, bin_sec: float = 0.5) -> SignalTrack | None:
    """Красный — пешеходы идут по зебре за стоп-линией; зелёный — машины пересекают линию на скорости,
    а на зебре никого; иначе неизвестно."""
    if ctx.scene is None or not ctx.scene.has_stop_lines() or not ctx.scene.has_crosswalks() or ctx.duration <= 0:
        return None
    from .scene import seg_intersect, point_in_poly
    bound = [(sl, bind_crosswalk(ctx.scene, sl)) for sl in ctx.scene.stop_lines]
    bound = [(sl, cw) for sl, cw in bound if cw is not None]
    if not bound:
        return None
    ts = np.arange(0.0, ctx.duration + bin_sec, bin_sec)
    nb = len(ts)
    min_h = PED_MIN_H_FRAC * ctx.meta["height"]
    ped_ids: list[set] = [set() for _ in ts]
    ped_long = np.zeros(nb, dtype=bool)
    crossing = np.zeros(nb)
    for sl, (cw_name, poly) in bound:
        for tr in ctx.persons():
            if tr.size < min_h:
                continue
            on = np.array([point_in_poly(poly, tr.cx[i], tr.by[i]) for i in range(len(tr.t))])
            for i in np.flatnonzero(on):
                b = int(tr.t[i] // bin_sec)
                if b < nb:
                    ped_ids[b].add(tr.id)
            # непрерывные проходы одного пешехода ≥ PED_ONE_SEC
            start = None
            for i in range(len(tr.t) + 1):
                cur = on[i] if i < len(tr.t) else False
                if cur and start is None:
                    start = i
                if not cur and start is not None:
                    if tr.t[i - 1] - tr.t[start] >= PED_ONE_SEC:
                        b0, b1 = int(tr.t[start] // bin_sec), int(tr.t[i - 1] // bin_sec)
                        ped_long[b0:min(nb, b1 + 1)] = True
                    start = None
        for tr in ctx.vehicles():
            thr = CROSS_MOVE_REL * tr.size
            for i in range(1, len(tr.t)):
                if tr.speed[i] < thr:
                    continue
                if seg_intersect((tr.cx[i - 1], tr.by[i - 1]), (tr.cx[i], tr.by[i]), sl.p1, sl.p2) \
                        and float(np.dot(np.array([tr.vx[i], tr.vy[i]]), sl.approach)) > 0:
                    b = int(tr.t[i] // bin_sec)
                    if b < nb:
                        crossing[b] += 1
    n_ped = np.array([len(s) for s in ped_ids])
    red = (n_ped >= PED_MANY) | ped_long
    pad = int(round(PED_PAD_SEC / bin_sec))
    red_pad = red.copy()
    for b in np.flatnonzero(red):
        red_pad[max(0, b - pad):b + pad + 1] = True
    win = int(round(GREEN_WINDOW_SEC / bin_sec))
    states: list[str] = []
    for b in range(nb):
        if red_pad[b]:
            states.append("red")
        elif crossing[max(0, b - win):b + 1].sum() >= 1 and n_ped[b] == 0:
            states.append("green")
        else:
            states.append("unknown")
    return SignalTrack(t=ts, state=states, source=["ped"] * nb, origin="ped")


# ---------------------------------------------------------------------------
# Фазы по трафику
# ---------------------------------------------------------------------------
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
        for tr in ctx.vehicles():
            for i in range(len(tr.t)):
                b = int(tr.t[i] // bin_sec)
                if b >= len(ts):
                    continue
                off = stop_line_offset(sl, tr.cx[i], tr.by[i])
                u = along_line(sl, tr.cx[i], tr.by[i])
                if -queue_depth_rel * tr.size < off < 0.3 * tr.size and -0.1 <= u <= 1.1 and tr.speed[i] < stop_rel * tr.size:
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
    return SignalTrack(t=ts, state=states, source=["traffic"] * len(ts), origin="traffic")


# ---------------------------------------------------------------------------
# Слияние
# ---------------------------------------------------------------------------
def phases(track: SignalTrack, grid: np.ndarray | None = None) -> list[tuple[str, float, float]]:
    """Фазы (состояние, начало, конец): одинаковые состояния, разделённые только «неизвестно», сшиваются."""
    if track is None or not len(track.t):
        return []
    ts = track.t if grid is None else grid
    states = track.state if grid is None else [track.at(float(t)) for t in grid]
    out: list[list] = []
    for i, s in enumerate(states):
        if s == "unknown":
            continue
        t = float(ts[i])
        if out and out[-1][0] == s:
            out[-1][2] = t
        else:
            out.append([s, t, t])
    return [(s, a, b) for s, a, b in out]


def lamp_plausible(lamp: SignalTrack | None, ped: SignalTrack | None, grid: np.ndarray) -> tuple[bool, str]:
    """Фазы лампы правдоподобны: есть и красный, и зелёный; внутренние фазы 10–150 с; жёлтый короткий;
    согласие с пешеходными фазами ≥ 80 % там, где известны обе."""
    if lamp is None or not len(lamp.t):
        return False, "no lamp"
    ph = phases(lamp, grid)
    kinds = {s for s, _, _ in ph}
    if "red" not in kinds or "green" not in kinds:
        return False, f"phases {sorted(kinds)}"
    t_end = float(grid[-1])
    for k, (s, a, b) in enumerate(ph):
        length = b - a
        interior = k > 0 and k < len(ph) - 1 and a > FUSE_STEP and b < t_end - FUSE_STEP
        if s in ("red", "green"):
            if length > PHASE_MAX or (interior and length < PHASE_MIN):
                return False, f"{s} phase {a:.0f}-{b:.0f} s"
        elif s == "amber" and length > AMBER_MAX:
            return False, f"amber phase {a:.0f}-{b:.0f} s"
    periodic = _periodic(ph, t_end)
    if ped is not None and len(ped.t):
        both = [(lamp.at(float(t)), ped.at(float(t))) for t in grid]
        both = [(a, b) for a, b in both if a in ("red", "green") and b in ("red", "green")]
        if len(both) * FUSE_STEP >= 5.0:
            agree = float(np.mean([a == b for a, b in both]))
            if periodic:
                return True, f"ok, periodic phases, agreement with pedestrians {agree:.2f}"
            if agree < AGREE_MIN:
                return False, f"agreement with pedestrians {agree:.2f}"
            return True, f"ok, agreement {agree:.2f} over {len(both) * FUSE_STEP:.0f} s"
    return True, "ok, periodic phases" if periodic else "ok, no pedestrian overlap"


def _periodic(ph: list[tuple[str, float, float]], t_end: float) -> bool:
    """Настоящий светофорный цикл: не меньше двух внутренних фаз красного и двух зелёного, длины каждого
    цвета в пределах ±PERIOD_TOL от медианы. Пешеходы такому источнику не указ."""
    inner = [(s, b - a) for k, (s, a, b) in enumerate(ph) if 0 < k < len(ph) - 1 and a > FUSE_STEP and b < t_end - FUSE_STEP]
    for colour in ("red", "green"):
        lens = np.array([d for s, d in inner if s == colour])
        if len(lens) < 2:
            return False
        med = float(np.median(lens))
        if med < PHASE_MIN or np.any(np.abs(lens - med) > PERIOD_TOL * med):
            return False
    return True


def fuse_signals(lamp: SignalTrack | None, ped: SignalTrack | None, traffic: SignalTrack | None,
                 duration: float, step: float = FUSE_STEP) -> SignalTrack:
    """Лампа, если правдоподобна; там, где она молчит (и когда её нет) — пешеходы, потом трафик.
    reliable=True только если в итоге есть хотя бы одна фаза красного и одна зелёного правдоподобной длины."""
    grid = np.arange(0.0, max(duration, step) + step / 2, step)
    ok, why = lamp_plausible(lamp, ped, grid)
    order = ([("lamp", lamp)] if ok else []) + [("ped", ped), ("traffic", traffic)]
    states, sources = [], []
    for t in grid:
        s, src = "unknown", ""
        for name, track in order:
            if track is None or not len(track.t):
                continue
            v = track.at(float(t))
            if v != "unknown":
                s, src = v, name
                break
        states.append(s)
        sources.append(src)
    fused = SignalTrack(t=grid, state=states, source=sources, origin="lamp" if ok else "ped/traffic")
    fused.note = why
    have = {s for s, a, b in fused.runs() if s in ("red", "green") and PHASE_MIN <= b - a <= PHASE_MAX}
    fused.reliable = {"red", "green"} <= have
    return fused

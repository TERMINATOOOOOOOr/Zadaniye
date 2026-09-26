"""Статические посторонние объекты на проезжей части по модели фона (без детектора).

Детектор COCO знает лишь несколько «препятствий» (животные, чемодан, стул, мяч); упавший груз, покрышка,
коробка, кусок бампера для него невидимы. Здесь они ищутся по фону:

- фон — приближение скользящей медианы (шаг один уровень серого за кадр) на уменьшенном кадре
  (длинная сторона work_side); обновляется только там, где кадр совпадает с фоном, поэтому предмет,
  оказавшийся на дороге, не «врастает» в фон за считанные секунды;
- карта `since` хранит момент, с которого пиксель отличается от фона; пятно, которое держится
  ≥ min_sec, компактно, имеет собственные края (которых нет в фоне) и по площади между area_min
  и area_max от кадра — кандидат;
- finish() выбрасывает кандидатов, которые накрыты треками детектора (машина, человек, велосипед:
  это стоящая машина или пешеход, а не предмет), чей центр не на дороге и кто стоит в штатных
  точках остановки (очередь у стоп-линии, если передан StopMap).

Мягкие тени и дрейф освещения отсеиваются тремя фильтрами: фон догоняет плавные изменения быстрее,
чем они превышают порог; резкий общий сдвиг (авто-экспозиция) сбрасывает фон целиком; у кромки тени
нет новых краёв, а её «застывшая» часть — тонкая полоса, не проходящая по компактности.

Стоимость: несколько операций над картинкой 480×270 на кадр — доли миллисекунды, детектор не нужен.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

from .tracking import Track

TRACKED_KINDS = ("vehicle", "person", "rider")


@dataclass
class StaticObject:
    """Неподвижный посторонний объект: интервал времени и бокс в пикселях исходного кадра."""

    t0: float
    t1: float
    x1: float
    y1: float
    x2: float
    y2: float
    contrast: float = 0.0   # средняя разница с фоном, уровни серого
    edge: float = 0.0       # средняя сила новых краёв на границе пятна

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class _Candidate:
    """Пятно, которое наблюдается из кадра в кадр (координаты рабочего разрешения)."""

    x1: float
    y1: float
    x2: float
    y2: float
    t_first: float
    t_last: float
    hits: int = 1
    fill: float = 0.0
    edge: float = 0.0
    contrast: float = 0.0
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    def iou(self, box: tuple[float, float, float, float]) -> float:
        return box_iou((self.x1, self.y1, self.x2, self.y2), box)

    def update(self, box: tuple[float, float, float, float], t: float, t_first: float, fill: float, edge: float, contrast: float) -> None:
        self.hits += 1
        self.t_last = t
        self.t_first = min(self.t_first, t_first)
        self.fill += fill
        self.edge += edge
        self.contrast += contrast
        self.boxes.append(box)
        a = 0.2   # бокс сглаживаем: границы пятна дрожат из-за проезжающих мимо машин
        self.x1 += a * (box[0] - self.x1); self.y1 += a * (box[1] - self.y1)
        self.x2 += a * (box[2] - self.x2); self.y2 += a * (box[3] - self.y2)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class StaticObjectDetector:
    """Копит модель фона по кадрам Part A (push) и отдаёт статические объекты (finish).

    width, height — размер исходного кадра (боксы возвращаются в его пикселях); frame в push() — кадр
    читателя любого размера с тем же соотношением сторон (он уменьшается до work_side по длинной стороне).
    """

    def __init__(self, width: int, height: int, work_side: int = 480, diff_thr: int = 25, min_sec: float = 5.0,
                 area_min: float = 0.0002, area_max: float = 0.03, absorb_sec: float = 180.0, gap_sec: float = 3.0,
                 fill_min: float = 0.35, aspect_max: float = 4.0, edge_min: float = 4.0, contrast_min: float = 30.0,
                 min_hits: int = 3, cover_max: float = 0.2, collect_every: int = 4) -> None:
        self.width, self.height = int(width), int(height)
        self.work_scale = work_side / float(max(self.width, self.height, 1))
        self.ww = max(8, int(round(self.width * self.work_scale)))
        self.wh = max(8, int(round(self.height * self.work_scale)))
        self.diff_thr = int(diff_thr)
        self.min_sec = float(min_sec)
        n_px = self.ww * self.wh
        self.area_min = max(4, int(round(area_min * n_px)))
        self.area_max = int(round(area_max * n_px))
        self.absorb_sec = float(absorb_sec)
        self.gap_sec = float(gap_sec)
        self.fill_min = float(fill_min)
        self.aspect_max = float(aspect_max)
        self.edge_min = float(edge_min)
        self.contrast_min = float(contrast_min)
        self.min_hits = int(min_hits)
        self.cover_max = float(cover_max)
        self.collect_every = max(1, int(collect_every))   # компоненты ищем не на каждом кадре: t0 всё равно берётся из карты since
        self.bg: np.ndarray | None = None          # uint8, рабочий размер
        self.since: np.ndarray | None = None       # float32: с какого момента пиксель отличается от фона
        self.fg_prev: np.ndarray | None = None     # uint8 0/255
        self.active: list[_Candidate] = []
        self.done: list[_Candidate] = []
        self.n_frames = 0
        self.n_resets = 0
        self.t_last = 0.0
        self._kernel = np.ones((3, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ кадры
    def push(self, frame: np.ndarray, t: float) -> None:
        """Один обработанный кадр Part A (BGR, любое разрешение с тем же соотношением сторон)."""
        if frame.shape[0] != self.wh or frame.shape[1] != self.ww:
            frame = cv2.resize(frame, (self.ww, self.wh), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self.n_frames += 1
        self.t_last = float(t)
        if self.bg is None:
            self.bg = g.copy()
            self.since = np.zeros(g.shape, dtype=np.float32)
            self.fg_prev = np.zeros(g.shape, dtype=np.uint8)
            return
        diff = cv2.absdiff(g, self.bg)
        _, fg = cv2.threshold(diff, self.diff_thr, 255, cv2.THRESH_BINARY)   # uint8 0/255
        n_fg = cv2.countNonZero(fg)
        # резкий общий сдвиг (авто-экспозиция, включение освещения): фон заново, иначе полкадра застынет в fg
        if n_fg > 0.5 * fg.size:
            self.bg = g.copy()
            self.since.fill(t)
            self.fg_prev = np.zeros(g.shape, dtype=np.uint8)
            self.active.clear()
            self.n_resets += 1
            return
        started = cv2.bitwise_and(fg, cv2.bitwise_not(self.fg_prev))
        self.since[started.view(bool)] = t
        # приближение скользящей медианы: ±1 уровень за кадр там, где кадр совпадает с фоном
        calm = cv2.bitwise_not(fg)
        cv2.add(self.bg, 1, dst=self.bg, mask=cv2.bitwise_and(cv2.compare(g, self.bg, cv2.CMP_GT), calm))
        cv2.subtract(self.bg, 1, dst=self.bg, mask=cv2.bitwise_and(cv2.compare(g, self.bg, cv2.CMP_LT), calm))
        fg_b = fg.view(bool)
        age = t - self.since
        # что торчит из фона дольше absorb_sec — новое состояние сцены (перестановка, тень здания)
        stale = fg_b & (age > self.absorb_sec)
        if stale.any():
            self.bg[stale] = g[stale]
            fg[stale] = 0
        self.fg_prev = fg
        if self.n_frames % self.collect_every == 0:
            static = (fg_b & (age >= self.min_sec)).view(np.uint8)
            static = cv2.morphologyEx(static, cv2.MORPH_OPEN, self._kernel)   # убираем крапинки и кромки тени
            if cv2.countNonZero(static):
                self._collect(static, g, diff, t)
        if self.active:
            self._expire(t)

    def _collect(self, static: np.ndarray, g: np.ndarray, diff: np.ndarray, t: float) -> None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
        if n <= 1:
            return
        x, y, w, h, area = (stats[1:, k] for k in range(5))
        fill = area / np.maximum(w * h, 1)
        aspect = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)
        ok = (area >= self.area_min) & (area <= self.area_max) & (w >= 3) & (h >= 3)             & (fill >= self.fill_min * 0.7) & (aspect <= self.aspect_max * 1.5)   # заведомо не объект отсеиваем сразу
        seen: list[tuple[tuple[float, float, float, float], float, float, float, float]] = []
        for j in np.flatnonzero(ok):
            i = int(j) + 1
            xi, yi, wi, hi = int(x[j]), int(y[j]), int(w[j]), int(h[j])
            # окно с запасом в 2 px: кольцо границы плюс апертура Собеля
            y0, y1 = max(0, yi - 2), min(self.wh, yi + hi + 2)
            x0, x1 = max(0, xi - 2), min(self.ww, xi + wi + 2)
            m = (labels[y0:y1, x0:x1] == i).view(np.uint8)
            ring = (cv2.dilate(m, self._kernel) - cv2.erode(m, self._kernel)).view(bool)
            if not ring.any():
                continue
            edge = float(self._grad(g[y0:y1, x0:x1])[ring].mean() - self._grad(self.bg[y0:y1, x0:x1])[ring].mean())
            mb = m.view(bool)
            contrast = float(diff[y0:y1, x0:x1][mb].mean())
            t_first = float(self.since[y0:y1, x0:x1][mb].min())
            seen.append(((float(xi), float(yi), float(xi + wi), float(yi + hi)), t_first, float(fill[j]), edge, contrast))
        for box, t_first, fl, edge, contrast in seen:
            best, best_iou = None, 0.0
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            for c in self.active:
                iou = c.iou(box)
                if iou < 0.3 and not (c.x1 <= cx <= c.x2 and c.y1 <= cy <= c.y2):
                    continue
                if best is None or iou > best_iou:
                    best, best_iou = c, iou
            if best is None:
                self.active.append(_Candidate(*box, t_first=t_first, t_last=t, fill=fl, edge=edge, contrast=contrast, boxes=[box]))
            elif best.t_last < t:
                best.update(box, t, t_first, fl, edge, contrast)

    @staticmethod
    def _grad(img: np.ndarray) -> np.ndarray:
        """Сила краёв: (|dx|+|dy|) Собеля / 8 — ступенька контраста D даёт ≈ D/2."""
        dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        return (np.abs(dx) + np.abs(dy)).astype(np.float32) / 8.0

    def _expire(self, t: float) -> None:
        keep = []
        for c in self.active:
            if t - c.t_last > self.gap_sec:
                self.done.append(c)
            else:
                keep.append(c)
        self.active = keep

    # ------------------------------------------------------------------ итог
    def candidates(self) -> list[_Candidate]:
        return self.done + self.active

    def finish(self, tracks: dict[int, Track] | Iterable[Track], on_road: Callable[[float, float], bool],
               stops=None, report: list | None = None, signal=None) -> list[StaticObject]:
        """Статические объекты после всех фильтров. report (если передан) получает (объект, причина|'ok') по каждому кандидату."""
        trs = list(tracks.values()) if isinstance(tracks, dict) else list(tracks)
        trs = [tr for tr in trs if tr.kind in TRACKED_KINDS and len(tr.t) > 0]
        out: list[StaticObject] = []
        s = 1.0 / self.work_scale
        for c in self.candidates():
            n = max(1, c.hits)
            obj = StaticObject(c.t_first, c.t_last, c.x1 * s, c.y1 * s, c.x2 * s, c.y2 * s, contrast=c.contrast / n, edge=c.edge / n)
            reason = self._reject_reason(c, obj, trs, on_road, stops, signal)
            if report is not None:
                report.append((obj, reason))
            if reason == "ok":
                out.append(obj)
        out.sort(key=lambda o: (o.t0, o.x1))
        return out

    def _reject_reason(self, c: _Candidate, obj: StaticObject, tracks: list[Track], on_road, stops, signal=None) -> str:
        n = max(1, c.hits)
        if c.hits < self.min_hits:
            return "hits"
        if obj.t1 - obj.t0 < self.min_sec:
            return "short"
        if c.fill / n < self.fill_min:
            return "fill"
        w, h = max(obj.x2 - obj.x1, 1e-6), max(obj.y2 - obj.y1, 1e-6)
        if max(w, h) / min(w, h) > self.aspect_max:
            return "aspect"
        if obj.edge < self.edge_min:
            return "edge"
        if obj.contrast < self.contrast_min:
            return "contrast"
        if not on_road(obj.cx, obj.cy):
            return "off_road"
        if stops is not None and stops.is_hotspot(obj.cx, obj.cy):
            return "stop_hotspot"
        cover = track_coverage(obj, tracks)
        if cover >= self.cover_max:
            return f"tracked({cover:.2f})"
        return context_reason(obj, tracks, signal)


def track_coverage(obj: StaticObject, tracks: list[Track], step: float = 0.5, iou_thr: float = 0.2,
                   margin: tuple[float, float] = (0.15, 0.25)) -> float:
    """Доля времени жизни объекта, когда его накрывает бокс трека: IoU > iou_thr или центр одного внутри другого.

    Для теста «центр внутри» бокс трека расширяется на margin (доли ширины и высоты): детектор часто режет
    крышу грузовика или автобуса за перекрытием, и кусок крыши оказывается прямо над боксом.
    """
    n_slots = max(1, int(np.ceil((obj.t1 - obj.t0) / step)) + 1)
    covered = np.zeros(n_slots, dtype=bool)
    bx1, by1, bx2, by2 = obj.box
    for tr in tracks:
        if tr.t1 < obj.t0 or tr.t0 > obj.t1:
            continue
        lo = int(np.searchsorted(tr.t, obj.t0, side="left"))
        hi = int(np.searchsorted(tr.t, obj.t1, side="right"))
        if hi <= lo:
            continue
        ob = tr.obs[lo:hi]
        x1 = np.array([o.x1 for o in ob]); y1 = np.array([o.y1 for o in ob])
        x2 = np.array([o.x2 for o in ob]); y2 = np.array([o.y2 for o in ob])
        ix = np.clip(np.minimum(x2, bx2) - np.maximum(x1, bx1), 0, None)
        iy = np.clip(np.minimum(y2, by2) - np.maximum(y1, by1), 0, None)
        inter = ix * iy
        union = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)
        tcx, tcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        mx, my = margin[0] * (x2 - x1), margin[1] * (y2 - y1)
        hit = (iou > iou_thr) | ((x1 - mx <= obj.cx) & (obj.cx <= x2 + mx) & (y1 - my <= obj.cy) & (obj.cy <= y2 + my)) \
            | ((bx1 <= tcx) & (tcx <= bx2) & (by1 <= tcy) & (tcy <= by2))
        if hit.any():
            slots = np.clip(((tr.t[lo:hi][hit] - obj.t0) / step).astype(int), 0, n_slots - 1)
            covered[slots] = True
    return float(covered.mean())


ADJACENT_FRAC = 0.5      # стоящий трек-сосед (в пределах своей высоты от объекта) дольше этой доли жизни объекта — не предмет
ADJACENT_SLOW_REL = 0.3  # «стоящий»: медленнее 30 % высоты бокса в секунду
PASSING_MIN = 2          # столько разных машин должно проехать мимо (в 4 размерах объекта) за время его жизни
PASSING_RADIUS = 4.0
PASSING_MOVE_REL = 0.5


RED_LIFE_FRAC = 0.6      # объект, живущий в основном во время красного, — часть очереди, которую детектор не дотрекал
WARMUP_SEC = 20.0        # фон строится с первого кадра: «призраки» машин из него живут первые секунды и не считаются


def red_fraction(obj: StaticObject, signal) -> float:
    """Доля жизни объекта, когда светофор (надёжный) показывал красный."""
    if signal is None or not getattr(signal, "reliable", False) or not len(signal.t):
        return 0.0
    ts = np.arange(obj.t0, obj.t1, 0.5)
    if len(ts) == 0:
        return 0.0
    return float(np.mean([signal.at(float(t)) == "red" for t in ts]))


def context_reason(obj: StaticObject, tracks: list[Track], signal=None) -> str:
    """Контекст вокруг предмета: тень стоящего человека и часть машины в очереди отличают от предмета соседи.
    'adjacent' — рядом почти всё время стоит трек (человек, машина); 'no_traffic' — мимо никто не проезжает
    (очередь на красный, а не предмет на живой дороге); иначе 'ok'."""
    life = max(obj.t1 - obj.t0, 1e-6)
    size = max(obj.x2 - obj.x1, obj.y2 - obj.y1, 1.0)
    if obj.t0 < WARMUP_SEC:
        return "warmup"
    if red_fraction(obj, signal) >= RED_LIFE_FRAC:
        return "queue_red"
    passing: set[int] = set()
    for tr in tracks:
        if tr.t1 < obj.t0 or tr.t0 > obj.t1:
            continue
        lo = int(np.searchsorted(tr.t, obj.t0, side="left"))
        hi = int(np.searchsorted(tr.t, obj.t1, side="right"))
        if hi <= lo:
            continue
        cx, by, h, sp = tr.cx[lo:hi], tr.by[lo:hi], tr.h[lo:hi], tr.speed[lo:hi]
        dist = np.hypot(cx - obj.cx, by - obj.cy)
        near_slow = (dist < np.maximum(h, size)) & (sp < ADJACENT_SLOW_REL * np.maximum(h, 1.0))
        if near_slow.any():
            covered = float(np.clip(tr.t[lo:hi][near_slow].max() - tr.t[lo:hi][near_slow].min(), 0, life)) + 0.5
            if covered / life >= ADJACENT_FRAC:
                return "adjacent"
        if tr.kind == "vehicle" and np.any((dist < PASSING_RADIUS * size) & (sp > PASSING_MOVE_REL * np.maximum(h, 1.0))):
            passing.add(tr.id)
    if len(passing) < PASSING_MIN:
        return "no_traffic"
    return "ok"


def refilter(objects: list[StaticObject], tracks, signal=None) -> list[StaticObject]:
    """Повторный контекстный фильтр для объектов, загруженных из кэша (кэш хранит результат finish())."""
    trs = list(tracks.values()) if isinstance(tracks, dict) else list(tracks)
    trs = [tr for tr in trs if tr.kind in TRACKED_KINDS and len(tr.t) > 0]
    return [o for o in objects if context_reason(o, trs, signal) == "ok"]


# ---------------------------------------------------------------------- кэш
def save_static_objects(path: str, objects: list[StaticObject]) -> None:
    """Кэш рядом с наблюдениями детектора: статические объекты считаются в том же проходе по кадрам."""
    data = np.array([[o.t0, o.t1, o.x1, o.y1, o.x2, o.y2, o.contrast, o.edge] for o in objects], dtype=float).reshape(-1, 8)
    np.savez_compressed(path, data=data)


def load_static_objects(path: str) -> list[StaticObject]:
    z = np.load(path, allow_pickle=False)
    return [StaticObject(*(float(v) for v in row)) for row in z["data"]]

"""Part B: причинная оценка риска аварии по кадрам. Ничего не знает о будущем и не открывает файл.

Сигналы: время до столкновения (TTC) по парам треков на встречном/пересекающемся курсе с поправкой
«успевает ли затормозить», резкое торможение, пешеход на проезжей части (если сцена размечена).
Точки штатных остановок (очереди у светофора) накапливаются онлайн по уже увиденным кадрам и
снижают риск «догнал очередь». Все парные вычисления векторизованы: в плотной сцене сотни объектов.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import Settings, seed_everything
from .scene import load_scene
from .tracking import Detector, kind_of
from .video import resize_frame, scale_for

TTC_T0 = 2.0            # риск растёт, когда TTC < 2 с
HIST_SEC = 1.5          # история трека для оценки скорости
DECAY = 0.8             # затухание оценки между обновлениями без сигнала
W_TTC, W_DEC, W_PED = 0.9, 0.35, 0.15
DEC_REL = 2.5           # торможение сильнее 2.5 высот бокса/с² = резкое
BRAKE_OK = 0.7          # если фактическое замедление ≥ 70 % требуемого — подъезд контролируемый
STOP_REL = 0.15
STOP_MARK_SEC = 1.5     # стоял ≥ 1.5 с → отмечаем ячейку как точку остановки
MAX_PAIR_REL = 8.0      # пары дальше 8 высот друг от друга не рассматриваем
MIN_H_FRAC = 0.06       # объекты ниже 6 % высоты кадра слишком далеко: скорость по ним шумная, зазоры в перспективе нулевые
LEADER_SLOW_REL = 0.5   # догон считаем опасным, только если лидер почти стоит...
CROSS_ANGLE_DEG = 30.0  # ...или курсы пересекаются под заметным углом (едущие друг за другом в потоке — норма)
FOLLOWER_MIN_REL = 1.5  # догоняющий должен ехать быстро (в высотах бокса/с), иначе это ползущая очередь
CLOSING_MIN_REL = 1.2

_DET: Detector | None = None
_DET_KEY: tuple = ()


def _detector(st: Settings) -> Detector:
    global _DET, _DET_KEY
    key = (st.model_b, st.device, st.imgsz_b, st.conf, st.iou, st.tracker)
    if _DET is None or _DET_KEY != key:
        _DET = Detector(st.weights(st.model_b), st.device, st.imgsz_b, st.conf, st.iou, st.tracker)
        _DET_KEY = key
    return _DET


class OnlineRisk:
    def __init__(self, st: Settings | None = None) -> None:
        self.st = st or Settings()
        self.det: Detector | None = None
        self.scene = load_scene(self.st.scene_path)
        self.hist: dict[int, deque] = {}
        self.score = 0.0
        self.i = 0
        self.scale = 1.0
        self.cell = 48
        self.min_h = 20.0
        self.stop_cells: dict[tuple[int, int], set[int]] = {}
        self.stopped_since: dict[int, float] = {}
        self.last = {"ttc": 0.0, "dec": 0.0, "ped": 0.0, "n": 0}   # компоненты последней оценки (диагностика)
        self.prev_ttc = 0.0

    def reset(self, meta: dict) -> None:
        seed_everything()
        self.det = _detector(self.st)
        self.det.reset()
        self.hist = {}
        self.score = 0.0
        self.i = 0
        w, h = max(1, int(meta.get("width", 1920) or 0)), max(1, int(meta.get("height", 1080) or 0))
        self.scale = scale_for(w, h, self.st.imgsz_b)
        self.cell = max(16, int(round(max(w, h) / 40)))
        self.min_h = MIN_H_FRAC * h
        self.stop_cells = {}
        self.stopped_since = {}
        self.prev_ttc = 0.0
        if self.scene is not None and tuple(self.scene.frame_size) != (w, h):
            self.scene = None   # сцена другой камеры — не используем

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.i += 1
        if (self.i - 1) % self.st.stride_b != 0:
            return self.score
        small = resize_frame(frame, self.scale)
        obs = self.det.track(small, self.i - 1, t_sec, self.scale)
        seen = set()
        for tid, o in obs:
            k = kind_of(o.cls)
            if k not in ("vehicle", "person", "rider"):
                continue
            h = self.hist.setdefault(tid, deque())
            h.append((t_sec, (o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2, o.x2 - o.x1, o.y2 - o.y1, k))
            while h and t_sec - h[0][0] > HIST_SEC:
                h.popleft()
            seen.add(tid)
        for tid in [k for k in self.hist if k not in seen]:
            h = self.hist[tid]
            if not h or t_sec - h[-1][0] > HIST_SEC:
                del self.hist[tid]
                self.stopped_since.pop(tid, None)
        S = self._states()
        self._update_stop_cells(S, t_sec)
        self._raw_risk(S)
        # TTC-сигнал должен держаться два обновления подряд: одиночный выброс скорости — не тревога
        ttc_eff = min(self.last["ttc"], self.prev_ttc)
        self.prev_ttc = self.last["ttc"]
        raw = min(1.0, W_TTC * ttc_eff + W_DEC * self.last["dec"] + W_PED * self.last["ped"]) ** 2   # квадрат: порог 0.5 только при сильном сигнале
        self.score = float(np.clip(max(raw, self.score * DECAY), 0.0, 1.0))
        return self.score

    # ---- состояния объектов: массивы, по одному элементу на трек ----
    def _states(self) -> dict:
        ids, X, Y, VX, VY, HH, WW, ACC, KIND = [], [], [], [], [], [], [], [], []
        for tid, h in self.hist.items():
            if len(h) < 3:
                continue
            arr = np.array([r[:5] for r in h], dtype=float)
            ts, xs, ys = arr[:, 0], arr[:, 1], arr[:, 2]
            span = ts[-1] - ts[0]
            if span < 0.2:
                continue
            dt = ts - ts.mean()
            den = float(np.dot(dt, dt)) or 1e-9
            vx = float(np.dot(dt, xs - xs.mean()) / den)
            vy = float(np.dot(dt, ys - ys.mean()) / den)
            speeds = np.hypot(np.gradient(xs, ts), np.gradient(ys, ts))
            ids.append(tid); X.append(xs[-1]); Y.append(ys[-1]); VX.append(vx); VY.append(vy)
            WW.append(float(arr[:, 3].mean())); HH.append(float(arr[:, 4].mean()))
            ACC.append(float((speeds[-1] - speeds[0]) / span)); KIND.append(h[-1][5])
        f = lambda a: np.asarray(a, dtype=float)
        return {"ids": ids, "x": f(X), "y": f(Y), "vx": f(VX), "vy": f(VY), "h": f(HH), "w": f(WW), "acc": f(ACC),
                "speed": np.hypot(f(VX), f(VY)), "kind": np.asarray(KIND, dtype=object)}

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int(y // self.cell), int(x // self.cell)

    def _update_stop_cells(self, S: dict, t: float) -> None:
        for i, tid in enumerate(S["ids"]):
            if S["kind"][i] != "vehicle":
                continue
            if S["speed"][i] < STOP_REL * S["h"][i]:
                since = self.stopped_since.setdefault(tid, t)
                if t - since >= STOP_MARK_SEC:
                    self.stop_cells.setdefault(self._cell_of(S["x"][i], S["y"][i] + S["h"][i] / 2), set()).add(tid)
            else:
                self.stopped_since.pop(tid, None)

    def _in_stop_cell(self, x: float, y: float) -> bool:
        r, c = self._cell_of(x, y)
        n = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = max(n, len(self.stop_cells.get((r + dr, c + dc), ())))
        return n >= 2

    def _raw_risk(self, S: dict) -> float:
        n = len(S["ids"])
        if n == 0:
            self.last = {"ttc": 0.0, "dec": 0.0, "ped": 0.0, "n": 0}
            return 0.0
        veh = S["kind"] == "vehicle"
        x, y, vx, vy, h, w, acc, speed = (S[k] for k in ("x", "y", "vx", "vy", "h", "w", "acc", "speed"))
        big = h >= self.min_h
        # резкое торможение
        f_dec = 0.0
        m = veh & big & (acc < -DEC_REL * h) & (speed > 0.5 * h)
        if m.any():
            f_dec = float(np.clip((-acc[m] / (2 * DEC_REL * h[m])).max(), 0.0, 1.0))
        # пешеход на проезжей части вне перехода (флаг на объект: нужен и для пар машина–пешеход)
        f_ped = 0.0
        ped_exposed = np.zeros(n, dtype=bool)
        if self.scene is not None and self.scene.road is not None:
            for i in np.flatnonzero(S["kind"] == "person"):
                by = y[i] + h[i] / 2
                if self.scene.in_road(x[i], by) and self.scene.crosswalk_at(x[i], by) is None:
                    ped_exposed[i] = True
                    f_ped = 1.0
        f_ttc = 0.0
        if n >= 2:
            rx = x[None, :] - x[:, None]; ry = y[None, :] - y[:, None]
            dist = np.hypot(rx, ry)
            np.fill_diagonal(dist, np.inf)
            size = (h[None, :] + h[:, None]) / 2
            pair_ok = np.triu((veh[:, None] | veh[None, :]) & (big[:, None] & big[None, :]) & (dist < MAX_PAIR_REL * size), 1)
            if pair_ok.any():
                rvx = vx[None, :] - vx[:, None]; rvy = vy[None, :] - vy[:, None]
                rel = np.hypot(rvx, rvy) + 1e-6
                dist_safe = np.where(np.isfinite(dist), dist, 1e9)
                closing = -(rx * rvx + ry * rvy) / np.maximum(dist_safe, 1e-6)
                miss = np.abs(rx * rvy - ry * rvx) / rel
                cosang = closing / rel
                gap = np.maximum(dist_safe - size, 0.0)
                ok = pair_ok & (closing >= CLOSING_MIN_REL * size) & (miss <= 0.45 * (w[None, :] + w[:, None])) & (cosang >= 0.9)
                if ok.any():
                    ttc = gap / np.maximum(closing, 1e-6)
                    f = np.clip((TTC_T0 - ttc) / TTC_T0, 0.0, 1.0) * np.minimum(1.0, closing / (2.0 * size))
                    # кто догоняет: у кого скорость направлена на партнёра
                    proj_a = vx[:, None] * rx + vy[:, None] * ry
                    proj_b = -(vx[None, :] * rx + vy[None, :] * ry)
                    a_is_follower = proj_a > proj_b
                    acc_f = np.where(a_is_follower, acc[:, None], acc[None, :])
                    a_req = closing ** 2 / (2 * np.maximum(gap, 0.3 * size))
                    controlled = np.maximum(0.0, -acc_f) >= BRAKE_OK * a_req
                    f = np.where(controlled, f * 0.2, f)
                    # догоняющий должен ехать быстро: ползущая очередь с нулевым зазором в перспективе — не риск
                    speed_f = np.where(a_is_follower, speed[:, None], speed[None, :])
                    h_f = np.where(a_is_follower, h[:, None], h[None, :])
                    f = np.where(speed_f >= FOLLOWER_MIN_REL * h_f, f, f * 0.15)
                    # догон едущего в ту же сторону лидера — обычное движение в потоке (зазор в перспективе нулевой)
                    speed_l = np.where(a_is_follower, speed[None, :], speed[:, None])
                    h_l = np.where(a_is_follower, h[None, :], h[:, None])
                    cos_v = (vx[:, None] * vx[None, :] + vy[:, None] * vy[None, :]) / np.maximum(speed[:, None] * speed[None, :], 1e-6)
                    crossing_course = cos_v < np.cos(np.radians(CROSS_ANGLE_DEG))
                    f = np.where((speed_l < LEADER_SLOW_REL * h_l) | crossing_course, f, f * 0.1)
                    # встречные машины на соседних проезжих частях в перспективе выглядят «лоб в лоб» — не курс на столкновение
                    f = np.where(cos_v < -0.7, 0.0, f)
                    # пешеход в паре: только если он на проезжей части вне перехода (у зебры машины проезжают рядом штатно)
                    is_ped = S["kind"] == "person"
                    ped_pair = is_ped[:, None] | is_ped[None, :]
                    ped_ok = ped_exposed[:, None] | ped_exposed[None, :]
                    ped_factor = 0.5 if self.scene is None else 0.1
                    f = np.where(ped_pair & ~ped_ok, f * ped_factor, f)
                    # лидер стоит в точке штатных остановок — вероятно очередь
                    idx = np.arange(n)
                    leader_idx = np.where(a_is_follower, idx[None, :], idx[:, None])
                    stopped_obj = speed < 0.3 * h
                    in_cell = np.array([bool(stopped_obj[i]) and self._in_stop_cell(x[i], y[i] + h[i] / 2) for i in range(n)])
                    f = np.where(in_cell[leader_idx], f * 0.4, f)
                    # лидер стоит в очереди (рядом другие стоящие машины) — «догнал очередь», а не авария
                    neigh = ((dist < 3.5 * size) & stopped_obj[None, :] & veh[None, :]).sum(axis=1)
                    in_queue = stopped_obj & (neigh >= 1)
                    f = np.where(in_queue[leader_idx], f * 0.2, f)
                    f_ttc = float(np.max(np.where(ok, f, 0.0)))
        self.last = {"ttc": f_ttc, "dec": f_dec, "ped": f_ped, "n": n}
        return float(min(1.0, W_TTC * f_ttc + W_DEC * f_dec + W_PED * f_ped))

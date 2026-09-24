"""Part B: причинная оценка риска аварии по кадрам. Ничего не знает о будущем и не открывает файл.

Сигналы: время до столкновения (TTC) по парам треков на встречном/пересекающемся курсе с поправкой
«успевает ли затормозить», резкое торможение, пешеход на проезжей части (если сцена размечена).
Точки штатных остановок (очереди у светофора) накапливаются онлайн по уже увиденным кадрам и
снижают риск «догнал очередь».
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
W_TTC, W_DEC, W_PED = 0.9, 0.35, 0.3
DEC_REL = 2.5           # торможение сильнее 2.5 высот бокса/с² = резкое
BRAKE_OK = 0.7          # если фактическое замедление ≥ 70 % требуемого — подъезд контролируемый
STOP_REL = 0.15
STOP_MARK_SEC = 1.5     # стоял ≥ 1.5 с → отмечаем ячейку как точку остановки

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
        self.stop_cells: dict[tuple[int, int], set[int]] = {}
        self.stopped_since: dict[int, float] = {}

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
        self.stop_cells = {}
        self.stopped_since = {}
        if self.scene is not None:
            self.scene = self.scene.scale_to(w, h)

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
        states = self._states()
        self._update_stop_cells(states, t_sec)
        raw = self._raw_risk(states) ** 2   # квадрат: порог тревоги 0.5 достигается только при сильном сигнале
        self.score = float(np.clip(max(raw, self.score * DECAY), 0.0, 1.0))
        return self.score

    # ---- признаки ----
    def _states(self) -> list[dict]:
        out = []
        for tid, h in self.hist.items():
            if len(h) < 3:
                continue
            ts = np.array([r[0] for r in h]); xs = np.array([r[1] for r in h]); ys = np.array([r[2] for r in h])
            if ts[-1] - ts[0] < 0.2:
                continue
            vx = np.polyfit(ts, xs, 1)[0]; vy = np.polyfit(ts, ys, 1)[0]
            speeds = np.hypot(np.gradient(xs, ts), np.gradient(ys, ts))
            acc = (speeds[-1] - speeds[0]) / max(ts[-1] - ts[0], 1e-3)
            hh = float(np.mean([r[4] for r in h])); ww = float(np.mean([r[3] for r in h]))
            out.append({"id": tid, "x": xs[-1], "y": ys[-1], "vx": vx, "vy": vy, "h": hh, "w": ww, "speed": float(np.hypot(vx, vy)),
                        "acc": float(acc), "kind": h[-1][5], "by": ys[-1] + hh / 2})
        return out

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int(y // self.cell), int(x // self.cell)

    def _update_stop_cells(self, states: list[dict], t: float) -> None:
        for s in states:
            if s["kind"] != "vehicle":
                continue
            if s["speed"] < STOP_REL * s["h"]:
                since = self.stopped_since.setdefault(s["id"], t)
                if t - since >= STOP_MARK_SEC:
                    self.stop_cells.setdefault(self._cell_of(s["x"], s["by"]), set()).add(s["id"])
            else:
                self.stopped_since.pop(s["id"], None)

    def _in_stop_cell(self, x: float, y: float) -> bool:
        r, c = self._cell_of(x, y)
        n = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = max(n, len(self.stop_cells.get((r + dr, c + dc), ())))
        return n >= 2

    def _raw_risk(self, S: list[dict]) -> float:
        f_ttc, f_dec, f_ped = 0.0, 0.0, 0.0
        for a in range(len(S)):
            sa = S[a]
            if sa["kind"] == "vehicle" and sa["acc"] < -DEC_REL * sa["h"] and sa["speed"] > 0.5 * sa["h"]:
                f_dec = max(f_dec, min(1.0, -sa["acc"] / (2 * DEC_REL * sa["h"])))
            if sa["kind"] == "person" and self.scene is not None and self.scene.road is not None:
                if self.scene.in_road(sa["x"], sa["by"]) and self.scene.crosswalk_at(sa["x"], sa["by"]) is None:
                    f_ped = 1.0
            for b in range(a + 1, len(S)):
                sb = S[b]
                if sa["kind"] != "vehicle" and sb["kind"] != "vehicle":
                    continue
                rx, ry = sb["x"] - sa["x"], sb["y"] - sa["y"]
                dist = float(np.hypot(rx, ry))
                if dist < 1e-6:
                    continue
                vx, vy = sb["vx"] - sa["vx"], sb["vy"] - sa["vy"]
                closing = -(rx * vx + ry * vy) / dist
                size = (sa["h"] + sb["h"]) / 2
                if closing < 0.8 * size:
                    continue
                rel_speed = float(np.hypot(vx, vy))
                miss = abs(rx * vy - ry * vx) / max(rel_speed, 1e-6)   # промах при прямолинейном движении
                if miss > 0.45 * (sa["w"] + sb["w"]):                   # разъедутся по соседним полосам
                    continue
                # курс должен вести на партнёра, а не мимо: угол между относительной скоростью и линией центров
                if (-(rx * vx + ry * vy)) / (dist * max(rel_speed, 1e-6)) < 0.9:
                    continue
                gap = max(dist - 0.5 * (sa["h"] + sb["h"]), 0.0)
                ttc = gap / closing
                f = float(np.clip((TTC_T0 - ttc) / TTC_T0, 0.0, 1.0)) * float(min(1.0, closing / (2.0 * size)))
                # кто догоняет: у него скорость направлена к партнёру
                follower, leader = (sa, sb) if (sa["vx"] * rx + sa["vy"] * ry) > (sb["vx"] * -rx + sb["vy"] * -ry) else (sb, sa)
                a_req = closing ** 2 / (2 * max(gap, 0.3 * size))
                a_act = max(0.0, -follower["acc"])
                if a_act >= BRAKE_OK * a_req:
                    f *= 0.2
                if leader["speed"] < 0.3 * leader["h"] and self._in_stop_cell(leader["x"], leader["by"]):
                    f *= 0.4
                f_ttc = max(f_ttc, f)
        return float(min(1.0, W_TTC * f_ttc + W_DEC * f_dec + W_PED * f_ped))

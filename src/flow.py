"""Поле направлений и маска дороги, построенные по трекам самого видео.

Нужны, когда сцена не размечена (нет configs/scene.json): направление движения в ячейке —
средний единичный вектор скоростей машин, прошедших через неё; согласованность — длина этого
среднего вектора (1 = все ехали в одну сторону, 0 = разнобой, например перекрёсток).
"""
from __future__ import annotations

import numpy as np

from .tracking import Track


class FlowField:
    def __init__(self, width: int, height: int, cell: int | None = None) -> None:
        self.width, self.height = width, height
        self.cell = cell or max(16, int(round(max(width, height) / 60)))
        self.nx = width // self.cell + 1
        self.ny = height // self.cell + 1
        self.sum_vx = np.zeros((self.ny, self.nx))
        self.sum_vy = np.zeros((self.ny, self.nx))
        self.count = np.zeros((self.ny, self.nx), dtype=int)
        self.presence = np.zeros((self.ny, self.nx), dtype=int)   # где вообще бывали машины (стоящие тоже)
        self.road_mask = np.zeros((self.ny, self.nx), dtype=bool)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(np.clip(y // self.cell, 0, self.ny - 1)), int(np.clip(x // self.cell, 0, self.nx - 1))

    def build(self, tracks: dict[int, Track], min_rel_speed: float = 0.5, min_presence: int = 3) -> "FlowField":
        for tr in tracks.values():
            if tr.kind != "vehicle":
                continue
            thr = min_rel_speed * tr.size
            for i in range(len(tr.t)):
                r, c = self._cell(tr.cx[i], tr.by[i])
                self.presence[r, c] += 1
                if tr.speed[i] > thr:
                    ux, uy = tr.vx[i] / tr.speed[i], tr.vy[i] / tr.speed[i]
                    self.sum_vx[r, c] += ux
                    self.sum_vy[r, c] += uy
                    self.count[r, c] += 1
        # маска дороги: ячейки, где машины бывали не реже min_presence раз, с расширением на одну ячейку
        m = self.count >= min_presence   # только движущиеся: обочина с припаркованными не дорога
        pad = np.pad(m, 1)
        dil = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dil |= pad[1 + dy:1 + dy + self.ny, 1 + dx:1 + dx + self.nx]
        self.road_mask = dil
        return self

    def direction(self, x: float, y: float) -> tuple[np.ndarray | None, float, int]:
        """(единичный вектор доминирующего направления | None, согласованность 0..1, число наблюдений)."""
        r, c = self._cell(x, y)
        n = int(self.count[r, c])
        if n == 0:
            return None, 0.0, 0
        mx, my = self.sum_vx[r, c] / n, self.sum_vy[r, c] / n
        cons = float(np.hypot(mx, my))
        if cons < 1e-6:
            return None, 0.0, n
        return np.array([mx / cons, my / cons]), cons, n

    def in_road(self, x: float, y: float) -> bool:
        r, c = self._cell(x, y)
        return bool(self.road_mask[r, c])

    def to_arrays(self) -> dict:
        """Для EDA: средние направления, согласованность, плотность."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = np.where(self.count > 0, self.sum_vx / np.maximum(self.count, 1), 0.0)
            my = np.where(self.count > 0, self.sum_vy / np.maximum(self.count, 1), 0.0)
        return {"cell": self.cell, "mean_vx": mx, "mean_vy": my, "consensus": np.hypot(mx, my),
                "count": self.count, "presence": self.presence, "road_mask": self.road_mask}


class StopMap:
    """Карта «точек остановки»: ячейки, где регулярно останавливаются разные машины (стоп-линии, очереди).

    Остановка в такой ячейке — штатная (светофор), а не событие; контакт с машиной, стоящей в такой ячейке,
    почти всегда «догнал очередь», а не авария.
    """

    def __init__(self, width: int, height: int, cell: int | None = None) -> None:
        self.cell = cell or max(16, int(round(max(width, height) / 40)))
        self.nx = width // self.cell + 1
        self.ny = height // self.cell + 1
        self.vehicles = np.zeros((self.ny, self.nx), dtype=int)   # сколько разных машин здесь стояло

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(np.clip(y // self.cell, 0, self.ny - 1)), int(np.clip(x // self.cell, 0, self.nx - 1))

    def build(self, tracks: dict[int, Track], stop_rel_speed: float = 0.15, min_stop_sec: float = 2.0) -> "StopMap":
        for tr in tracks.values():
            if tr.kind != "vehicle" or len(tr.t) < 3:
                continue
            thr = stop_rel_speed * tr.size
            cells: set[tuple[int, int]] = set()
            start = None
            for i in range(len(tr.t)):
                if tr.speed[i] < thr:
                    if start is None:
                        start = i
                else:
                    if start is not None and tr.t[i - 1] - tr.t[start] >= min_stop_sec:
                        cells.add(self._cell(tr.cx[(start + i - 1) // 2], tr.by[(start + i - 1) // 2]))
                    start = None
            if start is not None and tr.t[-1] - tr.t[start] >= min_stop_sec:
                cells.add(self._cell(tr.cx[(start + len(tr.t) - 1) // 2], tr.by[(start + len(tr.t) - 1) // 2]))
            for r, c in cells:
                self.vehicles[r, c] += 1
        return self

    def is_hotspot(self, x: float, y: float, min_vehicles: int = 3) -> bool:
        r, c = self._cell(x, y)
        r0, r1 = max(0, r - 1), min(self.ny, r + 2)
        c0, c1 = max(0, c - 1), min(self.nx, c + 2)
        return int(self.vehicles[r0:r1, c0:c1].max()) >= min_vehicles

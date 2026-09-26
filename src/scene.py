"""Геометрия сцены: дорога, полосы с направлениями, переходы, стоп-линии, сплошные, светофор.

Схема configs/scene.json (координаты в пикселях исходного кадра, y вниз):
{
  "frame_size": [w, h], "px_per_meter": null,
  "road": [[x,y],...], "intersection": [[x,y],...] | null,
  "lanes": [{"name","polygon","direction":[dx,dy],"allowed":["straight","left","right","uturn"]}],
  "crosswalks": [{"name","polygon"}],
  "stop_lines": [{"name","p1","p2","approach":[dx,dy],"lanes":[...]}],
  "solid_lines": [{"name","points":[[x,y],...]}],
  "signal": {"visible": bool, "roi": [x,y,w,h] | null}
}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def _poly(pts) -> np.ndarray:
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else np.array([0.0, 0.0])


def point_in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    if poly is None or len(poly) < 3:
        return False
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def side(px: float, py: float, a, b) -> float:
    """Знак векторного произведения: по какую сторону от прямой a→b лежит точка."""
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


def seg_intersect(p1, p2, q1, q2) -> bool:
    """Пересекаются ли отрезки p1p2 и q1q2."""
    d1 = side(q1[0], q1[1], p1, p2)
    d2 = side(q2[0], q2[1], p1, p2)
    d3 = side(p1[0], p1[1], q1, q2)
    d4 = side(p2[0], p2[1], q1, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def angle_deg(v1, v2) -> float:
    """Угол между векторами в градусах (0..180)."""
    a, b = _unit(v1), _unit(v2)
    if not a.any() or not b.any():
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


@dataclass
class Lane:
    name: str
    polygon: np.ndarray
    direction: np.ndarray
    allowed: set = field(default_factory=lambda: {"straight", "left", "right"})


@dataclass
class StopLine:
    name: str
    p1: np.ndarray
    p2: np.ndarray
    approach: np.ndarray
    lanes: list[str]


@dataclass
class Scene:
    frame_size: tuple[int, int]
    px_per_meter: float | None
    road: np.ndarray | None
    intersection: np.ndarray | None
    lanes: list[Lane]
    crosswalks: list[tuple[str, np.ndarray]]
    stop_lines: list[StopLine]
    solid_lines: list[tuple[str, np.ndarray]]
    islands: list[np.ndarray]
    signal_visible: bool
    signal_roi: tuple[int, int, int, int] | None
    notes: str = ""

    # --- запросы ---
    def in_road(self, x: float, y: float) -> bool:
        if self.road is None or not point_in_poly(self.road, x, y):
            return False
        return not any(point_in_poly(p, x, y) for p in self.islands)

    def in_intersection(self, x: float, y: float) -> bool:
        return point_in_poly(self.intersection, x, y) if self.intersection is not None else False

    def lane_at(self, x: float, y: float) -> Lane | None:
        for ln in self.lanes:
            if point_in_poly(ln.polygon, x, y):
                return ln
        return None

    def crosswalk_at(self, x: float, y: float) -> str | None:
        for name, poly in self.crosswalks:
            if point_in_poly(poly, x, y):
                return name
        return None

    def has_lanes(self) -> bool:
        return len(self.lanes) > 0

    def has_crosswalks(self) -> bool:
        return len(self.crosswalks) > 0

    def has_stop_lines(self) -> bool:
        return len(self.stop_lines) > 0

    def scale_to(self, width: int, height: int) -> "Scene":
        """Сцена размечена на кадре frame_size; если видео другого размера — масштабируем координаты."""
        sx = width / float(self.frame_size[0]); sy = height / float(self.frame_size[1])
        if abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6:
            return self
        m = np.array([sx, sy], dtype=np.float32)
        sc = lambda p: (None if p is None else (_poly(p) * m).astype(np.float32))
        return Scene(
            frame_size=(width, height), px_per_meter=(self.px_per_meter * (sx + sy) / 2 if self.px_per_meter else None),
            road=sc(self.road), intersection=sc(self.intersection),
            lanes=[Lane(l.name, sc(l.polygon), l.direction, l.allowed) for l in self.lanes],
            crosswalks=[(n, sc(p)) for n, p in self.crosswalks],
            stop_lines=[StopLine(s.name, s.p1 * m, s.p2 * m, s.approach, s.lanes) for s in self.stop_lines],
            solid_lines=[(n, sc(p)) for n, p in self.solid_lines],
            islands=[sc(p) for p in self.islands],
            signal_visible=self.signal_visible,
            signal_roi=(tuple(int(v * f) for v, f in zip(self.signal_roi, (sx, sy, sx, sy))) if self.signal_roi else None),
            notes=self.notes,
        )

    def transform(self, M, frame_size: tuple[int, int] | None = None) -> "Scene":
        """Применить подобие M (2×3: [[a, -b, tx], [b, a, ty]]) ко всей геометрии; вернуть новую сцену.
        Полигоны, полилинии и концы стоп-линий переносятся точками; векторы направлений только поворачиваются;
        ROI светофора: центр переносится, размер умножается на масштаб. frame_size — размер кадра видео."""
        M = np.asarray(M, dtype=np.float64).reshape(2, 3)
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        rot = M[:, :2] / scale if scale > 1e-9 else np.eye(2)
        pts = lambda p: (None if p is None else _transform_points(M, p))
        vec = lambda v: _unit(rot @ np.asarray(v, dtype=float))
        roi = None
        if self.signal_roi:
            x, y, w, h = self.signal_roi
            c = _transform_points(M, [[x + w / 2.0, y + h / 2.0]])[0]
            nw, nh = w * scale, h * scale
            roi = (int(round(c[0] - nw / 2.0)), int(round(c[1] - nh / 2.0)), max(1, int(round(nw))), max(1, int(round(nh))))
        return Scene(
            frame_size=tuple(frame_size) if frame_size else self.frame_size,
            px_per_meter=(self.px_per_meter * scale if self.px_per_meter else None),
            road=pts(self.road), intersection=pts(self.intersection),
            lanes=[Lane(l.name, pts(l.polygon), vec(l.direction), set(l.allowed)) for l in self.lanes],
            crosswalks=[(n, pts(p)) for n, p in self.crosswalks],
            stop_lines=[StopLine(s.name, _transform_points(M, [s.p1])[0], _transform_points(M, [s.p2])[0], vec(s.approach), list(s.lanes))
                        for s in self.stop_lines],
            solid_lines=[(n, pts(p)) for n, p in self.solid_lines],
            islands=[pts(p) for p in self.islands],
            signal_visible=self.signal_visible,
            signal_roi=roi,
            notes=self.notes,
        )


def _transform_points(M: np.ndarray, pts) -> np.ndarray:
    """Точки (N, 2) через аффинную матрицу 2×3 → float32 (N, 2)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return (p @ M[:, :2].T + M[:, 2]).astype(np.float32)


def load_scene(path: str | Path) -> Scene | None:
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    sig = d.get("signal") or {}
    return Scene(
        frame_size=tuple(d.get("frame_size") or (1920, 1080)),
        px_per_meter=d.get("px_per_meter"),
        road=_poly(d["road"]) if d.get("road") else None,
        intersection=_poly(d["intersection"]) if d.get("intersection") else None,
        lanes=[Lane(l["name"], _poly(l["polygon"]), _unit(l.get("direction", [0, 0])), set(l.get("allowed") or ["straight", "left", "right"]))
               for l in d.get("lanes", [])],
        crosswalks=[(c.get("name", f"cw{i}"), _poly(c["polygon"])) for i, c in enumerate(d.get("crosswalks", []))],
        stop_lines=[StopLine(s.get("name", f"sl{i}"), np.asarray(s["p1"], dtype=np.float32), np.asarray(s["p2"], dtype=np.float32),
                             _unit(s.get("approach", [0, 0])), list(s.get("lanes", []))) for i, s in enumerate(d.get("stop_lines", []))],
        solid_lines=[(s.get("name", f"s{i}"), _poly(s["points"])) for i, s in enumerate(d.get("solid_lines", []))],
        islands=[_poly(p["polygon"] if isinstance(p, dict) else p) for p in d.get("islands", [])],
        signal_visible=bool(sig.get("visible", False)),
        signal_roi=tuple(sig["roi"]) if sig.get("roi") else None,
        notes=d.get("notes", ""),
    )

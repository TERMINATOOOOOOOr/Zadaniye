"""Контекст одного видео, который получают все правила: треки, сцена, поле направлений, состояние светофора."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flow import FlowField, StopMap
from .scene import Scene
from .tracking import Track


@dataclass
class SignalTrack:
    """Состояние светофора по времени: 'red' | 'green' | 'amber' | 'unknown'.

    source — откуда взято состояние в каждой точке ('lamp' | 'ped' | 'traffic' | ''), origin — какой источник
    выиграл при слиянии, reliable — есть хотя бы одна фаза красного и одна зелёного правдоподобной длины
    (без этого правила red_light/stop_line молчат), note — почему принято такое решение (для отчёта)."""

    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    origin: str = ""
    reliable: bool = False
    note: str = ""

    def _index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def at(self, t: float) -> str:
        if len(self.t) == 0:
            return "unknown"
        return self.state[self._index(t)]

    def source_at(self, t: float) -> str:
        if len(self.t) == 0 or not self.source:
            return ""
        return self.source[self._index(t)]

    def runs(self) -> list[tuple[str, float, float]]:
        """Непрерывные участки одного состояния (включая 'unknown'): (состояние, начало, конец)."""
        out: list[list] = []
        for i, s in enumerate(self.state):
            t = float(self.t[i])
            if out and out[-1][0] == s:
                out[-1][2] = t
            else:
                out.append([s, t, t])
        return [(s, a, b) for s, a, b in out]

    def intervals(self, which: str) -> list[tuple[float, float]]:
        out, start = [], None
        for i in range(len(self.t)):
            on = self.state[i] == which
            if on and start is None:
                start = float(self.t[i])
            if not on and start is not None:
                out.append((start, float(self.t[i])))
                start = None
        if start is not None:
            out.append((start, float(self.t[-1])))
        return out


@dataclass
class Context:
    tracks: dict[int, Track]
    meta: dict
    scene: Scene | None
    flow: FlowField
    signal: SignalTrack | None = None       # слитое состояние (см. signal.fuse_signals), им пользуются правила
    stops: StopMap | None = None
    signal_roi: SignalTrack | None = None   # состояние по цвету лампы (для EDA и отчёта)
    signal_ped: SignalTrack | None = None   # состояние по пешеходным фазам (для отчёта)
    static_objects: list = field(default_factory=list)   # статические объекты на дороге (src/static_objects.py)

    @property
    def duration(self) -> float:
        return float(self.meta.get("duration", 0.0))

    def vehicles(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.kind == "vehicle"]

    def persons(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.kind == "person"]

    def on_road(self, x: float, y: float) -> bool:
        if self.scene is not None and self.scene.road is not None:
            return self.scene.in_road(x, y)
        return self.flow.in_road(x, y)

    def ref_direction(self, x: float, y: float) -> tuple[np.ndarray | None, float]:
        """Эталонное направление движения в точке: из полос сцены, иначе из поля направлений."""
        if self.scene is not None and self.scene.has_lanes():
            ln = self.scene.lane_at(x, y)
            if ln is not None and ln.direction.any():
                return ln.direction, 1.0
            # вне зон (полосы бывают размечены не везде) — поле направлений самого видео, как без сцены
        d, cons, n = self.flow.direction(x, y)
        if d is None or n < 5:
            return None, 0.0
        return d, cons

    def is_stop_hotspot(self, x: float, y: float) -> bool:
        return self.stops.is_hotspot(x, y) if self.stops is not None else False

    def stopped_neighbours(self, x: float, y: float, t: float, radius: float, exclude=(), rel_speed: float = 0.3) -> int:
        """Сколько других машин стоит в момент t в радиусе radius от точки (очередь). exclude — id, которые не считаем."""
        n = 0
        ex = {exclude} if isinstance(exclude, int) else set(exclude)
        for tr in self.tracks.values():
            if tr.kind != "vehicle" or tr.id in ex or not (tr.t0 <= t <= tr.t1):
                continue
            i = tr.index_at(t)
            if tr.speed[i] < rel_speed * tr.size and np.hypot(tr.cx[i] - x, tr.by[i] - y) < radius:
                n += 1
        return n

    def in_queue(self, x: float, y: float, t: float, size: float, exclude=()) -> bool:
        return self.is_stop_hotspot(x, y) or self.stopped_neighbours(x, y, t, 3.5 * size, exclude) >= 1

    def signal_state(self, t: float) -> str:
        return self.signal.at(t) if self.signal is not None else "unknown"

    def signal_reliable(self) -> bool:
        return self.signal is not None and bool(self.signal.reliable)

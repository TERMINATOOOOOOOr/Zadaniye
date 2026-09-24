"""Детектор YOLO11 + ByteTrack (ultralytics) и представление треков с гладкими скоростями."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import COCO_NAMES, DETECT_IDS, VEHICLE_NAMES, PERSON_NAMES, RIDER_NAMES, OBSTACLE_NAMES, resolve_device


@dataclass
class Obs:
    frame: int
    t: float
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    """Обёртка над ultralytics: один вызов = один кадр, состояние трекера живёт между кадрами одного видео."""

    def __init__(self, weights: str, device: str = "auto", imgsz: int = 960, conf: float = 0.25, iou: float = 0.5,
                 tracker: str = "bytetrack.yaml", classes: list[int] | None = None) -> None:
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.classes = classes or DETECT_IDS
        self.quantize = 16 if self.device.startswith("cuda") else None   # fp16 на GPU: в 1.5–2 раза быстрее, детерминизм не страдает

    def reset(self) -> None:
        """Сброс состояния трекера перед новым видео: id начинаются заново, прогон детерминирован."""
        pred = getattr(self.model, "predictor", None)
        trackers = getattr(pred, "trackers", None) if pred is not None else None
        if trackers:
            for tr in trackers:
                tr.reset()

    def track(self, frame: np.ndarray, frame_idx: int, t: float, scale: float) -> list[tuple[int, Obs]]:
        """Возвращает [(track_id, Obs)], координаты в пикселях исходного кадра."""
        res = self.model.track(frame, persist=True, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=self.classes,
                               tracker=self.tracker, device=self.device, quantize=self.quantize, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy() / scale
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        out = []
        for i in range(len(ids)):
            name = COCO_NAMES.get(int(cls[i]))
            if name is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            out.append((int(ids[i]), Obs(frame_idx, t, name, float(conf[i]), x1, y1, x2, y2)))
        return out


def kind_of(cls: str) -> str:
    if cls in VEHICLE_NAMES:
        return "vehicle"
    if cls in PERSON_NAMES:
        return "person"
    if cls in RIDER_NAMES:
        return "rider"
    if cls in OBSTACLE_NAMES:
        return "obstacle"
    return "other"


@dataclass
class Track:
    """Один объект во времени. Массивы заполняются в finalize()."""

    id: int
    obs: list[Obs] = field(default_factory=list)
    cls: str = ""
    kind: str = ""
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    cx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    by: np.ndarray = field(default_factory=lambda: np.zeros(0))   # нижняя граница бокса (точка касания дороги)
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(0))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def finalize(self, smooth_sec: float = 0.4) -> None:
        self.obs.sort(key=lambda o: o.frame)
        votes: dict[str, float] = {}
        for o in self.obs:
            votes[o.cls] = votes.get(o.cls, 0.0) + o.conf
        self.cls = max(votes, key=votes.get)
        self.kind = kind_of(self.cls)
        self.t = np.array([o.t for o in self.obs])
        self.frames = np.array([o.frame for o in self.obs], dtype=int)
        x1 = np.array([o.x1 for o in self.obs]); y1 = np.array([o.y1 for o in self.obs])
        x2 = np.array([o.x2 for o in self.obs]); y2 = np.array([o.y2 for o in self.obs])
        self.conf = np.array([o.conf for o in self.obs])
        self.w = x2 - x1
        self.h = y2 - y1
        k = self._window(smooth_sec)
        self.cx = _smooth((x1 + x2) / 2.0, k)
        self.cy = _smooth((y1 + y2) / 2.0, k)
        self.by = _smooth(y2, k)
        n = len(self.t)
        if n >= 2:
            dt = np.gradient(self.t)
            dt[dt <= 1e-6] = 1e-6
            self.vx = np.gradient(self.cx) / dt
            self.vy = np.gradient(self.cy) / dt
            self.speed = np.hypot(self.vx, self.vy)
            self.accel = np.gradient(_smooth(self.speed, k)) / dt
        else:
            self.vx = np.zeros(n); self.vy = np.zeros(n); self.speed = np.zeros(n); self.accel = np.zeros(n)

    def _window(self, smooth_sec: float) -> int:
        if len(self.t) < 3:
            return 1
        fps_eff = (len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-6)
        return max(1, int(round(smooth_sec * fps_eff)) | 1)

    @property
    def size(self) -> float:
        """Характерный размер объекта в пикселях (медиана высоты бокса)."""
        return float(np.median(self.h)) if len(self.h) else 1.0

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])

    def heading(self) -> np.ndarray:
        return np.arctan2(self.vy, self.vx)

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def box_at(self, i: int) -> tuple[float, float, float, float]:
        o = self.obs[i]
        return o.x1, o.y1, o.x2, o.y2


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.astype(float)
    pad = k // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def build_tracks(observations: list[tuple[int, Obs]], min_len: int = 3) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    for tid, o in observations:
        tracks.setdefault(tid, Track(tid)).obs.append(o)
    out = {}
    for tid, tr in tracks.items():
        if len(tr.obs) < min_len:
            continue
        tr.finalize()
        out[tid] = tr
    return out


def save_observations(path: str, observations: list[tuple[int, Obs]]) -> None:
    """Кэш наблюдений в npz: позволяет крутить правила без повторного детектирования."""
    if not observations:
        np.savez_compressed(path, data=np.zeros((0, 8)), cls=np.array([], dtype="U16"))
        return
    data = np.array([[tid, o.frame, o.t, o.conf, o.x1, o.y1, o.x2, o.y2] for tid, o in observations], dtype=float)
    cls = np.array([o.cls for _, o in observations], dtype="U16")
    np.savez_compressed(path, data=data, cls=cls)


def load_observations(path: str) -> list[tuple[int, Obs]]:
    z = np.load(path, allow_pickle=False)
    data, cls = z["data"], z["cls"]
    return [(int(r[0]), Obs(int(r[1]), float(r[2]), str(c), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7])))
            for r, c in zip(data, cls)]

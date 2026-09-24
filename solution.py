"""solution.py — интерфейс для харнесса организаторов.

Part A: detect_events(video_path) -> [[start_sec, end_sec, label], ...]
Part B: RiskEstimator().reset(meta); .step(frame, t_sec) -> float

Вся логика лежит в src/: детектор YOLO11 + ByteTrack → треки → правила на геометрии сцены → сегменты;
риск — TTC по трекам, причинно, на каждом k-м кадре.

Оба детектора загружаются и прогреваются при импорте модуля: харнесс запускает таймер бюджета
на каждое видео уже после импорта, а холодный старт CUDA и весов занимает несколько секунд.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")          # никаких сетевых проверок ultralytics
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CLASSES as _CLASSES, Settings  # noqa: E402
from src.pipeline import detect_events as _detect_events, get_detector  # noqa: E402
from src import risk as _risk  # noqa: E402

CLASSES: list[str] = list(_CLASSES)

RISK_HORIZON_SEC = 5.0


def _warm_up() -> None:
    """Загрузка весов, инициализация CUDA и первый прогон на пустом кадре — вне бюджета времени."""
    st = Settings()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    for det in (get_detector(st), _risk._detector(st)):
        try:
            det.track(dummy, 0, 0.0, 1.0)
        except Exception:
            pass
        det.reset()


_warm_up()


def detect_events(video_path: str) -> list[list]:
    """Part A — события одного видео."""
    return _detect_events(video_path, Settings())


class RiskEstimator:
    """Part B — причинная оценка риска; видит только кадры, которые ей передали."""

    def __init__(self) -> None:
        self._impl = _risk.OnlineRisk(Settings())

    def reset(self, meta: dict) -> None:
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._impl.step(frame, t_sec)

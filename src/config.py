"""Константы и настройки пайплайна. Всё, что можно переопределить переменными окружения, читается здесь."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
CONFIG_DIR = ROOT / "configs"

# Официальные классы событий (порядок как в задании).
CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

# COCO-классы детектора, которые нам нужны.
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light",
              15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 28: "suitcase", 56: "chair", 32: "sports ball"}
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
PERSON_NAMES = {"person"}
RIDER_NAMES = {"bicycle"}
OBSTACLE_NAMES = {"cat", "dog", "horse", "sheep", "cow", "suitcase", "chair", "sports ball"}
DETECT_IDS = sorted(COCO_NAMES)

SEED = 0


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v else default


class Settings:
    """Параметры прогона. Значения по умолчанию рассчитаны на T4 и бюджет 3× длительности видео."""

    def __init__(self) -> None:
        # Part A
        self.model_a = os.getenv("WIUT_MODEL_A", "yolo11s.pt")
        self.stride_a = _env_int("WIUT_STRIDE_A", 3)          # каждый 3-й кадр (8,3 к/с при 25 fps)
        self.imgsz_a = _env_int("WIUT_IMGSZ_A", 960)          # длинная сторона кадра для детектора
        self.conf = _env_float("WIUT_CONF", 0.15)     # ниже обычного: ByteTrack сам делит детекции на уверенные и слабые
        self.iou = _env_float("WIUT_IOU", 0.5)
        # Part B
        self.model_b = os.getenv("WIUT_MODEL_B", "yolo11n.pt")
        self.stride_b = _env_int("WIUT_STRIDE_B", 4)
        self.imgsz_b = _env_int("WIUT_IMGSZ_B", 640)
        # Общее
        self.device = os.getenv("WIUT_DEVICE", "auto")
        self.scene_path = os.getenv("WIUT_SCENE", str(CONFIG_DIR / "scene.json"))
        self.cache_dir = os.getenv("WIUT_CACHE", "")       # если задано — кэш наблюдений (ускоряет подбор порогов)
        self.tracker = os.getenv("WIUT_TRACKER", "bytetrack.yaml")

    def weights(self, name: str) -> str:
        p = WEIGHTS_DIR / name
        return str(p if p.exists() else name)


def resolve_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

"""Чтение видео: метаданные и последовательный обход кадров с шагом и ресайзом."""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "n_frames": n, "width": w, "height": h, "duration": n / float(fps) if fps else 0.0}


def scale_for(width: int, height: int, max_side: int | None) -> float:
    if not max_side:
        return 1.0
    return min(1.0, max_side / float(max(width, height, 1)))


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    # INTER_LINEAR: в 3–4 раза быстрее INTER_AREA на 4K, для детектора разницы нет
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)


def iter_frames(path: str, stride: int = 1, max_side: int | None = None) -> Iterator[tuple[int, float, np.ndarray, float]]:
    """Даёт (индекс кадра, время в секундах, кадр после ресайза, масштаб). Пропущенные кадры только grab()."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = scale_for(w, h, max_side)
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, resize_frame(frame, scale), scale
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()

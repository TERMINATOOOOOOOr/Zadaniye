"""Чтение видео: метаданные и последовательный обход кадров с шагом и ресайзом.

Два способа читать кадры:
- `iter_frames_cv2` — OpenCV: декодирует всё в полном размере, пропущенные кадры только grab().
- `iter_frames_ffmpeg` — внешний ffmpeg (бинарник из пакета imageio-ffmpeg или из PATH): прореживание
  и масштабирование внутри ffmpeg, при наличии NVDEC — аппаратное декодирование. На 4K это в разы быстрее.
`iter_frames` выбирает ffmpeg, если он доступен, и откатывается на OpenCV.

Необязательный `crop=(x, y, w, h)` в пикселях исходного кадра: вместе с уменьшенным кадром итератор отдаёт
вырез полного разрешения (пятый элемент кортежа). В ffmpeg это один граф фильтров и одно декодирование:
кадр масштабируется, снизу к нему подклеивается вырез, в трубу идёт один поток rawvideo.
"""
from __future__ import annotations

import os
import shutil
import subprocess
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


Crop = tuple[int, int, int, int]


def normalize_crop(crop: Crop | None, width: int, height: int) -> Crop | None:
    """Вырез (x, y, w, h) в границах кадра и с чётными координатами/размерами: так его одинаково режут
    ffmpeg (подвыборка цветности 4:2:0) и OpenCV. None — выреза нет."""
    if crop is None:
        return None
    x, y, w, h = (int(v) for v in crop)
    x = max(0, min(width - 2, x // 2 * 2))
    y = max(0, min(height - 2, y // 2 * 2))
    w = max(2, min(width - x, (w + 1) // 2 * 2))
    h = max(2, min(height - y, (h + 1) // 2 * 2))
    return x, y, w, h


def iter_frames_cv2(path: str, stride: int = 1, max_side: int | None = None, crop: Crop | None = None) -> Iterator[tuple]:
    """Даёт (индекс кадра, время в секундах, кадр после ресайза, масштаб[, вырез]). Пропущенные кадры только grab()."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = scale_for(w, h, max_side)
    crop = normalize_crop(crop, w, h)
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                if crop is None:
                    yield idx, idx / fps, resize_frame(frame, scale), scale
                else:
                    cx, cy, cw, ch = crop
                    piece = frame[cy:cy + ch, cx:cx + cw].copy()   # вырез из полного кадра до ресайза
                    yield idx, idx / fps, resize_frame(frame, scale), scale, piece
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
_FFMPEG: str | None = None
_HWACCEL_OK: bool | None = None


def ffmpeg_exe() -> str | None:
    """Бинарник ffmpeg: сначала из пакета imageio-ffmpeg (ставится pip-ом), потом из PATH."""
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG or None
    exe = None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    _FFMPEG = exe or ""
    return exe


def hwaccel_available(exe: str, path: str) -> bool:
    """Проверяем один раз на процесс: умеет ли этот ffmpeg декодировать файл через CUDA/NVDEC."""
    global _HWACCEL_OK
    if _HWACCEL_OK is not None:
        return _HWACCEL_OK
    if os.getenv("WIUT_NO_HWACCEL"):
        _HWACCEL_OK = False
        return False
    try:
        r = subprocess.run([exe, "-loglevel", "error", "-nostdin", "-hwaccel", "cuda", "-i", path, "-frames:v", "3", "-f", "null", "-"],
                           capture_output=True, timeout=60)
        _HWACCEL_OK = r.returncode == 0
    except Exception:
        _HWACCEL_OK = False
    return _HWACCEL_OK


def ffmpeg_filter(out_fps: float, ow: int, oh: int, crop: Crop | None) -> str:
    """Граф фильтров: прореживание и масштаб; с вырезом — split, вырез из полного кадра, подклейка снизу."""
    base = f"fps={out_fps:.6f}"
    if crop is None:
        return f"{base},scale={ow}:{oh}"
    cx, cy, cw, ch = crop
    pw = max(ow, cw)
    return (f"{base},split[a][b];[a]scale={ow}:{oh},pad={pw}:{oh + ch}:0:0:black[s];"
            f"[b]crop={cw}:{ch}:{cx}:{cy}[c];[s][c]overlay=0:{oh}")


def iter_frames_ffmpeg(path: str, stride: int = 1, max_side: int | None = None, meta: dict | None = None,
                       crop: Crop | None = None) -> Iterator[tuple]:
    """Кадры через ffmpeg: fps=src_fps/stride (каждый stride-й кадр), масштаб до max_side, BGR24 в трубу.
    С crop поток имеет высоту oh+ch: сверху кадр, снизу вырез полного разрешения (один декод на проход)."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not available")
    meta = meta or video_meta(path)
    fps, w, h = meta["fps"], meta["width"], meta["height"]
    scale = scale_for(w, h, max_side)
    ow, oh = (w, h) if scale >= 0.999 else (int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2)
    crop = normalize_crop(crop, w, h)
    out_fps = fps / stride
    args = [exe, "-loglevel", "error", "-nostdin"]
    if hwaccel_available(exe, path):
        args += ["-hwaccel", "cuda"]
    args += ["-threads", "0", "-i", path, "-vf", ffmpeg_filter(out_fps, ow, oh, crop), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    if crop is None:
        pw, ph, ch = ow, oh, 0
    else:
        ch = crop[3]
        pw, ph = max(ow, crop[2]), oh + ch
    frame_bytes = pw * ph * 3
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    k = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(ph, pw, 3)
            if crop is None:
                yield int(round(k * stride)), k / out_fps, raw, ow / float(w)
            else:
                yield int(round(k * stride)), k / out_fps, raw[:oh, :ow], ow / float(w), raw[oh:, :crop[2]]
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def iter_frames(path: str, stride: int = 1, max_side: int | None = None, crop: Crop | None = None) -> Iterator[tuple]:
    """ffmpeg, если есть (быстрее на 4K и умеет NVDEC), иначе OpenCV. Кортеж из 4 элементов, с crop — из 5."""
    if not os.getenv("WIUT_FORCE_CV2") and ffmpeg_exe():
        yield from iter_frames_ffmpeg(path, stride, max_side, crop=crop)
        return
    yield from iter_frames_cv2(path, stride, max_side, crop)

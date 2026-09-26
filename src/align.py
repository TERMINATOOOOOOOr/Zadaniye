"""Привязка размеченной сцены к конкретному видео: та ли это камера и на сколько сдвинут кадр.

Сцена (configs/scene.json) нарисована на кадре C3896. Другой клип той же камеры может быть сдвинут на десятки
пикселей (камеру трогали), а клип чужой камеры не должен получать эту сцену вообще. Здесь это решается по
изображению, без детектора:

1. Из видео берём не больше `N_FRAMES` кадров (ffmpeg с `-ss`: ближайший ключевой кадр, длинная сторона
   `WORK_SIDE` px), медиана по времени убирает машины и людей → статичный фон. Кадры читаются параллельными
   процессами: сэмплы 4K 4:2:2 10 бит NVDEC не берёт, а один программный декод кадра стоит ~0,5–1 с.
2. На фоне считаем ключевые точки (SIFT, при его отсутствии ORB) и сопоставляем с эталоном из
   configs/scene_ref.npz (тот же фон для C3896, построен tools/make_scene_ref.py).
3. По совпадениям с тестом отношений оцениваем подобие (масштаб, поворот, сдвиг) через RANSAC.
   Принимаем, только если инлаеров ≥ `MIN_INLIERS`, масштаб в `SCALE_RANGE` и поворот ≤ `MAX_ANGLE_DEG`;
   иначе камера другая — сцена выключается.
4. Если даны треки, статичные треки класса «traffic light» сверяются с эталонными положениями трёх
   светофоров: невязки до и после выравнивания попадают в отчёт (info), решение по ним не принимается.

Матрица преобразования действует в координатах видео: сцена сначала масштабируется к размеру кадра
(Scene.scale_to), затем к ней применяется подобие (Scene.transform). Для той же камеры и того же кадрирования
получается почти единичное преобразование.
"""
from __future__ import annotations

import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG_DIR
from .scene import Scene
from .video import ffmpeg_exe, video_meta

REF_PATH = CONFIG_DIR / "scene_ref.npz"
WORK_SIDE = 960            # длинная сторона рабочего кадра для фона и ключевых точек
N_FRAMES = 9               # сколько кадров берём для медианного фона
N_FEATURES = 3000          # верхний предел ключевых точек на кадр
RATIO = 0.75               # тест отношений Лоу
RANSAC_THR_WORK = 3.0      # порог RANSAC в пикселях рабочего кадра (пересчитывается в пиксели видео)
MIN_INLIERS = 40
SCALE_RANGE = (0.8, 1.25)
MAX_ANGLE_DEG = 5.0
SEEK_WORKERS = N_FRAMES    # все кадры разом: по процессу на кадр, ~230 МБ и 2 потока декодера на процесс
SEEK_THREADS = 2           # потоков декодера на процесс: больше не ускоряет при 9 параллельных процессах
SEEK_TIMEOUT = 60.0

# Эталонные центры трёх светофоров в 4K-координатах C3896 (медианы треков детектора).
REF_LIGHT_ANCHORS = np.array([[2310.0, 774.0], [1473.0, 589.0], [548.0, 998.0]], dtype=np.float32)


@dataclass
class SceneRef:
    """Эталон: ключевые точки фона в координатах полного кадра, дескрипторы, сам фон и якоря светофоров."""

    frame_size: tuple[int, int]
    keypoints: np.ndarray        # (N, 2) float32, координаты полного кадра
    descriptors: np.ndarray      # (N, 128) float32 для SIFT или (N, 32) uint8 для ORB
    background: np.ndarray       # серый фон, длинная сторона WORK_SIDE
    anchors: np.ndarray          # (3, 2) float32, координаты полного кадра
    method: str                  # 'sift' | 'orb'


# ---------------------------------------------------------------------------
# Кадры и фон
# ---------------------------------------------------------------------------
def work_size(width: int, height: int, side: int = WORK_SIDE) -> tuple[int, int]:
    """Размер рабочего кадра: длинная сторона `side`, обе стороны чётные (как в video.iter_frames_ffmpeg)."""
    s = min(1.0, side / float(max(width, height, 1)))
    if s >= 0.999:
        return width, height
    return int(round(width * s)) // 2 * 2, int(round(height * s)) // 2 * 2


def sample_times(duration: float, n: int = N_FRAMES) -> list[float]:
    """Моменты для выборки кадров: равномерно, без самого начала и конца клипа."""
    n = max(1, int(n))
    d = max(float(duration), 0.0)
    return [d * (k + 0.5) / n for k in range(n)]


def read_gray_at(video_path: str, t: float, out_w: int, out_h: int, exe: str | None = None) -> np.ndarray | None:
    """Один серый кадр около момента t через ffmpeg -ss: ключевой кадр, ближайший к t снизу (декодируется только он,
    без докрутки до точного времени — для фона это неважно, а на 4K экономит до целого GOP). Детерминировано."""
    exe = exe or ffmpeg_exe()
    if exe is None:
        return None
    args = [exe, "-loglevel", "error", "-nostdin", "-an", "-dn", "-threads", str(SEEK_THREADS),
            "-noaccurate_seek", "-skip_frame", "nokey", "-ss", f"{max(0.0, t):.3f}", "-i", video_path,
            "-frames:v", "1", "-vf", f"scale={out_w}:{out_h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        r = subprocess.run(args, capture_output=True, timeout=SEEK_TIMEOUT)
    except Exception:
        return None
    if len(r.stdout) != out_w * out_h:
        return None
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(out_h, out_w).copy()


def _read_gray_cv2(video_path: str, times: list[float], out_w: int, out_h: int) -> list[np.ndarray]:
    """Запасной путь без ffmpeg: OpenCV с позиционированием по времени."""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    if not cap.isOpened():
        return frames
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(g, (out_w, out_h), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    return frames


def sample_frames(video_path: str, meta: dict | None = None, n: int = N_FRAMES, side: int = WORK_SIDE,
                  workers: int = SEEK_WORKERS) -> tuple[list[np.ndarray], tuple[int, int]]:
    """До n серых кадров рабочего размера, равномерно по клипу. Возвращает (кадры, (out_w, out_h)).
    ffmpeg-процессы запускаются параллельно: они не держат GIL, а декодирование 4K-кадра дорогое."""
    meta = meta or video_meta(video_path)
    out_w, out_h = work_size(meta["width"], meta["height"], side)
    times = sample_times(meta.get("duration", 0.0), n)
    exe = ffmpeg_exe()
    if exe is None:
        return _read_gray_cv2(video_path, times, out_w, out_h), (out_w, out_h)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(times)))) as ex:
        frames = list(ex.map(lambda t: read_gray_at(video_path, t, out_w, out_h, exe), times))
    return [f for f in frames if f is not None], (out_w, out_h)


def median_background(frames: list[np.ndarray]) -> np.ndarray | None:
    """Медиана по времени: движущиеся объекты исчезают, остаётся дорога и застройка."""
    if not frames:
        return None
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ключевые точки
# ---------------------------------------------------------------------------
def make_detector(prefer: str = "sift", n_features: int = N_FEATURES):
    """SIFT из opencv-python-headless; если сборка без него — ORB. Возвращает (детектор, имя)."""
    if prefer == "sift" and hasattr(cv2, "SIFT_create"):
        try:
            return cv2.SIFT_create(nfeatures=n_features), "sift"
        except cv2.error:
            pass
    return cv2.ORB_create(nfeatures=n_features), "orb"


def detect_features(gray: np.ndarray, method: str = "sift", n_features: int = N_FEATURES
                    ) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Ключевые точки (N, 2) в координатах изображения и дескрипторы. Порядок точек фиксирован (сортировка),
    чтобы RANSAC на тех же данных давал тот же ответ."""
    det, name = make_detector(method, n_features)
    kps, desc = det.detectAndCompute(gray, None)
    if not kps or desc is None:
        return np.zeros((0, 2), dtype=np.float32), None, name
    xy = np.array([k.pt for k in kps], dtype=np.float32)
    resp = np.array([k.response for k in kps], dtype=np.float32)
    order = np.lexsort((xy[:, 1], xy[:, 0], -resp))
    return xy[order], desc[order], name


def _matcher_norm(method: str) -> int:
    return cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2


def match_descriptors(desc_ref: np.ndarray, desc_img: np.ndarray, method: str, ratio: float = RATIO) -> list[tuple[int, int]]:
    """Пары (индекс в эталоне, индекс в кадре), прошедшие тест отношений."""
    if desc_ref is None or desc_img is None or len(desc_ref) < 2 or len(desc_img) < 2:
        return []
    bf = cv2.BFMatcher(_matcher_norm(method), crossCheck=False)
    pairs = []
    for mm in bf.knnMatch(desc_ref, desc_img, k=2):
        if len(mm) == 2 and mm[0].distance < ratio * mm[1].distance:
            pairs.append((mm[0].queryIdx, mm[0].trainIdx))
    return pairs


def decompose(M: np.ndarray) -> dict:
    """Подобие [[a, -b, tx], [b, a, ty]] → масштаб, угол (градусы, по часовой стрелке при y вниз), сдвиг."""
    a, b = float(M[0, 0]), float(M[1, 0])
    return {"scale": float(math.hypot(a, b)), "angle_deg": float(math.degrees(math.atan2(b, a))),
            "tx": float(M[0, 2]), "ty": float(M[1, 2])}


def estimate_similarity(src: np.ndarray, dst: np.ndarray, thr: float) -> tuple[np.ndarray | None, int]:
    """Подобие src → dst через RANSAC. Возвращает (матрица 2×3 | None, число инлаеров)."""
    if len(src) < 3:
        return None, 0
    cv2.setRNGSeed(0)
    M, mask = cv2.estimateAffinePartial2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=cv2.RANSAC,
                                          ransacReprojThreshold=float(thr), maxIters=5000, confidence=0.999,
                                          refineIters=10)
    if M is None or mask is None:
        return None, 0
    return M.astype(np.float64), int(mask.sum())


def accept(params: dict, n_inliers: int) -> str:
    """'ok' либо причина отказа."""
    if n_inliers < MIN_INLIERS:
        return "few_inliers"
    if not (SCALE_RANGE[0] <= params["scale"] <= SCALE_RANGE[1]):
        return "bad_scale"
    if abs(params["angle_deg"]) > MAX_ANGLE_DEG:
        return "bad_angle"
    return "ok"


# ---------------------------------------------------------------------------
# Эталон
# ---------------------------------------------------------------------------
def save_ref(path: str | Path, ref: SceneRef) -> None:
    desc = ref.descriptors
    if ref.method == "sift":
        desc = np.clip(np.rint(desc), 0, 255).astype(np.uint8)   # значения SIFT в OpenCV целые 0..255: без потерь
    np.savez_compressed(str(path), frame_size=np.array(ref.frame_size, dtype=np.int32),
                        keypoints=ref.keypoints.astype(np.float32), descriptors=desc,
                        background=ref.background.astype(np.uint8), anchors=ref.anchors.astype(np.float32),
                        method=np.array(ref.method))


def load_ref(path: str | Path = REF_PATH) -> SceneRef | None:
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(str(p), allow_pickle=False)
    method = str(z["method"])
    desc = z["descriptors"]
    if method == "sift":
        desc = desc.astype(np.float32)
    return SceneRef(frame_size=tuple(int(v) for v in z["frame_size"]), keypoints=z["keypoints"].astype(np.float32),
                    descriptors=desc, background=z["background"], anchors=z["anchors"].astype(np.float32), method=method)


def build_ref(video_path: str, n: int = N_FRAMES, side: int = WORK_SIDE, anchors: np.ndarray = REF_LIGHT_ANCHORS,
              method: str = "sift") -> tuple[SceneRef, dict]:
    """Эталон из видео: медианный фон, ключевые точки в координатах полного кадра."""
    meta = video_meta(video_path)
    frames, (ow, oh) = sample_frames(video_path, meta, n, side)
    bg = median_background(frames)
    if bg is None:
        raise RuntimeError(f"no frames read from {video_path}")
    xy, desc, name = detect_features(bg, method)
    full = xy * np.array([meta["width"] / float(ow), meta["height"] / float(oh)], dtype=np.float32)
    ref = SceneRef(frame_size=(meta["width"], meta["height"]), keypoints=full, descriptors=desc, background=bg,
                   anchors=np.asarray(anchors, dtype=np.float32), method=name)
    return ref, {"frames": len(frames), "work_size": [ow, oh], "n_keypoints": int(len(xy)), "method": name}


# ---------------------------------------------------------------------------
# Оценка преобразования
# ---------------------------------------------------------------------------
def estimate_transform(ref: SceneRef, gray: np.ndarray, video_size: tuple[int, int]) -> tuple[np.ndarray | None, dict]:
    """Подобие «эталон, отмасштабированный к размеру видео» → «координаты видео» по серому рабочему кадру
    `gray` (уменьшенная копия кадра видео размером video_size). Возвращает (2×3 | None, info)."""
    vw, vh = int(video_size[0]), int(video_size[1])
    gh, gw = gray.shape[:2]
    t0 = time.perf_counter()
    xy, desc, name = detect_features(gray, ref.method)
    t_feat = time.perf_counter() - t0
    info: dict = {"method": name, "n_ref_kp": int(len(ref.keypoints)), "n_kp": int(len(xy)), "n_matches": 0,
                  "n_inliers": 0, "transform": None, "matrix": None, "reason": "few_matches",
                  "time_sec": {"features": round(t_feat, 3), "match": 0.0}}
    if name != ref.method:
        info["reason"] = "method_mismatch"
        return None, info
    t0 = time.perf_counter()
    pairs = match_descriptors(ref.descriptors, desc, name)
    info["n_matches"] = len(pairs)
    if len(pairs) < 3:
        info["time_sec"]["match"] = round(time.perf_counter() - t0, 3)
        return None, info
    # эталон → размер видео; кадр → полный размер видео
    pre = np.array([vw / float(ref.frame_size[0]), vh / float(ref.frame_size[1])], dtype=np.float32)
    up = np.array([vw / float(gw), vh / float(gh)], dtype=np.float32)
    src = ref.keypoints[[i for i, _ in pairs]] * pre
    dst = xy[[j for _, j in pairs]] * up
    thr = RANSAC_THR_WORK * float(max(up))
    M, n_in = estimate_similarity(src, dst, thr)
    info["time_sec"]["match"] = round(time.perf_counter() - t0, 3)
    info["n_inliers"] = n_in
    if M is None:
        info["reason"] = "no_model"
        return None, info
    params = decompose(M)
    info["transform"] = {k: round(v, 4) for k, v in params.items()}
    info["matrix"] = [[round(float(v), 6) for v in row] for row in M]
    info["reason"] = accept(params, n_in)
    return (M if info["reason"] == "ok" else None), info


# ---------------------------------------------------------------------------
# Сверка по светофорам
# ---------------------------------------------------------------------------
def static_lights(tracks, min_obs: int = 30, min_dur: float = 10.0, max_std_rel: float = 0.01) -> list[dict]:
    """Долгие неподвижные треки класса «traffic light»: медианный центр и размер."""
    out = []
    if not tracks:
        return out
    for tr in (tracks.values() if isinstance(tracks, dict) else tracks):
        if getattr(tr, "cls", "") != "traffic light" or len(tr.t) < min_obs or (tr.t1 - tr.t0) < min_dur:
            continue
        cx, cy = float(np.median(tr.cx)), float(np.median(tr.cy))
        size = max(float(np.median(tr.w)), float(np.median(tr.h)), 1.0)
        if max(float(np.std(tr.cx)), float(np.std(tr.cy))) > max(3.0, max_std_rel * 100.0 * size):
            continue
        out.append({"id": int(tr.id), "cx": round(cx, 1), "cy": round(cy, 1), "size": round(size, 1), "n": int(len(tr.t))})
    return out


def light_residuals(anchors: np.ndarray, lights: list[dict]) -> list[dict]:
    """Для каждого ожидаемого положения светофора — ближайший статичный трек и расстояние до него."""
    res = []
    for ax, ay in np.asarray(anchors, dtype=float).reshape(-1, 2):
        best, dist = None, None
        for lt in lights:
            d = math.hypot(lt["cx"] - ax, lt["cy"] - ay)
            if dist is None or d < dist:
                best, dist = lt, d
        res.append({"expected": [round(float(ax), 1), round(float(ay), 1)],
                    "found": ([best["cx"], best["cy"]] if best else None),
                    "residual_px": (round(float(dist), 1) if dist is not None else None)})
    return res


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return (p @ M[:, :2].T + M[:, 2]).astype(np.float32)


def check_lights(info: dict, tracks, video_size: tuple[int, int], ref: SceneRef | None = None,
                 ref_path: str | Path = REF_PATH) -> dict:
    """Сверка по светофорам после того, как треки построены: дополняет info["lights"] невязками до и после
    (по info["matrix"], если преобразование было принято). Нужна, когда align_scene вызвали до детектора."""
    ref = ref or load_ref(ref_path)
    if ref is None or info.get("reason") == "no_scene":
        return info
    vw, vh = int(video_size[0]), int(video_size[1])
    pre = np.array([vw / float(ref.frame_size[0]), vh / float(ref.frame_size[1])], dtype=np.float32)
    anchors = ref.anchors * pre
    lights = static_lights(tracks)
    M = np.asarray(info["matrix"], dtype=np.float64) if info.get("applied") and info.get("matrix") else None
    info["lights"] = {"n_tracks": len(lights), "before": light_residuals(anchors, lights),
                      "after": (light_residuals(apply_affine(M, anchors), lights) if M is not None else None)}
    return info


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------
def align_scene(scene: Scene | None, video_path: str, meta: dict | None = None, tracks=None,
                ref: SceneRef | None = None, ref_path: str | Path = REF_PATH) -> tuple[Scene | None, dict]:
    """Сцена, привязанная к видео, либо None (другая камера). info сериализуется в JSON целиком.

    Порядок: масштаб сцены к размеру кадра → медианный фон из ≤ 9 кадров → ключевые точки → подобие к эталону →
    проверка порогов → (если есть треки) невязки по светофорам до и после."""
    t_all = time.perf_counter()
    info: dict = {"applied": False, "reason": "no_scene", "video": Path(video_path).name}
    if scene is None:
        return None, info
    meta = meta or video_meta(video_path)
    vw, vh = int(meta["width"]), int(meta["height"])
    scene = scene.scale_to(vw, vh)
    info["prescale"] = [round(vw / float(scene.frame_size[0]), 4), round(vh / float(scene.frame_size[1]), 4)] \
        if scene.frame_size != (vw, vh) else [1.0, 1.0]
    ref = ref or load_ref(ref_path)
    if ref is None:
        info["reason"] = "no_reference"
        info["time_sec"] = {"total": round(time.perf_counter() - t_all, 3)}
        return None, info
    t0 = time.perf_counter()
    frames, _ = sample_frames(video_path, meta)
    bg = median_background(frames)
    t_frames = time.perf_counter() - t0
    info["frames"] = len(frames)
    if bg is None:
        info["reason"] = "no_frames"
        info["time_sec"] = {"frames": round(t_frames, 3), "total": round(time.perf_counter() - t_all, 3)}
        return None, info
    M, est = estimate_transform(ref, bg, (vw, vh))
    info.update(est)
    info["time_sec"]["frames"] = round(t_frames, 3)
    info["time_sec"]["total"] = round(time.perf_counter() - t_all, 3)
    info["applied"] = M is not None
    if tracks is not None:
        check_lights(info, tracks, (vw, vh), ref)
    if M is None:
        return None, info
    return scene.transform(M), info

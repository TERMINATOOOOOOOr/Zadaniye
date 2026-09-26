"""Part A целиком: видео → наблюдения детектора → треки → правила → события."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .align import align_scene, check_lights
from .config import Settings, seed_everything
from .context import Context
from .flow import FlowField, StopMap
from .rules import collision, congestion, impact, lines, obstacle, pedestrian, red_light, stopped, turns, wrong_way
from .scene import load_scene
from .segments import finalize_events
from .signal import FEATURES_VERSION, LAMP_CELL, LampReader, fuse_signals, lamp_crop_rect, signal_from_pedestrians, signal_from_traffic
from .static_objects import StaticObjectDetector, load_static_objects, refilter, save_static_objects
from .tracking import Detector, build_tracks, load_observations, save_observations
from .video import iter_frames, video_meta

Progress = Callable[[str, float], None]

_DETECTOR: Detector | None = None
_DETECTOR_KEY: tuple = ()


def get_detector(st: Settings) -> Detector:
    """Один детектор на процесс: загрузка весов дорогая, а трекер сбрасывается перед каждым видео."""
    global _DETECTOR, _DETECTOR_KEY
    key = (st.model_a, st.device, st.imgsz_a, st.conf, st.iou, st.tracker)
    if _DETECTOR is None or _DETECTOR_KEY != key:
        _DETECTOR = Detector(st.weights(st.model_a), st.device, st.imgsz_a, st.conf, st.iou, st.tracker)
        _DETECTOR_KEY = key
    return _DETECTOR


def _cache_paths(video_path: str, st: Settings, meta: dict, crop) -> tuple[Path | None, Path | None, Path | None]:
    """Файлы кэша: наблюдения детектора, признаки лампы (в ключе — вырез и версия признаков) и статические объекты."""
    if not st.cache_dir:
        return None, None, None
    key = hashlib.md5(f"{Path(video_path).name}|{meta['n_frames']}|{st.model_a}|{st.stride_a}|{st.imgsz_a}|{st.conf}".encode()).hexdigest()[:10]
    obs_cache = Path(st.cache_dir) / f"{Path(video_path).stem}_{key}.npz"
    lamp_cache = None
    if crop is not None:
        lkey = hashlib.md5(f"{key}|{tuple(crop)}|{LAMP_CELL}|{FEATURES_VERSION}".encode()).hexdigest()[:8]
        lamp_cache = Path(st.cache_dir) / f"{Path(video_path).stem}_{key}_lamp_{lkey}.npz"
    static_cache = Path(st.cache_dir) / f"{Path(video_path).stem}_{key}_static.npz"
    return obs_cache, lamp_cache, static_cache


class Aux:
    """Что ещё считается в проходе по кадрам кроме детекций: читатель лампы и детектор статических объектов.
    static=None означает, что объекты уже загружены из кэша (static_objects); иначе finish() зовут после треков."""

    def __init__(self) -> None:
        self.reader: LampReader | None = None
        self.static: StaticObjectDetector | None = None
        self.static_objects: list = []
        self.static_cache: Path | None = None


def lamp_setup(scene, meta: dict) -> tuple[tuple | None, LampReader | None]:
    """Вырез вокруг лампы и читатель, если в сцене есть видимый светофор с ROI и кадр того же размера."""
    if scene is None or not scene.signal_visible or not scene.signal_roi or tuple(scene.frame_size) != (meta["width"], meta["height"]):
        return None, None
    crop = lamp_crop_rect(scene.signal_roi, meta["width"], meta["height"])
    return crop, LampReader(scene.signal_roi, crop)


def collect_observations(video_path: str, st: Settings, scene, progress: Progress | None = None, meta: dict | None = None):
    """Прогон детектора+трекера по видео; в том же проходе по кадрам читается лампа (по вырезу полного разрешения)
    и копится модель фона для статических объектов. Возвращает (observations, Aux, meta)."""
    meta = meta or video_meta(video_path)
    crop, reader = lamp_setup(scene, meta)
    aux = Aux()
    aux.reader = reader
    obs_cache, lamp_cache, static_cache = _cache_paths(video_path, st, meta, crop)
    aux.static_cache = static_cache
    if obs_cache is not None and obs_cache.exists():
        obs = load_observations(str(obs_cache))
        need_lamp = reader is not None and not lamp_cache.exists()
        need_static = not static_cache.exists()
        if reader is not None and not need_lamp:
            z = np.load(lamp_cache)
            reader.load(z["t"], z["feats"])
        if not need_static:
            aux.static_objects = load_static_objects(str(static_cache))
        else:
            aux.static = StaticObjectDetector(meta["width"], meta["height"])
        if need_lamp or need_static:
            aux_pass(video_path, reader if need_lamp else None, aux.static, st)
            if need_lamp:
                _save_lamp(lamp_cache, reader)
        return obs, aux, meta
    det = get_detector(st)
    det.reset()
    aux.static = StaticObjectDetector(meta["width"], meta["height"])
    obs = []
    n_total = max(1, meta["n_frames"])
    t_start = time.perf_counter()
    for item in iter_frames(video_path, st.stride_a, st.imgsz_a, crop=crop):
        idx, t, frame, scale = item[:4]
        obs.extend(det.track(frame, idx, t, scale))
        if reader is not None:
            reader.push(item[4], t)
        aux.static.push(frame, t)
        if progress is not None and idx % (st.stride_a * 50) == 0:
            progress("детекция", idx / n_total)
    if obs_cache is not None:
        obs_cache.parent.mkdir(parents=True, exist_ok=True)
        save_observations(str(obs_cache), obs)
        if reader is not None:
            _save_lamp(lamp_cache, reader)
    meta["detect_sec"] = round(time.perf_counter() - t_start, 2)
    return obs, aux, meta


def aux_pass(video_path: str, reader: LampReader | None, static: StaticObjectDetector | None, st: Settings) -> None:
    """Проход по кадрам без детектора: признаки лампы по вырезу и/или модель фона. Нужен, когда детекции уже
    в кэше. Без детектора статических объектов кадр масштабируется до 64 px, чтобы не гонять его через трубу."""
    side = 480 if static is not None else 64
    crop = reader.crop if reader is not None else None
    for item in iter_frames(video_path, st.stride_a, side, crop=crop):
        if reader is not None:
            reader.push(item[4], item[1])
        if static is not None:
            static.push(item[2], item[1])


def _save_lamp(path: Path, reader: LampReader) -> None:
    t, feats = reader.arrays()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, feats=feats)


def build_context(video_path: str, st: Settings, progress: Progress | None = None) -> Context:
    scene = load_scene(st.scene_path)
    meta = video_meta(video_path)
    align_info: dict = {}
    if scene is not None:
        if progress is not None:
            progress("подгонка сцены", 0.01)
        aligned, align_info = align_scene(scene, video_path, meta)
        if aligned is None and align_info.get("reason") == "no_reference" \
                and tuple(scene.frame_size) == (meta["width"], meta["height"]):
            aligned = scene   # эталона нет: применяем сцену как есть, если размер кадра совпадает
        scene = aligned       # иначе None: другая камера или кадрирование не распозналось — правил на сцене не будет
    obs, aux, meta = collect_observations(video_path, st, scene, progress, meta)
    tracks = build_tracks(obs)
    if align_info and align_info.get("reason") not in ("no_scene", "no_reference"):
        check_lights(align_info, tracks, (meta["width"], meta["height"]))
    meta["align"] = align_info
    flow = FlowField(meta["width"], meta["height"]).build(tracks)
    stops = StopMap(meta["width"], meta["height"], cell=flow.cell).build(tracks)
    flow.road_mask |= stops.vehicles >= 3   # колонна/пробка стоит там, где никто не ехал — это тоже дорога
    ctx = Context(tracks=tracks, meta=meta, scene=scene, flow=flow, signal=None, stops=stops)
    # три источника состояния светофора и их слияние; без сцены/ROI остаётся только то, что есть
    lamp = aux.reader.finish(tracks) if aux.reader is not None else None
    ped = signal_from_pedestrians(ctx)
    traffic = signal_from_traffic(ctx)
    ctx.signal_roi, ctx.signal_ped = lamp, ped
    if lamp is not None or ped is not None or traffic is not None:
        ctx.signal = fuse_signals(lamp, ped, traffic, ctx.duration)
    # статические объекты на дороге: фильтр по трекам и дороге возможен только теперь
    if aux.static is not None:
        ctx.static_objects = aux.static.finish(tracks, ctx.on_road, stops, signal=ctx.signal)
        if aux.static_cache is not None:
            aux.static_cache.parent.mkdir(parents=True, exist_ok=True)
            save_static_objects(str(aux.static_cache), ctx.static_objects)
    else:
        ctx.static_objects = refilter(aux.static_objects, tracks, ctx.signal)
    return ctx


def run_rules(ctx: Context) -> dict[str, list[tuple[float, float]]]:
    accidents, near = collision.run(ctx)
    accidents = accidents + impact.run(ctx)   # парное правило и правило по точке удара дополняют друг друга
    return {
        "accident": accidents,
        "near_miss": near,
        "red_light": red_light.run_red_light(ctx),
        "wrong_way": wrong_way.run(ctx),
        "illegal_u_turn": turns.run_u_turn(ctx),
        "stopped_vehicle": stopped.run(ctx),
        "jaywalking": pedestrian.run_jaywalking(ctx),
        "failure_to_yield": pedestrian.run_failure_to_yield(ctx),
        "illegal_turn": turns.run_illegal_turn(ctx),
        "solid_line_crossing": lines.run(ctx),
        "stop_line": red_light.run_stop_line(ctx),
        "congestion": congestion.run(ctx),
        "road_obstacle": obstacle.run_obstacle(ctx),
        "fire_smoke": obstacle.run_fire_smoke(ctx),
    }


def detect_events(video_path: str, st: Settings | None = None, progress: Progress | None = None,
                  return_context: bool = False):
    seed_everything()
    st = st or Settings()
    ctx = build_context(video_path, st, progress)
    if progress is not None:
        progress("правила", 0.9)
    candidates = run_rules(ctx)
    events = finalize_events(candidates, ctx.duration)
    if progress is not None:
        progress("готово", 1.0)
    return (events, ctx) if return_context else events

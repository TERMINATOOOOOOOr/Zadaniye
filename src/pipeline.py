"""Part A целиком: видео → наблюдения детектора → треки → правила → события."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable

from .config import Settings, seed_everything
from .context import Context
from .flow import FlowField, StopMap
from .rules import collision, congestion, lines, obstacle, pedestrian, red_light, stopped, turns, wrong_way
from .scene import load_scene
from .segments import finalize_events
from .signal import SignalReader, signal_from_traffic
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


def collect_observations(video_path: str, st: Settings, scene, progress: Progress | None = None):
    """Прогон детектора+трекера по видео. Возвращает (observations, signal_track|None, meta)."""
    meta = video_meta(video_path)
    cache = None
    if st.cache_dir:
        key = hashlib.md5(f"{Path(video_path).name}|{meta['n_frames']}|{st.model_a}|{st.stride_a}|{st.imgsz_a}|{st.conf}".encode()).hexdigest()[:10]
        cache = Path(st.cache_dir) / f"{Path(video_path).stem}_{key}.npz"
        if cache.exists():
            obs = load_observations(str(cache))
            signal = _signal_from_cache(cache)
            if signal is None and scene is not None and scene.signal_visible and scene.signal_roi                     and tuple(scene.frame_size) == (meta["width"], meta["height"]):
                signal = signal_only_pass(video_path, scene.signal_roi, st)
                import numpy as np
                np.savez_compressed(str(cache).replace(".npz", "_signal.npz"), t=signal.t, state=np.array(signal.state))
            return obs, signal, meta
    det = get_detector(st)
    det.reset()
    same_camera = scene is not None and tuple(scene.frame_size) == (meta["width"], meta["height"])
    reader = SignalReader(scene.signal_roi) if (same_camera and scene.signal_visible and scene.signal_roi) else None
    obs = []
    n_total = max(1, meta["n_frames"])
    t_start = time.perf_counter()
    for idx, t, frame, scale in iter_frames(video_path, st.stride_a, st.imgsz_a):
        obs.extend(det.track(frame, idx, t, scale))
        if reader is not None:
            reader.push(frame, t, scale)
        if progress is not None and idx % (st.stride_a * 50) == 0:
            progress("детекция", idx / n_total)
    signal = reader.finish() if reader is not None else None
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_observations(str(cache), obs)
        if signal is not None:
            import numpy as np
            np.savez_compressed(str(cache).replace(".npz", "_signal.npz"), t=signal.t, state=np.array(signal.state))
    meta["detect_sec"] = round(time.perf_counter() - t_start, 2)
    return obs, signal, meta


def signal_only_pass(video_path: str, roi, st: Settings):
    """Только цвет светофора по кадрам (без детектора): нужен, когда детекции уже в кэше."""
    reader = SignalReader(roi)
    for idx, t, frame, scale in iter_frames(video_path, st.stride_a, st.imgsz_a):
        reader.push(frame, t, scale)
    return reader.finish()


def _signal_from_cache(cache: Path):
    import numpy as np
    from .context import SignalTrack
    p = Path(str(cache).replace(".npz", "_signal.npz"))
    if not p.exists():
        return None
    z = np.load(p)
    return SignalTrack(t=z["t"], state=[str(s) for s in z["state"]])


def build_context(video_path: str, st: Settings, progress: Progress | None = None) -> Context:
    scene = load_scene(st.scene_path)
    obs, signal, meta = collect_observations(video_path, st, scene, progress)
    if scene is not None and tuple(scene.frame_size) != (meta["width"], meta["height"]):
        scene = None   # сцена размечена для другой камеры/разрешения — не применяем, чтобы не плодить ложные события
    tracks = build_tracks(obs)
    flow = FlowField(meta["width"], meta["height"]).build(tracks)
    stops = StopMap(meta["width"], meta["height"], cell=flow.cell).build(tracks)
    flow.road_mask |= stops.vehicles >= 3   # колонна/пробка стоит там, где никто не ехал — это тоже дорога
    ctx = Context(tracks=tracks, meta=meta, scene=scene, flow=flow, signal=signal, stops=stops)
    # сигнал по поведению трафика надёжнее цвета лампы (блики, ракурс); цвет остаётся для EDA
    traffic_signal = signal_from_traffic(ctx)
    if traffic_signal is not None:
        ctx.signal_roi = signal
        ctx.signal = traffic_signal
    return ctx


def run_rules(ctx: Context) -> dict[str, list[tuple[float, float]]]:
    accidents, near = collision.run(ctx)
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

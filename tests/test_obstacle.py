"""road_obstacle по модели фона на синтетической последовательности: ответ известен заранее."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context import Context  # noqa: E402
from src.flow import FlowField, StopMap  # noqa: E402
from src.rules import obstacle  # noqa: E402
from src.static_objects import StaticObject, StaticObjectDetector, box_iou, load_static_objects, save_static_objects  # noqa: E402
from src.tracking import Obs, build_tracks  # noqa: E402

W, H = 1920, 1080          # исходный кадр
RW, RH = 960, 540          # кадр читателя (длинная сторона 960)
FPS_EFF = 10.0             # обработанных кадров в секунду (stride 3 при 30 fps)
ROAD_Y = (300, 800)        # дорога — горизонтальная полоса в исходных координатах
PATCH = (1000.0, 500.0, 1100.0, 570.0)   # тёмный предмет 100×70 px на дороге


def _background() -> np.ndarray:
    """Асфальт с лёгкой текстурой и разметкой, детерминированный."""
    rng = np.random.RandomState(0)
    bg = np.full((H, W, 3), 110, dtype=np.uint8)
    bg = np.clip(bg.astype(int) + rng.randint(-6, 7, (H, W, 1)), 0, 255).astype(np.uint8)
    bg[:ROAD_Y[0]] = 70; bg[ROAD_Y[1]:] = 70
    for y in range(ROAD_Y[0] + 100, ROAD_Y[1], 100):
        bg[y:y + 6, ::40] = 220
    return bg


def _cars(t: float) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Поток машин слева направо: (id, бокс) в момент t. Машина k стартует в 4k с, едет 8 с."""
    out = []
    for k in range(20):
        t0 = 4.0 * k
        if t0 <= t < t0 + 8.0:
            x = (t - t0) / 8.0 * (W + 200) - 200
            out.append((100 + k, (x, 380.0, x + 180.0, 480.0)))
    return out


def _render(t: float, bg: np.ndarray, patch: tuple | None, parked: tuple | None) -> np.ndarray:
    img = bg.copy()
    for _, (x1, y1, x2, y2) in _cars(t):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 190, 180), -1)
    if parked is not None:
        x1, y1, x2, y2 = (int(v) for v in parked)
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
    if patch is not None:
        x1, y1, x2, y2 = (int(v) for v in patch)
        cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
    return cv2.resize(img, (RW, RH), interpolation=cv2.INTER_AREA)


def _run(duration: float = 40.0, patch=PATCH, patch_from: float = 10.0, parked=None, parked_from: float = 10.0,
         track_parked: bool = False):
    bg = _background()
    det = StaticObjectDetector(W, H)
    obs = []
    n = int(duration * FPS_EFF)
    for k in range(n):
        t = k / FPS_EFF
        frame = _render(t, bg, patch if t >= patch_from else None, parked if (parked is not None and t >= parked_from) else None)
        det.push(frame, t)
        for tid, b in _cars(t):
            obs.append((tid, Obs(k * 3, t, "car", 0.9, *b)))
        if track_parked and parked is not None and t >= parked_from:
            obs.append((7, Obs(k * 3, t, "car", 0.9, *parked)))
    tracks = build_tracks(obs)
    on_road = lambda x, y: ROAD_Y[0] <= y <= ROAD_Y[1]   # noqa: E731
    report = []
    objs = det.finish(tracks, on_road, report=report)
    return objs, report, tracks


def test_static_patch_on_road_is_detected(monkeypatch):
    import src.static_objects as so
    monkeypatch.setattr(so, "WARMUP_SEC", 0.0)   # синтетика короткая: патч появляется на 10-й секунде
    objs, report, _ = _run()
    assert len(objs) == 1, [(o.t0, o.t1, o.box, r) for o, r in report]
    o = objs[0]
    assert abs(o.t0 - 10.0) < 1.0 and o.t1 >= 38.0, (o.t0, o.t1)
    assert box_iou(o.box, PATCH) > 0.5, o.box


def test_patch_under_tracked_vehicle_is_not_detected():
    # тот же тёмный прямоугольник, но детектор видит его как машину (трек 7 с тем же боксом)
    objs, report, _ = _run(patch=None, parked=PATCH, track_parked=True)
    assert objs == [], [(o.t0, o.t1, o.box, r) for o, r in report]
    assert any(r.startswith("tracked") for _, r in report)


def test_patch_off_road_is_not_detected():
    objs, report, _ = _run(patch=(1000.0, 100.0, 1100.0, 170.0))
    assert objs == []
    assert any(r == "off_road" for _, r in report)


def test_moving_traffic_alone_gives_nothing():
    objs, report, _ = _run(patch=None)
    assert objs == [] and report == []


def test_rule_adds_static_objects_to_coco_rule():
    flow = FlowField(W, H)
    ctx = Context(tracks={}, meta={"fps": 30.0, "width": W, "height": H, "duration": 40.0, "n_frames": 1200},
                  scene=None, flow=flow, signal=None, stops=StopMap(W, H, cell=flow.cell))
    assert obstacle.run_obstacle(ctx) == []
    ctx.static_objects = [StaticObject(10.0, 38.0, *PATCH), StaticObject(20.0, 20.0, *PATCH)]
    assert obstacle.run_obstacle(ctx) == [(10.0, 38.0)]


def test_cache_roundtrip(tmp_path):
    objs = [StaticObject(10.0, 38.0, *PATCH, contrast=70.0, edge=12.5)]
    p = tmp_path / "x_static.npz"
    save_static_objects(str(p), objs)
    back = load_static_objects(str(p))
    assert len(back) == 1 and back[0] == objs[0]
    save_static_objects(str(p), [])
    assert load_static_objects(str(p)) == []

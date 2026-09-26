"""Привязка сцены к видео (src/align.py) на синтетике: эталонный фон сдвигаем/масштабируем на известную величину
и проверяем, что преобразование восстанавливается; чужая картинка должна отвергаться. Видео не нужно."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import align  # noqa: E402
from src.scene import Lane, Scene, StopLine  # noqa: E402

REF = align.load_ref(align.REF_PATH)
needs_ref = pytest.mark.skipif(REF is None, reason="нет configs/scene_ref.npz (tools/make_scene_ref.py)")


def _work_matrix(scale: float, angle_deg: float, tx: float, ty: float, center) -> np.ndarray:
    """Подобие в координатах рабочего кадра: поворот+масштаб вокруг центра, затем сдвиг."""
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -angle_deg, scale)   # знак: у OpenCV угол против часовой
    M[:, 2] += (tx, ty)
    return M


def _full_from_work(M_work: np.ndarray, up: float) -> np.ndarray:
    """Та же матрица в координатах полного кадра (рабочий кадр = полный / up): линейная часть та же, сдвиг × up."""
    M = M_work.copy()
    M[:, 2] *= up
    return M


def _run(gray: np.ndarray):
    return align.estimate_transform(REF, gray, REF.frame_size)


@needs_ref
@pytest.mark.parametrize("scale,angle,tx,ty", [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, -20.0, 12.5), (1.05, 1.0, 15.0, -8.0), (0.9, -1.5, 30.0, 25.0)])
def test_recovers_known_similarity(scale, angle, tx, ty):
    bg = REF.background
    h, w = bg.shape[:2]
    up = REF.frame_size[0] / float(w)
    M_work = _work_matrix(scale, angle, tx, ty, (w / 2.0, h / 2.0))
    warped = cv2.warpAffine(bg, M_work, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M, info = _run(warped)
    assert M is not None, info
    assert info["reason"] == "ok" and info["n_inliers"] >= align.MIN_INLIERS
    expected = _full_from_work(M_work, up)
    got = align.decompose(M)
    want = align.decompose(expected)
    assert abs(got["scale"] - want["scale"]) <= 0.01 * want["scale"]          # 1 % по масштабу
    assert abs(got["angle_deg"] - want["angle_deg"]) <= 0.3
    assert abs(got["tx"] - want["tx"]) / up <= 2.0 and abs(got["ty"] - want["ty"]) / up <= 2.0   # 2 px рабочего кадра
    # точки сцены после переноса совпадают с истинными в пределах 2 px рабочего кадра
    pts = np.array([[500, 800], [2000, 1200], [3300, 600], [1000, 1900]], dtype=np.float32)
    err = np.linalg.norm(align.apply_affine(M, pts) - align.apply_affine(expected, pts), axis=1) / up
    assert float(err.max()) <= 2.0, err


@needs_ref
def test_rejects_noise():
    rng = np.random.default_rng(0)
    h, w = REF.background.shape[:2]
    noise = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    M, info = _run(noise)
    assert M is None
    assert info["reason"] in ("few_matches", "few_inliers", "no_model", "bad_scale", "bad_angle")
    assert info["n_inliers"] < align.MIN_INLIERS


@needs_ref
def test_rejects_unrelated_picture():
    h, w = REF.background.shape[:2]
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 256, size=(h, w), dtype=np.uint8), (0, 0), 6)
    for _ in range(60):     # «улица» из прямоугольников и кругов, к перекрёстку отношения не имеет
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 200)), y + int(rng.integers(10, 120))), int(rng.integers(0, 256)), -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 60)), int(rng.integers(0, 256)), -1)
    M, info = _run(img)
    assert M is None, info
    assert info["n_inliers"] < align.MIN_INLIERS


@needs_ref
def test_rejects_wrong_scale_and_angle():
    bg = REF.background
    h, w = bg.shape[:2]
    for scale, angle in ((0.6, 0.0), (1.0, 12.0)):
        M_work = _work_matrix(scale, angle, 0.0, 0.0, (w / 2.0, h / 2.0))
        warped = cv2.warpAffine(bg, M_work, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        M, info = _run(warped)
        assert M is None and info["reason"] in ("bad_scale", "bad_angle", "few_inliers", "few_matches", "no_model"), info


@needs_ref
def test_deterministic_and_json_serialisable():
    bg = REF.background
    h, w = bg.shape[:2]
    M_work = _work_matrix(1.0, 0.0, -20.0, 12.0, (w / 2.0, h / 2.0))
    warped = cv2.warpAffine(bg, M_work, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M1, info1 = _run(warped)
    M2, info2 = _run(warped)
    assert np.allclose(M1, M2) and info1["n_inliers"] == info2["n_inliers"]
    info1["lights"] = {"before": align.light_residuals(REF.anchors, [{"id": 1, "cx": 2300.0, "cy": 780.0, "size": 100.0, "n": 50}])}
    json.dumps(info1)


def test_scene_transform_moves_geometry():
    sc = Scene(frame_size=(3840, 2160), px_per_meter=10.0, road=np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32),
               intersection=None, lanes=[Lane("l1", np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32), np.array([1.0, 0.0]), {"straight"})],
               crosswalks=[("cw", np.array([[10, 10], [20, 10], [20, 20]], dtype=np.float32))],
               stop_lines=[StopLine("sl", np.array([0.0, 0.0], dtype=np.float32), np.array([10.0, 0.0], dtype=np.float32), np.array([0.0, 1.0]), [])],
               solid_lines=[("s", np.array([[0, 0], [5, 5]], dtype=np.float32))], islands=[], signal_visible=True, signal_roi=(100, 200, 40, 60))
    # чистый сдвиг
    M = np.array([[1.0, 0.0, -80.0], [0.0, 1.0, 50.0]])
    out = sc.transform(M, frame_size=(3840, 2160))
    assert np.allclose(out.road, sc.road + np.array([-80, 50], dtype=np.float32))
    assert np.allclose(out.crosswalks[0][1], sc.crosswalks[0][1] + np.array([-80, 50], dtype=np.float32))
    assert np.allclose(out.stop_lines[0].p1, [-80, 50]) and np.allclose(out.stop_lines[0].p2, [-70, 50])
    assert out.signal_roi == (20, 250, 40, 60)
    assert np.allclose(out.lanes[0].direction, [1.0, 0.0]) and np.allclose(out.stop_lines[0].approach, [0.0, 1.0])
    assert out.px_per_meter == pytest.approx(10.0)
    assert sc.road[0][0] == 0   # исходная сцена не изменилась
    # поворот на 90° по часовой (y вниз) с масштабом 2: вектор (1,0) → (0,1), ROI удваивается
    s, a = 2.0, math.radians(90.0)
    M = np.array([[s * math.cos(a), -s * math.sin(a), 0.0], [s * math.sin(a), s * math.cos(a), 0.0]])
    out = sc.transform(M)
    assert np.allclose(out.lanes[0].direction, [0.0, 1.0], atol=1e-6)
    assert np.allclose(out.stop_lines[0].approach, [-1.0, 0.0], atol=1e-6)
    assert out.signal_roi[2] == 80 and out.signal_roi[3] == 120
    assert out.px_per_meter == pytest.approx(20.0)
    assert np.allclose(out.road[1], [0.0, 200.0], atol=1e-4)


def test_static_lights_and_residuals():
    class T:
        def __init__(self, tid, cls, cx, cy, n=60, jitter=0.0):
            self.id, self.cls = tid, cls
            self.t = np.linspace(0, 30, n)
            self.cx = np.full(n, cx) + jitter * np.sin(self.t)
            self.cy = np.full(n, cy)
            self.w = np.full(n, 50.0); self.h = np.full(n, 120.0)
            self.t0, self.t1 = 0.0, 30.0
    tracks = {1: T(1, "traffic light", 2300.0, 780.0), 2: T(2, "car", 2300.0, 780.0), 3: T(3, "traffic light", 500.0, 500.0, n=5),
              4: T(4, "traffic light", 1400.0, 600.0, jitter=200.0)}
    lights = align.static_lights(tracks)
    assert [l["id"] for l in lights] == [1]
    res = align.light_residuals(np.array([[2310.0, 774.0]]), lights)
    assert res[0]["found"] == [2300.0, 780.0] and res[0]["residual_px"] == pytest.approx(math.hypot(10, 6), abs=0.1)
    assert align.light_residuals(np.array([[1.0, 1.0]]), [])[0]["found"] is None


@needs_ref
def test_check_lights_uses_matrix():
    class T:
        def __init__(self, tid, cx, cy, n=60):
            self.id, self.cls = tid, "traffic light"
            self.t = np.linspace(0, 30, n); self.cx = np.full(n, cx); self.cy = np.full(n, cy)
            self.w = np.full(n, 50.0); self.h = np.full(n, 120.0); self.t0, self.t1 = 0.0, 30.0
    # три светофора, сдвинутые на (-80, +50) относительно эталона; матрица чистого сдвига должна обнулить невязки
    tracks = {i: T(i, ax - 80.0, ay + 50.0) for i, (ax, ay) in enumerate(REF.anchors)}
    info = {"applied": True, "reason": "ok", "matrix": [[1.0, 0.0, -80.0], [0.0, 1.0, 50.0]]}
    align.check_lights(info, tracks, REF.frame_size, REF)
    assert info["lights"]["n_tracks"] == 3
    assert all(abs(r["residual_px"] - math.hypot(80, 50)) < 0.2 for r in info["lights"]["before"])
    assert all(r["residual_px"] < 0.2 for r in info["lights"]["after"])
    info = {"applied": False, "reason": "few_inliers", "matrix": None}
    align.check_lights(info, tracks, REF.frame_size, REF)
    assert info["lights"]["after"] is None and len(info["lights"]["before"]) == 3
    json.dumps(info)

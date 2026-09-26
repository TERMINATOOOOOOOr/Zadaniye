"""Правила на синтетических траекториях: сценарии, в которых ответ известен заранее."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context import Context  # noqa: E402
from src.flow import FlowField, StopMap  # noqa: E402
from src.rules import collision, stopped, turns, wrong_way  # noqa: E402
from src.tracking import Obs, build_tracks  # noqa: E402

W, H, FPS = 1920, 1080, 25.0
DT = 3 / FPS   # шаг наблюдений при stride 3


def _obs(tid, cls, pts, w=140.0, h=80.0):
    """pts: список (t, cx, cy) → наблюдения с боксом w×h вокруг центра."""
    out = []
    for t, cx, cy in pts:
        out.append((tid, Obs(int(round(t * FPS)), float(t), cls, 0.9, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def _line(t0, t1, x0, x1, y, step=DT):
    ts = np.arange(t0, t1, step)
    return [(t, x0 + (x1 - x0) * (t - t0) / max(t1 - t0, 1e-6), y) for t in ts]


def _ctx(observations, duration=60.0):
    tracks = build_tracks(observations)
    flow = FlowField(W, H).build(tracks)
    stops = StopMap(W, H, cell=flow.cell).build(tracks)
    flow.road_mask |= stops.vehicles >= 3
    return Context(tracks=tracks, meta={"fps": FPS, "width": W, "height": H, "duration": duration, "n_frames": int(duration * FPS)},
                   scene=None, flow=flow, signal=None, stops=stops)


def _traffic(n=12, y=500.0, t_gap=4.0, start_id=100):
    """Поток машин слева направо по одной полосе: задаёт поле направлений и маску дороги."""
    obs = []
    for k in range(n):
        obs += _obs(start_id + k, "car", _line(k * t_gap, k * t_gap + 8.0, 0.0, W, y))
    return obs


def test_stopped_vehicle_alone_vs_queue():
    obs = _traffic()
    # одиночная машина: приехала, встала на 25 с, поехала
    obs += _obs(1, "car", _line(15.0, 20.0, 0.0, 1000.0, 500.0) + [(t, 1000.0, 500.0) for t in np.arange(20.0, 45.0, DT)] + _line(45.0, 49.0, 1000.0, 1900.0, 500.0))
    ctx = _ctx(obs)
    ivs = stopped.run(ctx)
    assert len(ivs) == 1 and abs(ivs[0][0] - 20.0) < 1.0 and abs(ivs[0][1] - 45.0) < 1.0

    # та же остановка, но рядом стоят ещё две машины (очередь) — не событие
    obs2 = obs + _obs(2, "car", [(t, 1180.0, 500.0) for t in np.arange(18.0, 48.0, DT)]) \
               + _obs(3, "car", [(t, 1360.0, 500.0) for t in np.arange(18.0, 48.0, DT)])
    assert stopped.run(_ctx(obs2)) == []


def test_wrong_way_against_flow():
    obs = _traffic(n=14)
    obs += _obs(1, "car", _line(30.0, 38.0, W, 0.0, 500.0))   # справа налево по той же полосе
    ivs = wrong_way.run(_ctx(obs))
    assert len(ivs) == 1
    s, e = ivs[0]
    assert s <= 31.5 and e >= 36.5


def test_u_turn_segment_is_tight():
    pts = _line(0.0, 5.0, 0.0, 1000.0, 500.0)                       # едет вправо 5 с
    for k, t in enumerate(np.arange(5.0, 8.0, DT)):                    # разворот за 3 с по полуокружности радиуса 120
        a = np.pi * (t - 5.0) / 3.0
        pts.append((t, 1000.0 + 120.0 * np.sin(a), 500.0 + 120.0 * (1 - np.cos(a))))
    pts += _line(8.0, 13.0, 1000.0, 0.0, 740.0)                       # едет влево 5 с
    obs = _traffic() + _obs(1, "car", pts)
    ivs = turns.run_u_turn(_ctx(obs))
    assert len(ivs) == 1
    s, e = ivs[0]
    assert 4.0 <= s <= 6.0 and 7.0 <= e <= 9.5, ivs


def _rear_end(queue: bool):
    obs = _traffic()
    # A стоит одна на полосе с t=0
    obs += _obs(1, "car", [(t, 1000.0, 500.0) for t in np.arange(0.0, 20.0, DT)])
    if queue:
        obs += _obs(3, "car", [(t, 1180.0, 500.0) for t in np.arange(0.0, 20.0, DT)])
    # B подъезжает со скоростью 250 px/с и резко встаёт в контакте с A (центры на расстоянии 0.4 h)
    pts = []
    for t in np.arange(0.0, 20.0, DT):
        x = min(200.0 + 250.0 * t, 968.0)
        pts.append((t, x, 500.0))
    obs += _obs(2, "car", pts)
    return obs


def test_rear_end_accident_detected_but_queue_join_is_not():
    acc, near = collision.run(_ctx(_rear_end(queue=False)))
    assert len(acc) == 1, (acc, near)
    s, e = acc[0]
    assert 2.5 <= s <= 3.8 and e - s >= 3.0
    acc_q, _ = collision.run(_ctx(_rear_end(queue=True)))
    assert acc_q == []


# ---------------------------------------------------------------------------
# Правила, которым нужна сцена: зоны направлений и сплошные линии
# ---------------------------------------------------------------------------
from src.rules import lines  # noqa: E402
from src.scene import Lane, Scene  # noqa: E402


def _scene(lanes=(), solid_lines=(), intersection=None):
    poly = lambda pts: np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return Scene(frame_size=(W, H), px_per_meter=None, road=poly([[0, 0], [W, 0], [W, H], [0, H]]),
                 intersection=poly(intersection) if intersection is not None else None,
                 lanes=[Lane(n, poly(p), np.asarray(d, dtype=float), set(a)) for n, p, d, a in lanes],
                 crosswalks=[], stop_lines=[], solid_lines=[(n, poly(p)) for n, p in solid_lines],
                 islands=[], signal_visible=False, signal_roi=None)


def _ctx_scene(observations, scene, duration=60.0):
    ctx = _ctx(observations, duration)
    ctx.scene = scene
    return ctx


def _lane_change(t0, y_from, y_to, x0=0.0, speed=200.0, t_change=10.0, dur=1.0, t1=14.0):
    """Едет вправо со скоростью speed px/с, за dur секунд с t_change переходит с y_from на y_to."""
    pts = []
    for t in np.arange(t0, t1, DT):
        f = min(1.0, max(0.0, (t - t_change) / dur))
        pts.append((t, x0 + speed * (t - t0), y_from + (y_to - y_from) * f))
    return pts


def test_solid_line_crossing_needs_settled_sides():
    scene = _scene(solid_lines=[("s0", [[200, 500], [1700, 500]])])
    # настоящее перестроение через сплошную: одно событие вокруг момента пересечения (t около 10.5)
    obs = _traffic(y=300.0) + _obs(1, "car", _lane_change(4.0, 440.0, 560.0))
    ivs = lines.run(_ctx_scene(obs, scene))
    assert len(ivs) == 1, ivs
    s, e = ivs[0]
    assert 9.0 <= s <= 10.5 and 10.5 <= e <= 12.0, ivs
    # дрожание бокса: на один кадр заехал за линию и вернулся — не событие
    pts = _lane_change(4.0, 440.0, 440.0)
    k = int((10.0 - 4.0) / DT)
    pts[k] = (pts[k][0], pts[k][1], 530.0)
    assert lines.run(_ctx_scene(_traffic(y=300.0) + _obs(2, "car", pts), scene)) == []
    # едет по самой линии с шумом до 12 px — не событие
    rng = np.random.RandomState(0)
    pts = [(t, x, 500.0 + rng.uniform(-12, 12)) for t, x, _ in _lane_change(4.0, 500.0, 500.0)]
    assert lines.run(_ctx_scene(_traffic(y=300.0) + _obs(3, "car", pts), scene)) == []
    # та же траектория без сцены — правило молчит
    assert lines.run(_ctx(_traffic(y=300.0) + _obs(1, "car", _lane_change(4.0, 440.0, 560.0)))) == []


def test_wrong_way_uses_lane_zone_direction():
    scene = _scene(lanes=[("east", [[0, 400], [W, 400], [W, 600], [0, 600]], [1, 0], ["straight"])])
    obs = _traffic(n=6)                                                 # поток вправо по y=500 (внутри зоны)
    obs += _obs(1, "car", _line(30.0, 38.0, W, 0.0, 500.0))            # против зоны — событие
    obs += _obs(2, "car", _line(40.0, 48.0, 0.0, W, 520.0))            # по зоне — нет
    obs += _obs(3, "car", _line(50.0, 58.0, W, 0.0, 800.0))            # вне зоны: эталона нет — молчим
    ivs = wrong_way.run(_ctx_scene(obs, scene))
    assert len(ivs) == 1, ivs
    s, e = ivs[0]
    assert s <= 31.5 and e >= 36.5


def _right_turn_track(tid):
    """Едет вправо по y=500 до x=900, поворачивает направо (по часовой, вниз) по четверти окружности радиуса 150."""
    pts = _line(0.0, 4.5, 0.0, 900.0, 500.0)
    r, v = 150.0, 200.0
    t_arc = (np.pi * r / 2) / v
    for t in np.arange(4.5, 4.5 + t_arc, DT):
        a = (t - 4.5) / t_arc * np.pi / 2
        pts.append((t, 900.0 + r * np.sin(a), 500.0 + r * (1 - np.cos(a))))
    t_end = 4.5 + t_arc
    pts += [(t, 1050.0, 650.0 + v * (t - t_end)) for t in np.arange(t_end, t_end + 1.5, DT)]
    return _obs(tid, "car", pts)


def test_illegal_turn_uses_origin_lane_before_intersection():
    inter = [[800, 200], [1400, 200], [1400, 1000], [800, 1000]]
    zone = [[0, 400], [800, 400], [800, 600], [0, 600]]
    straight_only = _scene(lanes=[("east", zone, [1, 0], ["straight"])], intersection=inter)
    obs = _traffic(n=6) + _right_turn_track(1)
    ivs = turns.run_illegal_turn(_ctx_scene(obs, straight_only))
    assert len(ivs) == 1, ivs
    s, e = ivs[0]
    assert 4.0 <= s <= 5.5 and 5.0 <= e <= 7.0, ivs
    # тот же манёвр, но поворот направо из этой зоны разрешён
    right_ok = _scene(lanes=[("east", zone, [1, 0], ["straight", "right"])], intersection=inter)
    assert turns.run_illegal_turn(_ctx_scene(obs, right_ok)) == []
    # и разворотом это не считается
    assert turns.run_u_turn(_ctx_scene(obs, straight_only)) == []

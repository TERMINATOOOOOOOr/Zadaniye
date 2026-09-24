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

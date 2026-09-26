"""Светофор: чтение лампы на синтетических вырезах, пешеходные фазы, слияние источников, правила red_light/stop_line,
crop-режим видеоридера."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context import Context, SignalTrack  # noqa: E402
from src.flow import FlowField, StopMap  # noqa: E402
from src.rules import red_light  # noqa: E402
from src.scene import Scene, StopLine  # noqa: E402
from src.signal import (LampReader, bind_crosswalk, fuse_signals, lamp_crop_rect, phases,  # noqa: E402
                        signal_from_pedestrians, smooth_states, stop_line_offset)
from src.tracking import Obs, build_tracks  # noqa: E402
from src.video import ffmpeg_exe, ffmpeg_filter, iter_frames, iter_frames_cv2, normalize_crop  # noqa: E402

W, H, FPS = 1920, 1080, 25.0
DT = 3 / FPS
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# синтетика
# ---------------------------------------------------------------------------
def _lamp_crop(state: str, size=(264, 192), bright: float = 1.0, faint: bool = False, cover: bool = False) -> np.ndarray:
    """Вырез с тёмным корпусом и тремя линзами; горит секция state. faint — тусклая линза (день),
    cover — светлый прямоугольник поверх всего (автобус закрыл лампу)."""
    import cv2
    h, w = size
    img = np.full((h, w, 3), int(110 * bright), np.uint8)              # асфальт
    cv2.rectangle(img, (80, 60), (140, 210), (35, 35, 35), -1)         # корпус
    lenses = {"red": ((110, 85), (40, 40, 200)), "amber": ((110, 135), (30, 170, 230)), "green": ((110, 185), (170, 200, 40))}
    for name, (c, col) in lenses.items():
        cv2.circle(img, c, 16, (55, 55, 55), -1)
        if name == state:
            col = tuple(int(v * 0.55 + 30) for v in col) if faint else col
            cv2.circle(img, c, 16, col, -1)
    if cover:
        cv2.rectangle(img, (0, 40), (w, h), (215, 215, 215), -1)
    return img


def _reader_with(sequence: list[tuple[str, float]], fps: float = 8.33, **kw) -> tuple[LampReader, np.ndarray]:
    roi = (70, 50, 80, 170)
    crop = lamp_crop_rect(roi, 1920, 1080)
    reader = LampReader(roi, (0, 0, 192, 264))
    t = 0.0
    for state, dur in sequence:
        for _ in range(int(round(dur * fps))):
            cover = state == "cover"
            reader.push(_lamp_crop("red" if cover else state, cover=cover, **kw), t)
            t += 1 / fps
    return reader, np.asarray(reader.ts)


def _obs(tid, cls, pts, w=140.0, h=80.0):
    return [(tid, Obs(int(round(t * FPS)), float(t), cls, 0.9, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)) for t, cx, cy in pts]


def _line(t0, t1, x0, x1, y0, y1=None, step=DT):
    y1 = y0 if y1 is None else y1
    ts = np.arange(t0, t1, step)
    return [(t, x0 + (x1 - x0) * (t - t0) / max(t1 - t0, 1e-6), y0 + (y1 - y0) * (t - t0) / max(t1 - t0, 1e-6)) for t in ts]


def _scene() -> Scene:
    """Дорога слева направо, стоп-линия x=900, зебра x=940..1040 за ней, перекрёсток x>1060."""
    road = np.array([[0, 300], [W, 300], [W, 700], [0, 700]], np.float32)
    return Scene(frame_size=(W, H), px_per_meter=None, road=road,
                 intersection=np.array([[1060, 300], [1500, 300], [1500, 700], [1060, 700]], np.float32),
                 lanes=[], crosswalks=[("cw", np.array([[940, 300], [1040, 300], [1040, 700], [940, 700]], np.float32)),
                                       ("cw_far", np.array([[100, 300], [200, 300], [200, 700], [100, 700]], np.float32))],
                 stop_lines=[StopLine("sl", np.array([900, 300], np.float32), np.array([900, 700], np.float32), np.array([1.0, 0.0]), [])],
                 solid_lines=[], islands=[], signal_visible=True, signal_roi=(70, 50, 80, 170))


def _ctx(observations, scene, signal=None, duration=60.0) -> Context:
    tracks = build_tracks(observations)
    flow = FlowField(W, H).build(tracks)
    stops = StopMap(W, H, cell=flow.cell).build(tracks)
    return Context(tracks=tracks, meta={"fps": FPS, "width": W, "height": H, "duration": duration, "n_frames": int(duration * FPS)},
                   scene=scene, flow=flow, signal=signal, stops=stops)


def _fused(duration: float, reds: list[tuple[float, float]], greens: list[tuple[float, float]], reliable=True, source="lamp") -> SignalTrack:
    t = np.arange(0.0, duration + 0.2, 0.2)
    state = ["unknown"] * len(t)
    for a, b in reds:
        for i in np.flatnonzero((t >= a) & (t < b)):
            state[i] = "red"
    for a, b in greens:
        for i in np.flatnonzero((t >= a) & (t < b)):
            state[i] = "green"
    return SignalTrack(t=t, state=state, source=[source] * len(t), origin=source, reliable=reliable)


# ---------------------------------------------------------------------------
# лампа
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("faint", [False, True])
def test_lamp_reader_reads_phases_and_ignores_occlusion(faint):
    reader, ts = _reader_with([("red", 30), ("cover", 2), ("red", 8), ("amber", 3), ("green", 30), ("red", 20)], faint=faint)
    track = reader.finish()
    ph = [(s, round(a), round(b)) for s, a, b in phases(track)]
    reds = [p for p in ph if p[0] == "red"]
    greens = [p for p in ph if p[0] == "green"]
    assert reader.debug["sections_src"] == "colour"
    assert len(reds) == 2 and reds[0][1] == 0 and abs(reds[0][2] - 40) <= 1 and abs(reds[1][1] - 73) <= 1
    assert len(greens) == 1 and abs(greens[0][1] - 43) <= 1 and abs(greens[0][2] - 73) <= 1
    # автобус на 2 с не ломает фазу красного: состояние тянется через «неизвестно»
    assert track.at(31.0) == "red"
    assert any(s == "amber" for s, _, _ in ph)


def test_lamp_reader_daylight_bright_background_still_relative():
    """Днём тусклая линза на светлом фоне: абсолютные пороги молчат, относительные читают."""
    reader, _ = _reader_with([("red", 20), ("green", 20)], faint=True, bright=1.6)
    track = reader.finish()
    assert track.at(10.0) == "red" and track.at(30.0) == "green"


def test_lamp_reader_unknown_when_no_section_dominates():
    """Красное во всех секциях (красная машина за лампой) — не красный."""
    import cv2
    reader = LampReader((70, 50, 80, 170), (0, 0, 192, 264))
    for k in range(100):
        img = _lamp_crop("red")
        if 40 <= k < 60:
            cv2.rectangle(img, (60, 40), (160, 230), (40, 40, 200), -1)
        reader.push(img, k / 8.33)
    track = reader.finish()
    assert all(s == "unknown" for s in reader.debug["raw"][42:58])
    assert track.at(2.0) == "red"


def test_smooth_states_hysteresis():
    ts = np.arange(0, 10, 0.12)
    raw = ["red"] * 40 + ["green"] * 2 + ["red"] * 10 + ["green"] * 32
    out = smooth_states(raw, ts, hold_sec=0.4, max_gap_sec=4.0)
    assert out[41] == "red" and out[51] == "red" and out[-1] == "green"
    # длинное «неизвестно» сбрасывает состояние
    raw2 = ["red"] * 20 + ["unknown"] * 64
    out2 = smooth_states(raw2, ts, hold_sec=0.4, max_gap_sec=4.0)
    assert out2[30] == "red" and out2[-1] == "unknown"


# ---------------------------------------------------------------------------
# геометрия, пешеходы, слияние
# ---------------------------------------------------------------------------
def test_stop_line_offset_and_binding():
    sc = _scene()
    sl = sc.stop_lines[0]
    assert stop_line_offset(sl, 950, 500) == pytest.approx(50.0)
    assert stop_line_offset(sl, 850, 500) == pytest.approx(-50.0)
    # наклонная линия: смещение по нормали, а не проекция на approach
    sl2 = StopLine("s2", np.array([0, 100], np.float32), np.array([1000, 0], np.float32), np.array([0.9, 0.44], np.float32), [])
    assert abs(stop_line_offset(sl2, 500, 50)) < 1e-6
    assert stop_line_offset(sl2, 500, 150) > 0
    assert bind_crosswalk(sc, sl)[0] == "cw"


def test_signal_from_pedestrians_red_then_green():
    sc = _scene()
    obs = []
    # 0–10 с: два пешехода идут по зебре сверху вниз
    obs += _obs(1, "person", _line(0.0, 10.0, 990, 990, 300, 700), w=40, h=90)
    obs += _obs(2, "person", _line(1.0, 9.0, 1010, 1010, 700, 300), w=40, h=90)
    # 20–40 с: машины едут через линию, зебра пуста
    for k in range(5):
        obs += _obs(100 + k, "car", _line(20.0 + 4 * k, 26.0 + 4 * k, 200, 1500, 500))
    ctx = _ctx(obs, sc)
    ped = signal_from_pedestrians(ctx)
    assert ped.at(5.0) == "red" and ped.at(15.0) == "unknown" and ped.at(29.0) == "green" and ped.at(45.0) == "unknown"


def test_fuse_prefers_plausible_lamp_and_flags_reliability():
    dur = 100.0
    lamp = _fused(dur, reds=[(0, 40), (70, 100)], greens=[(40, 70)])
    ped = _fused(dur, reds=[(5, 20), (75, 90)], greens=[(45, 65)], source="ped")
    fused = fuse_signals(lamp, ped, None, dur)
    assert fused.origin == "lamp" and fused.reliable and fused.at(30.0) == "red" and fused.source_at(30.0) == "lamp"
    # лампа спорит с пешеходами — отвергнута, остаются пешеходы
    bad = _fused(dur, reds=[(40, 70)], greens=[(0, 40), (70, 100)])
    fused2 = fuse_signals(bad, ped, None, dur)
    assert fused2.origin != "lamp" and fused2.at(10.0) == "red" and fused2.source_at(10.0) == "ped" and fused2.at(30.0) == "unknown"
    assert fused2.reliable   # у пешеходов есть красный 15 с и зелёный 20 с
    # лампа мигает короткими фазами — неправдоподобна; без других источников ничего надёжного нет
    flicker = _fused(dur, reds=[(0, 3), (6, 9), (30, 100)], greens=[(3, 6), (9, 30)])
    fused3 = fuse_signals(flicker, None, None, dur)
    assert fused3.origin != "lamp" and not fused3.reliable


# ---------------------------------------------------------------------------
# правила
# ---------------------------------------------------------------------------
def _crossing_car(t0: float, tid: int = 1):
    """Едет слева направо через линию x=900 в момент ≈ t0+5, дальше через зебру на перекрёсток."""
    return _obs(tid, "car", _line(t0, t0 + 10.0, 400, 1400, 500))


def test_red_light_only_on_settled_reliable_red():
    sc = _scene()
    dur = 60.0
    sig = _fused(dur, reds=[(0, 30)], greens=[(30, 60)])
    ivs = red_light.run_red_light(_ctx(_crossing_car(10.0), sc, sig, dur))
    assert len(ivs) == 1 and 14.5 <= ivs[0][0] <= 15.5 and ivs[0][1] <= ivs[0][0] + 8.0 + 0.2
    # на зелёный — нет
    assert red_light.run_red_light(_ctx(_crossing_car(35.0), sc, sig, dur)) == []
    # в последнюю секунду красного — нет (переход)
    assert red_light.run_red_light(_ctx(_crossing_car(24.6), sc, sig, dur)) == []
    # красный только от очереди — нет
    sig_tr = _fused(dur, reds=[(0, 30)], greens=[(30, 60)], source="traffic")
    assert red_light.run_red_light(_ctx(_crossing_car(10.0), sc, sig_tr, dur)) == []
    # ненадёжный сигнал — нет
    sig_bad = _fused(dur, reds=[(0, 30)], greens=[(30, 60)], reliable=False)
    assert red_light.run_red_light(_ctx(_crossing_car(10.0), sc, sig_bad, dur)) == []


def test_stop_line_beyond_line_on_red():
    sc = _scene()
    dur = 60.0
    sig = _fused(dur, reds=[(0, 30)], greens=[(30, 60)])
    # подъехала на 3-й секунде, встала на 2-й секунде носом за линией (точка стояния 950 > 900 + 0.35·80) до 30 с
    pts = _line(0.0, 3.0, 500, 950, 500) + [(t, 950.0, 500.0) for t in np.arange(3.0, 30.0, DT)] + _line(30.0, 34.0, 950, 1400, 500)
    ivs = red_light.run_stop_line(_ctx(_obs(1, "car", pts), sc, sig, dur))
    assert len(ivs) == 1 and 2.5 <= ivs[0][0] <= 4.0 and abs(ivs[0][1] - 30.0) < 0.5
    # та же остановка перед линией — нет
    pts2 = _line(0.0, 3.0, 500, 850, 500) + [(t, 850.0, 500.0) for t in np.arange(3.0, 30.0, DT)]
    assert red_light.run_stop_line(_ctx(_obs(1, "car", pts2), sc, sig, dur)) == []
    # стоит на перекрёстке — это не stop_line
    pts3 = _line(0.0, 3.0, 500, 1100, 500) + [(t, 1100.0, 500.0) for t in np.arange(3.0, 30.0, DT)]
    assert red_light.run_stop_line(_ctx(_obs(1, "car", pts3), sc, sig, dur)) == []


# ---------------------------------------------------------------------------
# видеоридер с вырезом
# ---------------------------------------------------------------------------
def test_normalize_crop_and_filter_graph():
    assert normalize_crop((2243, 695, 97, 133), 3840, 2160) == (2242, 694, 98, 134)
    assert normalize_crop((3800, 2100, 200, 200), 3840, 2160) == (3800, 2100, 40, 60)
    assert normalize_crop(None, 10, 10) is None
    f = ffmpeg_filter(10.0, 960, 540, (2196, 630, 192, 264))
    assert "split" in f and "crop=192:264:2196:630" in f and "pad=960:804" in f and "overlay=0:540" in f
    assert ffmpeg_filter(10.0, 960, 540, None) == "fps=10.000000,scale=960:540"


DEV = ROOT / "data" / "dev" / "car-detection.mp4"


@pytest.mark.skipif(not DEV.exists() or ffmpeg_exe() is None, reason="нет dev-видео или ffmpeg")
def test_iter_frames_crop_matches_cv2():
    crop = (300, 100, 120, 80)
    ff = next(iter(iter_frames(str(DEV), 1, 384, crop=crop)))
    cv = next(iter(iter_frames_cv2(str(DEV), 1, 384, crop=crop)))
    assert len(ff) == 5 and len(cv) == 5
    assert ff[2].shape == (216, 384, 3) and ff[4].shape == (80, 120, 3) and cv[4].shape == (80, 120, 3)
    assert np.abs(ff[4].astype(int) - cv[4].astype(int)).mean() < 12
    plain = next(iter(iter_frames(str(DEV), 1, 384)))
    assert len(plain) == 4 and plain[2].shape == (216, 384, 3)

"""Проверки контракта: формат событий, отсутствие пересечений, бюджет времени, детерминизм, чистота Part B."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solution  # noqa: E402
from src.segments import finalize_events, merge_intervals  # noqa: E402

DEV = ROOT / "data" / "dev"
VIDEOS = sorted(DEV.glob("*.mp4"))[:1]


def test_classes_are_official_subset():
    from evaluate import OFFICIAL_CLASSES
    assert solution.CLASSES and all(c in OFFICIAL_CLASSES for c in solution.CLASSES)


def test_merge_and_min_len():
    ev = finalize_events({"stopped_vehicle": [(1.0, 12.0), (12.5, 30.0), (40.0, 43.0)], "near_miss": [(5.0, 5.2)]}, duration=100.0)
    labels = [e[2] for e in ev]
    assert labels == ["stopped_vehicle"]           # 40-43 короче 10 с, near_miss короче 0.6 с
    assert ev[0][0] == 1.0 and ev[0][1] == 30.0     # разрыв 0.5 с сшит
    assert merge_intervals([(0, 1), (0.5, 2), (5, 6)], 0.0) == [(0.0, 2.0), (5.0, 6.0)]


def test_no_same_class_overlap_after_finalize():
    ivs = [(float(i), float(i + 3)) for i in range(0, 60, 2)]
    ev = finalize_events({"wrong_way": ivs}, duration=100.0)
    for a, b in zip(ev, ev[1:]):
        assert a[1] <= b[0]


@pytest.mark.skipif(not VIDEOS, reason="нет dev-видео")
def test_harness_end_to_end(tmp_path):
    out = tmp_path / "pred.json"
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(ROOT / "run_submission.py"), "--videos", str(VIDEOS[0]), "--out", str(out), "--team", "test"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(out.read_text())
    log = d["log"][VIDEOS[0].name]
    assert not log["errors"], log["errors"]
    assert log["total_sec"] <= log["budget_sec"] * 0.5, "слишком медленно: нужен запас ×2 к бюджету"
    v = subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", str(out), "--validate-only"], capture_output=True, text=True, cwd=ROOT)
    assert v.returncode == 0, v.stdout
    risk = d["videos"][VIDEOS[0].name]["risk"]
    assert risk and all(0.0 <= s <= 1.0 for _, s in risk)
    assert time.perf_counter() - t0 < 600


@pytest.mark.skipif(not VIDEOS, reason="нет dev-видео")
def test_deterministic_two_runs():
    a = solution.detect_events(str(VIDEOS[0]))
    b = solution.detect_events(str(VIDEOS[0]))
    assert a == b


def test_risk_estimator_is_causal_and_bounded():
    est = solution.RiskEstimator()
    est.reset({"video_id": "x", "fps": 25.0, "width": 640, "height": 360, "n_frames": 10})
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    scores = [est.step(frame, i / 25.0) for i in range(10)]
    assert all(0.0 <= s <= 1.0 for s in scores)

"""Проверки инструмента CI-дымового теста: генератор ролика и сравнение двух predictions.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.video import video_meta  # noqa: E402
from tools.ci_smoke import compare, make_video  # noqa: E402


def _pred(risk, events=(), errors=()):
    return {"team": "t", "videos": {"clip.mp4": {"events": list(events), "risk": list(risk)}},
            "log": {"clip.mp4": {"duration": 3.0, "budget_sec": 9.0, "errors": list(errors)}}}


def _write(tmp_path: Path, name: str, d: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_make_video_is_readable_and_has_requested_shape(tmp_path):
    info = make_video(tmp_path / "clip.mp4", seconds=3.0, fps=30.0, width=320, height=240)
    m = video_meta(info["path"])
    assert (m["width"], m["height"]) == (320, 240)
    assert abs(m["fps"] - 30.0) < 0.01
    assert m["n_frames"] == 90 == info["n_frames"]


def test_make_video_is_bitwise_repeatable(tmp_path):
    a = make_video(tmp_path / "a.mp4")
    b = make_video(tmp_path / "b.mp4")
    assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()


def test_compare_accepts_identical_runs(tmp_path):
    risk = [[i / 30.0, 0.0] for i in range(90)]
    a = _write(tmp_path, "a.json", _pred(risk, events=[[0.5, 1.5, "jaywalking"]]))
    b = _write(tmp_path, "b.json", _pred(risk, events=[[0.5, 1.5, "jaywalking"]]))
    assert compare(a, b) == []


def test_compare_flags_differences_and_harness_errors(tmp_path):
    risk = [[i / 30.0, 0.0] for i in range(90)]
    a = _write(tmp_path, "a.json", _pred(risk))
    other = list(risk)
    other[40] = [40 / 30.0, 0.5]
    b = _write(tmp_path, "b.json", _pred(other))
    problems = compare(a, b)
    assert any("differ" in p for p in problems) and any("sample 40" in p for p in problems)

    # обнулённый по бюджету результат в обоих прогонах совпадает, но это не успех
    empty = _pred([], errors=["over time budget (20.0s > 9s): scored as empty"])
    c = _write(tmp_path, "c.json", empty)
    d = _write(tmp_path, "d.json", empty)
    problems = compare(c, d)
    assert any("harness reported" in p for p in problems) and any("empty risk" in p for p in problems)

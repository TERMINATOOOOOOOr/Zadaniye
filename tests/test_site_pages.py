"""Сайт команды: страницы отвечают 200, данные разбора ошибок и абляций корректны, расчёты панели верны."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
if str(SITE) not in sys.path:
    sys.path.insert(0, str(SITE))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
import dashboard  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from app import app  # noqa: WPS433 — импорт здесь, чтобы sys.path уже был настроен
    return TestClient(app)  # без lifespan: очередь задач страницам не нужна


def test_pages_return_200(client):
    for href, _label in config.NAV:
        r = client.get(href)
        assert r.status_code == 200, href
        assert "<main" in r.text, href
    assert ("/analysis", "Analysis") in config.NAV and ("/dashboard", "Dashboard") in config.NAV


def test_no_external_resources(client):
    for href in ("/analysis", "/dashboard", "/results"):
        html = client.get(href).text
        assert "https://cdn" not in html and "googleapis" not in html and "unpkg" not in html, href


def test_analysis_json_and_sheets(client):
    data = json.loads(config.ANALYSIS_JSON.read_text(encoding="utf-8"))
    entries = data["entries"]
    assert len(entries) >= 10
    for e in entries:
        assert e["verdict"] in ("true_positive", "false_positive", "unclear"), e["image"]
        assert e["cls"] in config.CLASSES, e["image"]
        assert e["why"] and e["learned"]
        img = config.STATIC_DIR / e["image"]
        assert img.is_file(), e["image"]
        assert img.stat().st_size <= 300 * 1024, e["image"]
        from PIL import Image  # noqa: WPS433
        with Image.open(img) as im:
            assert im.width <= 1400, e["image"]
    html = client.get("/analysis").text
    assert html.count("analysis-card") >= len(entries)
    assert "C3897_acc_315.jpg" in html and "false positive" in html


def test_ablation_json_and_rendering(client):
    data = json.loads(config.ABLATION_JSON.read_text(encoding="utf-8"))
    rows = data["rows"]
    keys = {"config", "video", "part_a_sec", "part_b_sec", "total_x_duration", "events_per_class", "notes", "status"}
    for r in rows:
        assert keys <= set(r), r
        assert r["status"] in ("measured", "pending")
        if r["status"] == "measured" and r["part_a_sec"] is not None:
            assert r["part_a_sec"] > 0 and r["part_b_sec"] > 0 and 0 < r["total_x_duration"] < 3
    n_measured = sum(r["status"] == "measured" for r in rows)
    assert n_measured >= 4
    assert sum(r["status"] == "pending" for r in rows) >= 1
    html = client.get("/results").text
    assert 'id="ablations"' in html
    assert html.count('<tr class="pending">') == sum(r["status"] == "pending" for r in rows)
    assert html.count('<tr class="measured">') == n_measured


def test_events_per_minute_bins():
    events = [[0.0, 5.0, "jaywalking"], [59.9, 61.0, "jaywalking"], [60.0, 70.0, "accident"], [150.0, 151.0, "near_miss"]]
    pm = dashboard.events_per_minute(events, duration=125.0)
    assert pm["minutes"] == 3
    assert pm["classes"] == ["accident", "near_miss", "jaywalking"]  # официальный порядок классов
    assert pm["bins"][0] == [0, 0, 2]      # две «прогулки» начались в нулевой минуте
    assert pm["bins"][1] == [1, 0, 0]
    assert pm["bins"][2] == [0, 1, 0]      # событие за пределами длительности — в последнюю минуту
    assert pm["max"] == 2


def test_alarm_runs():
    dt = 0.2
    scores = [0.0] * 10 + [0.6, 0.7, 0.5] + [0.1] * 5 + [0.9] + [0.0] * 3 + [0.55, 0.51]
    risk = [[round(i * dt, 3), s] for i, s in enumerate(scores)]
    a = dashboard.alarm_runs(risk, threshold=0.5)
    assert a["count"] == 3
    assert a["runs"][0]["start"] == pytest.approx(2.0) and a["runs"][0]["length"] == pytest.approx(0.6)
    assert a["runs"][1]["length"] == pytest.approx(dt)          # одиночный отсчёт = один шаг
    assert a["runs"][2]["end"] == pytest.approx(len(scores) * dt)  # серия до конца кривой
    assert a["runs"][0]["peak"] == pytest.approx(0.7)
    assert a["total_sec"] == pytest.approx(0.6 + 0.2 + 0.4)
    assert a["share"] == pytest.approx(6 / len(scores))
    assert dashboard.alarm_runs([])["count"] == 0


def test_dashboard_page_matches_results(client):
    data = json.loads(config.RESULTS_JSON.read_text(encoding="utf-8"))
    dash = dashboard.build(data)
    assert len(dash["videos"]) == len(data["videos"])
    for v, raw in zip(dash["videos"], data["videos"]):
        assert v["n_events"] == len(raw["events"])
        assert sum(sum(row) for row in v["per_minute"]["bins"]) == len(raw["events"])
        assert len(v["strip"]["segs"]) == len(raw["events"])
    assert dash["totals"]["grand_total"] == sum(len(v["events"]) for v in data["videos"])
    html = client.get("/dashboard").text
    assert html.count('class="svg-chart"') == len(data["videos"])
    assert html.count('class="svg-strip"') == len(data["videos"])
    assert "<script src=" not in html.split("<main")[1].split("</main>")[0]

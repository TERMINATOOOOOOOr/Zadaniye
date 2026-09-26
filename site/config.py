"""Настройки сайта. Всё, что захочется переименовать или подправить, живёт здесь.

Остальной код (app.py, bridge.py, шаблоны) читает значения отсюда.
"""
from __future__ import annotations

from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SITE_DIR.parent  # корень репозитория (там лежат solution.py и src/)

# --- Название и ссылки --------------------------------------------------------
SITE_NAME = "TrafficEye"
TAGLINE = "Traffic event detection from a fixed road camera"
TEAM_NAME = "Air MAX"
REPO_URL = "https://github.com/TERMINATOOOOOOOr/Zadaniye"
WEIGHTS_URL = "https://github.com/TERMINATOOOOOOOr/Zadaniye/tree/main/weights"
SITE_SOURCE_URL = f"{REPO_URL}/tree/main/site"
PREDICTIONS_SAMPLES = ROOT_DIR / "predictions_samples.json"

# --- Пути к данным сайта --------------------------------------------------------
CONTENT_DIR = SITE_DIR / "content"
STATIC_DIR = SITE_DIR / "static"
TEMPLATES_DIR = SITE_DIR / "templates"
EDA_JSON = STATIC_DIR / "eda" / "eda.json"
RESULTS_JSON = STATIC_DIR / "results" / "results.json"
EXAMPLES_DIR = STATIC_DIR / "results" / "examples"
ABLATION_JSON = STATIC_DIR / "results" / "ablation.json"  # таблица абляций на странице Results
ANALYSIS_JSON = STATIC_DIR / "analysis" / "analysis.json"  # разбор ошибок: контакт-листы + вердикты
# аннотированные ролики: корневой .gitignore не коммитит *.mp4, кроме site/static/media/*.mp4,
# поэтому ожидаем их здесь; results.json может и явно указать путь (относительно site/static/)
MEDIA_DIR = STATIC_DIR / "media"
ANNOTATED_DIRS = ("media", "results")  # где искать <name>_annotated.mp4, если путь не указан

# --- Демо / очередь задач ---------------------------------------------------
MAX_UPLOAD_MB = 10240  # 10 ГБ: 4K с этой камеры (147 Мбит/с) такой длины ≈ 9 минут
MAX_DURATION_SEC = 600  # 10 минут; сэмплы организаторов длятся до 340 с, YouTube качаем в 1080p
JOBS_DIR = SITE_DIR / "jobs_store"  # временные файлы задач (в .gitignore)
JOB_TTL_SEC = 2 * 3600  # задачи старше этого удаляются
JOB_CLEANUP_INTERVAL_SEC = 300
RISK_FRAME_STRIDE = 4  # каждый N-й кадр идёт в RiskEstimator.step (демо на CPU)
FULL_PRESET_MAX_SEC = 60  # клипы не длиннее этого гоняем с настройками сдачи (1280 px, каждый 3-й кадр, риск как в харнессе);
                          # длинные — с быстрым пресетом из переменных окружения (CPU)
FULL_PRESET_ENV = {"WIUT_IMGSZ_A": "1280", "WIUT_STRIDE_A": "3", "WIUT_IMGSZ_B": "640", "WIUT_STRIDE_B": "1"}
ANNOTATE_MAX_SIDE = 960  # длинная сторона аннотированного видео
API_WAIT_TIMEOUT_SEC = 900  # сколько /api/detect ждёт результат синхронно
EXPECTED_DETECT_FACTOR = 1.0  # оценка: detect_events ~ N x длительность видео (для прогресс-бара)
RISK_POINTS_FOR_UI = 1500  # кривая риска прореживается до этого числа точек на страницах

# --- Классы событий (официальный список, порядок фиксирован) ------------------
# Цвета подобраны под тёмный фон; подпись класса всегда выводится текстом рядом
# с цветом, так что цвет никогда не является единственным носителем смысла.
CLASS_INFO: dict[str, dict[str, str]] = {
    "accident":            {"desc": "Collision between road users or with a fixed object", "color": "#e04848"},
    "near_miss":           {"desc": "Sharp braking or swerving to avoid a collision, no contact", "color": "#e0862a"},
    "red_light":           {"desc": "Crossing the stop line on red", "color": "#d55181"},
    "wrong_way":           {"desc": "Driving against the traffic direction / in the oncoming lane", "color": "#9085e9"},
    "illegal_u_turn":      {"desc": "U-turn where prohibited", "color": "#3987e5"},
    "stopped_vehicle":     {"desc": "Stationary on the carriageway for 10 s or more, not queued at a signal", "color": "#c9a11c"},
    "jaywalking":          {"desc": "Pedestrian on the carriageway outside a crossing", "color": "#3bb54a"},
    "failure_to_yield":    {"desc": "Driving through a crossing while a pedestrian is on it", "color": "#1a9aa8"},
    "illegal_turn":        {"desc": "Turn from the wrong lane or in a prohibited direction", "color": "#a4b83a"},
    "solid_line_crossing": {"desc": "Lane change or manoeuvre across a solid marking", "color": "#35a9c9"},
    "stop_line":           {"desc": "Stopped past the stop line on red", "color": "#de8fbf"},
    "congestion":          {"desc": "Standstill or crawling traffic across all lanes of a direction", "color": "#8a9bc4"},
    "road_obstacle":       {"desc": "Debris, animal or fallen object on the carriageway", "color": "#a9713a"},
    "fire_smoke":          {"desc": "Visible fire or smoke from a vehicle or on the road", "color": "#f06a5a"},
}
CLASSES: list[str] = list(CLASS_INFO)

# --- Навигация ------------------------------------------------------------------
NAV: list[tuple[str, str]] = [
    ("/", "Home"),
    ("/team", "Team"),
    ("/approach", "Approach"),
    ("/eda", "EDA"),
    ("/results", "Results"),
    ("/analysis", "Analysis"),
    ("/dashboard", "Dashboard"),
    ("/demo", "Demo"),
    ("/report", "Report"),
    ("/links", "Links"),
]

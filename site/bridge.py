"""Обёртки над пайплайном команды (solution.py, src/render.py).

Все тяжёлые модули импортируются лениво внутри функций, чтобы сайт поднимался
даже без них (страницы команды/подхода/отчёта не зависят от ML-зависимостей).

Контракт, на который рассчитан сайт:

    solution.detect_events(video_path) -> [[start_sec, end_sec, label], ...]
    solution.RiskEstimator().reset(meta); .step(frame_bgr, t_sec) -> float
    src.render.annotate_video(video_path, events, out_path, risk=None, max_side=960) -> str

Необязательно: если ``detect_events`` принимает именованный аргумент
``progress_cb`` (или ``progress`` / ``on_progress``), сайт передаст туда функцию
``cb(fraction: float, stage: str | None = None)`` с долей выполнения 0..1.
Иначе прогресс части A оценивается по времени.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import config

log = logging.getLogger("site.pipeline")

RISK_META_KEYS = ("video_id", "fps", "width", "height", "n_frames")


def ensure_root_on_path() -> None:
    root = str(config.ROOT_DIR)
    if root not in sys.path:
        sys.path.append(root)


def _spec_ok(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — битые пакеты не должны ронять healthz
        return False


def availability() -> dict[str, bool]:
    """Дешёвая проверка (без импорта тяжёлых модулей), что есть на сервере."""
    ensure_root_on_path()
    return {
        "cv2": _spec_ok("cv2"),
        "solution": _spec_ok("solution"),
        "render": _spec_ok("src.render"),
    }


def has_cv2() -> bool:
    return _spec_ok("cv2")


# --- видео ------------------------------------------------------------------

def probe_video(path: str | Path) -> dict[str, Any]:
    """Метаданные видео через OpenCV: fps, размер, число кадров, длительность."""
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is not installed on the server") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("could not open the video (corrupted file or unsupported codec)")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or n_frames <= 0:
        raise RuntimeError("the video has no frames or no frame rate")
    return {
        "video_id": Path(path).name,
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration": n_frames / fps,
    }


# --- solution.py --------------------------------------------------------------

def load_solution():
    ensure_root_on_path()
    try:
        return importlib.import_module("solution")
    except ImportError as exc:
        raise RuntimeError(
            "solution.py is not available to the site yet (expected in the repository root)"
        ) from exc


def _progress_kwarg(fn: Callable) -> str | None:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    for name in ("progress_cb", "progress", "on_progress", "callback"):
        if name in params:
            return name
    return None


def clean_events(events: Any, duration: float, classes: list[str] | None = None) -> tuple[list, list[str]]:
    """Та же логика, что в run_submission.py: приводим типы и выбрасываем невалидное."""
    classes = classes or config.CLASSES
    out: list[list] = []
    problems: list[str] = []
    if events is None:
        return [], ["detect_events returned None (treated as [])"]
    for i, ev in enumerate(events):
        try:
            s, e, label = ev
            s, e, label = float(s), float(e), str(label)
        except Exception:  # noqa: BLE001
            problems.append(f"event {i} dropped: not [start, end, label]: {ev!r}")
            continue
        if label not in classes:
            problems.append(f"event {i} dropped: label {label!r} is not in the official class list")
            continue
        e = min(e, duration)
        if not (0.0 <= s < e):
            problems.append(f"event {i} dropped: bad times [{s}, {e}] for duration {duration:.2f}")
            continue
        out.append([round(s, 3), round(e, 3), label])
    out.sort(key=lambda x: (x[2], x[0]))
    kept: list[list] = []
    last_end: dict[str, float] = {}
    for s, e, label in out:
        if s < last_end.get(label, -1.0):
            problems.append(f"dropped {label} [{s}, {e}]: overlaps an earlier {label} segment")
            continue
        kept.append([s, e, label])
        last_end[label] = e
    kept.sort()
    return kept, problems


def detect_events(path: str | Path, meta: dict[str, Any], on_progress: Callable[[float, str | None], None] | None = None) -> list:
    """Часть A. Прогресс — через колбэк пайплайна, если он его принимает, иначе по времени."""
    sol = load_solution()
    fn = sol.detect_events
    kwarg = _progress_kwarg(fn)
    stop = threading.Event()
    if kwarg is None and on_progress is not None:
        expected = max(5.0, float(meta.get("duration", 0.0)) * config.EXPECTED_DETECT_FACTOR)
        t0 = time.time()

        def ticker() -> None:
            while not stop.wait(1.0):
                on_progress(min((time.time() - t0) / expected, 0.97), None)

        threading.Thread(target=ticker, name="detect-progress", daemon=True).start()
    kwargs = {kwarg: on_progress} if (kwarg and on_progress) else {}
    try:
        return fn(str(path), **kwargs)
    finally:
        stop.set()


def risk_curve(path: str | Path, meta: dict[str, Any], on_progress: Callable[[float], None] | None = None,
               stride: int | None = None) -> list[list[float]]:
    """Часть B: прогоняем кадры через RiskEstimator так же, как харнесс организаторов."""
    sol = load_solution()
    estimator_cls = getattr(sol, "RiskEstimator", None)
    if estimator_cls is None:
        return []
    import cv2  # noqa: WPS433

    stride = max(1, int(stride or config.RISK_FRAME_STRIDE))
    est = estimator_cls()
    est.reset({k: meta[k] for k in RISK_META_KEYS})
    cap = cv2.VideoCapture(str(path))
    fps = float(meta["fps"])
    n_frames = max(1, int(meta.get("n_frames") or 1))
    curve: list[list[float]] = []
    idx, last = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if idx % stride == 0:
                try:
                    last = min(1.0, max(0.0, float(est.step(frame, t))))
                except Exception:  # noqa: BLE001 — одна плохая оценка не должна ронять весь прогон
                    last = last
                curve.append([round(t, 3), round(last, 4)])
            idx += 1
            if on_progress and idx % 50 == 0:
                on_progress(min(idx / n_frames, 1.0))
    finally:
        cap.release()
    return curve


def downsample_risk(curve: list[list[float]], max_points: int | None = None) -> list[list[float]]:
    """Прореживание кривой для страниц: в каждой корзине берём максимум (пики не теряются)."""
    max_points = max_points or config.RISK_POINTS_FOR_UI
    n = len(curve)
    if n <= max_points:
        return curve
    bucket = n / max_points
    out: list[list[float]] = []
    i = 0.0
    while int(i) < n:
        chunk = curve[int(i): max(int(i) + 1, int(i + bucket))]
        best = max(chunk, key=lambda p: p[1])
        out.append([chunk[0][0], best[1]])
        i += bucket
    return out


# --- рендер ------------------------------------------------------------------

def annotate(path: str | Path, events: list, out_path: str | Path, risk: list | None = None,
             on_progress: Callable[[float], None] | None = None) -> str:
    ensure_root_on_path()
    try:
        from src.render import annotate_video  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError("src/render.py is not available yet") from exc
    kwargs: dict[str, Any] = {}
    if on_progress is not None:
        name = _progress_kwarg(annotate_video)
        if name:
            kwargs[name] = on_progress
    return annotate_video(str(path), events, str(out_path), risk=risk or None,
                          max_side=config.ANNOTATE_MAX_SIDE, **kwargs)


# --- полный прогон одной задачи -----------------------------------------------

def run_job(job) -> dict[str, Any]:
    """Обработчик для JobQueue: вход ``job.dir/input.mp4`` -> events.json (+ annotated.mp4)."""
    src = job.dir / "input.mp4"
    notes: list[str] = []
    want_risk = bool(job.options.get("risk", True))
    want_annotate = bool(job.options.get("annotate", True))

    job.set_progress(2, "reading metadata")
    meta = dict(job.meta or probe_video(src))
    meta["video_id"] = job.video_name or meta.get("video_id", "input.mp4")
    job.meta = meta
    duration = float(meta["duration"])

    # Часть A: 5 -> 60
    job.set_progress(5, "detecting events (Part A)")

    def on_detect(frac: float, stage: str | None = None) -> None:
        job.set_progress(5 + 55 * max(0.0, min(1.0, float(frac))), stage)

    t0 = time.time()
    raw_events = detect_events(src, meta, on_detect)
    events, problems = clean_events(raw_events, duration)
    notes += problems
    part_a = time.time() - t0
    job.set_progress(60, "detection finished")

    # Часть B: 60 -> 85
    risk: list[list[float]] = []
    part_b = 0.0
    if want_risk:
        job.set_progress(61, "estimating risk (Part B)")
        t1 = time.time()
        try:
            risk = risk_curve(src, meta, lambda f: job.set_progress(60 + 25 * f))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"risk curve not built: {type(exc).__name__}: {exc}")
            risk = []
        part_b = time.time() - t1
        if not risk:
            notes.append("RiskEstimator is missing or returned an empty curve")
    job.set_progress(85, "post-processing")

    # Рендер: 86 -> 98
    annotated: str | None = None
    if want_annotate:
        job.set_progress(86, "rendering the annotated video")

        def on_render(*args: Any) -> None:
            # принимаем progress(frac), progress(i, n), progress("стадия", frac) и проценты
            nums = [float(a) for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
            stage = next((a for a in args if isinstance(a, str)), None)
            if not nums:
                return
            frac = nums[0] / nums[1] if len(nums) >= 2 and nums[1] else nums[0]
            if frac > 1.0:
                frac /= 100.0
            job.set_progress(86 + 12 * max(0.0, min(1.0, frac)), stage or None)

        try:
            annotated = annotate(src, events, job.dir / "annotated.mp4", risk, on_progress=on_render)
            if not Path(annotated).exists():
                annotated = None
                notes.append("annotate_video did not create a file")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"annotated video unavailable: {exc}")
            annotated = None
    job.set_progress(98, "saving the result")

    payload = {
        "video": job.video_name,
        "duration": round(duration, 3),
        "meta": {k: meta[k] for k in RISK_META_KEYS},
        "events": events,
        "risk": risk,
        "timing_sec": {"part_a": round(part_a, 1), "part_b": round(part_b, 1)},
        "notes": notes,
    }
    (job.dir / "events.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    base = f"/jobs/{job.id}"
    return {
        "video": job.video_name,
        "duration": round(duration, 3),
        "meta": payload["meta"],
        "events": events,
        "risk": downsample_risk(risk),
        "n_risk": len(risk),
        "annotated": annotated is not None,
        "annotated_url": f"{base}/annotated.mp4" if annotated else None,
        "input_url": f"{base}/input.mp4",
        "events_url": f"{base}/events.json",
        "timing_sec": payload["timing_sec"],
        "notes": notes,
    }

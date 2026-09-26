"""Сайт команды: FastAPI-приложение.

Запуск из корня репозитория:

    uvicorn app:app --app-dir site --reload

(«site.app:app» не работает: стандартная библиотека Python уже занимает имя
``site``, поэтому каталог подключается через ``--app-dir``.)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SITE_DIR = Path(__file__).resolve().parent
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))
ROOT_DIR = SITE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

import config  # noqa: E402
import mdlite  # noqa: E402
import bridge  # noqa: E402  — обёртки над solution.py / src.render (ленивые импорты)
import dashboard  # noqa: E402
from jobs import Job, JobQueue  # noqa: E402
import fetch  # noqa: E402
from pydantic import BaseModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("site")

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
CHUNK = 1 << 20
JOB_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")
JOB_FILES = {"annotated.mp4": "video/mp4", "input.mp4": "video/mp4", "events.json": "application/json"}
VERDICT_LABELS = {"true_positive": "true positive", "false_positive": "false positive", "unclear": "unclear"}
ABLATION_STATUSES = ("measured", "pending")

FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='#121722'/>"
    "<path d='M8 32c6-11 14-16 24-16s18 5 24 16c-6 11-14 16-24 16S14 43 8 32z' fill='none' stroke='#5b9cff' stroke-width='4'/>"
    "<circle cx='32' cy='32' r='8' fill='#5b9cff'/><circle cx='35' cy='29' r='2.5' fill='#121722'/></svg>"
)

queue = JobQueue(
    config.JOBS_DIR,
    worker=bridge.run_job,
    ttl_sec=config.JOB_TTL_SEC,
    cleanup_interval_sec=config.JOB_CLEANUP_INTERVAL_SEC,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    queue.start()
    log.info("%s started; pipeline: %s", config.SITE_NAME, bridge.availability())
    yield
    queue.stop()


# Swagger UI отключён: он подгружает скрипты с CDN, а сайт живёт без внешних ресурсов. Схема — /api/openapi.json.
app = FastAPI(title=config.SITE_NAME, docs_url=None, redoc_url=None, openapi_url="/api/openapi.json",
              lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def fmt_time(value: Any) -> str:
    try:
        s = float(value)
    except (TypeError, ValueError):
        return "—"
    m = int(s // 60)
    return f"{m}:{s - 60 * m:04.1f}"


templates.env.filters["fmt_time"] = fmt_time
templates.env.globals.update(
    site={
        "name": config.SITE_NAME,
        "tagline": config.TAGLINE,
        "team": config.TEAM_NAME,
        "repo": config.REPO_URL,
        "weights": config.WEIGHTS_URL,
        "source": config.SITE_SOURCE_URL,
    },
    nav=config.NAV,
    class_info=config.CLASS_INFO,
    classes=config.CLASSES,
    year=datetime.now().year,
)


def render(request: Request, name: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
    ctx.setdefault("active", request.url.path)
    return templates.TemplateResponse(request=request, name=name, context=ctx, status_code=status_code)


def read_json(path: Path) -> Any:
    """None, если файла нет; dict с ключом __error__, если он битый."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", path, exc)
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def read_md(name: str) -> str:
    path = config.CONTENT_DIR / name
    if not path.exists():
        return f"<p class='muted'>File <code>site/content/{name}</code> not found.</p>"
    return mdlite.convert(path.read_text(encoding="utf-8"))


def static_url(rel: str | None) -> str | None:
    if not rel:
        return None
    if rel.startswith(("http://", "https://", "/")):
        return rel
    return f"/static/{rel}"


def find_annotated(video_name: str | None, given: str | None) -> str | None:
    """Путь к аннотированному ролику: явный из results.json, если файл есть, иначе
    <static>/media/<stem>_annotated.mp4 или <static>/results/<stem>_annotated.mp4."""
    if given and (given.startswith(("http://", "https://", "/")) or (config.STATIC_DIR / given).is_file()):
        return static_url(given)
    stem = Path(video_name or "").stem
    if stem:
        for d in config.ANNOTATED_DIRS:
            rel = f"{d}/{stem}_annotated.mp4"
            if (config.STATIC_DIR / rel).is_file():
                return static_url(rel)
    return static_url(given)  # пусть браузер покажет 404 — лучше, чем молча спрятать плеер


# --- страницы ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render(request, "index.html", avail=bridge.availability(),
                  has_results=config.RESULTS_JSON.exists(), has_eda=config.EDA_JSON.exists(),
                  max_mb=config.MAX_UPLOAD_MB, max_sec=config.MAX_DURATION_SEC)


@app.get("/team", response_class=HTMLResponse)
async def team(request: Request):
    data = read_json(config.CONTENT_DIR / "team.json")
    members: list[dict] = []
    error = None
    if isinstance(data, dict) and "__error__" in data:
        error = data["__error__"]
    elif isinstance(data, dict):
        members = list(data.get("members", []))
    elif isinstance(data, list):
        members = data
    return render(request, "team.html", members=members, error=error)


@app.get("/approach", response_class=HTMLResponse)
async def approach(request: Request):
    return render(request, "approach.html", body=read_md("approach.md"))


@app.get("/eda", response_class=HTMLResponse)
async def eda(request: Request):
    data = read_json(config.EDA_JSON)
    error = data.get("__error__") if isinstance(data, dict) else None
    videos = [] if (data is None or error) else list(data.get("videos", []))
    for v in videos:
        v["images"] = {k: static_url(p) for k, p in (v.get("images") or {}).items() if p}
        cot = v.get("counts_over_time") or {}
        v["series_names"] = [k for k in cot if k != "t"]
    return render(request, "eda.html", videos=videos, error=error, present=data is not None)


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    data = read_json(config.RESULTS_JSON)
    error = data.get("__error__") if isinstance(data, dict) else None
    videos = [] if (data is None or error) else list(data.get("videos", []))
    failures = [] if (data is None or error) else list(data.get("failures", []))
    for v in videos:
        v["src"] = find_annotated(v.get("name"), v.get("annotated_mp4"))
        events = v.get("events") or []
        v["events"] = events
        risk = v.get("risk") or []
        v["risk"] = bridge.downsample_risk(risk)
        if not v.get("duration"):
            ends = [float(e[1]) for e in events if len(e) >= 2] + [float(p[0]) for p in risk[-1:]]
            v["duration"] = max(ends) if ends else 0.0
        v["labels"] = [c for c in config.CLASSES if any(len(e) >= 3 and e[2] == c for e in events)]
    examples = []
    if config.EXAMPLES_DIR.exists():
        for p in sorted(config.EXAMPLES_DIR.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            cls = next((c for c in config.CLASSES if p.stem.lower().startswith(c)), None)
            examples.append({"url": f"/static/results/examples/{p.name}", "name": p.stem, "cls": cls})
    return render(request, "results.html", videos=videos, failures=failures, examples=examples,
                  error=error, present=data is not None, ablation=load_ablation())


def load_ablation() -> dict[str, Any]:
    """Таблица абляций для страницы Results: строки со status measured/pending (pending рисуются серым)."""
    data = read_json(config.ABLATION_JSON)
    if data is None:
        return {"present": False, "rows": [], "note": "", "error": None}
    if isinstance(data, dict) and "__error__" in data:
        return {"present": True, "rows": [], "note": "", "error": data["__error__"]}
    rows = []
    for r in (data.get("rows") if isinstance(data, dict) else data) or []:
        if not isinstance(r, dict):
            continue
        status = r.get("status") if r.get("status") in ABLATION_STATUSES else "pending"
        epc = r.get("events_per_class") or {}
        rows.append({
            "config": r.get("config", "—"), "video": r.get("video", "—"), "status": status,
            "duration_sec": r.get("duration_sec"), "part_a_sec": r.get("part_a_sec"), "part_b_sec": r.get("part_b_sec"),
            "total_x_duration": r.get("total_x_duration"), "notes": r.get("notes", ""),
            "events_per_class": [(c, epc[c]) for c in config.CLASSES if c in epc]
            + sorted((c, n) for c, n in epc.items() if c not in config.CLASSES),
        })
    return {"present": True, "rows": rows, "note": (data.get("note", "") if isinstance(data, dict) else ""), "error": None}


@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    """Разбор ошибок: контакт-листы кандидатов с вердиктом и выводом."""
    data = read_json(config.ANALYSIS_JSON)
    error = data.get("__error__") if isinstance(data, dict) else None
    raw = [] if (data is None or error) else list(data.get("entries", []) if isinstance(data, dict) else data)
    entries = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict) or not e.get("image"):
            continue
        verdict = e.get("verdict") if e.get("verdict") in VERDICT_LABELS else "unclear"
        t = e.get("t")
        entries.append({
            "id": f"sheet-{i + 1}", "url": static_url(e["image"]), "cls": e.get("cls", "—"), "video": e.get("video", "—"),
            "t": float(t) if isinstance(t, (int, float)) else None, "verdict": verdict,
            "why": e.get("why", ""), "learned": e.get("learned", ""), "in_output": bool(e.get("in_output")),
        })
    counts = {k: sum(1 for e in entries if e["verdict"] == k) for k in VERDICT_LABELS}
    return render(request, "analysis.html", entries=entries, counts=counts, verdict_labels=VERDICT_LABELS,
                  in_output=sum(1 for e in entries if e["in_output"]), error=error, present=data is not None,
                  how_to_read=(data.get("how_to_read", "") if isinstance(data, dict) else ""))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Операторская панель: события по минутам, итоги по классам, полосы времени, тревоги риска."""
    data = read_json(config.RESULTS_JSON)
    error = data.get("__error__") if isinstance(data, dict) else None
    dash = dashboard.build(data if (isinstance(data, dict) and not error) else {})
    return render(request, "dashboard.html", dash=dash, error=error, present=data is not None)


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    return render(request, "demo.html", max_mb=config.MAX_UPLOAD_MB, max_sec=config.MAX_DURATION_SEC,
                  avail=bridge.availability(), stats=queue.stats(), host=str(request.base_url).rstrip("/"))


@app.get("/report", response_class=HTMLResponse)
async def report(request: Request):
    return render(request, "report.html", body=read_md("report.md"))


@app.get("/links", response_class=HTMLResponse)
async def links(request: Request):
    return render(request, "links.html", predictions_exists=config.PREDICTIONS_SAMPLES.exists())


@app.get("/downloads/predictions_samples.json")
async def predictions_samples():
    if not config.PREDICTIONS_SAMPLES.exists():
        raise HTTPException(404, "predictions_samples.json has not been published yet")
    return FileResponse(str(config.PREDICTIONS_SAMPLES), media_type="application/json",
                        filename="predictions_samples.json")


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    return Response(FAVICON_SVG, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "site": config.SITE_NAME,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": bridge.availability(),
        "jobs": queue.stats(),
    }


# --- API -------------------------------------------------------------------------

@app.get("/api/classes")
async def api_classes():
    return {"classes": [{"id": c, **config.CLASS_INFO[c]} for c in config.CLASSES]}


async def create_job(request: Request, upload: UploadFile, options: dict[str, Any]) -> Job:
    """Сохраняет загрузку с проверками размера/длительности и ставит задачу в очередь."""
    name = Path(upload.filename or "video.mp4").name
    if not name.lower().endswith(".mp4"):
        raise HTTPException(400, "only .mp4 files are accepted")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BYTES + CHUNK:
        raise HTTPException(413, f"file is larger than {config.MAX_UPLOAD_MB} MB")
    if not bridge.has_cv2():
        raise HTTPException(503, "OpenCV is not installed on the server — the demo is temporarily unavailable")

    job = queue.create(video_name=name, options=options)
    dest = job.dir / "input.mp4"
    total = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, f"file is larger than {config.MAX_UPLOAD_MB} MB")
                f.write(chunk)
        if total == 0:
            raise HTTPException(400, "empty file")
        try:
            meta = await run_in_threadpool(bridge.probe_video, dest)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        if meta["duration"] > config.MAX_DURATION_SEC + 0.5:
            raise HTTPException(
                400, f"video is longer than {config.MAX_DURATION_SEC} s (this file is {meta['duration']:.1f} s)")
    except Exception:
        queue.discard(job)
        raise
    finally:
        await upload.close()
    job.meta = meta
    job.size_bytes = total
    queue.submit(job)
    log.info("job %s: %s, %.1f s, %.1f MB, options %s", job.id, name, meta["duration"], total / 1048576, options)
    return job


@app.post("/api/jobs", status_code=202)
async def api_jobs_create(request: Request, file: UploadFile = File(...)):
    job = await create_job(request, file, {"risk": True, "annotate": True})
    return {"id": job.id, "status_url": f"/api/jobs/{job.id}", "position": queue.position(job)}


class UrlJob(BaseModel):
    url: str


def _finish_download(job: Job, dest: Path) -> None:
    """Скачивание закончено: проверить длительность и поставить в очередь (вызывается из потока загрузки)."""
    try:
        meta = bridge.probe_video(dest)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if meta["duration"] > config.MAX_DURATION_SEC + 0.5:
        raise ValueError(f"video is longer than {config.MAX_DURATION_SEC} s (this file is {meta['duration']:.1f} s)")
    job.meta = meta
    queue.submit(job)
    log.info("job %s: downloaded %s, %.1f s, %.1f MB", job.id, job.video_name, meta["duration"], job.size_bytes / 1048576)


@app.post("/api/jobs_url", status_code=202)
async def api_jobs_from_url(body: UrlJob):
    """Тот же конвейер, но видео скачивает сервер: прямая ссылка или публичная ссылка Google Drive."""
    if not bridge.has_cv2():
        raise HTTPException(503, "OpenCV is not installed on the server — the demo is temporarily unavailable")
    try:
        url, name = fetch.resolve(body.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job = queue.create(video_name=name, options={"risk": True, "annotate": True, "source_url": body.url})
    fetch.download(job, url, job.dir / "input.mp4", MAX_BYTES, _finish_download, max_sec=config.MAX_DURATION_SEC)
    return {"id": job.id, "status_url": f"/api/jobs/{job.id}", "position": None}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found or already deleted (jobs are kept for 2 hours)")
    return job.to_dict(position=queue.position(job))


@app.api_route("/jobs/{job_id}/{filename}", methods=["GET", "HEAD"], include_in_schema=False)
async def job_file(job_id: str, filename: str):
    if not JOB_ID_RE.match(job_id) or filename not in JOB_FILES:
        raise HTTPException(404)
    path = config.JOBS_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(path), media_type=JOB_FILES[filename],
                        filename=filename if filename.endswith(".json") else None)


@app.post("/api/detect")
async def api_detect(request: Request, file: UploadFile = File(...), risk: int = 1, annotate: int = 0, wait: int = 1):
    """Программный интерфейс: загрузить .mp4 и получить события (и кривую риска) в JSON.

    Параметры запроса: ``risk=0`` — без части B; ``annotate=1`` — также
    отрендерить видео; ``wait=0`` — не ждать, вернуть id задачи для опроса
    ``/api/jobs/{id}``.
    """
    job = await create_job(request, file, {"risk": bool(risk), "annotate": bool(annotate)})
    base = {"job_id": job.id, "status_url": f"/api/jobs/{job.id}"}
    if not wait:
        return JSONResponse({**base, "state": job.state}, status_code=202)
    finished = await run_in_threadpool(job.done_event.wait, config.API_WAIT_TIMEOUT_SEC)
    if not finished:
        return JSONResponse({**base, "state": job.state, "progress": job.progress,
                             "detail": "the job is still running — poll status_url"}, status_code=202)
    if job.state == "error":
        raise HTTPException(500, job.error or "processing failed")
    payload = read_json(job.dir / "events.json") or {}
    result = job.result or {}
    return {**base, "state": "done", "video": payload.get("video", job.video_name),
            "duration": payload.get("duration"), "events": payload.get("events", []),
            "risk": payload.get("risk", []), "timing_sec": payload.get("timing_sec"),
            "notes": payload.get("notes", []), "annotated_url": result.get("annotated_url"),
            "events_url": result.get("events_url")}


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith(("/api/", "/jobs/", "/downloads/")):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    return render(request, "error.html", status_code=exc.status_code, status=exc.status_code, detail=exc.detail)


@app.middleware("http")
async def timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - t0) * 1000:.1f}"
    return response

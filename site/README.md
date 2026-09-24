# Team website (site/)

FastAPI + Jinja2, no external CDNs. Pages: `/` `/team` `/approach` `/eda` `/results` `/demo` `/report` `/links`;
service routes: `/healthz`, `/api/detect`, `/api/jobs/{id}`, `/api/classes`, `/api/openapi.json` (Swagger UI is
disabled on purpose — it loads scripts from a CDN).

## Run locally

From the **repository root** (where `solution.py` and `src/` live):

```bash
pip install -r site/requirements.txt          # plus the root requirements.txt for the demo
uvicorn app:app --app-dir site --reload       # http://127.0.0.1:8000
# or
python site/run.py --reload
```

Why not `uvicorn site.app:app`: the name `site` is taken by Python's standard library, so the directory is added
with `--app-dir site` and the module is simply called `app`.

The site starts without the ML dependencies: `solution.py`, `src/render.py` and OpenCV are imported lazily, only for
the demo. Without them `/demo` says so honestly and `/healthz` reports what was found.

## What to edit

| What | Where |
|---|---|
| Name, team, links, limits | `site/config.py` |
| Team | `site/content/team.json` |
| Approach, report | `site/content/approach.md`, `site/content/report.md` (Markdown) |
| EDA | `site/static/eda/eda.json` + images next to it |
| Results | `site/static/results/results.json`, `site/static/results/examples/*.jpg` |
| Annotated clips | `site/static/media/<name>_annotated.mp4` (see below) |
| Class colours | `CLASS_INFO` in `site/config.py` (feeds both the CSS variables and the JS map) |

## Expected data paths

| Artifact | Path | Notes |
|---|---|---|
| `results.json` | `site/static/results/results.json` | |
| annotated videos | `site/static/media/<video-stem>_annotated.mp4` | the root `.gitignore` ignores `*.mp4` **except** `site/static/media/*.mp4`, so this is the only committable place |
| class examples | `site/static/results/examples/<class_id>_*.jpg` | the class id prefix is picked up for the badge |
| `eda.json` + images | `site/static/eda/eda.json`, `site/static/eda/*.png` | |

`annotated_mp4` in `results.json` is relative to `site/static/`. If it is omitted (or the file is missing), the page
looks for `media/<stem>_annotated.mp4`, then `results/<stem>_annotated.mp4`, where `<stem>` is the video name without
extension. Videos must be H.264 `.mp4` for the browser.

`eda.json` format:

```json
{"videos": [{"name": "C3896.mp4", "width": 1920, "height": 1080, "fps": 25.0, "duration": 600.0, "n_frames": 15000,
  "counts_over_time": {"t": [0, 1, 2], "car": [3, 4, 4], "person": [0, 1, 0]},
  "images": {"heatmap": "eda/C3896_heatmap.png", "flow": "eda/C3896_flow.png"}}]}
```

`results.json` format (`risk` may be sampled at any spacing, e.g. every 5th frame — the chart uses the timestamps):

```json
{"videos": [{"name": "C3896.mp4", "annotated_mp4": "media/C3896_annotated.mp4", "duration": 600.0,
  "events": [[12.0, 15.5, "red_light"]], "risk": [[0.0, 0.01], [0.2, 0.02]]}],
 "failures": [{"video": "C3896.mp4", "t": 214.0, "note": "missed near_miss: pedestrian behind a pole"}]}
```

## Demo and API

- `/demo` form: `.mp4`, ≤ 2 min, ≤ 200 MB (enforced server-side: size while streaming, duration via OpenCV).
- In-process queue: one job at a time, states `queued / running / done / error`, progress 0–100.
  If `detect_events` accepts `progress_cb=`, Part A progress comes from there; otherwise it is estimated from elapsed time.
- Job files live in `site/jobs_store/<id>/` (input.mp4, annotated.mp4, events.json) and are deleted after 2 hours.
- `POST /api/detect` (`-F file=@video.mp4`) — JSON with events and the risk curve; parameters `risk=0`, `annotate=1`, `wait=0`.

## Deploy on Railway (Docker, CPU)

1. Service settings: **Config-as-code** → `site/railway.json` (or copy the file to the repository root).
   Dockerfile path `site/Dockerfile`, build context = repository root, leave Root Directory empty.
2. Healthcheck is `/healthz` (already in railway.json). No variables are required; Railway injects `PORT`.
3. The image installs `site/requirements.txt` and, if present, the root `requirements.txt` with the CPU torch index
   (`PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`) so the image stays a reasonable size.
4. One uvicorn worker (the job queue lives in process memory). Railway's disk is ephemeral — fine for a demo.

Local image build: `docker build -f site/Dockerfile -t trafficeye-site .` from the root. Add a root `.dockerignore`
with `.venv/`, `data/`, `out/` first, otherwise the virtual environment ends up in the build context.

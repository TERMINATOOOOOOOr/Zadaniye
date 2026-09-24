"""EDA сэмпл-видео: параметры, счётчики объектов во времени, тепловая карта движения, поле направлений, плотность.

    python -m src.eda data/samples --out site/static/eda
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .pipeline import build_context


def analyse(video_path: str, st: Settings, out_dir: Path) -> dict:
    ctx = build_context(video_path, st)
    meta, tracks, flow = ctx.meta, ctx.tracks, ctx.flow
    name = Path(video_path).name
    stem = Path(video_path).stem
    W, H = meta["width"], meta["height"]

    # счётчики по классам во времени (шаг 1 с)
    bins = np.arange(0.0, meta["duration"] + 1.0, 1.0)
    counts: dict[str, np.ndarray] = {}
    for tr in tracks.values():
        arr = counts.setdefault(tr.cls, np.zeros(len(bins)))
        seen = set()
        for t in tr.t:
            b = int(t)
            if b < len(bins) and b not in seen:
                arr[b] += 1
                seen.add(b)
    counts_over_time = {"t": [float(b) for b in bins]} | {k: [int(x) for x in v] for k, v in counts.items()}

    # тепловая карта присутствия (нижние точки боксов) и карта направлений
    fa = flow.to_arrays()
    heat = fa["presence"].astype(float)
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)
    heat_img = _colorize(heat, W, H)
    flow_img = _draw_flow(fa, W, H)
    density_img = _colorize(fa["count"].astype(float), W, H)
    frame = _first_frame(video_path)
    if frame is not None:
        heat_img = cv2.addWeighted(frame, 0.45, heat_img, 0.55, 0)
        flow_img = cv2.addWeighted(frame, 0.55, flow_img, 0.45, 0)
        density_img = cv2.addWeighted(frame, 0.45, density_img, 0.55, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for key, img in (("heatmap", heat_img), ("flow", flow_img), ("density", density_img)):
        p = out_dir / f"{stem}_{key}.jpg"
        cv2.imwrite(str(p), _fit(img, 1280), [cv2.IMWRITE_JPEG_QUALITY, 82])
        images[key] = f"eda/{p.name}"
    if frame is not None:
        p = out_dir / f"{stem}_frame.jpg"
        cv2.imwrite(str(p), _fit(frame, 1280), [cv2.IMWRITE_JPEG_QUALITY, 82])
        images["frame"] = f"eda/{p.name}"

    lens = [tr.t1 - tr.t0 for tr in tracks.values()]
    speeds = [float(np.median(tr.speed) / tr.size) for tr in tracks.values() if tr.kind == "vehicle" and len(tr.t) > 5]
    brightness = float(frame.mean()) if frame is not None else None
    return {
        "name": name, "width": W, "height": H, "fps": meta["fps"], "duration": round(meta["duration"], 2), "n_frames": meta["n_frames"],
        "brightness": None if brightness is None else round(brightness, 1),
        "n_tracks": len(tracks), "tracks_by_class": dict(Counter(tr.cls for tr in tracks.values())),
        "track_len_median_s": round(float(np.median(lens)), 2) if lens else 0.0,
        "vehicle_rel_speed_median": round(float(np.median(speeds)), 3) if speeds else None,
        "road_cells": int(fa["road_mask"].sum()), "directional_cells": int(((fa["consensus"] > 0.75) & (fa["count"] >= 5)).sum()),
        "signal_states": dict(Counter(ctx.signal.state)) if ctx.signal is not None else None,
        "counts_over_time": counts_over_time, "images": images,
    }


def _first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def _fit(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)


def _colorize(grid: np.ndarray, W: int, H: int) -> np.ndarray:
    g = grid / (grid.max() + 1e-9)
    img = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def _draw_flow(fa: dict, W: int, H: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cell = fa["cell"]
    mx, my, cons, cnt = fa["mean_vx"], fa["mean_vy"], fa["consensus"], fa["count"]
    for r in range(mx.shape[0]):
        for c in range(mx.shape[1]):
            if cnt[r, c] < 3:
                continue
            x, y = int(c * cell + cell / 2), int(r * cell + cell / 2)
            L = cell * 0.8 * max(cons[r, c], 0.2)
            ang = np.arctan2(my[r, c], mx[r, c])
            col = (int(120 + 135 * (np.cos(ang) + 1) / 2), int(80 + 175 * cons[r, c]), int(120 + 135 * (np.sin(ang) + 1) / 2))
            cv2.arrowedLine(img, (x, y), (int(x + L * np.cos(ang)), int(y + L * np.sin(ang))), col, 1, tipLength=0.4)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", help="папка с mp4 или один файл")
    ap.add_argument("--out", default="site/static/eda")
    ap.add_argument("--cache", default="out/cache")
    args = ap.parse_args()
    st = Settings()
    st.cache_dir = args.cache
    src = Path(args.videos)
    paths = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    out_dir = Path(args.out)
    report = {"videos": [analyse(str(p), st, out_dir) for p in paths]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eda.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for v in report["videos"]:
        print(f"{v['name']}: {v['width']}x{v['height']} {v['fps']} fps {v['duration']} s, tracks {v['n_tracks']} {v['tracks_by_class']}")
    print(f"wrote {out_dir / 'eda.json'}")


if __name__ == "__main__":
    main()

"""Дымовой тест детерминизма для CI: синтетический ролик → два прогона харнесса → одинаковый результат.

Одни и те же функции используются в .github/workflows/ci.yml и в tests/test_ci_smoke.py, чтобы CI и
локальная проверка гоняли один код.

    python tools/ci_smoke.py make-video out/smoke/clip/synthetic.mp4      # ролик 3 с, 320x240, 30 fps
    python run_submission.py --videos out/smoke/clip --out out/smoke/a.json --time-factor 60
    python run_submission.py --videos out/smoke/clip --out out/smoke/b.json --time-factor 60
    python tools/ci_smoke.py compare out/smoke/a.json out/smoke/b.json  # секции "videos" должны совпасть

Ролик рисуется OpenCV без случайности: серое полотно дороги и несколько прямоугольников, которые едут
с постоянной скоростью. Детектор на них почти ничего не находит, но весь путь (ffmpeg-ридер, YOLO,
ByteTrack, правила, Part B по каждому кадру) проходится целиком и должен давать бит-в-бит один ответ.
Множитель бюджета времени поднят намеренно: на CPU-раннере 3 секунды видео не укладываются в 3×, а без
этого харнесс молча обнулил бы оба результата и сравнение стало бы пустым (compare это ловит по логу).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Прямоугольники: (x0, y0, ширина, высота, vx, vy) в пикселях и пикселях/кадр; цвета BGR.
_BOXES = [
    ((10, 150, 60, 30), (1.5, 0.0), (40, 40, 200)),
    ((300, 100, 50, 28), (-2.0, 0.0), (200, 80, 40)),
    ((140, 20, 24, 40), (0.0, 1.2), (30, 160, 30)),
    ((200, 200, 70, 26), (-1.0, -0.3), (220, 220, 220)),
]


def make_video(path: str | Path, seconds: float = 3.0, fps: float = 30.0, width: int = 320, height: int = 240) -> dict:
    """Пишет синтетический mp4 (mp4v) с движущимися прямоугольниками. Возвращает его параметры."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(seconds * fps))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    try:
        for k in range(n_frames):
            frame = np.full((height, width, 3), 120, dtype=np.uint8)
            cv2.rectangle(frame, (0, height // 3), (width, height * 2 // 3), (90, 90, 90), thickness=-1)   # полоса «дороги»
            cv2.line(frame, (0, height // 2), (width, height // 2), (200, 200, 200), 1)
            for (x0, y0, w, h), (vx, vy), color in _BOXES:
                x = int(round(x0 + vx * k)) % width
                y = int(round(y0 + vy * k)) % height
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return {"path": str(path), "n_frames": n_frames, "fps": fps, "width": width, "height": height}


def compare(a_path: str | Path, b_path: str | Path) -> list[str]:
    """Сравнивает два predictions.json. Пустой список = прогоны совпали и оба прошли без ошибок харнесса."""
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, d in (("a", a), ("b", b)):
        videos = d.get("videos") or {}
        if not videos:
            problems.append(f"{name}: no videos in the output")
            continue
        for vid, entry in videos.items():
            errors = (d.get("log") or {}).get(vid, {}).get("errors") or []
            if errors:
                problems.append(f"{name}/{vid}: harness reported {len(errors)} problem(s): {errors[0].splitlines()[0]}")
            if not entry.get("risk"):
                problems.append(f"{name}/{vid}: empty risk curve")
    if a.get("videos") != b.get("videos"):
        problems.append("the 'videos' sections differ between the two runs")
        for vid in sorted(set(a.get("videos", {})) | set(b.get("videos", {}))):
            ea, eb = a.get("videos", {}).get(vid), b.get("videos", {}).get(vid)
            if ea is None or eb is None:
                problems.append(f"{vid}: present in only one run")
                continue
            if ea.get("events") != eb.get("events"):
                problems.append(f"{vid}: events differ ({len(ea.get('events', []))} vs {len(eb.get('events', []))})")
            ra, rb = ea.get("risk", []), eb.get("risk", [])
            if ra != rb:
                first = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
                problems.append(f"{vid}: risk curves differ (len {len(ra)} vs {len(rb)}, first difference at sample {first})")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-video", help="записать синтетический mp4")
    mk.add_argument("out")
    mk.add_argument("--seconds", type=float, default=3.0)
    mk.add_argument("--fps", type=float, default=30.0)
    mk.add_argument("--width", type=int, default=320)
    mk.add_argument("--height", type=int, default=240)
    cp = sub.add_parser("compare", help="сравнить секции videos двух predictions.json")
    cp.add_argument("a")
    cp.add_argument("b")
    args = ap.parse_args(argv)

    if args.cmd == "make-video":
        info = make_video(args.out, args.seconds, args.fps, args.width, args.height)
        print(f"wrote {info['path']}: {info['n_frames']} frames, {info['width']}x{info['height']} @ {info['fps']:g} fps")
        return 0
    problems = compare(args.a, args.b)
    if problems:
        for p in problems:
            print("!", p)
        return 1
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    for vid, entry in a["videos"].items():
        print(f"{vid}: {len(entry['events'])} events, {len(entry['risk'])} risk samples — identical in both runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

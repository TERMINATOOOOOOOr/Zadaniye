"""Кадровые полоски для ручной разметки: миниатюры с шагом fps, по cols в ряд, номер секунды в углу.

    python tools/strips.py VIDEO OUT_DIR [--fps 1] [--from 0] [--to DURATION] [--tile 320] [--cols 10] [--rows 8]

Пишет OUT_DIR/<stem>_<start>-<end>.jpg по rows рядов в файле, чтобы картинки оставались читаемыми.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.video import ffmpeg_exe, video_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--crop", default="", help="x:y:w:h в пикселях исходного кадра — вырез перед уменьшением")
    a = ap.parse_args()
    meta = video_meta(a.video)
    t_to = a.t_to if a.t_to is not None else meta["duration"]
    w = a.tile; h = int(round(w * meta["height"] / meta["width"]))
    if a.crop:
        cx, cy, cw, chh = (int(v) for v in a.crop.split(":"))
        h = int(round(w * chh / cw))
    vf = (f"crop={cw}:{chh}:{cx}:{cy}," if a.crop else "") + f"fps={a.fps},scale={w}:{h}"
    exe = ffmpeg_exe()
    p = subprocess.Popen([exe, "-loglevel", "error", "-nostdin", "-ss", f"{a.t_from:.3f}", "-t", f"{t_to - a.t_from:.3f}", "-i", a.video,
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []; k = 0; per_file = a.cols * a.rows; stem = Path(a.video).stem; written = []

    def flush(first_k: int) -> None:
        if not tiles:
            return
        while len(tiles) % a.cols:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
        t0 = a.t_from + first_k / a.fps; t1 = a.t_from + (k - 1) / a.fps
        name = out / f"{stem}_{t0:06.1f}-{t1:06.1f}.jpg"
        cv2.imwrite(str(name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82]); written.append(name.name)
        tiles.clear()

    first = 0
    while True:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        im = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        t = a.t_from + k / a.fps
        cv2.putText(im, f"{t:.1f}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im); k += 1
        if len(tiles) >= per_file:
            flush(first); first = k
    p.wait(); flush(first)
    print("\n".join(written))


if __name__ == "__main__":
    main()

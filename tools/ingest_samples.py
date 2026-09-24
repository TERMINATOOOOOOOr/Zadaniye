"""Приём сэмпл-видео организаторов: перенос в data/samples, метаданные, дубликаты, кадры сцены.

    python tools/ingest_samples.py "C:/Users/LOQ/Downloads" [--move]

Ищет C3*.MP4 в указанной папке, переносит (или копирует) в data/samples/, печатает
разрешение/fps/длительность, сравнивает хэши файлов одинакового размера, сохраняет по 6 кадров
каждого видео в out/scene_frames/ (для scene_editor и первого взгляда на сцену).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
FRAMES = ROOT / "out" / "scene_frames"


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,duration,codec_name,bit_rate", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip().split(",")
    keys = ["codec", "width", "height", "fps", "duration", "bit_rate", "nb_frames"]
    return dict(zip(keys, out))


def file_hash(path: Path, chunk: int = 1 << 24) -> str:
    """Хэш первых и последних 64 МБ + размера: достаточно, чтобы отличить дубликат, и не читает 6 ГБ целиком."""
    h = hashlib.sha1()
    size = path.stat().st_size
    with open(path, "rb") as f:
        h.update(f.read(chunk * 4))
        f.seek(max(0, size - chunk * 4))
        h.update(f.read(chunk * 4))
    h.update(str(size).encode())
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--move", action="store_true", help="переносить, а не копировать")
    args = ap.parse_args()
    src = Path(args.src)
    files = sorted(p for p in src.glob("C3*.MP4") if not p.name.endswith(".crdownload"))
    if not files:
        print("нет файлов C3*.MP4 в", src)
        sys.exit(1)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    placed = []
    for p in files:
        dst = SAMPLES / p.name
        if not dst.exists():
            (shutil.move if args.move else shutil.copy2)(str(p), str(dst))
        placed.append(dst)
    hashes: dict[str, list[str]] = {}
    for p in placed:
        m = probe(p)
        hs = file_hash(p)
        hashes.setdefault(hs, []).append(p.name)
        print(f"{p.name}: {p.stat().st_size / 1e9:.2f} GB, {m.get('codec')} {m.get('width')}x{m.get('height')} {m.get('fps')} fps, "
              f"{float(m.get('duration') or 0):.1f} s, hash {hs}")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-vf", "fps=1/60,scale=1280:-1", "-frames:v", "6",
                        str(FRAMES / f"{p.stem}_%02d.jpg")])
    for hs, names in hashes.items():
        if len(names) > 1:
            print("ДУБЛИКАТЫ (одинаковое содержимое):", names)
    print(f"кадры сцены: {FRAMES}")


if __name__ == "__main__":
    main()

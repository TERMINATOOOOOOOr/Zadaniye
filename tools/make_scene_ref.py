"""Эталон сцены для привязки видео к разметке: configs/scene_ref.npz из data/samples/C3896.MP4.

Берём ~9 кадров, равномерно разбросанных по клипу (ffmpeg -ss), длинная сторона 960 px, серый; медиана по
времени даёт фон без машин. На фоне считаем ключевые точки (SIFT, иначе ORB) и сохраняем: точки в координатах
полного кадра, дескрипторы, сам фон, размер кадра и эталонные центры трёх светофоров. Это производные данные
(признаки и один маленький фон), не видео.

    python tools/make_scene_ref.py [--video data/samples/C3896.MP4] [--out configs/scene_ref.npz] [--frames 9]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.align import N_FRAMES, REF_LIGHT_ANCHORS, REF_PATH, WORK_SIDE, build_ref, load_ref, save_ref  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="эталон фона и ключевых точек для src/align.py")
    ap.add_argument("--video", default="data/samples/C3896.MP4")
    ap.add_argument("--out", default=str(REF_PATH))
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--side", type=int, default=WORK_SIDE)
    ap.add_argument("--method", default="sift", choices=["sift", "orb"])
    ap.add_argument("--preview", default="", help="куда сохранить фон в jpg для проверки глазами (необязательно)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ref, info = build_ref(args.video, n=args.frames, side=args.side, anchors=REF_LIGHT_ANCHORS, method=args.method)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_ref(out, ref)
    back = load_ref(out)
    info.update({"frame_size": list(ref.frame_size), "out": str(out), "bytes": out.stat().st_size,
                 "anchors": ref.anchors.tolist(), "sec": round(time.perf_counter() - t0, 2),
                 "reload_ok": bool(back is not None and len(back.keypoints) == len(ref.keypoints))})
    if args.preview:
        import cv2
        cv2.imwrite(args.preview, ref.background)
        info["preview"] = args.preview
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

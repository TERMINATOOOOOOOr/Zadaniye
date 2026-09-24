"""Полный прогон по папке видео: официальный харнесс → predictions_*.json, потом материалы для сайта.

    python tools/run_samples.py data/samples --team NAME --out predictions_samples.json --site --labels data/labels/my_labels.json

--site   рендерит размеченные ролики в site/static/results/, собирает results.json и примеры кадров по классам
--labels если есть своя разметка — печатает отчёт evaluate.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("--team", default="team")
    ap.add_argument("--out", default="predictions_samples.json")
    ap.add_argument("--site", action="store_true")
    ap.add_argument("--labels", default="")
    ap.add_argument("--max-side", type=int, default=960)
    args = ap.parse_args()

    cmd = [sys.executable, str(ROOT / "run_submission.py"), "--videos", args.videos, "--out", args.out, "--team", args.team]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)
    subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", args.out, "--validate-only"], cwd=ROOT, check=True)
    if args.labels:
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", args.out, "--gt", args.labels, "--per-video",
                        "--json", str(ROOT / "out" / "eval_report.json")], cwd=ROOT)
    if args.site:
        build_site_results(Path(args.videos), Path(args.out), args.max_side)


def build_site_results(videos_dir: Path, pred_path: Path, max_side: int) -> None:
    from src.config import Settings
    from src.pipeline import build_context
    from src.render import annotate_video

    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    out_dir = ROOT / "site" / "static" / "results"
    media_dir = ROOT / "site" / "static" / "media"      # единственное место, откуда mp4 попадают в git
    ex_dir = out_dir / "examples"
    for d in (out_dir, media_dir, ex_dir):
        d.mkdir(parents=True, exist_ok=True)
    st = Settings()
    st.cache_dir = str(ROOT / "out" / "cache")
    results = {"videos": [], "failures": []}
    per_class_examples: dict[str, int] = {}
    for name, entry in pred["videos"].items():
        path = videos_dir / name if videos_dir.is_dir() else videos_dir
        events, risk = entry["events"], entry.get("risk", [])
        ctx = build_context(str(path), st)
        stem = Path(name).stem
        mp4 = media_dir / f"{stem}_annotated.mp4"
        print(f"render {name} → {mp4.name} ({len(events)} events)")
        annotate_video(str(path), events, str(mp4), risk=risk, max_side=max_side, tracks=ctx.tracks)
        for s, e, label in events:
            if per_class_examples.get(label, 0) >= 3:
                continue
            frame = _frame_at(str(path), (s + e) / 2, max_side)
            if frame is not None:
                cv2.putText(frame, f"{label} {s:.1f}-{e:.1f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(ex_dir / f"{label}_{stem}_{int(s)}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                per_class_examples[label] = per_class_examples.get(label, 0) + 1
        results["videos"].append({"name": name, "annotated_mp4": f"media/{mp4.name}", "events": events,
                                  "risk": risk[::5], "duration": ctx.duration})
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir / 'results.json'}")


def _frame_at(path: str, t: float, max_side: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = f.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return cv2.resize(f, (int(w * s), int(h * s))) if s < 1 else f


if __name__ == "__main__":
    main()

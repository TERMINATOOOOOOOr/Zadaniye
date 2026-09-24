"""Рендер размеченного видео для сайта: боксы с id, подписи активных событий, полоса-таймлайн и кривая риска."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .video import iter_frames, video_meta

COLORS = {
    "accident": (40, 40, 230), "near_miss": (0, 140, 255), "red_light": (60, 20, 220), "wrong_way": (200, 60, 200),
    "illegal_u_turn": (180, 100, 20), "stopped_vehicle": (30, 200, 230), "jaywalking": (0, 200, 120),
    "failure_to_yield": (20, 160, 60), "illegal_turn": (160, 120, 40), "solid_line_crossing": (120, 120, 200),
    "stop_line": (90, 60, 200), "congestion": (60, 160, 200), "road_obstacle": (100, 100, 100), "fire_smoke": (0, 80, 255),
}
KIND_COLOR = {"vehicle": (80, 220, 80), "person": (60, 120, 255), "rider": (255, 180, 60), "obstacle": (0, 200, 255), "other": (180, 180, 180)}


def annotate_video(video_path: str, events: list, out_path: str, risk: list | None = None, max_side: int = 960,
                   tracks: dict | None = None, stride: int = 2, progress=None) -> str:
    """Пишет H.264 mp4 (через ffmpeg, если есть; иначе mp4v). Боксы берёт из tracks, если переданы, иначе гоняет детектор."""
    meta = video_meta(video_path)
    st = Settings()
    if tracks is None:
        from .pipeline import build_context
        st.imgsz_a = min(st.imgsz_a, max_side)
        ctx = build_context(video_path, st)
        tracks = ctx.tracks
    boxes_by_frame = _index_boxes(tracks)
    risk_t = np.array([r[0] for r in risk]) if risk else None
    risk_s = np.array([r[1] for r in risk]) if risk else None

    out_path = str(out_path)
    tmp = out_path + ".raw.mp4"
    writer = None
    n_total = max(1, meta["n_frames"])
    for idx, t, frame, scale in iter_frames(video_path, stride, max_side):
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"] / stride, (w, h + 70))
        canvas = np.zeros((h + 70, w, 3), dtype=np.uint8)
        canvas[:h] = frame
        for tid, cls, kind, box in _boxes_near(boxes_by_frame, idx, stride):
            x1, y1, x2, y2 = (int(v * scale) for v in box)
            col = KIND_COLOR.get(kind, (200, 200, 200))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
            cv2.putText(canvas, f"{cls[:3]}{tid}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        active = [e for e in events if e[0] <= t <= e[1]]
        for k, e in enumerate(active[:4]):
            cv2.putText(canvas, f"{e[2]}  {e[0]:.1f}-{e[1]:.1f}", (8, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(e[2], (255, 255, 255)), 2, cv2.LINE_AA)
        _draw_timeline(canvas, h, w, events, t, meta["duration"], risk_t, risk_s)
        writer.write(canvas)
        if progress is not None and idx % (stride * 50) == 0:
            progress("рендер", idx / n_total)
    if writer is not None:
        writer.release()
    _to_h264(tmp, out_path)
    return out_path


def _index_boxes(tracks: dict) -> dict[int, list]:
    by_frame: dict[int, list] = {}
    for tr in tracks.values():
        for o in tr.obs:
            by_frame.setdefault(o.frame, []).append((tr.id, tr.cls, tr.kind, (o.x1, o.y1, o.x2, o.y2)))
    return by_frame


def _boxes_near(by_frame: dict, idx: int, stride: int) -> list:
    for d in range(0, stride + 1):
        if idx - d in by_frame:
            return by_frame[idx - d]
        if idx + d in by_frame:
            return by_frame[idx + d]
    return []


def _draw_timeline(canvas, h, w, events, t, duration, risk_t, risk_s) -> None:
    y0 = h + 6
    cv2.rectangle(canvas, (0, h), (w, h + 70), (25, 25, 25), -1)
    if duration <= 0:
        return
    for e in events:
        x1, x2 = int(w * e[0] / duration), int(w * e[1] / duration)
        cv2.rectangle(canvas, (x1, y0), (max(x2, x1 + 2), y0 + 14), COLORS.get(e[2], (200, 200, 200)), -1)
    if risk_t is not None and len(risk_t) > 1:
        pts = [(int(w * rt / duration), int(y0 + 60 - 36 * rs)) for rt, rs in zip(risk_t[::max(1, len(risk_t) // w)], risk_s[::max(1, len(risk_t) // w)])]
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 200, 255), 1)
        cv2.line(canvas, (0, int(y0 + 60 - 18)), (w, int(y0 + 60 - 18)), (70, 70, 70), 1)   # порог 0.5
    x = int(w * t / duration)
    cv2.line(canvas, (x, y0), (x, y0 + 62), (255, 255, 255), 2)
    cv2.putText(canvas, f"{t:6.1f}s", (min(x + 6, w - 70), y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def _to_h264(src: str, dst: str) -> None:
    ff = shutil.which("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])
        if r.returncode == 0:
            Path(src).unlink(missing_ok=True)
            return
    shutil.move(src, dst)

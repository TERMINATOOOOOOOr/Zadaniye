"""Загрузка видео по ссылке на стороне сервера: прямые http(s)-ссылки и публичные ссылки Google Drive.

Зачем: 2 минуты 4K с дорожной камеры — это ~2 ГБ; тянуть такое через браузер долго, а сервер
в дата-центре скачает за десятки секунд. Скачивание идёт в отдельном потоке, прогресс пишется в задачу.
"""
from __future__ import annotations

import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from jobs import Job

_DRIVE_ID = re.compile(r"(?:/d/|[?&]id=|/file/d/)([A-Za-z0-9_-]{20,})")
_HTML_HEAD = (b"<!doctype", b"<html", b"<head", b"<?xml")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) TrafficEye-demo/1.0"


def is_youtube(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(h in host for h in ("youtube.com", "youtu.be", "youtube-nocookie.com"))


def _download_youtube(job: Job, url: str, dest: Path, max_bytes: int, max_sec: int) -> None:
    """YouTube через yt-dlp (видео-дорожка mp4 до 1080p, без звука). YouTube может блокировать дата-центры."""
    import subprocess
    import sys
    job.set_progress(None, "downloading (YouTube)")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings",
           "-f", "bv*[height<=1080][ext=mp4]/b[height<=1080][ext=mp4]/b",
           "--match-filter", f"duration <= {max_sec}", "--max-filesize", str(max_bytes),
           "-o", str(dest), url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError as exc:
        raise ValueError("yt-dlp is not installed on the server") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("YouTube download took longer than 15 minutes") from exc
    if not dest.exists() or dest.stat().st_size == 0:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        tail = err[-1] if err else ""
        if "does not pass filter" in (r.stdout or "") + (r.stderr or ""):
            raise ValueError(f"the YouTube video is longer than {max_sec} s")
        if "confirm you" in tail.lower() or "bot" in tail.lower() or "429" in tail or "Sign in" in tail:
            raise ValueError("YouTube blocks downloads from this server (bot check). Upload the .mp4 file instead, "
                             "or use a Google Drive / direct link")
        raise ValueError(f"YouTube download failed: video longer than {max_sec} s, unavailable, or blocked "
                         f"({tail[:160]})")
    job.size_bytes = dest.stat().st_size


def resolve(url: str) -> tuple[str, str]:
    """(url для скачивания, имя файла для показа). Google Drive share-ссылки переводятся в прямую загрузку."""
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http(s) links are accepted")
    if is_youtube(url):
        return url, "youtube.mp4"
    if "drive.google.com" in parsed.netloc or "docs.google.com" in parsed.netloc:
        m = _DRIVE_ID.search(url)
        if not m:
            raise ValueError("cannot find the file id in this Google Drive link")
        fid = m.group(1)
        return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t", f"drive_{fid[:8]}.mp4"
    name = Path(parsed.path).name or "video.mp4"
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    return url, name


def download(job: Job, url: str, dest: Path, max_bytes: int, on_done: Callable[[Job, Path], None], max_sec: int = 180) -> None:
    """Скачивает в фоне; по завершении зовёт on_done(job, dest) (проверка длительности и постановка в очередь)."""

    def run() -> None:
        job.state = "queued"
        job.set_progress(0, "downloading")
        total = 0
        started = time.time()
        try:
            if is_youtube(url):
                _download_youtube(job, url, dest, max_bytes, max_sec)
                on_done(job, dest)
                return
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                length = resp.headers.get("Content-Length")
                expected = int(length) if length and length.isdigit() else None
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if expected is not None and expected > max_bytes:
                    raise ValueError(f"file is larger than {max_bytes // 1048576} MB")
                if "text/html" in ctype:
                    raise ValueError("the link returned a web page, not a video (Google Drive: the file is not shared "
                                     "publicly, or its download quota is exceeded)")
                with dest.open("wb") as f:
                    first = True
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        if first:
                            first = False
                            head = chunk[:16].lower()
                            if any(head.startswith(h) for h in _HTML_HEAD):
                                raise ValueError("the link returned a web page, not a video (Google Drive: not shared "
                                                 "publicly, or download quota exceeded)")
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"file is larger than {max_bytes // 1048576} MB")
                        f.write(chunk)
                        if expected:
                            job.set_progress(total / expected * 100, "downloading")
                        elif time.time() - started > 2:
                            job.set_progress(None, f"downloading ({total // 1048576} MB)")
            if total == 0:
                raise ValueError("empty download")
            job.size_bytes = total
            on_done(job, dest)
        except Exception as exc:  # noqa: BLE001 — любая ошибка = задача с ошибкой, не падение сервера
            job.state = "error"
            job.error = str(exc) if str(exc) else exc.__class__.__name__
            job.finished = time.time()
            job.done_event.set()

    threading.Thread(target=run, name=f"download-{job.id}", daemon=True).start()

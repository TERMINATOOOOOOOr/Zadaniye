"""Простая in-process очередь задач: один рабочий поток, состояния
queued / running / done / error, прогресс 0–100, уборка старых задач.

Очередь ничего не знает о видео и пайплайне: ей передают функцию-обработчик
``worker(job) -> dict``, результат которой кладётся в ``job.result``.
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("site.jobs")

STATES = ("queued", "running", "done", "error")


@dataclass
class Job:
    id: str
    dir: Path
    video_name: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    state: str = "queued"
    progress: int = 0
    stage: str = "queued"
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    result: dict[str, Any] | None = None
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def set_progress(self, value: float | int | None = None, stage: str | None = None) -> None:
        if value is not None:
            self.progress = int(max(0, min(100, round(value))))
        if stage:
            self.stage = stage

    def to_dict(self, position: int | None = None) -> dict[str, Any]:
        now = time.time()
        elapsed = (self.finished or now) - (self.started or now) if self.started else 0.0
        d: dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
            "video": self.video_name,
            "created": self.created,
            "elapsed_sec": round(elapsed, 1),
            "duration_sec": self.meta.get("duration"),
            "position": position,
            "result": self.result if self.state == "done" else None,
        }
        return d


class JobQueue:
    def __init__(
        self,
        root: Path,
        worker: Callable[[Job], dict[str, Any]],
        ttl_sec: float = 7200,
        cleanup_interval_sec: float = 300,
    ) -> None:
        self.root = Path(root)
        self.worker = worker
        self.ttl = ttl_sec
        self.cleanup_interval = cleanup_interval_sec
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self._started = False
        self._stop = threading.Event()
        self.current_id: str | None = None

    # --- жизненный цикл ---------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup()
        threading.Thread(target=self._run, name="job-worker", daemon=True).start()
        threading.Thread(target=self._cleanup_loop, name="job-cleanup", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    # --- API --------------------------------------------------------------
    def create(self, video_name: str, options: dict[str, Any] | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, dir=job_dir, video_name=video_name, options=dict(options or {}))
        with self._lock:
            self._jobs[job_id] = job
        return job

    def submit(self, job: Job) -> None:
        job.state = "queued"
        job.set_progress(0, "queued")
        self._q.put(job.id)

    def discard(self, job: Job) -> None:
        """Удалить задачу, которая ещё не была поставлена в очередь (например, не прошла проверку)."""
        with self._lock:
            self._jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def position(self, job: Job) -> int | None:
        """Позиция в очереди (1 = следующая), None если не в очереди."""
        if job.state != "queued":
            return None
        with self._lock:
            ahead = [j for j in self._jobs.values() if j.state == "queued" and j.created < job.created]
        return len(ahead) + 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_state = {s: 0 for s in STATES}
            for j in self._jobs.values():
                by_state[j.state] = by_state.get(j.state, 0) + 1
        return {"queued": by_state["queued"], "running": by_state["running"],
                "done": by_state["done"], "error": by_state["error"], "current": self.current_id}

    # --- уборка -----------------------------------------------------------
    def cleanup(self, now: float | None = None) -> int:
        """Удаляет задачи (и их каталоги) старше TTL. Возвращает число удалённых."""
        now = now or time.time()
        removed = 0
        with self._lock:
            stale = [j for j in self._jobs.values()
                     if now - j.created > self.ttl and j.state in ("done", "error", "queued")]
            for j in stale:
                self._jobs.pop(j.id, None)
        for j in stale:
            shutil.rmtree(j.dir, ignore_errors=True)
            removed += 1
        # каталоги от прошлых запусков процесса
        if self.root.exists():
            known = set(self._jobs.keys())
            for d in self.root.iterdir():
                if not d.is_dir() or d.name in known:
                    continue
                try:
                    if now - d.stat().st_mtime > self.ttl:
                        shutil.rmtree(d, ignore_errors=True)
                        removed += 1
                except OSError:
                    pass
        if removed:
            log.info("cleanup: removed %d job(s)", removed)
        return removed

    def _cleanup_loop(self) -> None:
        while not self._stop.wait(self.cleanup_interval):
            try:
                self.cleanup()
            except Exception:  # noqa: BLE001
                log.exception("cleanup failed")

    # --- рабочий поток ----------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self.get(job_id)
            if job is None:
                continue
            self.current_id = job.id
            job.state = "running"
            job.started = time.time()
            job.set_progress(1, "starting")
            try:
                job.result = self.worker(job)
                job.state = "done"
                job.set_progress(100, "done")
            except Exception as exc:  # noqa: BLE001
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                log.error("job %s failed:\n%s", job.id, traceback.format_exc())
            finally:
                job.finished = time.time()
                self.current_id = None
                job.done_event.set()

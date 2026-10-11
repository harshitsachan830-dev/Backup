
"""
ModelMind - Background Evaluation Job Manager

Phase 7A: bounded background execution.

Features:
- Background jobs using ThreadPoolExecutor
- Unique job IDs
- Queued / running / completed / failed / cancelled status
- Progress reporting
- Maximum concurrent and outstanding jobs
- Queued-job cancellation
- Cancellation requests for running jobs
- Bounded completed-job history

Limitations:
- Running sklearn operations cannot be forcibly interrupted
  by this thread-based implementation.
- This is a single-process, in-memory manager.
- Worker processes and hard execution timeouts will be
  implemented separately before shared deployment.
- Only trusted backend code should submit callables.
"""

from __future__ import annotations

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4


JobFunction = Callable[[Callable[[float], None]], Any]

TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
}


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class JobManager:
    """
    Thread-based evaluation manager.

    Thread safety:
        Shared job state is protected by a Lock.

    Job capacity:
        max_pending limits queued + running jobs.

    Do not submit arbitrary user-provided functions.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_pending: int = 3,
        max_history: int = 50,
    ) -> None:

        for name, value in {
            "max_workers": max_workers,
            "max_pending": max_pending,
            "max_history": max_history,
        }.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if max_pending < max_workers:
            raise ValueError(
                "max_pending must be >= max_workers."
            )

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="modelmind-eval",
        )

        self._lock = Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future] = {}

        self._max_pending = max_pending
        self._max_history = max_history
        self._closed = False

    def _prune_locked(self) -> None:
        """
        Remove oldest completed entries when the
        history limit has been exceeded.

        Call only while holding self._lock.
        """
        terminal_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in TERMINAL_STATES
        ]

        excess = len(terminal_ids) - self._max_history

        for job_id in terminal_ids[:max(0, excess)]:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)

    def submit(
        self,
        work: JobFunction,
    ) -> str:
        """
        Submit a trusted backend task.

        work receives a progress callback taking
        values from 0.0 to 1.0.
        """

        if not callable(work):
            raise TypeError(
                "work must be callable."
            )

        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Job manager has shut down."
                )

            active_count = sum(
                job["status"] not in TERMINAL_STATES
                for job in self._jobs.values()
            )

            if active_count >= self._max_pending:
                raise RuntimeError(
                    "Evaluation queue is full. "
                    "Try again after another job finishes."
                )

            job_id = uuid4().hex

            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "cancel_requested": False,
                "result": None,
                "error": None,
            }

            try:
                future = self._executor.submit(
                    self._execute,
                    job_id,
                    work,
                )
            except Exception:
                self._jobs.pop(job_id, None)
                raise

            self._futures[job_id] = future
            self._prune_locked()

            return job_id

    def _execute(
        self,
        job_id: str,
        work: JobFunction,
    ) -> None:

        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return

            if job["status"] == "cancelled":
                return

            job["status"] = "running"
            job["started_at"] = _now()

        def report_progress(value: float) -> None:
            if isinstance(value, bool):
                raise ValueError(
                    "Progress must be a number."
                )

            value = float(value)

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Progress must be between 0 and 1."
                )

            with self._lock:
                current = self._jobs.get(job_id)

                if (
                    current is not None
                    and current["status"] == "running"
                ):
                    current["progress"] = max(
                        current["progress"],
                        value,
                    )

        try:
            result = work(report_progress)

        except Exception as error:
            with self._lock:
                job = self._jobs.get(job_id)

                if job is not None:
                    job["status"] = "failed"
                    job["error"] = (
                        f"{type(error).__name__}: {error}"
                    )[:2000]
                    job["finished_at"] = _now()
                    self._prune_locked()
            return

        with self._lock:
            job = self._jobs.get(job_id)

            if job is not None:
                job["status"] = "completed"
                job["progress"] = 1.0
                job["result"] = result
                job["finished_at"] = _now()
                self._prune_locked()

    def get(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        """Return a copy of a job's current state."""

        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            return deepcopy(job)

    def cancel(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        """
        Cancel a queued job if possible.

        Running jobs receive a cancellation request
        flag but continue until their worker finishes.
        """

        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if job["status"] in TERMINAL_STATES:
                return deepcopy(job)

            job["cancel_requested"] = True

            future = self._futures.get(job_id)

            if (
                job["status"] == "queued"
                and future is not None
                and future.cancel()
            ):
                job["status"] = "cancelled"
                job["finished_at"] = _now()
                self._prune_locked()

            return deepcopy(job)

    def shutdown(
        self,
        *,
        wait: bool = True,
    ) -> None:
        """Stop accepting jobs and shut down the pool."""

        with self._lock:
            self._closed = True

        self._executor.shutdown(
            wait=wait,
            cancel_futures=False,
        )

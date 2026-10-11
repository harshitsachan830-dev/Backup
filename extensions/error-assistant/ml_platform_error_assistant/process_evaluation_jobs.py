
"""
ModelMind - Process-Based Evaluation Jobs

Phase 8A: local process isolation for ML evaluation.

Uses Windows-compatible multiprocessing spawn.

Features:
- Separate Python worker process per running evaluation
- Hard elapsed-time limit enforced by parent
- Termination of running jobs
- Cancellation of queued jobs
- Bounded concurrency and pending jobs
- Job status and results
- No changes to the existing Jupyter API

Security requirements:
- Only trusted server code may call submit()
- Validate and authorize dataset paths before submit()
- Bind jobs to authenticated owners at the API layer
- Enforce per-user quotas before public deployment

This is an experimental replacement for the existing
thread-based job manager. Do not expose it to users yet.
"""

from __future__ import annotations

import multiprocessing as mp
import time

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "timed_out",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_main(
    connection: Any,
    dataset_path: str,
    options: dict[str, Any],
) -> None:
    """
    Top-level worker entry point required by Windows spawn.

    The process performs trusted ModelMind evaluation.
    It does not accept arbitrary Python source code.
    """
    try:
        from .evaluated_recommendation import (
            recommend_models_evaluated,
        )

        result = recommend_models_evaluated(
            Path(dataset_path),
            target_column=options["target_column"],
            feature_columns=options.get("feature_columns"),
            task=options.get("task"),
            cv_folds=options.get("cv_folds", 3),
            evaluate_scalers=options.get(
                "evaluate_scalers",
                False,
            ),
        )

        if not result.get("handled"):
            raise ValueError(
                str(result.get("error", "Evaluation failed."))
            )

        connection.send({
            "ok": True,
            "result": result,
        })

    except Exception as error:
        try:
            connection.send({
                "ok": False,
                "error": (
                    f"{type(error).__name__}: {error}"
                )[:2000],
            })
        except (OSError, EOFError, BrokenPipeError):
            pass

    finally:
        connection.close()


def _stop_process(process: mp.Process) -> None:
    """Terminate a process, escalating if necessary."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)

    if process.is_alive():
        process.kill()
        process.join(timeout=2)

    if not process.is_alive():
        process.join(timeout=0.1)


class ProcessEvaluationManager:
    """
    Manage evaluations using child processes.

    One coordinator thread supervises each active
    child process. CPU-heavy fitting happens only
    inside the child process.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_pending: int = 3,
        timeout_seconds: float = 120,
        max_history: int = 50,
    ) -> None:
        if (
            type(max_workers) is not int
            or max_workers < 1
            or type(max_pending) is not int
            or max_pending < max_workers
            or type(max_history) is not int
            or max_history < 1
        ):
            raise ValueError(
                "Invalid worker, queue or history limits."
            )

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(
                timeout_seconds, (int, float)
            )
            or not 0 < timeout_seconds <= 3600
        ):
            raise ValueError(
                "timeout_seconds must be between 0 and 3600."
            )

        self._context = mp.get_context("spawn")

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="modelmind-process-supervisor",
        )

        self._max_pending = max_pending
        self._max_history = max_history
        self._timeout = float(timeout_seconds)

        self._lock = Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future] = {}
        self._closed = False

    def _prune_locked(self) -> None:
        completed = [
            job_id
            for job_id, state in self._jobs.items()
            if state["status"] in TERMINAL_STATES
        ]

        extra = len(completed) - self._max_history

        for job_id in completed[:max(extra, 0)]:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)

    def _finish(
        self,
        job_id: str,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None or job["status"] in TERMINAL_STATES:
                return

            job["status"] = status
            job["result"] = result
            job["error"] = error
            job["finished_at"] = _now()

            if status == "completed":
                job["progress"] = 1.0

            self._prune_locked()

    def submit(
        self,
        dataset_path: str | Path,
        *,
        target_column: str,
        feature_columns: list[str] | None = None,
        task: str | None = None,
        cv_folds: int = 3,
        evaluate_scalers: bool = False,
    ) -> str:
        """
        Submit an already authorized local dataset.

        IMPORTANT: This method is not an HTTP security
        boundary. The caller must validate the file path
        against the trusted Jupyter workspace first.
        """
        path = Path(dataset_path)

        if not path.is_file():
            raise ValueError(
                "Dataset file does not exist."
            )

        if not isinstance(target_column, str) or not target_column:
            raise ValueError(
                "Explicit target column required."
            )

        if task not in (
            None,
            "classification",
            "regression",
        ):
            raise ValueError("Invalid task.")

        if (
            type(cv_folds) is not int
            or not 2 <= cv_folds <= 5
        ):
            raise ValueError(
                "cv_folds must be between 2 and 5."
            )

        if type(evaluate_scalers) is not bool:
            raise ValueError(
                "evaluate_scalers must be boolean."
            )

        if feature_columns is not None and (
            not isinstance(feature_columns, list)
            or not all(
                isinstance(x, str)
                for x in feature_columns
            )
        ):
            raise ValueError(
                "feature_columns must be a list of strings."
            )

        options = {
            "target_column": target_column,
            "feature_columns": feature_columns,
            "task": task,
            "cv_folds": cv_folds,
            "evaluate_scalers": evaluate_scalers,
        }

        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Evaluation manager is shut down."
                )

            active = sum(
                state["status"] not in TERMINAL_STATES
                for state in self._jobs.values()
            )

            if active >= self._max_pending:
                raise RuntimeError(
                    "Evaluation queue is full."
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
                    self._run,
                    job_id,
                    str(path.resolve()),
                    options,
                )
            except Exception:
                self._jobs.pop(job_id, None)
                raise

            self._futures[job_id] = future
            self._prune_locked()

            return job_id

    def _run(
        self,
        job_id: str,
        path: str,
        options: dict[str, Any],
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)

            if state is None or state["status"] == "cancelled":
                return

            state["status"] = "running"
            state["started_at"] = _now()
            state["progress"] = 0.05

        process = None
        receiver = None
        sender = None

        try:
            receiver, sender = self._context.Pipe(
                duplex=False
            )

            process = self._context.Process(
                target=_worker_main,
                args=(sender, path, options),
                daemon=True,
            )

            process.start()

            sender.close()
            sender = None

            deadline = time.monotonic() + self._timeout

            while True:
                with self._lock:
                    state = self._jobs.get(job_id)
                    should_cancel = (
                        state is not None
                        and state["cancel_requested"]
                    )

                if should_cancel:
                    _stop_process(process)
                    self._finish(
                        job_id,
                        "cancelled",
                        error="Running evaluation cancelled.",
                    )
                    return

                if time.monotonic() >= deadline:
                    _stop_process(process)
                    self._finish(
                        job_id,
                        "timed_out",
                        error=(
                            "Evaluation exceeded the configured "
                            "execution time limit."
                        ),
                    )
                    return

                if receiver.poll(0.05):
                    try:
                        message = receiver.recv()
                    except EOFError:
                        message = None

                    if message is not None:
                        if message.get("ok"):
                            self._finish(
                                job_id,
                                "completed",
                                result=message["result"],
                            )
                        else:
                            self._finish(
                                job_id,
                                "failed",
                                error=message.get(
                                    "error",
                                    "Worker failed.",
                                ),
                            )
                        return

                if not process.is_alive():
                    self._finish(
                        job_id,
                        "failed",
                        error=(
                            "Worker process exited without "
                            "a valid result."
                        ),
                    )
                    return

        except Exception as error:
            self._finish(
                job_id,
                "failed",
                error=(
                    f"{type(error).__name__}: {error}"
                )[:2000],
            )

        finally:
            if process is not None:
                _stop_process(process)
                process.close()

            if sender is not None:
                sender.close()

            if receiver is not None:
                receiver.close()

    def get(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            state = self._jobs.get(job_id)

            return (
                deepcopy(state)
                if state is not None
                else None
            )

    def cancel(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            state = self._jobs.get(job_id)

            if state is None:
                return None

            if state["status"] in TERMINAL_STATES:
                return deepcopy(state)

            state["cancel_requested"] = True

            future = self._futures.get(job_id)

            if (
                state["status"] == "queued"
                and future is not None
                and future.cancel()
            ):
                state["status"] = "cancelled"
                state["finished_at"] = _now()

            return deepcopy(state)

    def shutdown(
        self,
        *,
        wait: bool = True,
    ) -> None:
        with self._lock:
            self._closed = True

            for state in self._jobs.values():
                if state["status"] not in TERMINAL_STATES:
                    state["cancel_requested"] = True

        self._executor.shutdown(
            wait=wait,
            cancel_futures=False,
        )

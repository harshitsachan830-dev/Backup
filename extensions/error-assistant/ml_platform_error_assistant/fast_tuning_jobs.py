
"""
ModelMind Fast Tune Top 5 — isolated background job manager.

Independent of the existing AutoML evaluation manager.

Features:
- Windows-safe spawned worker process
- One running worker, bounded queue
- Authenticated owner identity stored per job
- Progress/status polling
- Cancellation and hard process termination
- Hard supervisor timeout
- No arbitrary user-supplied code execution

The caller must authorize the dataset path against the Jupyter
workspace before calling submit().
"""

from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


TERMINAL = {"completed", "failed", "cancelled", "timed_out"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tuning_worker(
    connection: Any,
    dataset_path: str,
    options: dict[str, Any],
) -> None:
    """Top-level worker function compatible with Windows spawn."""
    try:
        import pandas as pd

        from .fast_hyperparameter_tuner import tune_top_five

        frame = pd.read_csv(dataset_path)

        result = tune_top_five(
            frame,
            model_names=options["model_names"],
            target_column=options["target_column"],
            feature_columns=options["feature_columns"],
            task=options["task"],
            time_budget_seconds=options["time_budget_seconds"],
            level=options["level"],
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


def _stop_worker(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)

    if process.is_alive():
        process.kill()
        process.join(timeout=2)

    if process.exitcode is None:
        process.join(timeout=0.1)


class FastTuningJobManager:
    """
    Separate worker manager exclusively for hyperparameter tuning.

    Owners can retrieve/cancel only their own jobs.
    """

    def __init__(
        self,
        *,
        max_pending: int = 3,
        max_history: int = 30,
        timeout_seconds: float = 180,
    ) -> None:
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Invalid max_pending.")

        if type(max_history) is not int or max_history < 1:
            raise ValueError("Invalid max_history.")

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 1 <= timeout_seconds <= 3600
        ):
            raise ValueError("Invalid worker timeout.")

        self._lock = threading.RLock()
        self._context = mp.get_context("spawn")
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="modelmind-fast-tuning",
        )

        self._jobs: dict[str, dict[str, Any]] = {}
        self._owners: dict[str, str] = {}
        self._futures: dict[str, Any] = {}

        self._max_pending = max_pending
        self._max_history = max_history
        self._timeout = float(timeout_seconds)
        self._closed = False

    def _prune(self) -> None:
        finished = [
            job_id
            for job_id, state in self._jobs.items()
            if state["status"] in TERMINAL
        ]

        excess = len(finished) - self._max_history

        for job_id in finished[:max(0, excess)]:
            self._jobs.pop(job_id, None)
            self._owners.pop(job_id, None)
            self._futures.pop(job_id, None)

    def submit(
        self,
        path: str | Path,
        *,
        owner: str,
        model_names: list[str],
        target_column: str,
        feature_columns: list[str],
        task: str,
        time_budget_seconds: float = 20,
        level: str = "basic",
    ) -> str:
        """
        Submit an already-authorized CSV dataset.

        The HTTP layer must independently verify path permissions,
        model names and the user's current Top-5 selection.
        """
        dataset = Path(path)

        if not dataset.is_file() or dataset.suffix.lower() != ".csv":
            raise ValueError("An existing CSV dataset is required.")

        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("Authenticated owner is required.")

        if (
            not isinstance(model_names, list)
            or not 1 <= len(model_names) <= 5
            or not all(
                isinstance(name, str) and name.strip()
                for name in model_names
            )
            or len(set(model_names)) != len(model_names)
        ):
            raise ValueError("One to five distinct models required.")

        if (
            not isinstance(target_column, str)
            or not target_column
            or not isinstance(feature_columns, list)
            or not feature_columns
            or not all(
                isinstance(name, str) and name
                for name in feature_columns
            )
            or len(set(feature_columns)) != len(feature_columns)
            or target_column in feature_columns
        ):
            raise ValueError("Invalid target or feature columns.")

        if task not in {"classification", "regression"}:
            raise ValueError("Invalid ML task.")

        if (
            isinstance(time_budget_seconds, bool)
            or not isinstance(time_budget_seconds, (int, float))
            or not math.isfinite(time_budget_seconds)
            or not 1 <= time_budget_seconds <= 150
        ):
            raise ValueError("Time budget must be 1–150 seconds.")

        if level not in {"basic", "advanced"}:
            raise ValueError(
                "Tuning level must be basic or advanced."
            )

        options = {
            "model_names": list(model_names),
            "target_column": target_column,
            "feature_columns": list(feature_columns),
            "task": task,
            "time_budget_seconds": float(time_budget_seconds),
            "level": level,
        }

        with self._lock:
            if self._closed:
                raise RuntimeError("Fast tuning manager is closed.")

            active = sum(
                job["status"] not in TERMINAL
                for job in self._jobs.values()
            )

            if active >= self._max_pending:
                raise RuntimeError("Fast tuning queue is full.")

            job_id = uuid4().hex

            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "created_at": _timestamp(),
                "started_at": None,
                "finished_at": None,
                "cancel_requested": False,
                "result": None,
                "error": None,
            }

            self._owners[job_id] = owner

            try:
                self._futures[job_id] = self._executor.submit(
                    self._run,
                    job_id,
                    str(dataset.resolve()),
                    options,
                )
            except Exception:
                self._jobs.pop(job_id, None)
                self._owners.pop(job_id, None)
                raise

            self._prune()
            return job_id

    def _finish(
        self,
        job_id: str,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)

            if state is None or state["status"] in TERMINAL:
                return

            state["status"] = status
            state["finished_at"] = _timestamp()
            state["progress"] = 1.0
            state["result"] = result
            state["error"] = error
            self._prune()

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
            state["started_at"] = _timestamp()
            state["progress"] = 0.05

        receiver = None
        sender = None
        process = None

        try:
            receiver, sender = self._context.Pipe(duplex=False)

            process = self._context.Process(
                target=_tuning_worker,
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
                    cancelled = (
                        state is None
                        or state["cancel_requested"]
                    )

                if cancelled:
                    _stop_worker(process)
                    self._finish(
                        job_id,
                        "cancelled",
                        error="Fast tuning cancelled.",
                    )
                    return

                if time.monotonic() >= deadline:
                    _stop_worker(process)
                    self._finish(
                        job_id,
                        "timed_out",
                        error="Fast tuning exceeded worker timeout.",
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
                                    "Tuning worker failed.",
                                ),
                            )
                        return

                if not process.is_alive():
                    self._finish(
                        job_id,
                        "failed",
                        error="Worker exited without a result.",
                    )
                    return

        except Exception as error:
            self._finish(
                job_id,
                "failed",
                error=f"{type(error).__name__}: {error}"[:2000],
            )

        finally:
            if process is not None:
                _stop_worker(process)
                process.close()

            if sender is not None:
                sender.close()

            if receiver is not None:
                receiver.close()

    def get(
        self,
        job_id: str,
        *,
        owner: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            if self._owners.get(job_id) != owner:
                return None

            state = self._jobs.get(job_id)
            return dict(state) if state else None

    def cancel(
        self,
        job_id: str,
        *,
        owner: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            if self._owners.get(job_id) != owner:
                return None

            state = self._jobs.get(job_id)

            if state is None:
                return None

            if state["status"] in TERMINAL:
                return dict(state)

            state["cancel_requested"] = True

            if state["status"] == "queued":
                future = self._futures.get(job_id)

                if future is not None and future.cancel():
                    state["status"] = "cancelled"
                    state["finished_at"] = _timestamp()
                    state["progress"] = 1.0
                    self._prune()

            return dict(state)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True

            for state in self._jobs.values():
                if state["status"] not in TERMINAL:
                    state["cancel_requested"] = True

        self._executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

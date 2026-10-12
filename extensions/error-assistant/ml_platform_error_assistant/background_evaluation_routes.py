
"""
ModelMind - Authenticated Background Evaluation API

Phase 7C: Separate HTTP handlers for background ML jobs.

Endpoints to be registered later:
    POST /api/dataset/evaluation-jobs
    GET  /api/dataset/evaluation-jobs/{job_id}
    POST /api/dataset/evaluation-jobs/{job_id}/cancel

Safeguards:
- Jupyter Server authentication
- Jobs associated with the authenticated user
- Workspace-restricted CSV paths
- Maximum request body size
- Bounded concurrent/pending evaluations
- Explicit target and feature validation

Limitations:
- In-memory job state, not persistent storage
- No forced termination of running sklearn training
- No hard execution timeout
- Only suitable for controlled local testing so far
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

import tornado.web

from jupyter_server.base.handlers import APIHandler

from .dataset_security import (
    DatasetAccessError,
    resolve_workspace_dataset,
)
from .evaluated_recommendation import (
    recommend_models_evaluated,
)
from .process_evaluation_jobs import ProcessEvaluationManager


MAX_REQUEST_BYTES = 16_384
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

# One bounded manager shared across the running server process.
EVALUATION_JOBS = ProcessEvaluationManager(
    max_workers=1,
    max_pending=3,
    max_history=50,
    timeout_seconds=120,
)

# Store ownership separately from public job results.
_JOB_OWNERS: dict[str, str] = {}
_OWNER_LOCK = Lock()


def _identity(user: Any) -> str:
    """
    Obtain a stable authenticated Jupyter username.

    Fail closed when there is no identifiable user.
    """
    if user is None:
        raise PermissionError("Authentication required.")

    if isinstance(user, bytes):
        user = user.decode("utf-8")

    if isinstance(user, str):
        identity = user.strip()
    elif isinstance(user, dict):
        identity = str(
            user.get("username")
            or user.get("name")
            or ""
        ).strip()
    else:
        identity = str(
            getattr(user, "username", None)
            or getattr(user, "name", None)
            or ""
        ).strip()

    if not identity:
        raise PermissionError(
            "Unable to identify the authenticated user."
        )

    return identity


def _workspace_root(handler: APIHandler) -> Path:
    """Get the trusted root from Jupyter Server settings."""
    contents_manager = handler.settings.get(
        "contents_manager"
    )

    root = getattr(
        contents_manager,
        "root_dir",
        None,
    )

    if not root:
        server_app = handler.settings.get("serverapp")
        root = getattr(server_app, "root_dir", None)

    if not root:
        raise RuntimeError(
            "Unable to determine the Jupyter workspace root."
        )

    return Path(root)


def normalize_job_request(payload: Any) -> dict[str, Any]:
    """Validate submitted evaluation options."""

    if not isinstance(payload, dict):
        raise ValueError(
            "Request body must be a JSON object."
        )

    allowed = {
        "file_path",
        "target_col",
        "target_column",
        "feature_columns",
        "task",
        "cv_folds",
        "evaluate_scalers",
    }

    unknown = set(payload) - allowed

    if unknown:
        raise ValueError(
            "Unsupported request fields: "
            + ", ".join(sorted(unknown))
        )

    file_path = payload.get("file_path")

    if not isinstance(file_path, str):
        raise ValueError(
            "file_path must be a string."
        )

    file_path = file_path.strip()

    if not file_path:
        raise ValueError(
            "file_path cannot be empty."
        )

    first_target = payload.get("target_col")
    second_target = payload.get("target_column")

    if (
        first_target is not None
        and second_target is not None
        and first_target != second_target
    ):
        raise ValueError(
            "Conflicting target names were provided."
        )

    target = (
        first_target
        if first_target is not None
        else second_target
    )

    if not isinstance(target, str) or not target.strip():
        raise ValueError(
            "Select a target column before training."
        )

    target = target.strip()

    features = payload.get("feature_columns")

    if features is not None:
        if (
            not isinstance(features, list)
            or not features
            or not all(
                isinstance(value, str) and value.strip()
                for value in features
            )
        ):
            raise ValueError(
                "feature_columns must be a non-empty "
                "list of strings."
            )

        if len(features) != len(set(features)):
            raise ValueError(
                "Duplicate feature selections are not allowed."
            )

        if target in features:
            raise ValueError(
                "Target cannot also be an input feature."
            )

    task = payload.get("task")

    if task not in (
        None,
        "classification",
        "regression",
    ):
        raise ValueError(
            "task must be classification or regression."
        )

    folds = payload.get("cv_folds", 5)

    if (
        not isinstance(folds, int)
        or isinstance(folds, bool)
        or not 2 <= folds <= 5
    ):
        raise ValueError(
            "cv_folds must be an integer from 2 to 5."
        )

    evaluate_scalers = payload.get(
        "evaluate_scalers",
        False,
    )

    if not isinstance(evaluate_scalers, bool):
        raise ValueError(
            "evaluate_scalers must be a boolean."
        )

    return {
        "file_path": file_path,
        "target_column": target,
        "feature_columns": features,
        "task": task,
        "cv_folds": folds,
        "evaluate_scalers": evaluate_scalers,
    }


def _prune_owner_registry() -> int:
    """
    Remove ownership entries for jobs no longer retained
    by the process evaluation manager.

    Do not hold _OWNER_LOCK while querying the manager.
    This avoids nested lock acquisition.
    """
    with _OWNER_LOCK:
        known_ids = tuple(_JOB_OWNERS)

    stale_ids = [
        job_id
        for job_id in known_ids
        if EVALUATION_JOBS.get(job_id) is None
    ]

    if not stale_ids:
        return 0

    removed = 0

    with _OWNER_LOCK:
        for job_id in stale_ids:
            if job_id in _JOB_OWNERS:
                del _JOB_OWNERS[job_id]
                removed += 1

    return removed


def _check_owner(job_id: str, user: str) -> bool:
    """
    Check ownership and reject jobs no longer retained.

    Unknown and other users' jobs are indistinguishable
    to the HTTP caller.
    """
    with _OWNER_LOCK:
        owns_job = _JOB_OWNERS.get(job_id) == user

    if not owns_job:
        return False

    if EVALUATION_JOBS.get(job_id) is None:
        with _OWNER_LOCK:
            if _JOB_OWNERS.get(job_id) == user:
                _JOB_OWNERS.pop(job_id, None)
        return False

    return True


def _job_work(
    path: Path,
    options: dict[str, Any],
):
    """Create the trusted callable executed by JobManager."""

    def work(report_progress):
        report_progress(0.05)

        result = recommend_models_evaluated(
            path,
            target_column=options["target_column"],
            feature_columns=options["feature_columns"],
            task=options["task"],
            cv_folds=options["cv_folds"],
            evaluate_scalers=options["evaluate_scalers"],
        )

        report_progress(0.95)

        if not result.get("handled"):
            raise ValueError(
                str(result.get("error", "Evaluation failed."))
            )

        return result

    return work


class EvaluationJobSubmitHandler(APIHandler):
    """Submit a dataset-specific background evaluation."""

    @tornado.web.authenticated
    def post(self):
        self.set_header(
            "Content-Type",
            "application/json",
        )

        if len(self.request.body) > MAX_REQUEST_BYTES:
            self.set_status(413)
            self.finish(json.dumps({
                "error": "Evaluation request is too large.",
            }))
            return

        try:
            owner = _identity(self.current_user)

            payload = json.loads(self.request.body)
            options = normalize_job_request(payload)

            root = _workspace_root(self)

            path = resolve_workspace_dataset(
                options["file_path"],
                root,
            )

        except PermissionError as error:
            self.set_status(403)
            self.finish(json.dumps({"error": str(error)}))
            return

        except DatasetAccessError as error:
            self.set_status(403)
            self.finish(json.dumps({"error": str(error)}))
            return

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
        ) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        except Exception:
            self.set_status(500)
            self.finish(json.dumps({
                "error": "Server workspace configuration error.",
            }))
            return

        try:
            job_id = EVALUATION_JOBS.submit(
                path,
                target_column=options["target_column"],
                feature_columns=options["feature_columns"],
                task=options["task"],
                cv_folds=options["cv_folds"],
                evaluate_scalers=options["evaluate_scalers"],
            )

            with _OWNER_LOCK:
                _JOB_OWNERS[job_id] = owner

            # Keep owner records aligned with retained jobs.
            _prune_owner_registry()

            # Optional Fast Tune dataset identity registration.
            # Failure must not change normal AutoML behavior.
            try:
                from .fast_tuning_dataset_binding import (
                    register_evaluation_dataset,
                )

                register_evaluation_dataset(
                    job_id=job_id,
                    owner=owner,
                    dataset_path=path,
                )

            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Optional Fast Tune dataset binding failed "
                    "for evaluation job %s. Normal AutoML "
                    "continues; tuning will require a new "
                    "successfully bound evaluation.",
                    job_id,
                    exc_info=True,
                )

        except RuntimeError as error:
            self.set_status(429)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.set_status(202)
        self.finish(json.dumps({
            "job_id": job_id,
            "status": "queued",
            "message": "Evaluation accepted.",
        }))


class EvaluationJobStatusHandler(APIHandler):
    """Retrieve only the authenticated user's job."""

    @tornado.web.authenticated
    def get(self, job_id: str):
        self.set_header(
            "Content-Type",
            "application/json",
        )

        try:
            owner = _identity(self.current_user)
        except PermissionError:
            self.set_status(403)
            self.finish(json.dumps({
                "error": "Authentication required.",
            }))
            return

        if (
            not JOB_ID_PATTERN.fullmatch(job_id)
            or not _check_owner(job_id, owner)
        ):
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Evaluation job not found.",
            }))
            return

        job = EVALUATION_JOBS.get(job_id)

        if job is None:
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Evaluation job not found.",
            }))
            return

        self.finish(json.dumps(job))


class EvaluationJobCancelHandler(APIHandler):
    """Cancel queued jobs or request running cancellation."""

    @tornado.web.authenticated
    def post(self, job_id: str):
        self.set_header(
            "Content-Type",
            "application/json",
        )

        try:
            owner = _identity(self.current_user)
        except PermissionError:
            self.set_status(403)
            self.finish(json.dumps({
                "error": "Authentication required.",
            }))
            return

        if (
            not JOB_ID_PATTERN.fullmatch(job_id)
            or not _check_owner(job_id, owner)
        ):
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Evaluation job not found.",
            }))
            return

        job = EVALUATION_JOBS.cancel(job_id)

        if job is None:
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Evaluation job not found.",
            }))
            return

        self.finish(json.dumps({
            "job_id": job_id,
            "status": job["status"],
            "cancel_requested": job["cancel_requested"],
            "message": (
                "Queued job cancelled."
                if job["status"] == "cancelled"
                else
                "Cancellation recorded. Running model training "
                "will be terminated by the process supervisor."
                if job["status"] == "running"
                else
                "Job is already in a terminal state."
            ),
        }))

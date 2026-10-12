
"""
ModelMind — Independent Fast Tuning HTTP API.

New endpoints, to be registered after validation:

POST /api/dataset/tuning-jobs
GET  /api/dataset/tuning-jobs/{job_id}
POST /api/dataset/tuning-jobs/{job_id}/cancel

This module does not modify existing evaluation routes.

Important:
Model names are checked against the existing AutoML catalog.
Binding those names to a previously completed user's Top-5
evaluation still requires a server-side verification step
before production deployment.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import tornado.web

from jupyter_server.base.handlers import APIHandler

from .background_evaluation_routes import (
    _identity,
    _workspace_root,
)
from .dataset_security import (
    DatasetAccessError,
    resolve_workspace_dataset,
)
from .fast_tuning_jobs import FastTuningJobManager
from .model_evaluator import get_candidate_models


MAX_REQUEST_BYTES = 16384
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

TUNING_JOBS = FastTuningJobManager(
    max_pending=3,
    max_history=30,
    timeout_seconds=180,
)


def normalize_tuning_request(payload: Any) -> dict[str, Any]:
    """Strict validation without executing user-supplied code."""

    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")

    allowed = {
        "file_path",
        "target_column",
        "feature_columns",
        "task",
        "model_names",
        "time_budget_seconds",
        "evaluation_job_id",
        "level",
    }

    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            "Unsupported fields: " + ", ".join(sorted(unknown))
        )

    file_path = payload.get("file_path")
    target = payload.get("target_column")
    features = payload.get("feature_columns")
    task = payload.get("task")
    names = payload.get("model_names")
    budget = payload.get("time_budget_seconds", 20)
    level = payload.get("level", "basic")

    if (
        not isinstance(file_path, str)
        or not file_path.strip()
        or len(file_path) > 1024
    ):
        raise ValueError("Invalid dataset path.")

    if (
        not isinstance(target, str)
        or not target.strip()
        or len(target) > 256
    ):
        raise ValueError("Invalid target column.")

    if task not in {"classification", "regression"}:
        raise ValueError("Invalid ML task.")

    if (
        not isinstance(features, list)
        or not 1 <= len(features) <= 200
        or not all(
            isinstance(name, str)
            and name.strip()
            and len(name) <= 256
            for name in features
        )
        or len(features) != len(set(features))
        or target in features
    ):
        raise ValueError("Invalid feature selection.")

    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= 5
        or not all(
            isinstance(name, str)
            and name.strip()
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise ValueError("Provide one to five distinct models.")

    available = {
        name
        for name, _, _ in get_candidate_models(task=task)
    }

    invalid_models = [
        name for name in names if name not in available
    ]

    if invalid_models:
        raise ValueError(
            "Unsupported models: " + ", ".join(invalid_models)
        )

    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or not 1 <= budget <= 150
    ):
        raise ValueError(
            "Time budget must be between 1 and 150 seconds."
        )

    if level not in {"basic", "advanced"}:
        raise ValueError(
            "Tuning level must be basic or advanced."
        )

    return {
        "file_path": file_path.strip(),
        "target_column": target,
        "feature_columns": list(features),
        "task": task,
        "model_names": list(names),
        "time_budget_seconds": float(budget),
        "level": level,
    }


class TuningJobSubmitHandler(APIHandler):
    """Submit an authenticated Fast Tune job."""

    @tornado.web.authenticated
    def post(self):
        self.set_header("Content-Type", "application/json")

        if len(self.request.body) > MAX_REQUEST_BYTES:
            self.set_status(413)
            self.finish(json.dumps({
                "error": "Tuning request is too large."
            }))
            return

        try:
            owner = _identity(self.current_user)

            payload = json.loads(self.request.body)
            options = normalize_tuning_request(payload)

            root = _workspace_root(self)

            path = resolve_workspace_dataset(
                options["file_path"],
                root,
            )

            if path.suffix.lower() != ".csv":
                raise ValueError(
                    "Fast tuning currently supports CSV datasets."
                )

            # SECURE_FAST_TUNING_BINDING_V1
            from .background_evaluation_routes import (
                EVALUATION_JOBS,
                _check_owner,
            )
            from .fast_tuning_binding import (
                verify_completed_top_five,
            )
            from .fast_tuning_dataset_binding import (
                verify_evaluation_dataset,
            )

            evaluation_job_id = payload.get("evaluation_job_id")

            verified = verify_completed_top_five(
                job_id=evaluation_job_id,
                owner=owner,
                evaluation_manager=EVALUATION_JOBS,
                check_owner=_check_owner,
            )

            # The client's values must match the completed
            # evaluation. The server's verified values will
            # be used to submit the tuning job.
            for key in (
                "task",
                "target_column",
                "feature_columns",
                "model_names",
            ):
                if options[key] != verified[key]:
                    raise ValueError(
                        "Fast Tune selections do not match "
                        "the completed Model Advisor job. "
                        "Run Model Advisor again."
                    )

            verify_evaluation_dataset(
                job_id=evaluation_job_id,
                owner=owner,
                dataset_path=path,
            )

        except PermissionError:
            self.set_status(403)
            self.finish(json.dumps({
                "error": "Authentication required."
            }))
            return

        except DatasetAccessError:
            self.set_status(403)
            self.finish(json.dumps({
                "error": "Dataset access denied."
            }))
            return

        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        except Exception:
            self.set_status(500)
            self.finish(json.dumps({
                "error": "Unable to prepare tuning request."
            }))
            return

        try:
            job_id = TUNING_JOBS.submit(
                path,
                owner=owner,
                model_names=verified["model_names"],
                target_column=verified["target_column"],
                feature_columns=verified["feature_columns"],
                task=verified["task"],
                time_budget_seconds=options[
                    "time_budget_seconds"
                ],
                level=options["level"],
            )

        except RuntimeError as error:
            self.set_status(429)
            self.finish(json.dumps({"error": str(error)}))
            return

        except ValueError as error:
            self.set_status(400)
            self.finish(json.dumps({"error": str(error)}))
            return

        self.set_status(202)
        self.finish(json.dumps({
            "job_id": job_id,
            "status": "queued",
            "message": "Fast tuning job accepted.",
        }))



def _json_safe(value):
    """
    Make tuning-job API results valid JSON without altering
    the original tuning measurements or stored job state.
    """
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # NumPy scalar types may appear in ML results.
    if hasattr(value, "item") and callable(value.item):
        converted = value.item()
        if converted is not value:
            return _json_safe(converted)

    return value


class TuningJobStatusHandler(APIHandler):
    """Return only the authenticated user's tuning job."""

    @tornado.web.authenticated
    def get(self, job_id: str):
        self.set_header("Content-Type", "application/json")

        try:
            owner = _identity(self.current_user)
        except PermissionError:
            self.set_status(403)
            self.finish(json.dumps({
                "error": "Authentication required."
            }))
            return

        if not JOB_ID_PATTERN.fullmatch(job_id):
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Tuning job not found."
            }))
            return

        state = TUNING_JOBS.get(
            job_id,
            owner=owner,
        )

        if state is None:
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Tuning job not found."
            }))
            return

        self.finish(json.dumps(_json_safe(state), allow_nan=False))


class TuningJobCancelHandler(APIHandler):
    """Cancel only the authenticated user's tuning job."""

    @tornado.web.authenticated
    def post(self, job_id: str):
        self.set_header("Content-Type", "application/json")

        try:
            owner = _identity(self.current_user)
        except PermissionError:
            self.set_status(403)
            self.finish(json.dumps({
                "error": "Authentication required."
            }))
            return

        if not JOB_ID_PATTERN.fullmatch(job_id):
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Tuning job not found."
            }))
            return

        state = TUNING_JOBS.cancel(
            job_id,
            owner=owner,
        )

        if state is None:
            self.set_status(404)
            self.finish(json.dumps({
                "error": "Tuning job not found."
            }))
            return

        self.finish(json.dumps({
            "job_id": job_id,
            "status": state["status"],
            "cancel_requested": state["cancel_requested"],
        }))

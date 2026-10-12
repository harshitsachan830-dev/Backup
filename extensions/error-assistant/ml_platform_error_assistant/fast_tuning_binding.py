
"""
ModelMind — Fast Tuning Top-5 Job Verification.

New isolated module. Does not alter existing AutoML jobs.

Only a completed, authenticated evaluation job can supply
the authoritative Top-5 model names.

Dataset path provenance must be checked separately before
an HTTP tuning submission is permitted.
"""

from __future__ import annotations

import re
from typing import Any


JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class TuningBindingError(ValueError):
    """Invalid or inaccessible evaluation job."""


def verify_completed_top_five(
    *,
    job_id: str,
    owner: str,
    evaluation_manager: Any,
    check_owner: Any,
) -> dict[str, Any]:
    """Resolve authoritative finalists from the existing job."""

    if (
        not isinstance(job_id, str)
        or not JOB_ID_PATTERN.fullmatch(job_id)
    ):
        raise TuningBindingError("Invalid evaluation job ID.")

    if not isinstance(owner, str) or not owner.strip():
        raise TuningBindingError("Authentication required.")

    # Ownership must be checked BEFORE reading job results.
    if not check_owner(job_id, owner):
        raise TuningBindingError("Evaluation job not found.")

    job = evaluation_manager.get(job_id)

    if not isinstance(job, dict):
        raise TuningBindingError("Evaluation job not found.")

    if job.get("status") != "completed":
        raise TuningBindingError(
            "Model Advisor evaluation must be completed first."
        )

    result = job.get("result")

    if not isinstance(result, dict) or result.get("handled") is not True:
        raise TuningBindingError(
            "Completed job has no valid evaluation result."
        )

    automl = result.get("automl_recommendations")

    if not isinstance(automl, dict) or automl.get("success") is not True:
        raise TuningBindingError(
            "Completed job has no successful AutoML ranking."
        )

    recommendations = automl.get("recommendations")

    if (
        not isinstance(recommendations, list)
        or not 1 <= len(recommendations) <= 5
    ):
        raise TuningBindingError("Invalid AutoML finalists.")

    names = []

    for item in recommendations:
        if not isinstance(item, dict):
            raise TuningBindingError("Invalid model recommendation.")

        name = item.get("name")

        if not isinstance(name, str) or not name.strip():
            raise TuningBindingError("Invalid finalist model name.")

        if name in names:
            raise TuningBindingError("Duplicate finalist model.")

        names.append(name)

    empirical = result.get("empirical_evaluation")

    if not isinstance(empirical, dict) or empirical.get("success") is not True:
        raise TuningBindingError("Empirical evaluation unavailable.")

    task = empirical.get("task")
    target = empirical.get("target")
    features = empirical.get("features")

    if task not in {"classification", "regression"}:
        raise TuningBindingError("Invalid evaluated task.")

    if not isinstance(target, str) or not target:
        raise TuningBindingError("Invalid evaluated target.")

    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(x, str) and x for x in features)
        or len(set(features)) != len(features)
        or target in features
    ):
        raise TuningBindingError("Invalid evaluated features.")

    if automl.get("task") != task:
        raise TuningBindingError("Inconsistent evaluation task.")

    ranking = empirical.get("ranking")

    if not isinstance(ranking, list):
        raise TuningBindingError("Empirical ranking missing.")

    ranked_names = [
        row.get("model")
        for row in ranking
        if isinstance(row, dict)
        and row.get("status") == "success"
    ]

    if names != ranked_names:
        raise TuningBindingError(
            "Displayed finalists differ from measured ranking."
        )

    return {
        "evaluation_job_id": job_id,
        "model_names": names,
        "task": task,
        "target_column": target,
        "feature_columns": list(features),
        "verified": True,
        "dataset_identity_verified": False,
    }

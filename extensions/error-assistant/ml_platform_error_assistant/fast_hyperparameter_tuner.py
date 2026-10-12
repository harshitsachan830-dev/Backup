
"""
ModelMind Smart Hyperparameter Tuning.

Opt-in tuning for the existing measured AutoML Top 5.

Basic:
    Up to 8 important configurations per model.

Advanced:
    Up to 20 important configurations per model.

Preserves:
    - Existing AutoML ranking
    - Existing model catalog
    - Existing preprocessing and cross-validation evaluator
    - Existing background job interface
    - Notebook code generation and visualizations

Only recommend changed hyperparameters when the measured
improvement over defaults exceeds a small practical threshold.

The search does not guarantee globally optimal parameters.
"""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Real
from time import monotonic
from typing import Any

import pandas as pd

from .model_evaluator import (
    _evaluate_one,
    _sample_data,
    get_candidate_models,
)

from .smart_tuning_search import (
    build_smart_trials,
    describe_changes,
)


MAX_MODELS = 5
MAX_TUNING_ROWS = 700

BASIC_MAX_TRIALS = 8
ADVANCED_MAX_TRIALS = 20

# Small improvements can be cross-validation noise.
MIN_CLASSIFICATION_GAIN = 0.002
MIN_REGRESSION_RELATIVE_GAIN = 0.005


def _finite_score(value: Any) -> float | None:
    """Accept only real, finite model scores."""

    if isinstance(value, bool):
        return None

    if not isinstance(value, Real):
        return None

    score = float(value)

    if not math.isfinite(score):
        return None

    return score


def _to_json_value(value: Any) -> Any:
    """Convert small parameter/result values into JSON-safe data."""

    if hasattr(value, "item") and callable(value.item):
        value = value.item()

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _to_json_value(item)
            for key, item in value.items()
        }

    return str(value)


def _meaningful_improvement(
    baseline_score: float,
    candidate_score: float,
    *,
    task: str,
    higher_is_better: bool,
) -> bool:
    """Check whether changing defaults has useful measured gain."""

    improvement = (
        candidate_score - baseline_score
        if higher_is_better
        else baseline_score - candidate_score
    )

    if task == "classification":
        return improvement > MIN_CLASSIFICATION_GAIN

    threshold = (
        MIN_REGRESSION_RELATIVE_GAIN
        * max(abs(baseline_score), 1e-9)
    )

    return improvement > threshold


def _better(
    candidate_score: float,
    current_score: float,
    higher_is_better: bool,
) -> bool:
    if higher_is_better:
        return candidate_score > current_score

    return candidate_score < current_score


def _candidate_configurations(
    estimator: Any,
    level: str = "basic",
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Preserve a callable configuration helper for compatibility.

    First configuration always represents estimator defaults.
    """

    if level not in {"basic", "advanced"}:
        raise ValueError("Invalid tuning level.")

    limit = (
        BASIC_MAX_TRIALS
        if level == "basic"
        else ADVANCED_MAX_TRIALS
    )

    return build_smart_trials(
        estimator,
        level=level,
        max_trials=limit,
        random_state=random_state,
    )


def tune_top_five(
    frame: pd.DataFrame,
    *,
    model_names: list[str],
    target_column: str,
    feature_columns: list[str],
    task: str,
    scoring: str | None = None,
    higher_is_better: bool | None = None,
    metric: str | None = None,
    random_state: int = 42,
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
    time_budget_seconds: float = 20.0,
    level: str = "basic",
) -> dict[str, Any]:
    """
    Tune only previously selected AutoML finalists.

    Uses the existing leakage-safe ModelMind evaluator.

    Each model starts with a default-parameter baseline,
    then tries important Basic or Advanced configurations.

    The deadline is cooperative and checked between fits.
    The background worker must enforce a hard timeout.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(
            "A nonempty pandas DataFrame is required."
        )

    if not frame.columns.is_unique:
        raise ValueError(
            "Duplicate dataset column names are unsupported."
        )

    if task not in {"classification", "regression"}:
        raise ValueError("Invalid task.")

    if level not in {"basic", "advanced"}:
        raise ValueError(
            "Tuning level must be basic or advanced."
        )

    if target_column not in frame.columns:
        raise ValueError("Target column does not exist.")

    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or len(feature_columns) != len(set(feature_columns))
        or target_column in feature_columns
        or any(
            not isinstance(col, str)
            or col not in frame.columns
            for col in feature_columns
        )
    ):
        raise ValueError("Invalid or leaking feature selection.")

    if (
        not isinstance(model_names, list)
        or not 1 <= len(model_names) <= MAX_MODELS
        or not all(
            isinstance(name, str) and name.strip()
            for name in model_names
        )
        or len(set(model_names)) != len(model_names)
    ):
        raise ValueError(
            "Provide one to five distinct AutoML finalist names."
        )

    if (
        isinstance(time_budget_seconds, bool)
        or not isinstance(time_budget_seconds, (int, float))
        or not math.isfinite(time_budget_seconds)
        or not 1 <= time_budget_seconds <= 300
    ):
        raise ValueError(
            "Time budget must be between 1 and 300 seconds."
        )

    if scoring is None:
        scoring = (
            "accuracy"
            if task == "classification"
            else "neg_root_mean_squared_error"
        )

    if higher_is_better is None:
        higher_is_better = task == "classification"

    if not isinstance(scoring, str) or not scoring:
        raise ValueError("Invalid scoring name.")

    if not isinstance(higher_is_better, bool):
        raise ValueError(
            "higher_is_better must be boolean."
        )

    if metric is None:
        metric = (
            "Accuracy"
            if task == "classification"
            else "RMSE"
        )

    if frame[target_column].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    candidates = {
        name: (estimator, scaler)
        for name, estimator, scaler in get_candidate_models(
            task=task,
            random_state=random_state,
        )
    }

    unknown = [
        name
        for name in model_names
        if name not in candidates
    ]

    if unknown:
        raise ValueError(
            f"Unknown AutoML finalists: {unknown}"
        )

    data = _sample_data(
        frame,
        target_column=target_column,
        task=task,
        max_rows=MAX_TUNING_ROWS,
        min_class_count=2,
        random_state=random_state,
    )

    # FAIR_TOP_FIVE_TUNING_SCHEDULER_V1
    # Every model gets a baseline opportunity before deeper search.
    start = monotonic()
    results = []
    timed_out = False

    states = []

    for name in model_names:
        estimator, scaler = candidates[name]
        states.append({
            "name": name,
            "estimator": estimator,
            "scaler": scaler,
            "configurations": _candidate_configurations(
                estimator,
                level=level,
                random_state=random_state,
            ),
            "attempts": [],
            "default_result": None,
            "best_result": None,
        })

    total_rounds = max(
        len(state["configurations"])
        for state in states
    )

    for round_index in range(total_rounds):
        for state in states:
            configurations = state["configurations"]

            if round_index >= len(configurations):
                continue

            if monotonic() - start >= time_budget_seconds:
                timed_out = True
                break

            name = state["name"]
            estimator = state["estimator"]
            scaler = state["scaler"]
            changes = configurations[round_index]

            tuned_estimator = deepcopy(estimator)

            if changes:
                tuned_estimator.set_params(**changes)

            evaluated = _evaluate_one(
                model_name=name,
                estimator=tuned_estimator,
                scaler=scaler,
                data=data,
                target_column=target_column,
                feature_columns=feature_columns,
                task=task,
                folds=2,
                random_state=random_state,
                numeric_imputation=numeric_imputation,
                categorical_encoding=categorical_encoding,
                scoring=scoring,
                higher_is_better=higher_is_better,
            )

            score = _finite_score(
                evaluated.get("mean_score")
            )
            std_score = _finite_score(
                evaluated.get("std_score")
            )

            valid = (
                evaluated.get("status") == "success"
                and score is not None
            )

            entry = {
                "changes": _to_json_value(changes),
                "parameters": _to_json_value(
                    tuned_estimator.get_params(deep=False)
                ),
                "status": "success" if valid else "failed",
                "score": score,
                "mean_score": score,
                "std_score": std_score,
                "error": (
                    None if valid
                    else str(
                        evaluated.get("error")
                        or "Invalid or non-finite CV score."
                    )
                ),
            }

            state["attempts"].append(entry)

            if not valid:
                continue

            if not changes:
                state["default_result"] = entry

            best = state["best_result"]

            if (
                best is None
                or _better(
                    score,
                    best["score"],
                    higher_is_better,
                )
            ):
                state["best_result"] = entry

        if timed_out:
            break

    # Reconstruct the existing public response format.
    for state in states:
        attempts = state["attempts"]

        if not attempts:
            continue

        estimator = state["estimator"]
        default_result = state["default_result"]
        best_result = state["best_result"]

        completed = (
            len(attempts) == len(state["configurations"])
        )

        selected = best_result
        improvement = None
        recommendation_status = "no_valid_configuration"

        if best_result is not None:
            recommendation_status = "best_tested"

            if default_result is not None:
                improvement = (
                    best_result["score"]
                    - default_result["score"]
                    if higher_is_better
                    else default_result["score"]
                    - best_result["score"]
                )

                if not _meaningful_improvement(
                    default_result["score"],
                    best_result["score"],
                    task=task,
                    higher_is_better=higher_is_better,
                ):
                    selected = default_result
                    recommendation_status = (
                        "defaults_recommended"
                        if completed
                        else "partial_defaults_recommended"
                    )

                elif best_result["changes"]:
                    recommendation_status = (
                        "parameter_changes_recommended"
                        if completed
                        else "partial_parameter_changes"
                    )

                else:
                    recommendation_status = (
                        "defaults_recommended"
                        if completed
                        else "partial_defaults_recommended"
                    )

            else:
                recommendation_status = "baseline_unavailable"

        recommended_changes = (
            dict(selected["changes"])
            if (
                selected is not None
                and recommendation_status
                in {
                    "parameter_changes_recommended",
                    "partial_parameter_changes",
                }
            )
            else {}
        )

        explained_changes = describe_changes(
            estimator,
            recommended_changes,
        )

        results.append({
            "model": state["name"],
            "scaler": state["scaler"],
            "status": (
                "success"
                if selected is not None
                else "failed"
            ),
            "best": selected,
            "best_score": (
                selected["score"]
                if selected is not None
                else None
            ),
            "mean_score": (
                selected["score"]
                if selected is not None
                else None
            ),
            "score": (
                selected["score"]
                if selected is not None
                else None
            ),
            "std_score": (
                selected["std_score"]
                if selected is not None
                else None
            ),
            "baseline_score": (
                default_result["score"]
                if default_result is not None
                else None
            ),
            "best_tested_score": (
                best_result["score"]
                if best_result is not None
                else None
            ),
            "improvement_over_default": improvement,
            "best_params": recommended_changes,
            "recommended_changes": explained_changes,
            "recommendation_status": recommendation_status,
            "attempts": attempts,
            "configurations_tested": len(attempts),
            "configurations_planned": len(
                state["configurations"]
            ),
            "completed": completed,
        })

    successful = [
        item
        for item in results
        if item["status"] == "success"
    ]

    successful.sort(
        key=lambda item: item["best_score"],
        reverse=higher_is_better,
    )

    elapsed = monotonic() - start

    return {
        "success": bool(successful),
        "method": "smart_top_five_cv_search",
        "level": level,
        "task": task,
        "metric": metric,
        "scoring": scoring,
        "higher_is_better": higher_is_better,
        "rows_evaluated": len(data),
        "cv_folds": 2,
        "requested_models": list(model_names),
        "models_attempted": len(results),
        "models_completed": sum(
            bool(item["completed"])
            for item in results
        ),
        "winner": (
            successful[0]
            if successful
            else None
        ),
        "ranking": successful,
        "results": results,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "note": (
            "Only important parameter configurations were tested. "
            "Changes are suggested when they improve cross-validation "
            "performance beyond a small practical threshold. "
            "A tuned result is not guaranteed to improve on new data. "
            "This is a separate 2-fold tuning experiment; its scores "
            "must not be directly compared with original AutoML "
            "finalist scores that used different validation settings."
        ),
    }

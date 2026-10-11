
"""
ModelMind - Empirical Preprocessing Validator

Compares scaling methods using the CURRENT dataset
and the SAME cross-validation folds.

No scaler or imputer is fitted on validation data.

This module does not modify the original dataset
or existing JupyterLab functionality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
)

from .pipeline_builder import (
    SCALING_OPTIONS,
    build_model_pipeline,
)


def _convert_number(value: float | None):
    """Convert numeric results to JSON-safe values."""
    if value is None:
        return None

    number = float(value)

    if not np.isfinite(number):
        return None

    return round(number, 6)


def validate_preprocessing(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    estimator: BaseEstimator,
    task: str,
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
    cv_folds: int = 5,
    random_state: int = 42,
    tie_tolerance: float = 0.005,
) -> dict[str, Any]:
    """
    Compare scaler choices using cross-validation.

    Classification metric:
        Balanced accuracy (higher is better)

    Regression metric:
        RMSE (lower is better)

    Scaling methods:
        none, standard, robust, minmax

    Returns measured scores and a recommendation.

    A near-tie is resolved in favor of the simpler
    configuration, with the tie clearly disclosed.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if frame.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if not frame.columns.is_unique:
        raise ValueError(
            "Duplicate column names are not supported."
        )

    if task not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            "task must be classification or regression."
        )

    if not isinstance(
        target_column, str
    ) or not target_column.strip():
        raise ValueError(
            "Choose an explicit target column."
        )

    if target_column not in frame.columns:
        raise ValueError(
            f"Target '{target_column}' does not exist."
        )

    if not feature_columns:
        raise ValueError(
            "Select at least one input feature."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "Duplicate feature selections are not allowed."
        )

    if target_column in feature_columns:
        raise ValueError(
            "Target cannot also be an input feature."
        )

    unknown = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if unknown:
        raise ValueError(
            "Unknown input features: "
            + ", ".join(unknown)
        )

    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "estimator must be a scikit-learn estimator."
        )

    if not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError(
            "cv_folds must be at least 2."
        )

    if tie_tolerance < 0:
        raise ValueError(
            "tie_tolerance cannot be negative."
        )

    if task == "classification" and not (
        hasattr(estimator, "predict")
        and hasattr(estimator, "fit")
    ):
        raise ValueError(
            "Estimator must support fit and predict."
        )

    # Missing target rows cannot be used for
    # supervised learning.
    data = frame.dropna(
        subset=[target_column]
    ).copy()

    if data.empty:
        raise ValueError(
            "No rows have a valid target."
        )

    X = data[feature_columns]
    y = data[target_column]

    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression target must be numeric."
            )

        target_numbers = y.to_numpy(dtype=float)

        if not np.isfinite(target_numbers).all():
            raise ValueError(
                "Regression target contains "
                "infinite or invalid values."
            )

        if len(data) < cv_folds:
            raise ValueError(
                "Not enough target-complete rows "
                "for the requested CV folds."
            )

        cv = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        scoring = "neg_root_mean_squared_error"
        metric_name = "RMSE"
        higher_is_better = False

    else:
        counts = y.value_counts()

        if len(counts) < 2:
            raise ValueError(
                "Classification requires at least "
                "two target classes."
            )

        smallest_class = int(counts.min())

        if smallest_class < cv_folds:
            raise ValueError(
                "The smallest class has "
                f"{smallest_class} rows, but "
                f"{cv_folds} folds were requested. "
                "Reduce CV folds or provide "
                "more examples."
            )

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        scoring = "balanced_accuracy"
        metric_name = "Balanced accuracy"
        higher_is_better = True

    results: list[dict[str, Any]] = []

    for scaler_name in SCALING_OPTIONS:
        try:
            pipeline = build_model_pipeline(
                frame=data,
                target_column=target_column,
                feature_columns=feature_columns,
                estimator=clone(estimator),
                scaler=scaler_name,
                numeric_imputation=numeric_imputation,
                categorical_encoding=categorical_encoding,
            )

            # cross_val_score clones the full pipeline.
            # Imputation and scaling are fitted inside
            # EACH training fold only.
            raw_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

            if task == "regression":
                measured_scores = -raw_scores
            else:
                measured_scores = raw_scores

            if not np.isfinite(
                measured_scores
            ).all():
                raise ValueError(
                    "Non-finite validation metric."
                )

            results.append({
                "scaler": scaler_name,
                "status": "success",
                "mean_score": _convert_number(
                    np.mean(measured_scores)
                ),
                "std_score": _convert_number(
                    np.std(measured_scores)
                ),
                "fold_scores": [
                    _convert_number(score)
                    for score in measured_scores
                ],
                "error": None,
            })

        except Exception as exc:
            results.append({
                "scaler": scaler_name,
                "status": "failed",
                "mean_score": None,
                "std_score": None,
                "fold_scores": [],
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    successful = [
        item
        for item in results
        if item["status"] == "success"
        and item["mean_score"] is not None
    ]

    if not successful:
        return {
            "success": False,
            "recommended_scaler": None,
            "metric": metric_name,
            "higher_is_better": higher_is_better,
            "results": results,
            "error": (
                "All preprocessing candidates failed. "
                "Inspect the individual errors."
            ),
        }

    # Rank candidates by actual validation score.
    ranked = sorted(
        successful,
        key=lambda item: item["mean_score"],
        reverse=higher_is_better,
    )

    absolute_best = ranked[0]
    best_score = absolute_best["mean_score"]

    # Simpler preprocessing is preferred only when
    # its score is within the disclosed tolerance.
    simplicity_order = [
        "none",
        "standard",
        "robust",
        "minmax",
    ]

    near_best = [
        item
        for item in successful
        if abs(
            item["mean_score"] - best_score
        ) <= tie_tolerance
    ]

    selected = min(
        near_best,
        key=lambda item: simplicity_order.index(
            item["scaler"]
        ),
    )

    near_tie = len(near_best) > 1

    if near_tie:
        reason = (
            f"Multiple preprocessing methods scored "
            f"within {tie_tolerance} {metric_name} "
            "units of the best observed mean. "
            f"Selected '{selected['scaler']}' "
            "using the declared simplicity preference. "
            "This does not establish statistical "
            "superiority."
        )
    else:
        reason = (
            f"'{selected['scaler']}' had the best "
            f"observed mean cross-validation "
            f"{metric_name} among successful "
            "candidates. Results may vary "
            "across datasets and validation splits."
        )

    return {
        "success": True,
        "task": task,
        "target": target_column,
        "features": feature_columns,
        "model": type(estimator).__name__,
        "rows_evaluated": int(len(data)),
        "cv_folds": cv_folds,
        "random_state": random_state,
        "metric": metric_name,
        "higher_is_better": higher_is_better,
        "recommended_scaler": selected["scaler"],
        "recommended_score": selected["mean_score"],
        "absolute_best_scaler": absolute_best["scaler"],
        "absolute_best_score": absolute_best["mean_score"],
        "near_tie": near_tie,
        "tie_tolerance": tie_tolerance,
        "reason": reason,
        "results": results,
        "method": "cross_validation",
        "empirically_validated": True,
        "warnings": [
            "Validation results depend on the "
            "chosen model, scoring metric, "
            "dataset and cross-validation folds.",
            "Selecting preprocessing from these "
            "folds requires separate held-out "
            "evaluation for an unbiased final "
            "performance estimate.",
            "Target leakage, temporal dependencies "
            "and grouped observations require "
            "additional dataset-specific checks.",
        ],
    }

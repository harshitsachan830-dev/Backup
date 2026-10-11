
"""
ModelMind - Backward-Compatible Evaluated Recommendations

Keeps the existing educational recommendation output and
adds measured model and preprocessing results separately.

Does not modify:
- uploaded CSV files
- existing generated model code
- JupyterLab frontend
- original heuristic suitability scores

This is a standalone adapter until integrated into routes.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import load_dataset
from .dataset_validation import validate_dataframe
from .model_evaluator import evaluate_models
from .automl_result_adapter import build_automl_recommendations
from .automl_code_adapter import attach_automl_code
from .model_recommender import recommend_models
from .preprocessing_validator import validate_preprocessing
from .pipeline_builder import build_model_pipeline


def _choose_cv_folds(
    frame: pd.DataFrame,
    target: str,
    task: str,
    requested: int,
) -> int:
    """
    Choose a supported number of CV folds.

    Never allow folds with too few target observations
    or too few observations from any classification class.
    """
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested < 2
        or requested > 10
    ):
        raise ValueError(
            "cv_folds must be an integer between 2 and 10."
        )

    y = frame[target].dropna()

    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )
        maximum = int(counts.min())
    else:
        maximum = len(y)

    if maximum < 2:
        raise ValueError(
            "Not enough observations for cross-validation."
        )

    return min(requested, maximum)


def recommend_models_evaluated(
    path: str | Path,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    task: str | None = None,
    cv_folds: int = 5,
    evaluate_scalers: bool = False,
) -> dict[str, Any]:
    """
    Combine legacy educational recommendations and
    real measured cross-validation evidence.

    The original 'models' list is preserved.

    New top-level keys:
        dataset_validation
        empirical_evaluation
        preprocessing_evaluation
        recommendation_method

    The caller must enforce appropriate workspace
    authorization for the supplied file path.
    """

    path = Path(path)

    if not isinstance(target_column, str):
        return {
            "handled": False,
            "error": "Select a target column.",
        }

    target_column = target_column.strip()

    if not target_column:
        return {
            "handled": False,
            "error": (
                "Explicit target selection is required "
                "for evaluated recommendations."
            ),
        }

    if feature_columns is not None and (
        not isinstance(feature_columns, list)
        or not all(
            isinstance(name, str)
            for name in feature_columns
        )
    ):
        return {
            "handled": False,
            "error": "feature_columns must be a list of strings.",
        }

    if task not in (None, "classification", "regression"):
        return {
            "handled": False,
            "error": "Invalid ML task.",
        }

    try:
        frame = load_dataset(path)

        if not frame.columns.is_unique:
            raise ValueError(
                "Duplicate column names are not supported."
            )

        if target_column not in frame.columns:
            raise ValueError(
                f"Target '{target_column}' not found."
            )

        # Explicit task is preferable to automatic inference.
        # When omitted, preserve the legacy task inference.
        legacy = recommend_models(
            path,
            target_col=target_column,
        )

        if not legacy.get("handled"):
            return legacy

        resolved_task = task or legacy["task"]

        selected_features = (
            list(feature_columns)
            if feature_columns is not None
            else [
                name
                for name in frame.columns
                if name != target_column
            ]
        )

        validation = validate_dataframe(
            frame,
            target_column=target_column,
            feature_columns=selected_features,
            task=resolved_task,
        )

        if not validation["valid"]:
            return {
                "handled": False,
                "error": "Dataset validation failed.",
                "validation": validation,
            }

        folds = _choose_cv_folds(
            frame,
            target_column,
            resolved_task,
            cv_folds,
        )

        empirical = evaluate_models(
            frame,
            target_column=target_column,
            feature_columns=selected_features,
            task=resolved_task,
            cv_folds=folds,
        )

        result = dict(legacy)

        # Keep the existing models/code/tags/score/rank
        # exactly as the legacy system produced them.
        result["dataset_validation"] = validation
        result["empirical_evaluation"] = empirical
        result["automl_recommendations"] = build_automl_recommendations(empirical)
        if empirical.get("success"):
            result["automl_recommendations"] = attach_automl_code(
                result["automl_recommendations"],
                frame,
                target_column=target_column,
                feature_columns=selected_features,
                csv_path=str(path),
            )
        result["recommendation_method"] = {
            "legacy_model_scores": "heuristic_suitability",
            "measured_model_scores": "cross_validation",
            "note": (
                "Heuristic suitability scores are not "
                "measured accuracy or regression metrics. "
                "See empirical_evaluation for real "
                "validation performance."
            ),
        }

        result["preprocessing_evaluation"] = None

        if evaluate_scalers and empirical.get("success"):
            recommended_name = empirical["recommended_model"]

            # Resolve the same model-family estimator
            # used in the measured evaluator.
            from .model_evaluator import get_candidate_models

            candidates = get_candidate_models(
                resolved_task,
                random_state=42,
            )

            matching = [
                estimator
                for name, estimator, _ in candidates
                if name == recommended_name
            ]

            if matching:
                result["preprocessing_evaluation"] = (
                    validate_preprocessing(
                        frame,
                        target_column=target_column,
                        feature_columns=selected_features,
                        estimator=matching[0],
                        task=resolved_task,
                        cv_folds=folds,
                    )
                )

        return result

    except (ValueError, TypeError, OSError) as exc:
        return {
            "handled": False,
            "error": str(exc),
        }


"""
ModelMind - Verified Recommendation Adapter

Preserves original model recommendations and generates
verified Python code with optional dependency handling.

Existing ranking, scores, explanations and metadata
remain unchanged.

XGBoost recommendations receive dependency information
and a helpful runtime dependency guard.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .model_recommender import recommend_models
from .optional_dependencies import (
    check_model_dependency,
    add_runtime_dependency_guard,
)
from .verified_model_code import (
    MODEL_SPECS,
    generate_verified_model_code,
)


def recommend_models_with_verified_code(
    path: str | Path,
    target_col: str | None = None,
) -> dict[str, Any]:
    """
    Return original ModelMind recommendations with
    verified generated code.

    The caller must authorize the dataset path before
    calling this function from an HTTP endpoint.
    """

    dataset_path = Path(path).resolve(strict=True)

    # 1. Get the original model recommendations.
    original = recommend_models(
        dataset_path,
        target_col=target_col,
    )

    if not original.get("handled"):
        return original

    # Preserve the original data.
    result = deepcopy(original)

    selected_target = result.get("target_column")
    task = result.get("task")

    if (
        not isinstance(selected_target, str)
        or not selected_target
    ):
        raise ValueError(
            "Recommender did not return a valid target column."
        )

    if task not in {"classification", "regression"}:
        raise ValueError(
            "Recommender returned an unsupported task."
        )

    # 2. Read the dataset.
    df = pd.read_csv(dataset_path)

    if selected_target not in df.columns:
        raise ValueError(
            "Selected target is missing from the dataset."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Dataset contains duplicate column names."
        )

    # 3. Identify numerical and categorical features.
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in df.columns:
        if column == selected_target:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "No input features are available."
        )

    # 4. Update only the generated code and optional
    # dependency information for each recommendation.
    for model in result.get("models", []):
        model_id = model.get("id")

        if model_id not in MODEL_SPECS:
            raise ValueError(
                f"No verified generator for model: {model_id}"
            )

        expected_task = MODEL_SPECS[model_id][0]

        if expected_task != task:
            raise ValueError(
                f"Model {model_id} does not match task {task}."
            )

        verified_code = generate_verified_model_code(
            model_id=model_id,
            target_column=selected_target,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            csv_path=str(dataset_path),
        )

        # 5. Check optional library requirements.
        dependency = check_model_dependency(model_id)

        # All models preserve their existing recommendation
        # metadata. The generated code is updated.
        model["code"] = add_runtime_dependency_guard(
            model_id,
            verified_code,
        )

        # Only models requiring optional libraries
        # receive an additional dependency-status field.
        if dependency["optional_dependency"] is not None:
            model["dependency_status"] = dependency

    return result

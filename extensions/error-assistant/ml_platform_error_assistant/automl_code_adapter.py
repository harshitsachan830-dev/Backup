
"""
ModelMind AutoML code adapter.

Adds the correct executable Python script to each measured
AutoML finalist.

Does not alter model rankings or original ModelMind cards.
Supports all currently registered regression and classification
AutoML candidates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .verified_model_code import (
    generate_verified_model_code,
    MODEL_SPECS,
)


MODEL_IDS: dict[str, dict[str, str]] = {
    "regression": {
        "Linear Regression": "linear_regression",
        "Ridge Regression": "ridge_reg",
        "Lasso Regression": "lasso_reg",
        "Elastic Net": "elastic_net_reg",
        "Decision Tree": "decision_tree_reg",
        "Random Forest": "random_forest_reg",
        "Extra Trees": "extra_trees_reg",
        "Gradient Boosting": "gradient_boosting_reg",
        "Hist Gradient Boosting": "hist_gradient_boosting_reg",
        "K-Nearest Neighbors": "knn_reg",
        "Support Vector Regression": "svr",
        "XGBoost Regressor": "xgboost_reg",
    },
    "classification": {
        "Logistic Regression": "logistic_regression",
        "Ridge Classifier": "ridge_clf",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest_clf",
        "Extra Trees": "extra_trees_clf",
        "Gradient Boosting": "gradient_boosting_clf",
        "Hist Gradient Boosting": "hist_gradient_boosting_clf",
        "K-Nearest Neighbors": "knn_clf",
        "Linear SVM": "linear_svm_clf",
        "RBF SVM": "svm_clf",
        "Gaussian Naive Bayes": "naive_bayes",
        "XGBoost Classifier": "xgboost_clf",
    },
}


def attach_automl_code(
    automl: dict[str, Any],
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    csv_path: str,
) -> dict[str, Any]:
    """
    Return AutoML recommendations enriched with generated code.

    Rankings and scores are preserved exactly.
    A failed code generation never substitutes another model.
    """

    if not isinstance(automl, dict):
        raise TypeError("AutoML result must be a dictionary.")

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset must be a DataFrame.")

    task = automl.get("task")

    if task not in MODEL_IDS:
        raise ValueError(f"Unsupported AutoML task: {task}")

    if not isinstance(target_column, str):
        raise TypeError("Target column must be a string.")

    if target_column not in frame.columns:
        raise ValueError("Target column is missing from dataset.")

    if not isinstance(feature_columns, list):
        raise TypeError("Feature columns must be a list.")

    if not feature_columns:
        raise ValueError("At least one feature is required.")

    if target_column in feature_columns:
        raise ValueError("Target cannot also be a feature.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature columns are not allowed.")

    missing = [
        col
        for col in feature_columns
        if not isinstance(col, str) or col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Invalid feature columns: {missing}"
        )

    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(frame[col])
    ]

    categorical_columns = [
        col
        for col in feature_columns
        if col not in numeric_columns
    ]

    output = dict(automl)
    enriched = []

    recommendations = automl.get("recommendations", [])

    if not isinstance(recommendations, list):
        raise ValueError("Recommendations must be a list.")

    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue

        item = dict(recommendation)
        name = item.get("name")

        model_id = MODEL_IDS[task].get(name)

        item["model_id"] = model_id
        item["code"] = None
        item["code_available"] = False
        item["code_error"] = None

        if model_id is None:
            item["code_error"] = (
                "No verified code generator exists for this model."
            )
            enriched.append(item)
            continue

        if model_id not in MODEL_SPECS:
            item["code_error"] = (
                "The model is not registered for code generation."
            )
            enriched.append(item)
            continue

        if MODEL_SPECS[model_id][0] != task:
            item["code_error"] = (
                "Model task does not match the selected dataset task."
            )
            enriched.append(item)
            continue

        try:
            code = generate_verified_model_code(
                model_id=model_id,
                target_column=target_column,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                csv_path=csv_path,
            )

            if not isinstance(code, str) or not code.strip():
                raise ValueError("Generated code is empty.")

            item["code"] = code
            item["code_available"] = True

        except (ValueError, TypeError, SyntaxError, KeyError) as exc:
            item["code_error"] = str(exc)

        enriched.append(item)

    output["recommendations"] = enriched

    if enriched:
        output["winner"] = enriched[0]

    output["code_generation"] = {
        "total": len(enriched),
        "available": sum(
            item["code_available"]
            for item in enriched
        ),
        "note": (
            "Generated code uses the selected model family. "
            "Training configuration or holdout metrics may "
            "differ from the AutoML cross-validation settings."
        ),
    }

    return output


"""
ModelMind - Target-Aware Dataset Validation

Phase 1 of the backend reliability upgrade.

This module:
- Requires explicit target selection
- Separates features and targets
- Checks missing targets and feature quality
- Flags possible identifier and leakage columns
- Returns explainable, deterministic diagnostics

It does NOT:
- Train a model
- Automatically delete columns
- Fit preprocessing to the full dataset
- Claim to identify the best ML algorithm
"""

from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import load_dataset


def _is_missing(value: Any) -> bool:
    """Return True for a scalar missing value."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _id_like_name(name: str) -> bool:
    """Detect common identifier naming patterns."""
    normalized = str(name).strip().lower()

    return (
        normalized in {
            "id",
            "uuid",
            "guid",
            "index",
            "row_id",
            "record_id",
        }
        or normalized.endswith("_id")
        or normalized.startswith("id_")
        or normalized.endswith("_uuid")
    )


def _possible_leakage_name(
    feature_name: str,
    target_name: str,
) -> bool:
    """
    Identify suspicious names only.

    A name-based warning is NOT proof of leakage.
    """
    feature = feature_name.strip().lower()
    target = target_name.strip().lower()

    if feature == target:
        return True

    if feature in {
        f"{target}_prediction",
        f"{target}_predicted",
        f"predicted_{target}",
        f"{target}_label",
        f"{target}_outcome",
    }:
        return True

    return False


def validate_dataframe(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """
    Validate supervised ML dataset configuration.

    Returns a structured report containing:
      valid
      errors
      warnings
      target
      features
      dataset
      feature_analysis

    Does not mutate the input DataFrame.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if not isinstance(target_column, str):
        raise ValueError(
            "Select a target column before analysis."
        )

    target_column = target_column.strip()

    if not target_column:
        errors.append(
            "Target column is required. "
            "ModelMind must not silently "
            "choose the last CSV column."
        )

    if df.empty:
        errors.append(
            "The dataset contains no observations."
        )

    if len(df.columns) == 0:
        errors.append(
            "The dataset contains no columns."
        )

    if not df.columns.is_unique:
        errors.append(
            "Duplicate column names were detected. "
            "Rename them before training."
        )

    all_columns = list(df.columns)

    if any(
        not isinstance(column, str)
        for column in all_columns
    ):
        errors.append(
            "All column names must be strings."
        )

    if (
        target_column
        and target_column not in all_columns
    ):
        errors.append(
            f"Target '{target_column}' "
            "does not exist in the dataset."
        )

    allowed_tasks = {
        None,
        "classification",
        "regression",
    }

    if task not in allowed_tasks:
        errors.append(
            "Task must be classification, "
            "regression, or None."
        )

    if feature_columns is None:
        selected_features = [
            column
            for column in all_columns
            if column != target_column
        ]
    else:
        if not isinstance(feature_columns, list):
            raise TypeError(
                "feature_columns must be a list."
            )

        selected_features = list(feature_columns)

    if len(set(selected_features)) != len(
        selected_features
    ):
        errors.append(
            "Duplicate feature selections detected."
        )

    if target_column in selected_features:
        errors.append(
            "The target column cannot also "
            "be an input feature."
        )

    unknown_features = [
        name
        for name in selected_features
        if name not in all_columns
    ]

    if unknown_features:
        errors.append(
            "Unknown feature columns: "
            + ", ".join(map(str, unknown_features))
        )

    if not selected_features:
        errors.append(
            "Select at least one input feature."
        )

    total_rows = int(len(df))
    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows} duplicate rows detected. "
            "Duplicates can inflate validation scores "
            "if near-identical records occur across splits."
        )

    target_missing = None
    target_unique = None
    target_dtype = None
    target_numeric = None
    complete_target_rows = None

    if (
        target_column
        and target_column in all_columns
        and df.columns.is_unique
    ):
        y = df[target_column]

        target_missing = int(y.isna().sum())

        target_unique = int(
            y.dropna().nunique()
        )

        target_dtype = str(y.dtype)

        target_numeric = bool(
            pd.api.types.is_numeric_dtype(y)
        )

        complete_target_rows = (
            total_rows - target_missing
        )

        if target_missing:
            warnings.append(
                f"Target '{target_column}' contains "
                f"{target_missing} missing values. "
                "Rows without a supervised target "
                "must be handled before training."
            )

        if target_unique < 2:
            errors.append(
                "The selected target has fewer "
                "than two distinct non-missing values."
            )

        if (
            task == "regression"
            and not target_numeric
        ):
            errors.append(
                "Regression requires a numeric target. "
                "Choose a numerical target or "
                "classification instead."
            )

        if task == "classification":
            counts = y.dropna().value_counts()

            if not counts.empty:
                smallest = int(counts.min())
                largest = int(counts.max())

                if smallest < 2:
                    warnings.append(
                        "At least one target class "
                        "contains fewer than two examples. "
                        "Stratified validation may fail."
                    )

                if (
                    smallest > 0
                    and largest / smallest >= 5
                ):
                    warnings.append(
                        "Class imbalance detected. "
                        "Consider stratified splitting "
                        "and metrics such as macro F1 "
                        "or balanced accuracy."
                    )

        if _id_like_name(target_column):
            warnings.append(
                f"Target '{target_column}' looks "
                "like an identifier. Confirm that "
                "this is intentional."
            )

    feature_analysis: list[dict[str, Any]] = []

    for name in selected_features:
        if name not in all_columns:
            continue

        if not df.columns.is_unique:
            continue

        if name == target_column:
            continue

        column = df[name]

        non_missing = column.dropna()
        non_missing_count = int(
            len(non_missing)
        )

        missing_count = int(
            column.isna().sum()
        )

        unique_count = int(
            non_missing.nunique()
        )

        numeric = bool(
            pd.api.types.is_numeric_dtype(column)
        )

        unique_ratio = (
            unique_count / non_missing_count
            if non_missing_count
            else 0.0
        )

        id_like = _id_like_name(name)

        high_cardinality = bool(
            not numeric
            and unique_count >= 20
            and unique_ratio >= 0.8
        )

        constant = unique_count <= 1

        suspicious_leakage = (
            _possible_leakage_name(
                name,
                target_column,
            )
        )

        reasons: list[str] = []

        if id_like:
            reasons.append(
                "Identifier-like column name."
            )

        if constant:
            reasons.append(
                "Constant or entirely missing feature."
            )

        if high_cardinality:
            reasons.append(
                "High-cardinality categorical feature."
            )

        if suspicious_leakage:
            reasons.append(
                "Feature name resembles target-derived "
                "information. Investigate leakage."
            )

        if missing_count == total_rows and total_rows:
            reasons.append(
                "All values are missing."
            )

        for reason in reasons:
            warnings.append(
                f"Feature '{name}': {reason}"
            )

        feature_analysis.append({
            "name": name,
            "dtype": str(column.dtype),
            "kind": (
                "numeric"
                if numeric
                else "categorical_or_other"
            ),
            "missing_count": missing_count,
            "missing_percentage": round(
                100 * missing_count / total_rows,
                2,
            ) if total_rows else 0.0,
            "unique_count": unique_count,
            "unique_ratio": round(
                unique_ratio,
                4,
            ),
            "is_constant": constant,
            "is_likely_identifier": id_like,
            "high_cardinality": high_cardinality,
            "possible_leakage": suspicious_leakage,
            "warnings": reasons,
        })

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "target": {
            "name": target_column,
            "dtype": target_dtype,
            "is_numeric": target_numeric,
            "missing_count": target_missing,
            "unique_count": target_unique,
            "usable_rows": complete_target_rows,
        },
        "task": task,
        "features": selected_features,
        "dataset": {
            "total_rows": total_rows,
            "total_columns": len(all_columns),
            "duplicate_rows": duplicate_rows,
        },
        "feature_analysis": feature_analysis,
        "method": "deterministic_dataset_validation",
        "empirically_validated": False,
    }


def validate_dataset(
    path: Path,
    target_column: str,
    feature_columns: list[str] | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """
    Load a dataset and validate its configuration.

    Caller must authorize and resolve paths.
    This function is not a filesystem access
    security boundary.
    """
    frame = load_dataset(Path(path))

    return validate_dataframe(
        df=frame,
        target_column=target_column,
        feature_columns=feature_columns,
        task=task,
    )

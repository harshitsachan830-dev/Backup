
"""
ModelMind - Leakage-Safe ML Pipeline Builder

Build preprocessing pipelines for the CURRENT dataset.

Features:
- Numeric missing-value imputation
- StandardScaler, RobustScaler, MinMaxScaler, or no scaler
- Categorical imputation and encoding
- Explicit input feature selection
- Optional estimator integration

IMPORTANT:
This module builds UNFITTED pipelines.
The caller must fit them only on training data or
inside cross-validation folds.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


SCALING_OPTIONS = (
    "none",
    "standard",
    "robust",
    "minmax",
)

NUMERIC_IMPUTATION_OPTIONS = (
    "mean",
    "median",
    "most_frequent",
)

CATEGORICAL_ENCODING_OPTIONS = (
    "onehot",
    "ordinal",
)


def _select_scaler(name: str):
    """Create a new, unfitted scaler instance."""

    if name == "none":
        return "passthrough"

    if name == "standard":
        return StandardScaler()

    if name == "robust":
        return RobustScaler()

    if name == "minmax":
        return MinMaxScaler()

    raise ValueError(
        f"Unsupported scaler: {name}. "
        f"Choose one of {SCALING_OPTIONS}."
    )


def _select_encoder(name: str):
    """Create an unfitted categorical encoder."""

    if name == "onehot":
        return OneHotEncoder(
            handle_unknown="ignore",
        )

    if name == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    raise ValueError(
        f"Unsupported encoder: {name}. "
        f"Choose one of {CATEGORICAL_ENCODING_OPTIONS}."
    )


def inspect_feature_types(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, list[str]]:
    """
    Identify numeric and categorical input columns.

    This function does not fit or transform data.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if not frame.columns.is_unique:
        raise ValueError(
            "Dataset contains duplicate column names."
        )

    if not feature_columns:
        raise ValueError(
            "Select at least one input feature."
        )

    if len(feature_columns) != len(
        set(feature_columns)
    ):
        raise ValueError(
            "Duplicate input features are not allowed."
        )

    missing_columns = [
        name
        for name in feature_columns
        if name not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "Unknown input columns: "
            + ", ".join(map(str, missing_columns))
        )

    numeric: list[str] = []
    categorical: list[str] = []

    for name in feature_columns:
        column = frame[name]

        if column.isna().all():
            raise ValueError(
                f"Feature '{name}' is entirely missing. "
                "Remove or repair this feature before training."
            )

        if pd.api.types.is_numeric_dtype(
            column
        ):
            numeric.append(name)
        else:
            categorical.append(name)

    return {
        "numeric": numeric,
        "categorical": categorical,
    }


def build_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    scaler: str = "standard",
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
) -> ColumnTransformer:
    """
    Return an unfitted preprocessing transformer.

    This function does NOT call fit(), fit_transform(),
    or learn statistics from the dataset.

    DataFrame inspection is limited to feature types
    and validating selected columns.
    """

    if scaler not in SCALING_OPTIONS:
        raise ValueError(
            f"Invalid scaler: {scaler}"
        )

    if (
        numeric_imputation
        not in NUMERIC_IMPUTATION_OPTIONS
    ):
        raise ValueError(
            "Invalid numeric imputation strategy."
        )

    if (
        categorical_encoding
        not in CATEGORICAL_ENCODING_OPTIONS
    ):
        raise ValueError(
            "Invalid categorical encoding strategy."
        )

    types = inspect_feature_types(
        frame,
        feature_columns,
    )

    numeric_columns = types["numeric"]
    categorical_columns = types["categorical"]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=numeric_imputation,
                ),
            ),
        ]

        numeric_steps.append(
            (
                "scaler",
                _select_scaler(scaler),
            )
        )

        numeric_pipeline = Pipeline(
            steps=numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    _select_encoder(
                        categorical_encoding
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable input features were selected."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    estimator: BaseEstimator,
    *,
    scaler: str = "standard",
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
) -> Pipeline:
    """
    Return an unfitted preprocessing + model pipeline.

    Target must be specified explicitly.

    Example:
        model = build_model_pipeline(
            frame,
            target_column="price",
            feature_columns=["size", "location"],
            estimator=Ridge(),
        )

        X_train, X_test, y_train, y_test = ...
        model.fit(X_train, y_train)

    During cross-validation, sklearn fits a fresh
    preprocessing pipeline inside each training fold.
    """

    if not isinstance(target_column, str):
        raise ValueError(
            "Explicit target column is required."
        )

    if not target_column.strip():
        raise ValueError(
            "Explicit target column is required."
        )

    if target_column not in frame.columns:
        raise ValueError(
            f"Target '{target_column}' not found."
        )

    if target_column in feature_columns:
        raise ValueError(
            "Target must never be used "
            "as an input feature."
        )

    if estimator is None:
        raise ValueError(
            "A machine learning estimator is required."
        )

    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "estimator must be a scikit-learn estimator."
        )

    preprocessor = build_preprocessor(
        frame,
        feature_columns,
        scaler=scaler,
        numeric_imputation=numeric_imputation,
        categorical_encoding=categorical_encoding,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", estimator),
        ]
    )

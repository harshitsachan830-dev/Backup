
"""
ModelMind - Fast AutoML Model Evaluation Engine

Purpose:
- Evaluate a broader set of regression/classification algorithms
- Include XGBoost when available
- Use fast candidate screening
- Validate the strongest candidates using cross-validation
- Preserve the existing ModelMind evaluation response structure

All preprocessing is fitted inside cross-validation folds.

Important:
The chosen model is the best among successfully evaluated
finalists, not necessarily the globally best ML algorithm.

No notebook UI, routes, or original heuristic recommender
are modified by this file.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    RidgeClassifier,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.svm import (
    LinearSVC,
    SVC,
    SVR,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from .pipeline_builder import build_model_pipeline


# Conservative defaults for the existing process worker.
# The supervisor in process_evaluation_jobs.py still enforces
# the actual hard timeout.
MAX_SCREENING_ROWS = 700
MAX_EVALUATION_ROWS = 2000
MAX_FINALISTS = 5


def safe_number(value: Any) -> float | None:
    """Convert finite numeric values into JSON-safe floats."""
    try:
        number = float(value)

        if np.isfinite(number):
            return round(number, 6)

    except (TypeError, ValueError, OverflowError):
        pass

    return None


def _xgboost_available() -> bool:
    """Check whether XGBoost is importable in this environment."""
    return importlib.util.find_spec("xgboost") is not None


def get_candidate_models(
    task: str,
    random_state: int = 42,
) -> list[tuple[str, Any, str]]:
    """
    Return candidates for the requested ML task.

    The list is not a performance ranking.

    Tuple:
        (display name, estimator, scaler)
    """

    if task == "regression":
        models: list[tuple[str, Any, str]] = [
            (
                "Linear Regression",
                LinearRegression(),
                "standard",
            ),
            (
                "Ridge Regression",
                Ridge(alpha=1.0),
                "standard",
            ),
            (
                "Lasso Regression",
                Lasso(alpha=1.0, max_iter=5000),
                "standard",
            ),
            (
                "Elastic Net",
                ElasticNet(
                    alpha=1.0,
                    l1_ratio=0.5,
                    max_iter=5000,
                ),
                "standard",
            ),
            (
                "Decision Tree",
                DecisionTreeRegressor(
                    max_depth=10,
                    random_state=random_state,
                ),
                "none",
            ),
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=40,
                    max_depth=12,
                    random_state=random_state,
                    n_jobs=1,
                ),
                "none",
            ),
            (
                "Extra Trees",
                ExtraTreesRegressor(
                    n_estimators=40,
                    max_depth=12,
                    random_state=random_state,
                    n_jobs=1,
                ),
                "none",
            ),
            (
                "Gradient Boosting",
                GradientBoostingRegressor(
                    n_estimators=60,
                    random_state=random_state,
                ),
                "none",
            ),
            (
                "Hist Gradient Boosting",
                HistGradientBoostingRegressor(
                    max_iter=60,
                    max_depth=6,
                    random_state=random_state,
                ),
                "none",
            ),
            (
                "K-Nearest Neighbors",
                KNeighborsRegressor(n_neighbors=5),
                "standard",
            ),
            (
                "Support Vector Regression",
                SVR(kernel="rbf"),
                "standard",
            ),
        ]

        if _xgboost_available():
            from xgboost import XGBRegressor

            models.append(
                (
                    "XGBoost Regressor",
                    XGBRegressor(
                        n_estimators=60,
                        max_depth=4,
                        learning_rate=0.1,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=1,
                        random_state=random_state,
                        verbosity=0,
                    ),
                    "none",
                )
            )

        return models

    if task == "classification":
        models = [
            (
                "Logistic Regression",
                LogisticRegression(max_iter=1500),
                "standard",
            ),
            (
                "Ridge Classifier",
                RidgeClassifier(),
                "standard",
            ),
            (
                "Decision Tree",
                DecisionTreeClassifier(
                    max_depth=10,
                    random_state=random_state,
                ),
                "none",
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=40,
                    max_depth=12,
                    random_state=random_state,
                    n_jobs=1,
                ),
                "none",
            ),
            (
                "Extra Trees",
                ExtraTreesClassifier(
                    n_estimators=40,
                    max_depth=12,
                    random_state=random_state,
                    n_jobs=1,
                ),
                "none",
            ),
            (
                "Gradient Boosting",
                GradientBoostingClassifier(
                    n_estimators=60,
                    random_state=random_state,
                ),
                "none",
            ),
            (
                "Hist Gradient Boosting",
                HistGradientBoostingClassifier(
                    max_iter=60,
                    max_depth=6,
                    random_state=random_state,
                ),
                "none",
            ),
            (
                "K-Nearest Neighbors",
                KNeighborsClassifier(n_neighbors=5),
                "standard",
            ),
            (
                "Linear SVM",
                LinearSVC(max_iter=3000),
                "standard",
            ),
            (
                "RBF SVM",
                SVC(kernel="rbf"),
                "standard",
            ),
            (
                "Gaussian Naive Bayes",
                GaussianNB(),
                "standard",
            ),
        ]

        if _xgboost_available():
            from xgboost import XGBClassifier

            models.append(
                (
                    "XGBoost Classifier",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=4,
                        learning_rate=0.1,
                        tree_method="hist",
                        eval_metric="logloss",
                        n_jobs=1,
                        random_state=random_state,
                        verbosity=0,
                    ),
                    "none",
                )
            )

        return models

    raise ValueError(
        "Task must be classification or regression."
    )


def _make_cv(
    task: str,
    folds: int,
    random_state: int,
):
    if task == "classification":
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )


def _sample_data(
    data: pd.DataFrame,
    *,
    target_column: str,
    task: str,
    max_rows: int,
    min_class_count: int,
    random_state: int,
) -> pd.DataFrame:
    """Create a reproducible bounded subset if it is safe."""

    if len(data) <= max_rows:
        return data.copy()

    if task == "classification":
        try:
            selected, _ = train_test_split(
                data,
                train_size=max_rows,
                stratify=data[target_column],
                random_state=random_state,
            )

            if (
                selected[target_column]
                .value_counts()
                .min() >= min_class_count
            ):
                return selected.copy()

        except ValueError:
            pass

        # A representative bounded stratified sample could
        # not preserve enough rows in every class.
        # Refuse to silently drop minority classes.
        raise ValueError(
            "Could not create a safe bounded classification "
            "sample while preserving enough examples from "
            "every class. Reduce class imbalance, increase "
            "the evaluation budget, or select suitable data."
        )

    return data.sample(
        n=max_rows,
        random_state=random_state,
    ).copy()


def _evaluate_one(
    *,
    model_name: str,
    estimator: Any,
    scaler: str,
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    task: str,
    folds: int,
    random_state: int,
    numeric_imputation: str,
    categorical_encoding: str,
    scoring: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Fit one candidate through leakage-safe CV."""

    try:
        model = clone(estimator)

        pipeline = build_model_pipeline(
            frame=data,
            target_column=target_column,
            feature_columns=feature_columns,
            estimator=model,
            scaler=scaler,
            numeric_imputation=numeric_imputation,
            categorical_encoding=categorical_encoding,
        )

        X = data[feature_columns]
        y = data[target_column]

        # XGBoost classification requires integer classes.
        # Encoding the labels does not fit feature
        # preprocessing on validation folds.
        if (
            task == "classification"
            and model_name == "XGBoost Classifier"
        ):
            label_encoder = LabelEncoder()
            y = pd.Series(
                label_encoder.fit_transform(y),
                index=y.index,
            )

        cv = _make_cv(
            task,
            folds,
            random_state,
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        measured = (
            scores if higher_is_better else -scores
        )

        if not np.isfinite(measured).all():
            raise ValueError(
                "Non-finite cross-validation scores."
            )

        return {
            "model": model_name,
            "status": "success",
            "scaler": scaler,
            "mean_score": safe_number(np.mean(measured)),
            "std_score": safe_number(np.std(measured)),
            "fold_scores": [
                safe_number(score)
                for score in measured
            ],
            "error": None,
        }

    except Exception as error:
        return {
            "model": model_name,
            "status": "failed",
            "scaler": scaler,
            "mean_score": None,
            "std_score": None,
            "fold_scores": [],
            "error": (
                f"{type(error).__name__}: {error}"
            ),
        }


def evaluate_models(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    task: str,
    cv_folds: int = 5,
    random_state: int = 42,
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
) -> dict[str, Any]:
    """
    Two-stage model comparison.

    Stage 1:
        Screen all eligible candidates with up to
        700 rows and 2-fold CV.

    Stage 2:
        Validate up to 5 finalists with the requested
        CV fold count and up to 2000 rows.

    Existing response fields remain available.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if frame.empty:
        raise ValueError("Dataset is empty.")

    if not frame.columns.is_unique:
        raise ValueError(
            "Duplicate column names are not supported."
        )

    if task not in {"regression", "classification"}:
        raise ValueError("Invalid ML task.")

    if (
        not isinstance(target_column, str)
        or target_column not in frame.columns
    ):
        raise ValueError(
            "A valid target column is required."
        )

    if not isinstance(feature_columns, list):
        raise ValueError(
            "feature_columns must be a list."
        )

    if not feature_columns:
        raise ValueError(
            "At least one feature is required."
        )

    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError(
            "Duplicate features are not allowed."
        )

    if target_column in feature_columns:
        raise ValueError(
            "Target leakage: target is included in features."
        )

    unknown = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}"
        )

    if (
        isinstance(cv_folds, bool)
        or not isinstance(cv_folds, int)
        or cv_folds < 2
    ):
        raise ValueError(
            "cv_folds must be an integer >= 2."
        )

    data = frame.dropna(
        subset=[target_column]
    ).copy()

    if len(data) < cv_folds:
        raise ValueError(
            "Not enough rows for cross-validation."
        )

    y = data[target_column]

    if task == "classification":
        counts = y.value_counts()

        if len(counts) < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

        if counts.min() < cv_folds:
            raise ValueError(
                "Smallest class has fewer rows than CV folds."
            )

        scoring = "balanced_accuracy"
        metric = "Balanced accuracy"
        higher_is_better = True

    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression requires a numeric target."
            )

        values = y.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Regression target contains invalid values."
            )

        scoring = "neg_root_mean_squared_error"
        metric = "RMSE"
        higher_is_better = False

    candidates = get_candidate_models(
        task=task,
        random_state=random_state,
    )

    # Stage 1 uses at most two CV folds.
    screening_folds = 2

    screening_data = _sample_data(
        data,
        target_column=target_column,
        task=task,
        max_rows=MAX_SCREENING_ROWS,
        min_class_count=screening_folds,
        random_state=random_state,
    )

    screening_results = []

    for name, estimator, scaler in candidates:
        result = _evaluate_one(
            model_name=name,
            estimator=estimator,
            scaler=scaler,
            data=screening_data,
            target_column=target_column,
            feature_columns=feature_columns,
            task=task,
            folds=screening_folds,
            random_state=random_state,
            numeric_imputation=numeric_imputation,
            categorical_encoding=categorical_encoding,
            scoring=scoring,
            higher_is_better=higher_is_better,
        )

        screening_results.append(result)

    screened_success = [
        result
        for result in screening_results
        if (
            result["status"] == "success"
            and result["mean_score"] is not None
        )
    ]

    screened_success.sort(
        key=lambda row: row["mean_score"],
        reverse=higher_is_better,
    )

    finalist_names = {
        result["model"]
        for result in screened_success[:MAX_FINALISTS]
    }

    # Stage 2 evaluates finalists on a larger bounded
    # dataset with the requested number of folds.
    final_data = _sample_data(
        data,
        target_column=target_column,
        task=task,
        max_rows=MAX_EVALUATION_ROWS,
        min_class_count=cv_folds,
        random_state=random_state,
    )

    final_results = []

    for name, estimator, scaler in candidates:
        if name not in finalist_names:
            continue

        result = _evaluate_one(
            model_name=name,
            estimator=estimator,
            scaler=scaler,
            data=final_data,
            target_column=target_column,
            feature_columns=feature_columns,
            task=task,
            folds=cv_folds,
            random_state=random_state,
            numeric_imputation=numeric_imputation,
            categorical_encoding=categorical_encoding,
            scoring=scoring,
            higher_is_better=higher_is_better,
        )

        final_results.append(result)

    ranked = [
        row
        for row in final_results
        if (
            row["status"] == "success"
            and row["mean_score"] is not None
        )
    ]

    ranked.sort(
        key=lambda row: row["mean_score"],
        reverse=higher_is_better,
    )

    warnings = [
        "Model selection depends on the current dataset.",
        "Only finalists were evaluated with the full "
        "cross-validation procedure.",
        "Screened-out models have not been proven inferior.",
        "The chosen model is the best observed finalist, "
        "not necessarily the best possible algorithm.",
        "For an unbiased final estimate, use a separate "
        "test set or nested cross-validation.",
        "Time-series and grouped data require appropriate "
        "cross-validation strategies.",
    ]

    if not _xgboost_available():
        warnings.append(
            "XGBoost was skipped because its optional "
            "Python package is not installed."
        )

    if len(data) > MAX_EVALUATION_ROWS:
        warnings.append(
            "Evaluation used a bounded representative sample "
            "rather than every dataset row."
        )

    if not ranked:
        return {
            "success": False,
            "task": task,
            "target": target_column,
            "metric": metric,
            "higher_is_better": higher_is_better,
            "recommended_model": None,
            "recommended_score": None,
            "recommended_scaler": None,
            "ranking": [],
            "results": final_results,
            "screening_results": screening_results,
            "rows_evaluated": len(final_data),
            "cv_folds": cv_folds,
            "error": "No finalist completed successfully.",
            "empirically_validated": False,
            "warnings": warnings,
        }

    winner = ranked[0]

    return {
        "success": True,
        "task": task,
        "target": target_column,
        "features": list(feature_columns),
        "rows_evaluated": len(final_data),
        "cv_folds": cv_folds,
        "random_state": random_state,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "recommended_model": winner["model"],
        "recommended_score": winner["mean_score"],
        "recommended_scaler": winner["scaler"],
        "ranking": ranked,
        "results": final_results,
        "screening_results": screening_results,
        "screening_rows": len(screening_data),
        "screening_folds": screening_folds,
        "candidate_count": len(candidates),
        "finalist_count": len(final_results),
        "method": "two_stage_cross_validation",
        "empirically_validated": True,
        "reason": (
            f"{winner['model']} achieved the best observed "
            f"{metric} among the successfully evaluated "
            "finalists."
        ),
        "warnings": warnings,
    }

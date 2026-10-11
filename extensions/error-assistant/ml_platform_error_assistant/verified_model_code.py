
"""
ModelMind - Verified ML Code Generator

Generate self-contained Python scripts for supported models.

This module is separate from the existing recommender.
No existing API or UI behavior is changed yet.

Generated scripts:
- Load a CSV
- Select features and target explicitly
- Remove observations with missing targets
- Split into train/test subsets before fitting transforms
- Impute numeric and categorical features
- Encode categories safely
- Apply model-appropriate scaling
- Train the selected algorithm
- Evaluate predictions on held-out test data
"""

from __future__ import annotations

import ast
from typing import Any


MODEL_SPECS: dict[str, tuple[str, str, str]] = {
    "linear_regression": (
        "regression",
        "from sklearn.linear_model import LinearRegression",
        "LinearRegression()",
    ),
    "ridge_lasso": (
        "regression",
        "from sklearn.linear_model import Ridge, Lasso",
        "Ridge(alpha=1.0)",
    ),
    "random_forest_reg": (
        "regression",
        "from sklearn.ensemble import RandomForestRegressor",
        "RandomForestRegressor(n_estimators=40, random_state=42)",
    ),
    "gradient_boosting_reg": (
        "regression",
        "from sklearn.ensemble import GradientBoostingRegressor",
        "GradientBoostingRegressor(random_state=42)",
    ),
    "xgboost_reg": (
        "regression",
        "from xgboost import XGBRegressor",
        "XGBRegressor(n_estimators=60, random_state=42, n_jobs=1)",
    ),
    "logistic_regression": (
        "classification",
        "from sklearn.linear_model import LogisticRegression",
        "LogisticRegression(max_iter=1500)",
    ),
    "random_forest_clf": (
        "classification",
        "from sklearn.ensemble import RandomForestClassifier",
        "RandomForestClassifier(n_estimators=40, random_state=42)",
    ),
    "decision_tree": (
        "classification",
        "from sklearn.tree import DecisionTreeClassifier",
        "DecisionTreeClassifier(max_depth=10, random_state=42)",
    ),
    "svm_clf": (
        "classification",
        "from sklearn.svm import SVC",
        "SVC(kernel='rbf', C=1.0)",
    ),
    "svr": (
        "regression",
        "from sklearn.svm import SVR",
        "SVR(C=1.0)",
    ),
    "knn_clf": (
        "classification",
        "from sklearn.neighbors import KNeighborsClassifier",
        "KNeighborsClassifier(n_neighbors=5)",
    ),
    "naive_bayes": (
        "classification",
        "from sklearn.naive_bayes import GaussianNB",
        "GaussianNB()",
    ),
    "xgboost_clf": (
        "classification",
        "from xgboost import XGBClassifier",
        "XGBClassifier(n_estimators=60, random_state=42, n_jobs=1, eval_metric='logloss')",
    ),
    'ridge_reg': (
        'regression',
        'from sklearn.linear_model import Ridge',
        'Ridge(alpha=1.0)',
    ),
    'lasso_reg': (
        'regression',
        'from sklearn.linear_model import Lasso',
        'Lasso(alpha=1.0, max_iter=5000)',
    ),
    'elastic_net_reg': (
        'regression',
        'from sklearn.linear_model import ElasticNet',
        'ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)',
    ),
    'decision_tree_reg': (
        'regression',
        'from sklearn.tree import DecisionTreeRegressor',
        'DecisionTreeRegressor(max_depth=10, random_state=42)',
    ),
    'extra_trees_reg': (
        'regression',
        'from sklearn.ensemble import ExtraTreesRegressor',
        'ExtraTreesRegressor(n_estimators=40, max_depth=12, random_state=42, n_jobs=1)',
    ),
    'hist_gradient_boosting_reg': (
        'regression',
        'from sklearn.ensemble import HistGradientBoostingRegressor',
        'HistGradientBoostingRegressor(max_iter=60, max_depth=6, random_state=42)',
    ),
    'knn_reg': (
        'regression',
        'from sklearn.neighbors import KNeighborsRegressor',
        'KNeighborsRegressor(n_neighbors=5)',
    ),
    'ridge_clf': (
        'classification',
        'from sklearn.linear_model import RidgeClassifier',
        'RidgeClassifier()',
    ),
    'extra_trees_clf': (
        'classification',
        'from sklearn.ensemble import ExtraTreesClassifier',
        'ExtraTreesClassifier(n_estimators=40, max_depth=12, random_state=42, n_jobs=1)',
    ),
    'gradient_boosting_clf': (
        'classification',
        'from sklearn.ensemble import GradientBoostingClassifier',
        'GradientBoostingClassifier(n_estimators=60, random_state=42)',
    ),
    'hist_gradient_boosting_clf': (
        'classification',
        'from sklearn.ensemble import HistGradientBoostingClassifier',
        'HistGradientBoostingClassifier(max_iter=60, max_depth=6, random_state=42)',
    ),
    'linear_svm_clf': (
        'classification',
        'from sklearn.svm import LinearSVC',
        'LinearSVC(max_iter=3000)',
    ),
}

# The original recommender also produces some models
# through task-specific generators. These IDs cover
# its current public recommendation identifiers.

SCALING_MODELS = {
    "linear_regression",
    "ridge_lasso",
    "logistic_regression",
    "svm_clf",
    "svr",
    "knn_clf",
    "naive_bayes",
    'elastic_net_reg',
    'knn_reg',
    'lasso_reg',
    'linear_svm_clf',
    'ridge_clf',
    'ridge_reg',
}


def generate_verified_model_code(
    model_id: str,
    target_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    csv_path: str = "your_dataset.csv",
) -> str:
    """
    Return syntactically validated, self-contained code.

    Does not execute the generated code.

    Runtime success depends on valid dataset values,
    installed dependencies, and adequate sample size.
    """

    if model_id not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported model ID: {model_id}"
        )

    if not isinstance(target_column, str) or not target_column:
        raise ValueError(
            "Explicit target column is required."
        )

    if not isinstance(csv_path, str) or not csv_path.strip():
        raise ValueError(
            "A CSV path is required."
        )

    if (
        not isinstance(numeric_columns, list)
        or not isinstance(categorical_columns, list)
        or not all(
            isinstance(name, str) and name
            for name in numeric_columns + categorical_columns
        )
    ):
        raise ValueError(
            "Feature columns must be lists of non-empty strings."
        )

    features = numeric_columns + categorical_columns

    if not features:
        raise ValueError(
            "At least one feature is required."
        )

    if len(features) != len(set(features)):
        raise ValueError(
            "Duplicate input feature names are not allowed."
        )

    if target_column in features:
        raise ValueError(
            "Target cannot be included in input features."
        )

    task, model_import, estimator = MODEL_SPECS[model_id]

    scaler_step = (
        '("scaler", StandardScaler()),'
        if model_id in SCALING_MODELS
        else ""
    )

    if task == "classification":
        split_extra = "stratify=y"
        target_preparation = """
from sklearn.preprocessing import LabelEncoder

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

if len(target_encoder.classes_) < 2:
    raise ValueError("Classification needs at least two classes.")

class_counts = pd.Series(y).value_counts()
if class_counts.min() < 2:
    raise ValueError(
        "Each class needs at least two rows for stratified splitting."
    )
"""
        evaluation = """
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)

print("Classification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

decoded_predictions = target_encoder.inverse_transform(
    predictions.astype(int)
)
print("Predictions:", decoded_predictions[:10])
"""
    else:
        split_extra = "shuffle=True"
        target_preparation = """
y = pd.to_numeric(y, errors="raise")
if not np.isfinite(y.to_numpy(dtype=float)).all():
    raise ValueError("Regression targets must be finite numbers.")
"""
        evaluation = """
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))
"""

    # OneHotEncoder produces dense numeric output for
    # compatibility with GaussianNB and all other models.
    # This is intended for manageable educational datasets,
    # not very large/high-cardinality production workloads.

    script = f'''
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
{model_import}

# 1. Load the dataset
df = pd.read_csv({csv_path!r})

target = {target_column!r}
numeric_features = {numeric_columns!r}
categorical_features = {categorical_columns!r}
feature_columns = numeric_features + categorical_features

required = feature_columns + [target]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError("Missing columns: " + str(missing_columns))

# 2. Remove only observations with unknown targets
df = df.dropna(subset=[target]).copy()

if len(df) < 10:
    raise ValueError("At least 10 target-complete rows are required.")

X = df[feature_columns]
y = df[target].copy()

{target_preparation}

# 3. Split BEFORE fitting preprocessing statistics
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    {split_extra},
)

# 4. Create preprocessing steps
transformers = []

if numeric_features:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        {scaler_step}
    ])
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

# 5. Fit preprocessing and model on training rows only
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", {estimator}),
])

pipeline.fit(X_train, y_train)

# 6. Evaluate on held-out test rows
predictions = pipeline.predict(X_test)

print("Model:", {model_id!r})
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

{evaluation}
'''

    if model_id == "ridge_lasso":
        # Both models use identical training-only preprocessing.
        # Choose by CV on the training data, never by test RMSE.
        original_training = """pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", Ridge(alpha=1.0)),
])

pipeline.fit(X_train, y_train)

# 6. Evaluate on held-out test rows
predictions = pipeline.predict(X_test)"""

        replacement_training = """from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

candidates = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
}

cv_results = {}
candidate_pipelines = {}

for name, estimator in candidates.items():
    candidate_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])

    # All CV folds are within X_train; the held-out test
    # set is never used to select the algorithm.
    fold_scores = cross_val_score(
        candidate_pipeline,
        X_train,
        y_train,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise",
    )

    mean_rmse = float(-fold_scores.mean())
    cv_results[name] = mean_rmse
    candidate_pipelines[name] = candidate_pipeline

    print(
        name,
        "training CV RMSE:",
        round(mean_rmse, 4),
    )

best_name = min(cv_results, key=cv_results.get)
pipeline = candidate_pipelines[best_name]

print("Selected algorithm:", best_name)

# Refit the selected algorithm on the full training split.
pipeline.fit(X_train, y_train)

# 6. Evaluate once on the untouched test split.
predictions = pipeline.predict(X_test)"""

        if script.count(original_training) != 1:
            raise RuntimeError(
                "Ridge/Lasso training template does not match "
                "the expected original code."
            )

        script = script.replace(
            original_training,
            replacement_training,
            1,
        )

    # Fail immediately if the generated program is invalid.
    ast.parse(
        script,
        filename=f"modelmind_generated_{model_id}.py",
    )

    return script


def supported_models() -> list[str]:
    """Return the supported original model identifiers."""
    return sorted(MODEL_SPECS)

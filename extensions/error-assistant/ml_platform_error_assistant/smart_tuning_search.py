
"""
ModelMind Smart Tuning - Phase 1

Choose important hyperparameter configurations for each
selected AutoML model.

Basic: a small, high-impact search.
Advanced: a broader search of important parameters.

This module does NOT change existing evaluations, jobs,
ranking, notebook code, or model defaults.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.model_selection import ParameterSampler


# Each dictionary maps an estimator class to important
# hyperparameters. Values are SEARCH CANDIDATES, not
# recommendations until cross-validation verifies them.

BASIC_SPACES: dict[str, dict[str, list[Any]]] = {
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 10, 15, 21],
        "weights": ["uniform", "distance"],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 10, 15, 21],
        "weights": ["uniform", "distance"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "DecisionTreeRegressor": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 10, 20, None],
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 10, 20, None],
    },
    "ExtraTreesClassifier": {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 10, 20, None],
    },
    "ExtraTreesRegressor": {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 10, 20, None],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [75, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [75, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "HistGradientBoostingClassifier": {
        "max_iter": [75, 100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "HistGradientBoostingRegressor": {
        "max_iter": [75, 100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "LogisticRegression": {
        "C": [0.1, 0.5, 1.0, 2.0, 10.0],
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 2.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.2, 0.5, 0.8],
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", 0.01, 0.1],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.1, 0.5],
    },
    "XGBClassifier": {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
    },
    "XGBRegressor": {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
    },
    "GaussianNB": {
        "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],
    },
}


ADVANCED_EXTRAS: dict[str, dict[str, list[Any]]] = {
    "KNeighborsClassifier": {
        "p": [1, 2],
    },
    "KNeighborsRegressor": {
        "p": [1, 2],
    },
    "DecisionTreeClassifier": {
        "min_samples_split": [2, 5, 10],
        "ccp_alpha": [0.0, 0.001, 0.01],
    },
    "DecisionTreeRegressor": {
        "min_samples_split": [2, 5, 10],
        "ccp_alpha": [0.0, 0.001, 0.01],
    },
    "RandomForestClassifier": {
        "max_features": ["sqrt", 0.7, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForestRegressor": {
        "max_features": [0.5, 0.7, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTreesClassifier": {
        "max_features": ["sqrt", 0.7, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTreesRegressor": {
        "max_features": [0.5, 0.7, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoostingClassifier": {
        "max_depth": [2, 3, 5],
        "subsample": [0.7, 0.9, 1.0],
    },
    "GradientBoostingRegressor": {
        "max_depth": [2, 3, 5],
        "subsample": [0.7, 0.9, 1.0],
    },
    "HistGradientBoostingClassifier": {
        "max_leaf_nodes": [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "HistGradientBoostingRegressor": {
        "max_leaf_nodes": [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "LogisticRegression": {
        "class_weight": [None, "balanced"],
    },
    "ElasticNet": {
        "max_iter": [1000, 3000],
    },
    "SVC": {
        "class_weight": [None, "balanced"],
    },
    "SVR": {
        "gamma": ["scale", 0.01, 0.1],
    },
    "XGBClassifier": {
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0],
    },
    "XGBRegressor": {
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0],
    },
}


def get_smart_search_space(
    estimator: Any,
    *,
    level: str = "basic",
) -> dict[str, list[Any]]:
    """Return supported, important parameter search values."""

    if level not in ("basic", "advanced"):
        raise ValueError("Level must be basic or advanced.")

    class_name = type(estimator).__name__
    supported = estimator.get_params(deep=False)

    space = {
        key: list(values)
        for key, values in BASIC_SPACES.get(
            class_name, {}
        ).items()
        if key in supported
    }

    if level == "advanced":
        space.update({
            key: list(values)
            for key, values in ADVANCED_EXTRAS.get(
                class_name, {}
            ).items()
            if key in supported
        })

    return space


def build_smart_trials(
    estimator: Any,
    *,
    level: str = "basic",
    max_trials: int | None = None,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Build candidate parameter changes.

    Include the default configuration as the baseline.
    Search trials contain only values different from
    the estimator's defaults.

    This is search planning, NOT score evaluation.
    """

    if max_trials is None:
        max_trials = 8 if level == "basic" else 20

    if (
        not isinstance(max_trials, int)
        or isinstance(max_trials, bool)
        or not 1 <= max_trials <= 100
    ):
        raise ValueError(
            "max_trials must be an integer from 1 to 100."
        )

    space = get_smart_search_space(
        estimator,
        level=level,
    )

    # An empty dictionary represents original defaults.
    trials: list[dict[str, Any]] = [{}]

    if not space or max_trials == 1:
        return trials

    defaults = estimator.get_params(deep=False)

    # Reserve capacity for the baseline.
    sampled = list(
        ParameterSampler(
            space,
            n_iter=max_trials - 1,
            random_state=random_state,
        )
    )

    for proposal in sampled:
        changes = {
            key: value
            for key, value in proposal.items()
            if defaults.get(key) != value
        }

        # Validate compatibility without fitting the model.
        candidate = clone(estimator)
        candidate.set_params(**changes)

        if changes not in trials:
            trials.append(changes)

    return trials


def describe_changes(
    estimator: Any,
    selected_changes: dict[str, Any],
) -> list[dict[str, Any]]:
    """Produce a simple old-value -> new-value explanation."""

    defaults = estimator.get_params(deep=False)
    result = []

    for key, value in selected_changes.items():
        if key not in defaults or defaults[key] == value:
            continue

        result.append({
            "parameter": key,
            "default": defaults[key],
            "recommended": value,
        })

    return result

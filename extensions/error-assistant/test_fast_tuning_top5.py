
"""
ModelMind Fast Tune Top 5 — Stage 2 tests.

Tests the separate tuning engine only.
Does not modify existing AutoML or notebook features.
"""

import json
import math
import time

import pandas as pd

from sklearn.datasets import make_classification, make_regression

from ml_platform_error_assistant.fast_hyperparameter_tuner import (
    tune_top_five,
)


def build_data(task):
    if task == "classification":
        X, y = make_classification(
            n_samples=150,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=42,
        )
    else:
        X, y = make_regression(
            n_samples=150,
            n_features=6,
            noise=8,
            random_state=42,
        )

    columns = [f"feature_{i}" for i in range(6)]
    frame = pd.DataFrame(X, columns=columns)
    frame["target"] = y

    return frame, columns


def test_top_five(task, names):
    frame, features = build_data(task)

    start = time.perf_counter()

    result = tune_top_five(
        frame,
        model_names=names,
        target_column="target",
        feature_columns=features,
        task=task,
        time_budget_seconds=20,
    )

    elapsed = time.perf_counter() - start

    assert result["success"], result
    assert result["models_attempted"] <= 5
    assert result["cv_folds"] == 2
    assert len(result["results"]) <= 5
    assert result["winner"] is not None

    seen = set()

    for item in result["results"]:
        assert item["model"] in names
        assert item["model"] not in seen
        seen.add(item["model"])

        # Basic Smart Tuning tests up to 8 configurations,
# including the original default configuration.
    assert 1 <= item["configurations_tested"] <= 8

    for attempt in item["attempts"]:
            if attempt["status"] == "success":
                assert attempt["score"] is not None
                assert math.isfinite(attempt["score"])
                assert isinstance(attempt["parameters"], dict)

    # The eventual API needs serializable results.
    json.dumps(result, allow_nan=False)

    ranking = result["ranking"]

    for earlier, later in zip(ranking, ranking[1:]):
        first = earlier["best"]["score"]
        second = later["best"]["score"]

        if result["higher_is_better"]:
            assert first >= second
        else:
            assert first <= second

    print(f"\n{task.upper()} TOP-5 TEST PASSED")
    print("Requested:", len(names))
    print("Attempted:", result["models_attempted"])
    print("Completed:", result["models_completed"])
    print("Timed out:", result["timed_out"])
    print("Winner:", result["winner"]["model"])
    print("Best score:", result["winner"]["best"]["score"])
    print("Best parameter changes:",
          result["winner"]["best"]["changes"])
    print("Elapsed:", round(elapsed, 3), "seconds")


def test_validation():
    frame, features = build_data("classification")

    def request(**overrides):
        kwargs = {
            "model_names": ["Decision Tree"],
            "target_column": "target",
            "feature_columns": features,
            "task": "classification",
            "time_budget_seconds": 5,
        }
        kwargs.update(overrides)
        return tune_top_five(frame, **kwargs)

    invalid_requests = [
        {"model_names": []},
        {"model_names": ["Decision Tree"] * 2},
        {"model_names": ["Unknown Model"]},
        {"model_names": ["Decision Tree"] * 6},
        {"feature_columns": ["target"]},
        {"feature_columns": ["nonexistent_feature"]},
        {"target_column": "missing_target"},
        {"task": "unknown"},
        {"time_budget_seconds": 0},
    ]

    for params in invalid_requests:
        try:
            request(**params)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"Invalid request was accepted: {params}"
            )

    print("\nINPUT VALIDATION PASSED")


if __name__ == "__main__":
    test_top_five(
        "classification",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
        ],
    )

    test_top_five(
        "regression",
        [
            "Linear Regression",
            "Ridge Regression",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors",
        ],
    )

    test_validation()

    print("\nALL STAGE-2 TOP-5 TESTS PASSED")

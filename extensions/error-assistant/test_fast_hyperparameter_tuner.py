
"""
Isolated smoke tests for ModelMind Fast Tune Top 5.
Uses synthetic data; no existing notebook files are changed.
"""

from sklearn.datasets import (
    make_classification,
    make_regression,
)
import pandas as pd

from ml_platform_error_assistant.fast_hyperparameter_tuner import (
    tune_top_five,
)


def check_classification():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    frame = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(5)]
    )
    frame["target"] = y

    result = tune_top_five(
        frame,
        model_names=[
            "Decision Tree",
            "Logistic Regression",
        ],
        target_column="target",
        feature_columns=[f"feature_{i}" for i in range(5)],
        task="classification",
        time_budget_seconds=20,
    )

    assert result["success"], result
    assert result["winner"]["model"] in {
        "Decision Tree",
        "Logistic Regression",
    }
    assert result["cv_folds"] == 2

    print("CLASSIFICATION PASSED")
    print("Winner:", result["winner"]["model"])
    print("Accuracy:", result["winner"]["best"]["score"])
    print("Time:", result["elapsed_seconds"], "seconds")


def check_regression():
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        noise=5,
        random_state=42,
    )

    frame = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(5)]
    )
    frame["target"] = y

    result = tune_top_five(
        frame,
        model_names=[
            "Decision Tree",
            "Ridge Regression",
        ],
        target_column="target",
        feature_columns=[f"feature_{i}" for i in range(5)],
        task="regression",
        time_budget_seconds=20,
    )

    assert result["success"], result
    assert result["winner"]["model"] in {
        "Decision Tree",
        "Ridge Regression",
    }
    assert result["winner"]["best"]["score"] >= 0

    print("REGRESSION PASSED")
    print("Winner:", result["winner"]["model"])
    print("RMSE:", result["winner"]["best"]["score"])
    print("Time:", result["elapsed_seconds"], "seconds")


if __name__ == "__main__":
    check_classification()
    check_regression()
    print("ALL FAST TUNING SMOKE TESTS PASSED")

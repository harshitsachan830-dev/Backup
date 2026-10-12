
"""
ModelMind Fast Tuning API validation tests.

No server startup, route registration, or network calls.
"""

from ml_platform_error_assistant.background_tuning_routes import (
    normalize_tuning_request,
)


def main():
    valid = {
        "file_path": "test_dataset.csv",
        "target_column": "target",
        "feature_columns": ["age", "income"],
        "task": "classification",
        "model_names": [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
        ],
        "time_budget_seconds": 20,
    }

    result = normalize_tuning_request(valid)

    assert len(result["model_names"]) == 5
    assert result["time_budget_seconds"] == 20

    print("VALID TOP-5 REQUEST PASSED")

    invalid_cases = [
        {"model_names": ["Nonexistent Model"]},
        {"model_names": ["Random Forest"] * 2},
        {"model_names": []},
        {"model_names": ["Random Forest"] * 6},
        {"feature_columns": ["target"]},
        {"feature_columns": []},
        {"task": "invalid"},
        {"time_budget_seconds": 0},
        {"time_budget_seconds": 151},
        {"time_budget_seconds": float("nan")},
        {"time_budget_seconds": True},
        {"unexpected_setting": "value"},
    ]

    for override in invalid_cases:
        request = dict(valid)
        request.update(override)

        try:
            normalize_tuning_request(request)
        except (ValueError, TypeError):
            continue

        raise AssertionError(
            f"Invalid request accepted: {override}"
        )

    print("INVALID REQUEST REJECTION PASSED")
    print("MODEL WHITELIST VALIDATION PASSED")
    print("TIME BUDGET VALIDATION PASSED")
    print("ALL FAST TUNING API VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()

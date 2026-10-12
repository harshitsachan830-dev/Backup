
"""
Independent background Fast Tuning tests.

Run as a Python script, not inside an interactive Python shell,
because Windows multiprocessing uses spawn.
"""

import json
import tempfile
import time

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from ml_platform_error_assistant.fast_tuning_jobs import (
    FastTuningJobManager,
)


def wait_for_job(manager, job_id, owner, timeout=60):
    started = time.monotonic()

    while time.monotonic() - started < timeout:
        state = manager.get(job_id, owner=owner)

        if state is None:
            raise AssertionError("Job unexpectedly missing.")

        if state["status"] in {
            "completed", "failed", "cancelled", "timed_out"
        }:
            return state

        time.sleep(0.1)

    raise TimeoutError("Test job did not finish.")


def main():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    features = [f"feature_{i}" for i in range(6)]

    frame = pd.DataFrame(X, columns=features)
    frame["target"] = y

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "test_dataset.csv"
        frame.to_csv(path, index=False)

        manager = FastTuningJobManager(
            max_pending=3,
            timeout_seconds=45,
        )

        try:
            job_id = manager.submit(
                path,
                owner="student_a",
                model_names=[
                    "Logistic Regression",
                    "Decision Tree",
                    "Random Forest",
                    "K-Nearest Neighbors",
                    "Gaussian Naive Bayes",
                ],
                target_column="target",
                feature_columns=features,
                task="classification",
                time_budget_seconds=20,
            )

            assert len(job_id) == 32

            # A different authenticated user must not see this job.
            assert manager.get(
                job_id, owner="student_b"
            ) is None

            assert manager.cancel(
                job_id, owner="student_b"
            ) is None

            state = wait_for_job(
                manager, job_id, "student_a"
            )

            print("JOB STATUS:", state["status"])

            if state["status"] != "completed":
                raise AssertionError(
                    f"Worker failed: {state['error']}"
                )

            result = state["result"]

            assert result["success"]
            assert result["models_attempted"] == 5
            assert result["winner"] is not None

            json.dumps(result, allow_nan=False)

            print(
                "BEST TUNED MODEL:",
                result["winner"]["model"],
            )
            print(
                "BEST PARAMETER CHANGES:",
                result["winner"]["best"]["changes"],
            )
            print(
                "TUNING TIME:",
                result["elapsed_seconds"],
            )

            print("OWNER ISOLATION PASSED")
            print("BACKGROUND EXECUTION PASSED")

            cancelled_id = manager.submit(
                path,
                owner="student_a",
                model_names=["Random Forest"],
                target_column="target",
                feature_columns=features,
                task="classification",
                time_budget_seconds=20,
            )

            cancelled = manager.cancel(
                cancelled_id,
                owner="student_a",
            )

            assert cancelled is not None

            cancelled_state = wait_for_job(
                manager, cancelled_id, "student_a"
            )

            assert cancelled_state["status"] in {
                "cancelled",
                "completed",
            }

            print(
                "CANCELLATION TEST:",
                cancelled_state["status"],
            )
            print("ALL BACKGROUND TESTS PASSED")

        finally:
            manager.shutdown()


if __name__ == "__main__":
    main()

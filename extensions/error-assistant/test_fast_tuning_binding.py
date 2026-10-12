
"""
Test Fast Tuning binding without starting Jupyter.

No existing source files are modified.
"""

from copy import deepcopy

from ml_platform_error_assistant.fast_tuning_binding import (
    TuningBindingError,
    verify_completed_top_five,
)


JOB_ID = "a" * 32


class FakeManager:
    def __init__(self, job):
        self.job = job

    def get(self, job_id):
        return deepcopy(self.job)


def make_job():
    names = [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Gaussian Naive Bayes",
    ]

    return {
        "status": "completed",
        "result": {
            "handled": True,
            "automl_recommendations": {
                "success": True,
                "task": "classification",
                "recommendations": [
                    {"name": name}
                    for name in names
                ],
            },
            "empirical_evaluation": {
                "success": True,
                "task": "classification",
                "target": "target",
                "features": ["age", "income"],
                "ranking": [
                    {
                        "model": name,
                        "status": "success",
                    }
                    for name in names
                ],
            },
        },
    }


def verify(job, owner="student_a", owner_check=None):
    if owner_check is None:
        owner_check = (
            lambda job_id, user:
            job_id == JOB_ID and user == "student_a"
        )

    return verify_completed_top_five(
        job_id=JOB_ID,
        owner=owner,
        evaluation_manager=FakeManager(job),
        check_owner=owner_check,
    )


def must_reject(job, owner="student_a", owner_check=None):
    try:
        verify(job, owner=owner, owner_check=owner_check)
    except TuningBindingError:
        return

    raise AssertionError("Invalid evaluation was accepted.")


def main():
    original = make_job()

    result = verify(original)

    assert result["verified"] is True
    assert result["dataset_identity_verified"] is False
    assert len(result["model_names"]) == 5
    assert result["model_names"][0] == "Logistic Regression"

    print("VALID TOP-5 BINDING PASSED")

    must_reject(original, owner="student_b")
    must_reject(
        original,
        owner_check=lambda job_id, user: False,
    )

    print("OWNER ISOLATION PASSED")

    changed = make_job()
    changed["status"] = "running"
    must_reject(changed)

    changed = make_job()
    changed["result"]["empirical_evaluation"]["success"] = False
    must_reject(changed)

    changed = make_job()
    changed["result"]["automl_recommendations"][
        "recommendations"
    ][0]["name"] = "Another Model"
    must_reject(changed)

    changed = make_job()
    changed["result"]["empirical_evaluation"][
        "features"
    ] = ["target"]
    must_reject(changed)

    changed = make_job()
    changed["result"]["automl_recommendations"][
        "recommendations"
    ].append({"name": "Extra Model"})
    must_reject(changed)

    print("INCOMPLETE/MISMATCHED JOB REJECTION PASSED")
    print("ALL TOP-5 BINDING TESTS PASSED")


if __name__ == "__main__":
    main()

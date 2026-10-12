
"""
ModelMind Fast Tuning — dataset identity tests.

No Jupyter server or existing AutoML files modified.
"""

import tempfile
from pathlib import Path

from ml_platform_error_assistant.fast_tuning_dataset_binding import (
    DatasetBindingError,
    register_evaluation_dataset,
    verify_evaluation_dataset,
    remove_evaluation_dataset,
)


JOB_ID = "a" * 32


def expect_rejected(callback):
    try:
        callback()
    except DatasetBindingError:
        return

    raise AssertionError("Invalid dataset binding was accepted.")


def main():
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)

        original = directory / "training.csv"
        different = directory / "different.csv"

        original.write_text(
            "age,income,target\n"
            "20,10000,0\n"
            "30,20000,1\n",
            encoding="utf-8",
        )

        different.write_text(
            "age,income,target\n"
            "20,10000,0\n"
            "30,20000,1\n",
            encoding="utf-8",
        )

        register_evaluation_dataset(
            job_id=JOB_ID,
            owner="student_a",
            dataset_path=original,
        )

        verified = verify_evaluation_dataset(
            job_id=JOB_ID,
            owner="student_a",
            dataset_path=original,
        )

        assert verified["verified"]
        assert verified["dataset_identity_verified"]

        print("ORIGINAL DATASET VERIFICATION PASSED")

        expect_rejected(
            lambda: verify_evaluation_dataset(
                job_id=JOB_ID,
                owner="student_b",
                dataset_path=original,
            )
        )

        print("OWNER ISOLATION PASSED")

        expect_rejected(
            lambda: verify_evaluation_dataset(
                job_id=JOB_ID,
                owner="student_a",
                dataset_path=different,
            )
        )

        print("DIFFERENT DATASET PATH REJECTED")

        original.write_text(
            "age,income,target\n"
            "20,10000,0\n"
            "30,99999,1\n",
            encoding="utf-8",
        )

        expect_rejected(
            lambda: verify_evaluation_dataset(
                job_id=JOB_ID,
                owner="student_a",
                dataset_path=original,
            )
        )

        print("MODIFIED DATASET CONTENT REJECTED")

        remove_evaluation_dataset(JOB_ID)

        expect_rejected(
            lambda: verify_evaluation_dataset(
                job_id=JOB_ID,
                owner="student_a",
                dataset_path=original,
            )
        )

        print("EXPIRED DATASET BINDING REJECTED")
        print("ALL DATASET IDENTITY TESTS PASSED")


if __name__ == "__main__":
    main()


"""
ModelMind — Fast Tuning Dataset Identity Binding.

Independent module for recording and verifying the dataset
associated with a completed AutoML evaluation.

Does not change existing AutoML selection or model training.

Each evaluation job is bound to:
- Authenticated owner
- Canonical workspace-authorized CSV path
- SHA-256 hash of the dataset contents

In-memory metadata is lost when Jupyter restarts.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from pathlib import Path
from threading import RLock
from typing import Any


_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_LOCK = RLock()
_BINDINGS: dict[str, dict[str, str]] = {}

MAX_BINDINGS = 100


class DatasetBindingError(ValueError):
    """Dataset identity cannot be verified."""


def _hash_dataset(path: Path) -> str:
    """Compute a streaming SHA-256 digest of the CSV contents."""

    if not path.is_file() or path.suffix.lower() != ".csv":
        raise DatasetBindingError(
            "An existing CSV dataset is required."
        )

    digest = hashlib.sha256()

    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

    except OSError as error:
        raise DatasetBindingError(
            "Cannot read the selected dataset."
        ) from error

    return digest.hexdigest()


def _validate_identity(
    job_id: str,
    owner: str,
) -> None:

    if (
        not isinstance(job_id, str)
        or not _JOB_ID_RE.fullmatch(job_id)
    ):
        raise DatasetBindingError(
            "Invalid evaluation job ID."
        )

    if not isinstance(owner, str) or not owner.strip():
        raise DatasetBindingError(
            "Authenticated owner required."
        )


def register_evaluation_dataset(
    *,
    job_id: str,
    owner: str,
    dataset_path: str | Path,
) -> None:
    """
    Register dataset metadata at evaluation submission.

    Caller MUST:
    - authenticate the owner
    - authorize the path against Jupyter's workspace
    - use the real submitted evaluation job ID
    """

    _validate_identity(job_id, owner)

    path = Path(dataset_path).resolve(strict=True)
    digest = _hash_dataset(path)

    binding = {
        "owner": owner,
        "path": str(path),
        "sha256": digest,
    }

    with _LOCK:
        if job_id in _BINDINGS:
            raise DatasetBindingError(
                "Evaluation dataset already registered."
            )

        if len(_BINDINGS) >= MAX_BINDINGS:
            raise DatasetBindingError(
                "Dataset-binding registry is full."
            )

        _BINDINGS[job_id] = binding


def verify_evaluation_dataset(
    *,
    job_id: str,
    owner: str,
    dataset_path: str | Path,
) -> dict[str, Any]:
    """
    Verify that tuning uses the exact recorded dataset.

    A matching path alone is insufficient: the file contents
    must also match the originally recorded SHA-256.
    """

    _validate_identity(job_id, owner)

    with _LOCK:
        stored = _BINDINGS.get(job_id)

        if stored is None:
            raise DatasetBindingError(
                "Evaluation dataset binding not found. "
                "Run Model Advisor again."
            )

        binding = dict(stored)

    if not hmac.compare_digest(binding["owner"], owner):
        raise DatasetBindingError(
            "Evaluation dataset binding not found."
        )

    path = Path(dataset_path).resolve(strict=True)

    if path != Path(binding["path"]):
        raise DatasetBindingError(
            "Dataset path differs from AutoML evaluation."
        )

    current_digest = _hash_dataset(path)

    if not hmac.compare_digest(
        binding["sha256"],
        current_digest,
    ):
        raise DatasetBindingError(
            "Dataset contents changed after Model Advisor "
            "evaluation. Run Model Advisor again."
        )

    return {
        "verified": True,
        "evaluation_job_id": job_id,
        "dataset_identity_verified": True,
        "dataset_path": str(path),
    }


def remove_evaluation_dataset(
    job_id: str,
) -> None:
    """Remove metadata for an expired evaluation job."""

    with _LOCK:
        _BINDINGS.pop(job_id, None)

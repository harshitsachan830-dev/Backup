
"""
ModelMind - Dataset Workspace Security

Validate a dataset path before passing it to
a ModelMind dataset analysis or evaluation function.

This module:
- Restricts access to an authorized workspace root
- Rejects directory traversal outside the workspace
- Resolves symlinks before checking access
- Accepts CSV files only for the initial integration
- Limits input file size

The Jupyter server must supply the trusted workspace
root. Never accept that root from an HTTP request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


PathInput = Union[str, Path]

DEFAULT_MAX_BYTES = 25 * 1024 * 1024


class DatasetAccessError(ValueError):
    """Dataset path failed an access or validation check."""


def resolve_workspace_dataset(
    file_path: PathInput,
    workspace_root: PathInput,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """
    Return the canonical path to an authorized CSV.

    Absolute paths are supported only when located
    inside the trusted workspace.

    Relative paths are resolved from workspace_root,
    not from an arbitrary HTTP-supplied directory.

    This is an authorization check on the resolved
    filesystem path, not a sandbox against concurrent
    filesystem modification or untrusted file contents.
    """

    if not isinstance(file_path, (str, Path)):
        raise DatasetAccessError(
            "Dataset path must be a string or Path."
        )

    if not isinstance(workspace_root, (str, Path)):
        raise DatasetAccessError(
            "Workspace root must be a string or Path."
        )

    raw_path = str(file_path).strip()

    if not raw_path:
        raise DatasetAccessError(
            "Dataset path cannot be empty."
        )

    if "\x00" in raw_path:
        raise DatasetAccessError(
            "Dataset path contains an invalid character."
        )

    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
    ):
        raise ValueError(
            "max_bytes must be a positive integer."
        )

    try:
        root = Path(workspace_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DatasetAccessError(
            "The configured workspace cannot be resolved."
        ) from error

    if not root.is_dir():
        raise DatasetAccessError(
            "The configured workspace is not a directory."
        )

    candidate = Path(raw_path)

    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DatasetAccessError(
            "Dataset file does not exist or cannot be resolved."
        ) from error

    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DatasetAccessError(
            "Access denied: dataset is outside "
            "the authorized Jupyter workspace."
        ) from error

    if not resolved.is_file():
        raise DatasetAccessError(
            "Dataset path must reference a regular file."
        )

    if resolved.suffix.lower() != ".csv":
        raise DatasetAccessError(
            "Only CSV dataset files are allowed "
            "for this evaluation endpoint."
        )

    try:
        file_size = resolved.stat().st_size
    except OSError as error:
        raise DatasetAccessError(
            "Unable to inspect the dataset file."
        ) from error

    if file_size == 0:
        raise DatasetAccessError(
            "The dataset file is empty."
        )

    if file_size > max_bytes:
        raise DatasetAccessError(
            f"Dataset exceeds the configured "
            f"{max_bytes:,}-byte limit."
        )

    return resolved

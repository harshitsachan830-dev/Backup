
"""
ModelMind - Optional ML Dependency Intelligence

Phase 10A

Provides:
- Detection of optional ML packages
- Student-friendly missing-package explanations
- Available scikit-learn alternatives
- Runtime dependency checks in generated notebook code

Does not install packages, modify rankings, or execute ML training.
"""

from __future__ import annotations

import ast
import importlib.util
from typing import Any


OPTIONAL_MODEL_DEPENDENCIES = {
    "xgboost_clf": {
        "package": "xgboost",
        "display_name": "XGBoost",
        "alternatives": [
            "Random Forest Classifier",
            "Decision Tree",
            "Logistic Regression",
        ],
    },
    "xgboost_reg": {
        "package": "xgboost",
        "display_name": "XGBoost",
        "alternatives": [
            "Gradient Boosting Regressor",
            "Random Forest Regressor",
            "Linear Regression",
        ],
    },
}


def check_model_dependency(model_id: str) -> dict[str, Any]:
    """
    Check package availability in the current Python interpreter.

    This reflects the interpreter executing this function.
    A separate Jupyter kernel can have different packages.
    """
    spec = OPTIONAL_MODEL_DEPENDENCIES.get(model_id)

    if spec is None:
        return {
            "model_id": model_id,
            "optional_dependency": None,
            "available": True,
            "message": "No special optional dependency required.",
            "alternatives": [],
        }

    package = spec["package"]

    try:
        available = importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        available = False

    if available:
        message = (
            f"{spec['display_name']} is available in this "
            "Python environment."
        )
    else:
        message = (
            f"{spec['display_name']} requires the optional "
            f"Python package '{package}', which was not found "
            "in this Python environment."
        )

    return {
        "model_id": model_id,
        "optional_dependency": package,
        "available": available,
        "message": message,
        "alternatives": list(spec["alternatives"]),
    }


def add_runtime_dependency_guard(
    model_id: str,
    generated_code: str,
) -> str:
    """
    Add a dependency check to generated notebook code.

    The generated code checks the actual notebook kernel
    environment before importing XGBoost.

    This does not install anything automatically.
    """
    if not isinstance(generated_code, str):
        raise TypeError("Generated code must be a string.")

    ast.parse(generated_code)

    spec = OPTIONAL_MODEL_DEPENDENCIES.get(model_id)

    if spec is None:
        return generated_code

    package = spec["package"]
    display_name = spec["display_name"]

    if not any(
        isinstance(node, ast.ImportFrom)
        and node.module == package
        for node in ast.parse(generated_code).body
    ):
        raise ValueError(
            f"Expected an import from '{package}' "
            "in the generated model code."
        )

    marker = "# ModelMind optional dependency check"

    if generated_code.startswith(marker):
        return generated_code

    alternatives = ", ".join(spec["alternatives"])

    guard = f'''{marker}
import importlib.util as _modelmind_importlib

if _modelmind_importlib.find_spec({package!r}) is None:
    raise ImportError(
        {display_name!r} + " is not installed in this "
        "notebook's Python kernel. "
        "To use this model, install the optional "
        {package!r} + " package into the active kernel "
        "environment, then restart the kernel if required. "
        "Or choose a model that uses scikit-learn: "
        {alternatives!r}
    )

'''

    updated = guard + generated_code

    ast.parse(updated)

    return updated

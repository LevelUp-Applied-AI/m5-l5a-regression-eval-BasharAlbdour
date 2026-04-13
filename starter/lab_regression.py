"""Compatibility wrapper for test import path.

Tests expect ``starter/lab_regression.py`` to exist. This shim loads the
project's root ``lab_regression.py`` and re-exports the lab functions.
"""

from pathlib import Path
import importlib.util


_ROOT_FILE = Path(__file__).resolve().parents[1] / "lab_regression.py"
_SPEC = importlib.util.spec_from_file_location("_root_lab_regression", _ROOT_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load module from {_ROOT_FILE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


load_data = _MODULE.load_data
split_data = _MODULE.split_data
build_logistic_pipeline = _MODULE.build_logistic_pipeline
build_ridge_pipeline = _MODULE.build_ridge_pipeline
evaluate_classifier = _MODULE.evaluate_classifier
evaluate_regressor = _MODULE.evaluate_regressor
run_cross_validation = _MODULE.run_cross_validation

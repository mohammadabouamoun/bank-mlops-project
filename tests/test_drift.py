# tests/test_drift.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from unittest.mock import patch
from platform_service.drift import DriftDetector, _psi, _chi2_pvalue

# -------------------------------------------------------------------
# Unit tests for utility functions
# -------------------------------------------------------------------
def test_psi_identical():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.2, 0.3, 0.5])
    assert _psi(p, q) < 1e-10

def test_psi_different():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.5, 0.3, 0.2])
    assert _psi(p, q) > 0.1

def test_chi2_no_drift():
    obs = {"cat1": 50, "cat2": 50}
    exp_props = {"cat1": 0.5, "cat2": 0.5}
    pval = _chi2_pvalue(obs, exp_props)
    assert pval > 0.05

def test_chi2_significant():
    obs = {"cat1": 90, "cat2": 10}
    exp_props = {"cat1": 0.5, "cat2": 0.5}
    pval = _chi2_pvalue(obs, exp_props)
    assert pval < 0.05

# -------------------------------------------------------------------
# DriftDetector integration test
# -------------------------------------------------------------------
@pytest.fixture
def reference_data(tmp_path):
    """Simplified reference – single bin per numeric/output, single category for job."""
    ref = {
        "numeric": {
            "age": {
                "bin_edges": [0, 100],
                "ref_proportions": [1.0]
            }
        },
        "categorical": {
            "job": {
                "categories": ["admin"],
                "ref_proportions": {"admin": 1.0}
            }
        },
        "output": {
            "bin_edges": [0, 1],
            "ref_proportions": [1.0]
        }
    }
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(ref))
    return ref_path

def test_drift_detector(reference_data, tmp_path):
    window_path = tmp_path / "window.csv"
    detector = DriftDetector(
        reference_path=reference_data,
        window_path=window_path,
        window_size=500,
        min_window_size=5,
    )
    # Add 10 identical predictions
    features = {"age": 45, "job": "admin"}
    proba = 0.1
    for _ in range(10):
        report = detector.check_and_report(features, proba)
    # The window is 10 rows, min_window_size=5, so a report should be generated.
    # The distribution should still be "low" severity (no drift).
    report = detector.compute_drift_report()
    assert report["severity"] == "low"
    assert "age" in report["psi_values"]
    assert "job" in report["chi2_values"]
    assert report["window_size"] == 10
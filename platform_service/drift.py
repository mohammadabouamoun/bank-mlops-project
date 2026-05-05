import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, Dict
from scipy.stats import chisquare
import structlog

log = structlog.get_logger(__name__)

# ----------------------------------------------------------------------
# 1. Utility functions
# ----------------------------------------------------------------------
def _psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-10) -> float:
    expected = np.where(expected == 0, epsilon, expected)
    actual   = np.where(actual == 0, epsilon, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _chi2_pvalue(
    observed_counts: dict,
    expected_proportions: dict,
    min_expected: int = 5,
) -> float:
    """
    Chi‑squared test with merging of small expected frequencies.
    Categories with expected count < min_expected are merged into '__other__'.
    """
    total_obs = sum(observed_counts.values())
    if total_obs == 0:
        return 1.0

    obs_merged = defaultdict(int)
    exp_merged = defaultdict(float)

    all_cats = set(observed_counts) | set(expected_proportions)
    for cat in sorted(all_cats):
        obs = observed_counts.get(cat, 0)
        exp_ratio = expected_proportions.get(cat, 0.0)
        exp_count = exp_ratio * total_obs
        if exp_count >= min_expected:
            obs_merged[cat] = obs
            exp_merged[cat] = exp_count
        else:
            obs_merged["__other__"] += obs
            exp_merged["__other__"] += exp_count

    categories = sorted(obs_merged.keys())
    observed = [obs_merged[c] for c in categories]
    expected = [exp_merged[c] for c in categories]
    if sum(expected) == 0 or len(observed) < 2:
        return 1.0
    _, p_value = chisquare(f_obs=observed, f_exp=expected)
    return p_value


# ----------------------------------------------------------------------
# 2. DriftDetector class
# ----------------------------------------------------------------------
class DriftDetector:
    def __init__(
        self,
        reference_path: Optional[Path] = None,
        window_path: Optional[Path] = None,
        window_size: int = 500,
        min_window_size: int = 100,
        psi_threshold_medium: float = 0.15,
        psi_threshold_high: float = 0.25,
        chi2_alpha: float = 0.05,
    ):
        self.reference_path = reference_path or Path("models/reference.json")
        self.window_path = window_path or Path("data/prediction_window.csv")
        self.window_size = window_size
        self.min_window_size = min_window_size

        self.psi_threshold_medium = psi_threshold_medium
        self.psi_threshold_high = psi_threshold_high
        self.chi2_alpha = chi2_alpha

        # Load reference distributions
        with open(self.reference_path, "r") as f:
            self.ref = json.load(f)

        # Load / initialise rolling window
        self.window = self._load_window()
        self._last_severity: str = "low"   # prevent webhook on first report if severity is low
    # ------------------------------------------------------------------
    # Window persistence
    # ------------------------------------------------------------------
    def _load_window(self) -> pd.DataFrame:
        if self.window_path.exists():
            df = pd.read_csv(self.window_path)
            return df.tail(self.window_size)
        return pd.DataFrame()

    def _save_window(self) -> None:
        self.window_path.parent.mkdir(parents=True, exist_ok=True)
        self.window.to_csv(self.window_path, index=False)

    # ------------------------------------------------------------------
    # Add a prediction
    # ------------------------------------------------------------------
    def add_prediction(self, features: Dict[str, float | str], probability: float) -> None:
        """Record a new prediction, trim window, and persist."""
        row = {**features, "probability": probability}
        new_row = pd.DataFrame([row])
        self.window = pd.concat([self.window, new_row], ignore_index=True)
        self.window = self.window.tail(self.window_size)
        self._save_window()       # ← persist after every new prediction

    # ------------------------------------------------------------------
    # Drift report
    # ------------------------------------------------------------------
    def compute_drift_report(self) -> dict:
        """Return drift metrics unless the window is too small."""
        if len(self.window) < self.min_window_size:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "low",
                "window_size": len(self.window),
                "psi_values": {},
                "chi2_values": {},
                "output_psi": 0.0,
                "drifted_features": {"numeric": [], "categorical": []},
            }

        psi_values = {}
        chi2_values = {}
        drifted_num = []
        drifted_cat = []

        # Numeric – PSI
        for col, info in self.ref["numeric"].items():
            if col not in self.window.columns:
                psi_values[col] = 0.0
                continue
            bin_edges = np.array(info["bin_edges"])
            ref_props = np.array(info["ref_proportions"])
            actual_counts, _ = np.histogram(self.window[col].astype(float), bins=bin_edges)
            if actual_counts.sum() == 0:
                psi = 0.0
            else:
                actual_props = actual_counts / actual_counts.sum()
                psi = _psi(ref_props, actual_props)
            psi_values[col] = psi
            if psi >= self.psi_threshold_medium:
                drifted_num.append(col)

        # Categorical – chi‑squared
        for col, info in self.ref["categorical"].items():
            if col not in self.window.columns:
                chi2_values[col] = 1.0
                continue
            ref_proportions = info["ref_proportions"]
            observed = self.window[col].astype(str).value_counts().to_dict()
            p_val = _chi2_pvalue(observed, ref_proportions, min_expected=5)
            chi2_values[col] = p_val
            if p_val < self.chi2_alpha:
                drifted_cat.append(col)

        # Output PSI
        output_psi = 0.0
        if "probability" in self.window.columns:
            bin_edges = np.array(self.ref["output"]["bin_edges"])
            ref_props = np.array(self.ref["output"]["ref_proportions"])
            actual_counts, _ = np.histogram(self.window["probability"].astype(float), bins=bin_edges)
            if actual_counts.sum() > 0:
                actual_props = actual_counts / actual_counts.sum()
                output_psi = _psi(ref_props, actual_props)

        severity = self._compute_severity(psi_values, chi2_values, output_psi,
                                          drifted_num, drifted_cat)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "psi_values": psi_values,
            "chi2_values": chi2_values,
            "output_psi": output_psi,
            "drifted_features": {"numeric": drifted_num, "categorical": drifted_cat},
            "window_size": len(self.window),
        }

    def _compute_severity(self, psi_values, chi2_values, output_psi,
                          drifted_num, drifted_cat) -> str:
        if any(v >= self.psi_threshold_high for v in psi_values.values()) \
                or output_psi >= self.psi_threshold_high:
            return "high"
        if any(v >= self.psi_threshold_medium for v in psi_values.values()) \
                or output_psi >= self.psi_threshold_medium \
                or any(p < self.chi2_alpha for p in chi2_values.values()):
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Public API: check after each prediction
    # ------------------------------------------------------------------
    def check_and_report(
        self, features: Dict[str, float | str], probability: float
    ) -> Optional[dict]:
        self.add_prediction(features, probability)
        report = self.compute_drift_report()
        new_sev = report["severity"]

        if self._last_severity != new_sev:
            old_sev = self._last_severity
            self._last_severity = new_sev
            log.info("drift.severity_changed",
                     old=old_sev, new=new_sev,
                     psi=report.get("psi_values", {}))
            return report
        return None
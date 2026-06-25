# Sprint 4: Monitoring module
# Tracks model performance and data quality metrics

import pandas as pd
import numpy as np

def compute_psi(baseline, target, bins=10):
    """Compute Population Stability Index for drift detection."""
    breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    target_pct   = np.histogram(target, bins=breakpoints)[0] / len(target)
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    target_pct   = np.where(target_pct   == 0, 0.0001, target_pct)
    psi = np.sum((target_pct - baseline_pct) * np.log(target_pct / baseline_pct))
    return psi

def check_missing_rate(df, column='TotalCharges', threshold=0.05):
    """Monitor missing value rate in incoming data."""
    missing_rate = df[column].isnull().mean()
    status = "OK" if missing_rate < threshold else "ALERT"
    return {"column": column, "missing_rate": missing_rate, "status": status}

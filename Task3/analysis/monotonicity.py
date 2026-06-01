"""
this file computes monotonicity scores for each perturbation group and metric

it contains one helper that measures how cleanly each metric moves with the perturbation scale
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Task3.analysis.shared import METRICS


def compute_monotonicity(curve_agg, metrics=None):
    metrics = metrics or METRICS
    rows = []

    for group_name in sorted(curve_agg["perturbation_group"].dropna().astype(str).unique()):
        group_df = curve_agg[curve_agg["perturbation_group"] == group_name].sort_values("scale")
        scale_values = group_df["scale"].to_numpy(dtype=float)

        for metric in metrics:
            metric_values = group_df[f"{metric}_norm"].to_numpy(dtype=float)
            valid = np.isfinite(scale_values) & np.isfinite(metric_values)

            if valid.sum() >= 2 and np.unique(scale_values[valid]).size >= 2:
                rho, p_value = spearmanr(scale_values[valid], metric_values[valid])
            else:
                rho, p_value = np.nan, np.nan

            rows.append(
                {
                    "perturbation_group": group_name,
                    "metric": metric,
                    "rho_spearman": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "abs_rho": abs(float(rho)) if np.isfinite(rho) else np.nan,
                }
            )

    return pd.DataFrame(rows)

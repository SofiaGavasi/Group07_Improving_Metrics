"""
this file computes sensitivity scores for each perturbation group and metric

it contains one helper that measures how strongly each normalized metric moves away from baseline
"""

import numpy as np
import pandas as pd

from Task3.analysis.shared import METRICS


def compute_sensitivity(curve_df, metrics=None):
    metrics = metrics or METRICS
    rows = []

    for group_name in sorted(curve_df["perturbation_group"].dropna().astype(str).unique()):
        group_df = curve_df[curve_df["perturbation_group"] == group_name]

        for metric in metrics:
            values = group_df[f"{metric}_norm"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            rows.append(
                {
                    "perturbation_group": group_name,
                    "metric": metric,
                    "max_abs_norm_change": float(np.max(np.abs(values))) if values.size else np.nan,
                    "mean_abs_norm_change": float(np.mean(np.abs(values))) if values.size else np.nan,
                }
            )

    return pd.DataFrame(rows)

"""
this file computes reliability scores for each metric

reliability is defined as the mean bootstrap CI width per metric across all experiments


experiments where the ci columns are nan (bootstrapping was not run) are skipped
per metric. the output is one row per metric with the mean ci width and the number
of experiments that had valid ci data.
"""

import numpy as np
import pandas as pd

from Task3.analysis.shared import METRICS


def compute_reliability(analysis_df, metrics=None):
    metrics = metrics or METRICS

    rows = []
    for metric in metrics:
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"

        if low_col not in analysis_df.columns or high_col not in analysis_df.columns:
            rows.append(
                {
                    "metric": metric,
                    "mean_ci_width": np.nan,
                    "n_experiments": 0,
                }
            )
            continue

        low_values = pd.to_numeric(analysis_df[low_col], errors="coerce")
        high_values = pd.to_numeric(analysis_df[high_col], errors="coerce")
        widths = high_values - low_values

        valid_widths = widths[np.isfinite(widths)]

        rows.append(
            {
                "metric": metric,
                "mean_ci_width": float(valid_widths.mean()) if valid_widths.size > 0 else np.nan,
                "n_experiments": int(valid_widths.size),
            }
        )

    return pd.DataFrame(rows)
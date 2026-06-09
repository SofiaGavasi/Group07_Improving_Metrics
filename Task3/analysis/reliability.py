"""
this file computes reliability scores for each metric

reliability is defined as the mean bootstrap CI width per metric across all experiments,
normalised by the absolute baseline value of each metric to make widths comparable
across metrics with different scales (e.g. FID vs Precision).

epsilon=0.1 is added to the baseline denominator for consistency with the
normalised delta formula and to handle near-zero baselines (e.g. KID).

experiments where the ci columns are nan (bootstrapping was not run) are skipped
per metric. the output is one row per metric with the mean relative ci width and
the number of experiments that had valid ci data.
"""

import numpy as np
import pandas as pd

from Task3.analysis.shared import METRICS

CI_COLUMN_OVERRIDE = {
    "kid_mean": "kid",
    "is_mean":  "is",
}

EPSILON = 0.1


def compute_reliability(analysis_df, metrics=None):
    metrics = metrics or METRICS

    # compute per-metric baseline means for normalisation
    baseline_mask = analysis_df["name"].astype(str).str.contains("baseline", case=False, na=False)
    baseline_rows = analysis_df[baseline_mask] if baseline_mask.any() else pd.DataFrame()

    baseline_means = {}
    for metric in metrics:
        if not baseline_rows.empty and metric in baseline_rows.columns:
            vals = pd.to_numeric(baseline_rows[metric], errors="coerce").dropna()
            baseline_means[metric] = float(vals.mean()) if vals.size > 0 else np.nan
        else:
            baseline_means[metric] = np.nan

    rows = []
    for metric in metrics:
        ci_key = CI_COLUMN_OVERRIDE.get(metric, metric)
        low_col  = f"{ci_key}_ci_low"
        high_col = f"{ci_key}_ci_high"

        if low_col not in analysis_df.columns or high_col not in analysis_df.columns:
            rows.append(
                {
                    "metric": metric,
                    "mean_ci_width": np.nan,
                    "mean_relative_ci_width": np.nan,
                    "n_experiments": 0,
                }
            )
            continue

        low_values  = pd.to_numeric(analysis_df[low_col],  errors="coerce")
        high_values = pd.to_numeric(analysis_df[high_col], errors="coerce")
        widths = high_values - low_values
        valid_widths = widths[np.isfinite(widths)]

        baseline_val = baseline_means.get(metric, np.nan)
        denom = abs(baseline_val) + EPSILON if np.isfinite(baseline_val) else np.nan
        relative_widths = valid_widths / denom if np.isfinite(denom) else pd.Series(np.nan, index=valid_widths.index)

        rows.append(
            {
                "metric": metric,
                "mean_ci_width": float(valid_widths.mean()) if valid_widths.size > 0 else np.nan,
                "mean_relative_ci_width": float(relative_widths.mean()) if valid_widths.size > 0 else np.nan,
                "n_experiments": int(valid_widths.size),
            }
        )

    return pd.DataFrame(rows)
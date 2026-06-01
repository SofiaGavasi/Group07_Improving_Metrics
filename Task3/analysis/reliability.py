"""
this file computes the reliability table for nuisance perturbations

it contains the part of the old notebook analysis that measured spread across sample size preprocessing and domain shift rows
"""

import numpy as np
import pandas as pd

from Task3.analysis.shared import METRICS


def compute_reliability(analysis_df, metrics=None):
    metrics = metrics or METRICS

    source = analysis_df[
        analysis_df["perturbation_group"].astype(str).str.startswith("sample_size")
        | analysis_df["perturbation_group"].astype(str).str.startswith("preprocessing::")
        | analysis_df["perturbation_group"].astype(str).str.startswith("domain_shift::")
    ].copy()

    rows = []
    if source.empty:
        return pd.DataFrame(rows)

    # I collapse all domain shift variants into one bucket (they are not really severities)
    source["reliability_group"] = np.where(
        source["perturbation_group"].astype(str).str.startswith("domain_shift::"),
        "domain_shift",
        source["perturbation_group"],
    )

    for group_name, group_df in source.groupby("reliability_group", dropna=False):
        for metric in metrics:
            values = group_df[f"{metric}_norm"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size >= 2:
                mean_value = float(np.mean(values))
                std_value = float(np.std(values))
                cv_value = abs(std_value / mean_value) if mean_value != 0 else np.nan
                span_value = float(np.max(values) - np.min(values))
            else:
                mean_value = np.nan
                std_value = np.nan
                cv_value = np.nan
                span_value = np.nan

            rows.append(
                {
                    "perturbation_group": group_name,
                    "metric": metric,
                    "n_points": int(values.size),
                    "mean_norm": mean_value,
                    "std_norm": std_value,
                    "cv_norm": cv_value,
                    "span_norm": span_value,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    return frame[frame["n_points"] >= 2].copy()

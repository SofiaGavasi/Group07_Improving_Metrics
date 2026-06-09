"""
this file computes sensitivity scores for each perturbation group and metric

sensitivity measures how strongly each metric responds to perturbations it is
intended to detect. following eq. 5 of the task3 report, only rows belonging to
the target perturbation set T(m) for metric m are used:

    S_raw_m = mean over p in T(m), s of |delta_m(p, s)|

the target sets are defined in TARGET_PREFIXES below, mirroring the primary
metrics table in specificity.py but keyed by metric rather than by family.
"""

import numpy as np
import pandas as pd

from Task3.analysis.shared import METRICS


# for each metric, the perturbation group prefixes that belong to its target set T(m)
TARGET_PREFIXES = {
    "fid":       ["degradation_", "memoisation"],
    "kid_mean":  ["degradation_", "memoisation"],
    "is_mean":   ["class_removal", "class_imbalance"],
    "precision": ["degradation_", "memoisation"],
    "recall":    ["class_removal", "class_imbalance", "memoisation"],
    "density":   ["degradation_", "memoisation"],
    "coverage":  ["class_removal", "class_imbalance", "memoisation"],
}


def _is_on_target(group_label, prefixes):
    text = str(group_label)
    return any(text.startswith(prefix) for prefix in prefixes)


def compute_sensitivity(curve_df, metrics=None):
    """
    Returns one row per (perturbation_group, metric).
    For on-target (group, metric) pairs: mean and max |norm change| are filled.
    For off-target pairs: values are nan (mirrors specificity table structure).
    sensitivity_kind column marks 'on_target' or 'off_target'.
    """
    metrics = metrics or METRICS
    groups = sorted(curve_df["perturbation_group"].dropna().astype(str).unique())
    rows = []

    for group_name in groups:
        group_df = curve_df[curve_df["perturbation_group"] == group_name]

        for metric in metrics:
            prefixes = TARGET_PREFIXES.get(metric, [])
            on_target = _is_on_target(group_name, prefixes)

            if not on_target:
                rows.append({
                    "perturbation_group": group_name,
                    "metric": metric,
                    "mean_abs_norm_change": np.nan,
                    "max_abs_norm_change": np.nan,
                    "sensitivity_kind": "off_target",
                })
                continue

            values = group_df[f"{metric}_norm"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            rows.append({
                "perturbation_group": group_name,
                "metric": metric,
                "mean_abs_norm_change": float(np.mean(np.abs(values))) if values.size else np.nan,
                "max_abs_norm_change": float(np.max(np.abs(values))) if values.size else np.nan,
                "sensitivity_kind": "on_target",
            })

    return pd.DataFrame(rows)


def compute_sensitivity_summary(sensitivity_df):
    """
    Aggregates the per-group sensitivity table into one row per metric
    (mean of on-target rows only), for use in rwfas weight computation.
    """
    on_target = sensitivity_df[sensitivity_df["sensitivity_kind"] == "on_target"]
    rows = []
    for metric in METRICS:
        metric_df = on_target[on_target["metric"] == metric]
        values = metric_df["mean_abs_norm_change"].dropna().to_numpy(dtype=float)
        rows.append({
            "metric": metric,
            "mean_abs_norm_change": float(np.mean(values)) if values.size else np.nan,
            "max_abs_norm_change": float(np.max(values)) if values.size else np.nan,
            "n_on_target_rows": int(values.size),
        })
    return pd.DataFrame(rows)
"""
this file computes sensitivity scores for each metric

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
    "recall":    ["class_removal", "class_imbalance"],
    "density":   ["degradation_", "memoisation"],
    "coverage":  ["class_removal", "class_imbalance"],
}


def _is_on_target(group_label, prefixes):
    text = str(group_label)
    return any(text.startswith(prefix) for prefix in prefixes)


def compute_sensitivity(curve_df, metrics=None):
    metrics = metrics or METRICS
    rows = []

    for metric in metrics:
        prefixes = TARGET_PREFIXES.get(metric, [])

        on_target_mask = curve_df["perturbation_group"].apply(
            lambda group: _is_on_target(group, prefixes)
        )
        on_target_df = curve_df[on_target_mask]

        values = on_target_df[f"{metric}_norm"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        rows.append(
            {
                "metric": metric,
                "max_abs_norm_change": float(np.max(np.abs(values))) if values.size else np.nan,
                "mean_abs_norm_change": float(np.mean(np.abs(values))) if values.size else np.nan,
                "n_on_target_rows": int(values.size),
            }
        )

    return pd.DataFrame(rows)
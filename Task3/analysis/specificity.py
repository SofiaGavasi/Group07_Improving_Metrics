"""
this file computes specificity scores for each perturbation group and metric

it contains the rules that decide which metrics are on target for a perturbation family and how much off target drift is left
"""

import numpy as np
import pandas as pd

from Task3.analysis.shared import METRICS


PRIMARY_METRICS_BY_FAMILY = {
    "degradation_noise": {"fid", "kid_mean", "precision", "density"},
    "degradation_blur": {"fid", "kid_mean", "precision", "density"},
    "degradation_jpeg": {"fid", "kid_mean", "precision", "density"},
    "degradation_all": {"fid", "kid_mean", "precision", "density"},
    "memoisation": {"fid", "kid_mean", "precision", "density", "recall", "coverage"},
    "class_removal":   {"recall", "coverage", "is_mean"},
    "class_imbalance": {"recall", "coverage", "is_mean"},
    "sample_size": set(),
    "preprocessing": set(),
    "domain_shift": {"fid", "kid_mean", "is_mean", "precision", "recall", "density", "coverage"},
}


def _base_family(group_label):# this lets us avoid having to do a lot of manual labeling of which perturbation groups belong to which families
    text = str(group_label).lower()

    if text.startswith("degradation_"):
        return text
    if text.startswith("class_imbalance::"):
        return "class_imbalance"
    if text.startswith("preprocessing::"):
        return "preprocessing"
    if text.startswith("sample_size"):
        return "sample_size"
    if text.startswith("class_removal"):
        return "class_removal"
    if text.startswith("memo"):
        return "memoisation"

    return text


def compute_specificity(curve_df, perturbation_groups=None, metrics=None):
    metrics = metrics or METRICS
    if perturbation_groups is None:
        perturbation_groups = sorted(curve_df["perturbation_group"].dropna().astype(str).unique())

    rows = []

    for group_name in perturbation_groups:
        family = _base_family(group_name)
        primary_metrics = PRIMARY_METRICS_BY_FAMILY.get(family, set())
        group_df = curve_df[curve_df["perturbation_group"] == group_name]

        for metric in metrics:
            if metric in primary_metrics:
                # on-target: compute mean abs norm for use in ratio-based specificity
                values = group_df[f"{metric}_norm"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "perturbation_group": group_name,
                        "metric": metric,
                        "on_target_mean_abs_norm": float(np.mean(np.abs(values))) if values.size else np.nan,
                        "off_target_max_abs_norm": np.nan,
                        "off_target_mean_abs_norm": np.nan,
                        "specificity_kind": "primary_metric",
                    }
                )
                continue

            values = group_df[f"{metric}_norm"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size:
                rows.append(
                    {
                        "perturbation_group": group_name,
                        "metric": metric,
                        "on_target_mean_abs_norm": np.nan,
                        "off_target_max_abs_norm": float(np.max(np.abs(values))),
                        "off_target_mean_abs_norm": float(np.mean(np.abs(values))),
                        "specificity_kind": "off_target",
                    }
                )
            else:
                rows.append(
                    {
                        "perturbation_group": group_name,
                        "metric": metric,
                        "on_target_mean_abs_norm": np.nan,
                        "off_target_max_abs_norm": np.nan,
                        "off_target_mean_abs_norm": np.nan,
                        "specificity_kind": "missing_metric_data",
                    }
                )

    return pd.DataFrame(rows)
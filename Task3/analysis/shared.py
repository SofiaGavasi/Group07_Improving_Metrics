"""
this file holds the shared helpers used by task3 perturbation analysis
such as parsing naming convention for each perturbation

it contains metric names baseline handling perturbation parsing and the logic that turns raw experiment rows into grouped analysis tables
"""

import re

import numpy as np
import pandas as pd


METRICS = ["fid", "kid_mean", "is_mean", "precision", "recall", "density", "coverage"]

METRIC_LABELS = {
    "fid": "FID",
    "kid_mean": "KID",
    "is_mean": "IS",
    "precision": "Precision",
    "recall": "Recall",
    "density": "Density",
    "coverage": "Coverage",
}

LOWER_BETTER = {"fid", "kid_mean"}


def validate_analysis_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Expected a pandas DataFrame for `df`.")

    needed = ["name", "model", "dataset", "perturbation_family"] + METRICS
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise RuntimeError(f"Dataframe missing required columns: {missing}")


def select_experiment_rows(df):
    frame = (
        df.dropna(subset=METRICS, how="all")
        .drop_duplicates(subset=["name"], keep="last")
        .copy()
    )

    if frame.empty:
        raise RuntimeError("No experiment rows with metrics are available.")

    return frame


def build_baseline_table(frame):
    baseline_mask = frame["name"].astype(str).str.contains("baseline", case=False, na=False) | (
        frame["perturbation_family"].astype(str).str.lower() == "baseline"
    )

    baseline = (
        frame[baseline_mask]
        .groupby(["model", "dataset"], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )

    if baseline.empty:
        raise RuntimeError("Could not identify baseline rows.")

    return baseline


def add_normalized_metric_columns(frame):
    # i drop old baseline and norm columns here so this helper can be called on raw frames and on delta frames without merge suffix collisions
    cleanup_cols = [f"{metric}_baseline" for metric in METRICS]
    cleanup_cols.extend(f"{metric}_norm" for metric in METRICS)
    working = frame.drop(columns=cleanup_cols, errors="ignore").copy()

    baseline = build_baseline_table(frame)

    merged = working.merge(
        baseline.rename(columns={metric: f"{metric}_baseline" for metric in METRICS}),
        on=["model", "dataset"],
        how="left",
    )

    for metric in METRICS:
        baseline_values = merged[f"{metric}_baseline"]
        raw_delta = merged[metric] - baseline_values
        relative_delta = np.where(
            np.isfinite(baseline_values) & (baseline_values != 0),
            raw_delta / np.abs(baseline_values),
            np.nan,
        )
        merged[f"{metric}_norm"] = relative_delta if metric in LOWER_BETTER else -relative_delta

    return merged


def _parse_memo_fraction(name):
    match = re.search(r"memo_frac_(\d+)pct", str(name).lower())
    return float(match.group(1)) / 100.0 if match else np.nan


def _parse_imbalance_balance(name):
    text = str(name).lower()

    match_decimal = re.search(r"_(\d+)p(\d+)$", text)
    if match_decimal:
        return float(f"{match_decimal.group(1)}.{match_decimal.group(2)}")

    match_percent = re.search(r"_(\d+)pct$", text)
    if match_percent:
        return float(match_percent.group(1)) / 100.0

    return np.nan


def _count_target_tokens(targets):
    tokens = [token for token in re.split(r"[,_]+", str(targets).strip()) if token]
    return float(len(tokens)) if tokens else np.nan


def _parse_removal_strength(name):
    text = str(name).lower()

    label_match = re.search(r"class_removal_label_(.+)$", text)
    if label_match:
        return _count_target_tokens(label_match.group(1))

    kmeans_match = re.search(r"class_removal_kmeans_k\d+_cluster_(.+)$", text)
    if kmeans_match:
        return _count_target_tokens(kmeans_match.group(1))

    return np.nan


def _parse_class_imbalance_group(name):
    text = str(name).lower()

    label_match = re.search(r"class_imbalance_label_(.+?)_(\d+p\d+|\d+pct)$", text)
    if label_match:
        targets = label_match.group(1).replace("_", ",")
        return f"class_imbalance::label::{targets}"

    kmeans_match = re.search(r"class_imbalance_kmeans_k(\d+)_cluster_(.+?)_(\d+p\d+|\d+pct)$", text)
    if kmeans_match:
        k_value = kmeans_match.group(1)
        targets = kmeans_match.group(2).replace("_", ",")
        return f"class_imbalance::kmeans_k{k_value}::{targets}"

    return "class_imbalance::unknown"


def _parse_class_removal_group(name):
    text = str(name).lower()

    if re.search(r"class_removal_label_(.+)$", text):
        return "class_removal::label"

    kmeans_match = re.search(r"class_removal_kmeans_k(\d+)_cluster_(.+)$", text)
    if kmeans_match:
        return f"class_removal::kmeans_k{kmeans_match.group(1)}"

    return "class_removal"


def _parse_preprocessing_scale(name):
    match = re.search(r"_scale(\d+)p(\d+)$", str(name).lower())
    return float(f"{match.group(1)}.{match.group(2)}") if match else np.nan


def _parse_domain_shift_dataset(name):
    match = re.search(r"domain_shift_(.+)$", str(name).lower())
    return match.group(1) if match else "unknown"


def infer_group_and_scale(row):
    family = str(row.get("perturbation_family", "")).lower()
    name = str(row.get("name", ""))

    if family == "baseline":
        return "baseline", np.nan

    if family.startswith("degradation_"):
        severity = row.get("degradation_severity", np.nan)
        if pd.isna(severity):
            match = re.search(r"sev(\d+)", name.lower())
            severity = float(match.group(1)) if match else np.nan
        return family, float(severity) if pd.notna(severity) else np.nan

    if family.startswith("memo"):
        return "memoisation", _parse_memo_fraction(name)

    if family.startswith("class_removal"):
        return _parse_class_removal_group(name), _parse_removal_strength(name)

    if family.startswith("class_imbalance"):
        return _parse_class_imbalance_group(name), _parse_imbalance_balance(name)

    if family.startswith("sample_size"):
        sample_size = row.get("sample_size_n", np.nan)
        if pd.isna(sample_size):
            match = re.search(r"sample_size_(\d+)$", name.lower())
            sample_size = float(match.group(1)) if match else np.nan
        return "sample_size", float(sample_size) if pd.notna(sample_size) else np.nan

    if family.startswith("preprocessing"):
        variant = str(row.get("preprocessing_variant", "") or "").strip()
        if not variant:
            match = re.search(r"preprocessing_([a-z0-9_]+)_scale", name.lower())
            variant = match.group(1) if match else "unknown"

        scale = row.get("preprocessing_scale", np.nan)
        if pd.isna(scale):
            scale = _parse_preprocessing_scale(name)

        return f"preprocessing::{variant}", float(scale) if pd.notna(scale) else np.nan

    if family.startswith("domain_shift"):
        dataset_name = _parse_domain_shift_dataset(name)
        return f"domain_shift::{dataset_name}", np.nan

    severity = row.get("severity_from_name", np.nan)
    return family, float(severity) if pd.notna(severity) else np.nan


def add_group_and_scale_columns(frame):
    grouped = frame.apply(
        lambda row: pd.Series(infer_group_and_scale(row), index=["perturbation_group", "scale"]),
        axis=1,
    )

    merged = frame.copy()
    merged[["perturbation_group", "scale"]] = grouped
    return merged


def prepare_analysis_tables(df):
    validate_analysis_dataframe(df)

    experiments = select_experiment_rows(df)
    experiments = add_normalized_metric_columns(experiments)
    experiments = add_group_and_scale_columns(experiments)

    analysis_df = experiments[~experiments["perturbation_group"].isin(["baseline"])].copy()

    curve_df = analysis_df[
        (~analysis_df["perturbation_group"].astype(str).str.startswith("domain_shift::"))
        & analysis_df["scale"].notna()
    ].copy()

    if curve_df.empty:
        raise RuntimeError("No perturbation rows with inferred scales found for curve analysis.")

    curve_df = curve_df.sort_values(["perturbation_group", "scale", "name"])
    curve_agg = (
        curve_df.groupby(["perturbation_group", "scale"], dropna=False)[METRICS + [f"{metric}_norm" for metric in METRICS]]
        .mean(numeric_only=True)
        .reset_index()
    )

    perturbation_groups = sorted(curve_agg["perturbation_group"].unique())
    domain_shift_df = analysis_df[
        analysis_df["perturbation_group"].astype(str).str.startswith("domain_shift::")
    ].copy()

    return {
        "experiments": experiments,
        "analysis_df": analysis_df,
        "curve_df": curve_df,
        "curve_agg": curve_agg,
        "perturbation_groups": perturbation_groups,
        "domain_shift_df": domain_shift_df,
    }


def class_imbalance_disturbed_count(group_label):
    text = str(group_label)
    if not text.startswith("class_imbalance::"):
        return np.nan

    parts = text.split("::")
    if len(parts) < 3:
        return np.nan

    targets = [target for target in parts[-1].split(",") if target]
    return float(len(targets)) if targets else np.nan

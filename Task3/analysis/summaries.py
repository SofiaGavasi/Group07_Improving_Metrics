"""
this file contains summary views built from the parsed batch dataframe

baseline deltas top results and metric bar plots
"""

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)

from Task3.analysis.shared import METRICS


def _get_plt():
    import matplotlib.pyplot as plt

    return plt


def summarize_by_family(df, batch_name):
    plt = _get_plt()

    family_summary = (
        df.groupby("perturbation_family", dropna=False)
        .agg(
            experiments=("name", "nunique"),
            rows=("name", "size"),
            successful_rows=("status", lambda values: int((values == "completed").sum())),
            with_metrics=("has_metrics_report", lambda values: int(values.sum())),
            fid_mean=("fid", "mean"),
            is_mean=("is_mean", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
        )
        .sort_values("experiments", ascending=False)
    )

    display(family_summary)

    plt.figure(figsize=(10, 4))
    family_counts = df["perturbation_family"].value_counts()
    plt.bar(family_counts.index.astype(str), family_counts.values)
    plt.title(f"Experiment Rows per Perturbation Family ({batch_name})")
    plt.ylabel("Rows")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return family_summary


def compute_baseline_deltas(df, metrics_cols=None, show_table=True):
    metrics_cols = metrics_cols or METRICS

    baseline_mask = (
        df["name"].astype(str).str.contains("baseline", case=False, na=False)
        | (df["perturbation_family"] == "baseline")
    )

    baseline = (
        df[baseline_mask]
        .groupby(["model", "dataset"], dropna=False)[metrics_cols]
        .mean()
        .rename(columns={column: f"{column}_baseline" for column in metrics_cols})
        .reset_index()
    )

    df_delta = df.merge(baseline, on=["model", "dataset"], how="left")
    for column in metrics_cols:
        df_delta[f"{column}_delta"] = df_delta[column] - df_delta[f"{column}_baseline"]

    delta_cols = [f"{column}_delta" for column in metrics_cols]
    delta_summary = (
        df_delta.groupby("perturbation_family", dropna=False)[delta_cols]
        .mean()
        .sort_index()
    )

    if show_table:
        display(delta_summary)

    return df_delta, delta_summary


def show_top_worst(df_delta):
    def topn(frame, column, n=10, ascending=True):
        subset = frame.dropna(subset=[column]).sort_values(column, ascending=ascending)
        return subset[["name", "perturbation_family", "degradation_variant", "model", "dataset", column]].head(n)

    print("Best FID (lowest):")
    display(topn(df_delta, "fid", n=10, ascending=True))

    print("Worst FID (highest):")
    display(topn(df_delta, "fid", n=10, ascending=False))

    print("Best Recall:")
    display(topn(df_delta, "recall", n=10, ascending=False))

    print("Largest FID increase vs baseline (delta):")
    display(topn(df_delta, "fid_delta", n=10, ascending=False))


def plot_metric_bars(df_delta):
    plt = _get_plt()

    bar_specs = [
        ("fid", "fid_ci_low", "fid_ci_high", "FID (lower better)", True),
        ("kid_mean", "kid_ci_low", "kid_ci_high", "KID mean (lower better)", True),
        ("is_mean", "is_ci_low", "is_ci_high", "IS mean (higher better)", False),
        ("precision", "precision_ci_low", "precision_ci_high", "Precision (higher better)", False),
        ("recall", "recall_ci_low", "recall_ci_high", "Recall (higher better)", False),
        ("density", "density_ci_low", "density_ci_high", "Density (higher better)", False),
        ("coverage", "coverage_ci_low", "coverage_ci_high", "Coverage (higher better)", False),
    ]

    families = sorted([family for family in df_delta["perturbation_family"].dropna().unique()])
    family_order = ["baseline"] + [family for family in families if family != "baseline"]
    color_map = {family: plt.cm.tab20(index % 20) for index, family in enumerate(families)}

    max_bars = 220

    for metric_key, low_key, high_key, title, lower_is_better in bar_specs:
        columns = ["name", "perturbation_family", metric_key, low_key, high_key]
        plot_df = df_delta[columns].dropna(subset=[metric_key]).copy()
        if plot_df.empty:
            print(f"Skipping {metric_key}: no valid values.")
            continue

        plot_df["_family_order"] = pd.Categorical(
            plot_df["perturbation_family"],
            categories=family_order,
            ordered=True,
        )
        plot_df = plot_df.sort_values(["_family_order", metric_key], ascending=[True, lower_is_better])
        if len(plot_df) > max_bars:
            plot_df = plot_df.head(max_bars)

        values = plot_df[metric_key].to_numpy(dtype=float)
        lows = plot_df[low_key].to_numpy(dtype=float)
        highs = plot_df[high_key].to_numpy(dtype=float)

        lower_error = np.where(np.isfinite(lows), np.maximum(0.0, values - lows), np.nan)
        upper_error = np.where(np.isfinite(highs), np.maximum(0.0, highs - values), np.nan)

        x_values = np.arange(len(plot_df))
        colors = [color_map.get(family, "gray") for family in plot_df["perturbation_family"]]

        plt.figure(figsize=(20, 7))
        plt.bar(x_values, values, color=colors, alpha=0.9)

        has_ci = np.isfinite(lower_error) & np.isfinite(upper_error)
        if has_ci.any():
            plt.errorbar(
                x_values[has_ci],
                values[has_ci],
                yerr=np.vstack([lower_error[has_ci], upper_error[has_ci]]),
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=2,
                alpha=0.8,
            )

        plt.title(f"{title} by experiment, color-coded by perturbation family")
        plt.ylabel(title)
        plt.xlabel("Experiments (sorted)")
        plt.xticks(x_values, plot_df["name"], rotation=45, ha="right", fontsize=7)

        handles = []
        labels = []
        for family in families:
            if family in set(plot_df["perturbation_family"]):
                handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[family]))
                labels.append(family)

        if handles:
            plt.legend(handles, labels, loc="best", fontsize=8, ncols=2)

        plt.tight_layout()
        plt.show()

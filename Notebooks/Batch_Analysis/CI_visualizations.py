import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("fid", "fid_ci_low", "fid_ci_high", "FID"),
    ("kid_mean", "kid_ci_low", "kid_ci_high", "KID"),
    ("is_mean", "is_ci_low", "is_ci_high", "IS"),
    ("precision", "precision_ci_low", "precision_ci_high", "Precision"),
    ("recall", "recall_ci_low", "recall_ci_high", "Recall"),
    ("density", "density_ci_low", "density_ci_high", "Density"),
    ("coverage", "coverage_ci_low", "coverage_ci_high", "Coverage"),
]

GROUP_LABELS = {
    "model": "Model",
    "dataset": "Dataset",
    "perturbation_family": "Family",
}

NUMERIC_AXIS_CANDIDATES = [
    ("sample_size_n", "Sample size"),
    ("degradation_severity", "Degradation severity"),
    ("preprocessing_scale", "Preprocessing scale"),
    ("severity_from_name", "Severity"),
]

LABEL_AXIS_CANDIDATES = [
    ("domain_shift_dataset", "Target dataset"),
]


def _baseline_mask(df):
    family_mask = pd.Series(False, index=df.index)
    if "perturbation_family" in df.columns:
        family_mask = df["perturbation_family"].astype(str).str.lower().eq("baseline")
    name_mask = df["name"].astype(str).str.contains("baseline", case=False, na=False)
    return family_mask | name_mask


def _clean_text(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def _format_numeric_tick(value):
    if not np.isfinite(value):
        return "NA"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3g}"


def _available_metric_specs(df, metrics=None):
    allowed = set(metrics) if metrics is not None else None
    specs = []
    for metric_key, low_key, high_key, label in METRIC_SPECS:
        if allowed is not None and metric_key not in allowed:
            continue
        if all(col in df.columns for col in [metric_key, low_key, high_key]):
            specs.append((metric_key, low_key, high_key, label))
    return specs


def _build_group_title(group_key, group_cols):
    parts = []
    for col, value in zip(group_cols, group_key):
        parts.append(f"{GROUP_LABELS.get(col, col)}: {_clean_text(value)}")
    return " | ".join(parts)


def _pick_axis(group_df):
    for col, label in NUMERIC_AXIS_CANDIDATES:
        if col not in group_df.columns:
            continue
        values = pd.to_numeric(group_df[col], errors="coerce")
        if values.notna().all():
            return col, label, values

    for col, label in LABEL_AXIS_CANDIDATES:
        if col not in group_df.columns:
            continue
        labels = group_df[col].map(_clean_text)
        if (labels != "unknown").all() and labels.is_unique:
            return col, label, labels

    return "name", "Experiment", group_df["name"].map(_clean_text)


def _chunk_frame(frame, chunk_size):
    if len(frame) <= chunk_size:
        return [frame]
    return [frame.iloc[start : start + chunk_size].copy() for start in range(0, len(frame), chunk_size)]


def _build_lookup_key(row, cols):
    return tuple(row[col] for col in cols)


def _build_baseline_reference_map(
    df,
    metric_specs,
    lookup_cols,
):
    if df.empty:
        return {}

    value_cols = []
    for metric_key, low_key, high_key, _ in metric_specs:
        value_cols.extend([metric_key, low_key, high_key])

    if lookup_cols:
        baseline_df = (
            df.groupby(list(lookup_cols), dropna=False)[value_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        return {
            _build_lookup_key(row, lookup_cols): row
            for _, row in baseline_df.iterrows()
        }

    baseline_row = df[value_cols].mean(numeric_only=True)
    return {(): baseline_row}


def _unique_legend_entries(ax_list):
    seen = set()
    out_handles = []
    out_labels = []
    for ax in ax_list:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label in seen:
                continue
            seen.add(label)
            out_handles.append(handle)
            out_labels.append(label)
    return out_handles, out_labels


def _plot_chunk(
    group_df,
    metric_specs,
    group_title,
    baseline_row,
    chunk_index,
    chunk_count,
):
    axis_col, axis_label, axis_values = _pick_axis(group_df)
    plot_df = group_df.copy()
    plot_df["_axis_value"] = axis_values

    is_numeric_axis = pd.api.types.is_numeric_dtype(plot_df["_axis_value"])
    sort_cols = ["_axis_value", "name"] if is_numeric_axis else ["_axis_value"]
    plot_df = plot_df.sort_values(sort_cols, kind="stable")

    n_cols = 3
    n_rows = int(math.ceil(len(metric_specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, (metric_key, low_key, high_key, metric_label) in enumerate(metric_specs):
        ax = axes_flat[idx]
        metric_df = plot_df.dropna(subset=[metric_key, low_key, high_key]).copy()

        if metric_df.empty:
            ax.set_title(metric_label, fontsize=10)
            ax.text(0.5, 0.5, "No CI available", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        values = metric_df[metric_key].to_numpy(dtype=float)
        lows = metric_df[low_key].to_numpy(dtype=float)
        highs = metric_df[high_key].to_numpy(dtype=float)
        lower_err = np.maximum(0.0, values - lows)
        upper_err = np.maximum(0.0, highs - values)

        if is_numeric_axis:
            x = pd.to_numeric(metric_df["_axis_value"], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(x)
            x = x[order]
            values = values[order]
            lower_err = lower_err[order]
            upper_err = upper_err[order]
            ax.errorbar(
                x,
                values,
                yerr=np.vstack([lower_err, upper_err]),
                fmt="-o",
                color="tab:blue",
                ecolor="black",
                linewidth=1.4,
                elinewidth=1.0,
                capsize=3,
                markersize=4,
                label="Experiment values",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([_format_numeric_tick(v) for v in x], fontsize=8)
        else:
            labels = metric_df["_axis_value"].astype(str).tolist()
            x = np.arange(len(labels), dtype=float)
            ax.errorbar(
                x,
                values,
                yerr=np.vstack([lower_err, upper_err]),
                fmt="o",
                color="tab:blue",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
                markersize=4,
                label="Experiment values",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

        if baseline_row is not None:
            baseline_value = baseline_row.get(metric_key, np.nan)
            baseline_low = baseline_row.get(low_key, np.nan)
            baseline_high = baseline_row.get(high_key, np.nan)

            if np.isfinite(baseline_value):
                ax.axhline(
                    baseline_value,
                    color="tab:red",
                    linewidth=1.2,
                    linestyle="-",
                    alpha=0.9,
                    label="Baseline",
                )
            if np.isfinite(baseline_low):
                ax.axhline(
                    baseline_low,
                    color="tab:red",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.8,
                    label="Baseline CI low",
                )
            if np.isfinite(baseline_high):
                ax.axhline(
                    baseline_high,
                    color="tab:red",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.8,
                    label="Baseline CI high",
                )

        ax.set_title(metric_label, fontsize=10)
        ax.set_xlabel(axis_label, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

    for idx in range(len(metric_specs), len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, labels = _unique_legend_entries(axes_flat)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(4, len(labels)), fontsize=8)

    suffix = f" (part {chunk_index} of {chunk_count})" if chunk_count > 1 else ""
    fig.suptitle(f"Experiment confidence intervals: {group_title}{suffix}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92 if handles else 0.96])
    plt.show()


def plot_experiment_confidence_intervals(
    df,
    metrics=None,
    group_cols=("model", "dataset", "perturbation_family"),
    max_experiments_per_figure=8,
):
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Expected a pandas DataFrame for `df`.")
    if max_experiments_per_figure < 1:
        raise ValueError("`max_experiments_per_figure` must be at least 1.")

    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise RuntimeError(f"Dataframe missing required grouping columns: {missing_group_cols}")
    if "name" not in df.columns:
        raise RuntimeError("Dataframe missing required column: ['name']")

    metric_specs = _available_metric_specs(df, metrics=metrics)
    if not metric_specs:
        raise RuntimeError("No metric CI columns were found in the dataframe.")

    dedupe_cols = list(group_cols) + ["name"]
    plot_df = df.drop_duplicates(subset=dedupe_cols, keep="last").copy()
    baseline_mask = _baseline_mask(plot_df)
    baseline_lookup_cols = tuple(col for col in group_cols if col != "perturbation_family")
    baseline_map = _build_baseline_reference_map(
        plot_df[baseline_mask].copy(),
        metric_specs=metric_specs,
        lookup_cols=baseline_lookup_cols,
    )

    ci_mask = pd.Series(False, index=plot_df.index)
    for metric_key, low_key, high_key, _ in metric_specs:
        ci_mask |= plot_df[[metric_key, low_key, high_key]].notna().all(axis=1)

    ci_df = plot_df[ci_mask & ~baseline_mask].copy()
    if ci_df.empty:
        print("No non-baseline experiment-level confidence intervals found in the dataframe.")
        return ci_df

    plotted_group_count = 0
    for group_key, group_df in ci_df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_title = _build_group_title(group_key, group_cols)
        axis_col, _, axis_values = _pick_axis(group_df)
        sorted_group_df = group_df.copy()
        sorted_group_df["_axis_value_for_chunking"] = axis_values
        is_numeric_axis = pd.api.types.is_numeric_dtype(sorted_group_df["_axis_value_for_chunking"])
        sort_cols = ["_axis_value_for_chunking", "name"] if is_numeric_axis else ["_axis_value_for_chunking"]
        sorted_group_df = (
            sorted_group_df.sort_values(sort_cols, kind="stable")
            .drop(columns=["_axis_value_for_chunking"])
        )

        if axis_col == "sample_size_n":
            chunked_frames = [sorted_group_df]
        else:
            chunked_frames = _chunk_frame(sorted_group_df, max_experiments_per_figure)

        baseline_key = _build_lookup_key(group_df.iloc[0], baseline_lookup_cols) if baseline_lookup_cols else ()
        baseline_row = baseline_map.get(baseline_key)
        for chunk_index, chunk_df in enumerate(chunked_frames, start=1):
            _plot_chunk(
                chunk_df,
                metric_specs=metric_specs,
                group_title=group_title,
                baseline_row=baseline_row,
                chunk_index=chunk_index,
                chunk_count=len(chunked_frames),
            )
        plotted_group_count += 1

    print(
        "Plotted confidence intervals for "
        f"{len(ci_df)} non-baseline experiment rows across {plotted_group_count} grouped plot set(s)."
    )
    return ci_df

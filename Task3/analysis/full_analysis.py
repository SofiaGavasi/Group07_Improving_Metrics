"""
this file runs the full task3 perturbation analysis flow

it prepares the shared tables calls the split metric modules draws the plots and returns the result tables in one place
"""

import json

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)

from Task3.analysis.shared import (
    METRICS,
    METRIC_LABELS,
    class_imbalance_disturbed_count,
    prepare_analysis_tables,
)
from Task3.analysis.monotonicity import compute_monotonicity
from Task3.analysis.reliability import compute_reliability
from Task3.analysis.sensitivity import compute_sensitivity
from Task3.analysis.specificity import compute_specificity

from Task3.analysis.plot_components import (
    plot_monotonicity_heatmap,
    plot_sensitivity_bars,
    plot_reliability_bars,
    plot_specificity_heatmap,
)


def _get_plt():
    import matplotlib.pyplot as plt
    return plt


def _line_axis_mode(group_labels):
    labels = [str(group_label) for group_label in group_labels]
    if labels and all(label == "sample_size" for label in labels):
        return "sample_size"
    if labels and all(label.startswith("preprocessing::") for label in labels):
        return "preprocessing"
    return "default"


def _sample_size_reference(group_df):
    scale_values = pd.to_numeric(group_df["scale"], errors="coerce")
    max_scale = scale_values.max()
    if pd.notna(max_scale) and float(max_scale) > 0:
        return float(max_scale)
    return 1280.0


def _transform_line_scales(scale_series, axis_mode, sample_size_reference):
    scale_values = pd.to_numeric(scale_series, errors="coerce").to_numpy(dtype=float)
    if axis_mode == "sample_size":
        return scale_values / float(sample_size_reference)
    if axis_mode == "preprocessing":
        return 1.0 - scale_values
    return scale_values


def _line_axis_label(axis_mode, sample_size_reference):
    if axis_mode == "sample_size":
        return f"Sample Size / N (N={int(sample_size_reference)})"
    if axis_mode == "preprocessing":
        return "Severity (1 - scale)"
    return "Scale / Severity"


def _format_axis_tick(value):
    if not np.isfinite(value):
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _apply_line_axis_ticks(ax, axis_mode, tick_values):
    tick_values_sorted = sorted(value for value in tick_values if np.isfinite(value))
    if not tick_values_sorted:
        return

    if axis_mode == "sample_size":
        ax.set_xscale("log")
        ax.set_xlim(tick_values_sorted[0] * 0.9, tick_values_sorted[-1] * 1.05)
        rotation = 35
        alignment = "right"
        tick_fontsize = 6.5
    else:
        rotation = 0
        alignment = "center"
        tick_fontsize = 7

    ax.set_xticks(tick_values_sorted)
    ax.set_xticklabels(
        [_format_axis_tick(value) for value in tick_values_sorted],
        rotation=rotation,
        ha=alignment,
        fontsize=tick_fontsize,
    )


def _build_plot_groups(perturbation_groups):
    class_imbalance_groups = [
        group_name
        for group_name in perturbation_groups
        if group_name.startswith("class_imbalance::")
    ]

    class_counts = sorted(
        {
            int(count)
            for count in (class_imbalance_disturbed_count(group_name) for group_name in class_imbalance_groups)
            if pd.notna(count)
        }
    )

    plot_groups = {
        "Degradation": [group_name for group_name in perturbation_groups if group_name.startswith("degradation_")],
        "Class Removal": [group_name for group_name in perturbation_groups if group_name.startswith("class_removal")],
    }

    for disturbed_count in class_counts:
        label = "class" if disturbed_count == 1 else "classes"
        plot_groups[f"Class Imbalance ({disturbed_count} disturbed {label})"] = [
            group_name
            for group_name in class_imbalance_groups
            if class_imbalance_disturbed_count(group_name) == disturbed_count
        ]

    unknown_groups = [
        group_name
        for group_name in class_imbalance_groups
        if pd.isna(class_imbalance_disturbed_count(group_name))
    ]
    if unknown_groups:
        plot_groups["Class Imbalance (unknown disturbed classes)"] = unknown_groups

    plot_groups["Memoisation"] = [group_name for group_name in perturbation_groups if group_name == "memoisation"]
    plot_groups["Sample Size"] = [group_name for group_name in perturbation_groups if group_name == "sample_size"]
    plot_groups["Preprocessing"] = [
        group_name
        for group_name in perturbation_groups
        if group_name.startswith("preprocessing::")
    ]

    return plot_groups


def _unique_legend_entries(axes):
    seen = set()
    handles_out = []
    labels_out = []

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            handles_out.append(handle)
            labels_out.append(label)

    return handles_out, labels_out


def _plot_group_metric_panels(curve_df, group_name, group_labels):
    plt = _get_plt()

    if not group_labels:
        print(f"No groups to plot for: {group_name}")
        return

    plot_df = curve_df[curve_df["perturbation_group"].isin(group_labels)].copy()
    axis_mode = _line_axis_mode(group_labels)
    sample_size_reference = _sample_size_reference(plot_df) if axis_mode == "sample_size" else np.nan

    n_cols = 4
    n_rows = int(np.ceil(len(METRICS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for index, metric in enumerate(METRICS):
        ax = axes_flat[index]
        panel_base = plot_df[f"{metric}_baseline"].mean()
        tick_values = set()

        for group_label in group_labels:
            group_df = curve_df[curve_df["perturbation_group"] == group_label]
            stats = (
                group_df.groupby("scale", dropna=False)[metric]
                .agg(y_mean="mean", y_std="std", n="count")
                .reset_index()
            )

            stats["x_plot"] = _transform_line_scales(
                stats["scale"],
                axis_mode=axis_mode,
                sample_size_reference=sample_size_reference,
            )
            stats = stats.sort_values("x_plot")

            x_values = stats["x_plot"].to_numpy(dtype=float)
            y_values = stats["y_mean"].to_numpy(dtype=float)
            if np.isfinite(x_values).sum() == 0 or np.isfinite(y_values).sum() == 0:
                continue

            ci_values = np.where(
                stats["n"].to_numpy(dtype=float) >= 2,
                1.96 * stats["y_std"].to_numpy(dtype=float) / np.sqrt(stats["n"].to_numpy(dtype=float)),
                np.nan,
            )

            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=group_label,
            )
            tick_values.update(value for value in x_values if np.isfinite(value))

            if np.isfinite(ci_values).any():
                ax.fill_between(x_values, y_values - ci_values, y_values + ci_values, alpha=0.15)

        if np.isfinite(panel_base):
            ax.axhline(
                panel_base,
                linestyle="--",
                linewidth=1.0,
                color="black",
                alpha=0.8,
                label="baseline",
            )

        ax.grid(alpha=0.2)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlabel(_line_axis_label(axis_mode, sample_size_reference), fontsize=8)
        ax.set_ylabel("Metric value", fontsize=8)
        if tick_values and axis_mode in {"sample_size", "preprocessing"}:
            _apply_line_axis_ticks(ax, axis_mode, tick_values)
        ax.tick_params(labelsize=8)

    for index in range(len(METRICS), len(axes_flat)):
        axes_flat[index].axis("off")

    handles, labels = _unique_legend_entries(axes_flat)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8)

    fig.suptitle(f"{group_name}: metric response by perturbation type", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def _plot_group_average_metric_panels(curve_df, group_name, group_labels):
    plt = _get_plt()

    plot_df = curve_df[curve_df["perturbation_group"].isin(group_labels)].copy()
    if plot_df.empty:
        print(f"No rows available for averaged plotting: {group_name}")
        return

    axis_mode = _line_axis_mode(group_labels)
    sample_size_reference = _sample_size_reference(plot_df) if axis_mode == "sample_size" else np.nan

    n_cols = 4
    n_rows = int(np.ceil(len(METRICS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for index, metric in enumerate(METRICS):
        ax = axes_flat[index]
        stats = (
            plot_df.groupby("scale", dropna=False)[metric]
            .agg(y_mean="mean", y_std="std", n="count")
            .reset_index()
        )
        stats["x_plot"] = _transform_line_scales(
            stats["scale"],
            axis_mode=axis_mode,
            sample_size_reference=sample_size_reference,
        )
        stats = stats.sort_values("x_plot")

        x_values = stats["x_plot"].to_numpy(dtype=float)
        y_values = stats["y_mean"].to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)

        if valid.any():
            ci_values = np.where(
                stats["n"].to_numpy(dtype=float) >= 2,
                1.96 * stats["y_std"].to_numpy(dtype=float) / np.sqrt(stats["n"].to_numpy(dtype=float)),
                np.nan,
            )

            ax.plot(
                x_values[valid],
                y_values[valid],
                marker="o",
                linewidth=1.6,
                markersize=4,
                color="tab:blue",
                label="average across target sets",
            )

            ci_valid = valid & np.isfinite(ci_values)
            if ci_valid.any():
                ax.fill_between(
                    x_values[ci_valid],
                    y_values[ci_valid] - ci_values[ci_valid],
                    y_values[ci_valid] + ci_values[ci_valid],
                    alpha=0.15,
                    color="tab:blue",
                )

            if axis_mode in {"sample_size", "preprocessing"}:
                _apply_line_axis_ticks(ax, axis_mode, set(x_values[valid]))

        panel_base = plot_df[f"{metric}_baseline"].mean()
        if np.isfinite(panel_base):
            ax.axhline(
                panel_base,
                linestyle="--",
                linewidth=1.0,
                color="black",
                alpha=0.8,
                label="baseline",
            )

        ax.grid(alpha=0.2)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlabel(_line_axis_label(axis_mode, sample_size_reference), fontsize=8)
        ax.set_ylabel("Metric value", fontsize=8)
        ax.tick_params(labelsize=8)

    for index in range(len(METRICS), len(axes_flat)):
        axes_flat[index].axis("off")

    handles, labels = _unique_legend_entries(axes_flat)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8)

    fig.suptitle(f"{group_name}: averaged metric response across target sets", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def _baseline_metric_stats(experiments_df, domain_shift_df, metric):
    baseline_series = pd.Series(dtype=float)
    if experiments_df is not None and {"perturbation_group", metric}.issubset(experiments_df.columns):
        baseline_series = pd.to_numeric(
            experiments_df.loc[experiments_df["perturbation_group"] == "baseline", metric],
            errors="coerce",
        ).dropna()

    if baseline_series.empty:
        baseline_series = pd.to_numeric(domain_shift_df.get(f"{metric}_baseline"), errors="coerce").dropna()

    baseline_mean = baseline_series.mean() if not baseline_series.empty else np.nan
    if baseline_series.size >= 2:
        baseline_ci = 1.96 * baseline_series.std(ddof=1) / np.sqrt(baseline_series.size)
    else:
        baseline_ci = np.nan
    return float(baseline_mean) if pd.notna(baseline_mean) else np.nan, float(baseline_ci) if pd.notna(baseline_ci) else np.nan


def _plot_domain_shift_panels(domain_shift_df, experiments_df=None):
    plt = _get_plt()

    if domain_shift_df.empty:
        print("No domain-shift rows found for dedicated plotting.")
        return

    plot_df = domain_shift_df.copy()
    plot_df["domain_dataset"] = plot_df["perturbation_group"].astype(str).str.replace(
        "domain_shift::",
        "",
        regex=False,
    )

    domain_datasets = sorted(plot_df["domain_dataset"].dropna().astype(str).unique())
    x_values = np.arange(len(domain_datasets), dtype=float)

    n_cols = 4
    n_rows = int(np.ceil(len(METRICS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for index, metric in enumerate(METRICS):
        ax = axes_flat[index]
        stats = (
            plot_df.groupby("domain_dataset", dropna=False)[metric]
            .agg(y_mean="mean", y_std="std", n="count")
            .reset_index()
            .sort_values("domain_dataset")
        )

        stats = stats[stats["domain_dataset"].astype(str).isin(domain_datasets)].copy()
        metric_values = stats["y_mean"].to_numpy(dtype=float)
        valid = np.isfinite(metric_values)
        baseline_mean, baseline_ci = _baseline_metric_stats(experiments_df, plot_df, metric)

        ci_values = np.where(
            stats["n"].to_numpy(dtype=float) >= 2,
            1.96 * stats["y_std"].to_numpy(dtype=float) / np.sqrt(stats["n"].to_numpy(dtype=float)),
            np.nan,
        )

        if np.isfinite(baseline_mean):
            ax.bar(
                [0],
                [baseline_mean],
                width=0.7,
                color="black",
                alpha=0.8,
                label="baseline",
            )
            if np.isfinite(baseline_ci):
                ax.errorbar(
                    [0],
                    [baseline_mean],
                    yerr=[baseline_ci],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                    alpha=0.8,
                )

        if valid.any():
            domain_positions = x_values[valid] + 1.0
            ax.bar(
                domain_positions,
                metric_values[valid],
                width=0.7,
                color="tab:blue",
                alpha=0.85,
                label="domain_shift",
            )

            ci_valid = valid & np.isfinite(ci_values)
            if ci_valid.any():
                ax.errorbar(
                    x_values[ci_valid] + 1.0,
                    metric_values[ci_valid],
                    yerr=ci_values[ci_valid],
                    fmt="none",
                    ecolor="tab:blue",
                    elinewidth=1.0,
                    capsize=3,
                    alpha=0.8,
                )

        ax.grid(alpha=0.2, axis="y")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlabel("Baseline + domain-shift dataset", fontsize=8)
        ax.set_ylabel("Metric value", fontsize=8)
        ax.set_xticks(np.arange(len(domain_datasets) + 1, dtype=float))
        ax.set_xticklabels(["baseline"] + domain_datasets, rotation=20, ha="right", fontsize=7)
        ax.tick_params(labelsize=8)

    for index in range(len(METRICS), len(axes_flat)):
        axes_flat[index].axis("off")

    handles, labels = _unique_legend_entries(axes_flat)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8)

    fig.suptitle("Domain Shift: metric response across target datasets", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def _show_table_head(title, frame):
    print(title)
    display(frame)


def _normalize_sequence(value):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [token.strip() for token in text.split(",") if token.strip()]
        if isinstance(parsed, list):
            return list(parsed)
        return [parsed]
    if pd.isna(value):
        return []
    return [value]


def _normalize_count_mapping(value):
    if isinstance(value, dict):
        return {str(key): int(val) for key, val in value.items() if pd.notna(val)}
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return {str(key): int(val) for key, val in parsed.items() if pd.notna(val)}
    return {}


def _lookup_count(mapping, class_index, class_name):
    candidate_keys = []
    if class_index is not None and str(class_index) != "":
        candidate_keys.append(str(class_index))
    if class_name is not None and str(class_name) != "":
        candidate_keys.append(str(class_name))

    for key in candidate_keys:
        if key in mapping:
            return float(mapping[key])
    return np.nan


def _per_class_fraction_list(value, count):
    if count <= 0:
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(item) for item in list(value)]
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return [np.nan] * count
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [np.nan] * count
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                scalar = float(text)
            except Exception:
                return [np.nan] * count
            return [scalar] * count
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
        try:
            scalar = float(parsed)
        except Exception:
            return [np.nan] * count
        return [scalar] * count
    try:
        scalar = float(value)
    except Exception:
        return [np.nan] * count
    return [scalar] * count


def _build_removed_image_overviews(experiments_df, family_prefix):
    removed_col = f"{family_prefix}_removed_count"
    kept_col = f"{family_prefix}_kept_count"
    survivor_col = f"{family_prefix}_survivor_count"
    evaluation_col = f"{family_prefix}_evaluation_count"
    returned_col = f"{family_prefix}_returned_count"
    pool_col = f"{family_prefix}_pool_count"
    strategy_col = f"{family_prefix}_strategy"
    label_mode_col = f"{family_prefix}_label_mode"
    drop_classes_col = f"{family_prefix}_drop_classes"
    drop_names_col = f"{family_prefix}_drop_class_names"
    pred_hist_col = f"{family_prefix}_predicted_label_histogram_fake"
    pred_pos_col = f"{family_prefix}_predicted_positive_counts"
    drop_fraction_col = f"{family_prefix}_drop_fraction"

    if removed_col not in experiments_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    summary_source = experiments_df[
        experiments_df["perturbation_family"].astype(str).str.startswith(family_prefix)
        & pd.to_numeric(experiments_df[removed_col], errors="coerce").notna()
    ].copy()

    if summary_source.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_source["affected_class_count"] = summary_source[drop_classes_col].apply(
        lambda value: len(_normalize_sequence(value))
    )
    summary_source["affected_classes"] = summary_source[drop_names_col].apply(
        lambda value: ", ".join(str(item) for item in _normalize_sequence(value))
    )

    summary_columns = [
        "name",
        "model",
        "dataset",
        "perturbation_group",
        "scale",
        strategy_col,
        label_mode_col,
        "affected_class_count",
        "affected_classes",
    ]
    if family_prefix == "class_imbalance":
        summary_columns.append(drop_fraction_col)
    summary_columns.extend(
        [
            removed_col,
            kept_col,
            survivor_col,
            evaluation_col,
            returned_col,
            pool_col,
        ]
    )

    summary = (
        summary_source[summary_columns]
        .rename(
            columns={
                "scale": "intensity",
                strategy_col: "strategy",
                label_mode_col: "label_mode",
                drop_fraction_col: "drop_fraction",
                removed_col: "removed_count",
                kept_col: "kept_count",
                survivor_col: "survivor_count",
                evaluation_col: "evaluation_count",
                returned_col: "returned_count",
                pool_col: "pool_count",
            }
        )
        .sort_values(["affected_class_count", "intensity", "name"], na_position="last")
        .reset_index(drop=True)
    )

    per_class_rows = []
    for row in summary_source.itertuples(index=False):
        class_indices = _normalize_sequence(getattr(row, drop_classes_col))
        class_names = _normalize_sequence(getattr(row, drop_names_col))
        hist_map = _normalize_count_mapping(getattr(row, pred_hist_col))
        positive_map = _normalize_count_mapping(getattr(row, pred_pos_col))
        fractions = _per_class_fraction_list(getattr(row, drop_fraction_col, None), len(class_indices))

        max_len = max(len(class_indices), len(class_names))
        for index in range(max_len):
            class_index = class_indices[index] if index < len(class_indices) else None
            class_name = class_names[index] if index < len(class_names) else class_index

            affected_pool_count = np.nan
            count_kind = ""
            exact_removed_count = np.nan
            if hist_map:
                affected_pool_count = _lookup_count(hist_map, class_index, class_name)
                count_kind = "predicted_label_histogram_fake"
                if family_prefix == "class_removal" and pd.notna(affected_pool_count):
                    exact_removed_count = float(affected_pool_count)
                elif (
                    family_prefix == "class_imbalance"
                    and pd.notna(affected_pool_count)
                    and index < len(fractions)
                    and pd.notna(fractions[index])
                    and str(getattr(row, label_mode_col)) == "single_label"
                ):
                    exact_removed_count = float(int(affected_pool_count * float(fractions[index])))
            elif positive_map:
                affected_pool_count = _lookup_count(positive_map, class_index, class_name)
                count_kind = "predicted_positive_counts"

            drop_fraction = fractions[index] if index < len(fractions) else np.nan
            per_class_rows.append(
                {
                    "name": row.name,
                    "model": row.model,
                    "dataset": row.dataset,
                    "perturbation_group": row.perturbation_group,
                    "intensity": row.scale,
                    "strategy": getattr(row, strategy_col),
                    "label_mode": getattr(row, label_mode_col),
                    "affected_class_index": class_index,
                    "affected_class": class_name,
                    "drop_fraction": drop_fraction,
                    "affected_pool_count": affected_pool_count,
                    "exact_removed_count_if_available": exact_removed_count,
                    "count_kind": count_kind,
                }
            )

    per_class = pd.DataFrame(per_class_rows)
    if not per_class.empty:
        per_class = per_class.sort_values(
            ["perturbation_group", "intensity", "affected_class"],
            na_position="last",
        ).reset_index(drop=True)

    return summary, per_class


def run_full_perturbation_analysis(df, show_plots=True, show_tables=True, verbose=True):
    prepared = prepare_analysis_tables(df)

    monotonicity = compute_monotonicity(prepared["curve_agg"])
    sensitivity = compute_sensitivity(prepared["curve_df"])
    reliability = compute_reliability(prepared["analysis_df"])
    specificity = compute_specificity(prepared["curve_df"], prepared["perturbation_groups"])

    if show_plots:
        plot_groups = _build_plot_groups(prepared["perturbation_groups"])
        for group_name, group_labels in plot_groups.items():
            _plot_group_metric_panels(prepared["curve_df"], group_name, group_labels)
            if group_name.startswith("Class Imbalance"):
                _plot_group_average_metric_panels(prepared["curve_df"], group_name, group_labels)

        _plot_domain_shift_panels(prepared["domain_shift_df"], experiments_df=prepared["experiments"])

        plot_monotonicity_heatmap(monotonicity)
        plot_sensitivity_bars(sensitivity)

        if reliability.empty or reliability["n_experiments"].sum() == 0:
            print("Reliability: no bootstrap CI data found in this batch.")
        else:
            plot_reliability_bars(reliability)

        plot_specificity_heatmap(specificity)

    if verbose:
        print("Perturbation groups included in curve analysis:")
        print(sorted(prepared["perturbation_groups"]))

    if show_tables:
        class_removal_overview, class_removal_per_class = _build_removed_image_overviews(
            prepared["experiments"],
            family_prefix="class_removal",
        )
        class_imbalance_overview, class_imbalance_per_class = _build_removed_image_overviews(
            prepared["experiments"],
            family_prefix="class_imbalance",
        )

        specificity_diagnostics = (
            specificity.assign(is_na=specificity["off_target_max_abs_norm"].isna())
            .groupby(["specificity_kind", "is_na"], dropna=False)
            .size()
            .reset_index(name="count")
        )

        _show_table_head(
            "Class Removal: removed-image overview:",
            class_removal_overview,
        )
        _show_table_head(
            "Class Removal: per-class detail:",
            class_removal_per_class,
        )
        _show_table_head(
            "Class Imbalance: removed-image overview:",
            class_imbalance_overview,
        )
        _show_table_head(
            "Class Imbalance: per-class detail:",
            class_imbalance_per_class,
        )
        _show_table_head(
            "Specificity NA diagnostics:",
            specificity_diagnostics,
        )
        _show_table_head(
            "Monotonicity (head):",
            monotonicity.sort_values(["perturbation_group", "metric"]).head(40),
        )
        _show_table_head(
            "Sensitivity (head):",
            sensitivity.sort_values(["perturbation_group", "metric"]).head(40),
        )
        _show_table_head(
            "Reliability (head):",
            reliability.sort_values(["perturbation_group", "metric"]).head(40) if not reliability.empty else pd.DataFrame(),
        )
        _show_table_head(
            "Specificity (head):",
            specificity.sort_values(["perturbation_group", "metric"]).head(40),
        )

    return {
        "monotonicity": monotonicity,
        "sensitivity": sensitivity,
        "reliability": reliability,
        "robustness": reliability,
        "specificity": specificity,
    }

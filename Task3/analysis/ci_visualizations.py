"""
this file plots experiment level confidence intervals from the parsed batch table

"""

import math

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

LABEL_AXIS_CANDIDATES = [
    ("domain_shift_dataset", "Target dataset"),
]


def _get_plt():
    import matplotlib.pyplot as plt
    return plt


# ── axis mode helpers (identical logic to full_analysis.py) ──────────────────

def _axis_mode(perturbation_groups):
    labels = [str(g) for g in perturbation_groups if pd.notna(g)]
    if labels and all(g == "sample_size" for g in labels):
        return "sample_size"
    if labels and all(g.startswith("preprocessing::") for g in labels):
        return "preprocessing"
    return "default"


def _sample_size_reference(scale_series):
    values = pd.to_numeric(scale_series, errors="coerce")
    max_val = values.max()
    return float(max_val) if pd.notna(max_val) and float(max_val) > 0 else 1280.0


def _transform_scales(scale_series, mode, sample_size_ref):
    values = pd.to_numeric(scale_series, errors="coerce").to_numpy(dtype=float)
    if mode == "sample_size":
        return values / float(sample_size_ref)
    if mode == "preprocessing":
        return 1.0 - values
    return values


def _axis_label(mode, sample_size_ref):
    if mode == "sample_size":
        return f"Sample Size / N (N={int(sample_size_ref)})"
    if mode == "preprocessing":
        return "Severity (1 - scale)"
    return "Scale / Severity"


def _format_tick(value):
    if not np.isfinite(value):
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _apply_axis_ticks(ax, mode, tick_values):
    tick_values_sorted = sorted(v for v in tick_values if np.isfinite(v))
    if not tick_values_sorted:
        return
    if mode == "sample_size":
        rotation, alignment, fontsize = 35, "right", 6.5
    else:
        rotation, alignment, fontsize = 0, "center", 7
    ax.set_xticks(tick_values_sorted)
    ax.set_xticklabels(
        [_format_tick(v) for v in tick_values_sorted],
        rotation=rotation,
        ha=alignment,
        fontsize=fontsize,
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _baseline_mask(df):
    family_mask = pd.Series(False, index=df.index)
    if "perturbation_family" in df.columns:
        family_mask = df["perturbation_family"].astype(str).str.lower().eq("baseline")
    name_mask = df["name"].astype(str).str.contains("baseline", case=False, na=False)
    return family_mask | name_mask


def _clean_text(value):
    if pd.isna(value):
        return "unknown"
    return str(value).strip() or "unknown"


def _available_metric_specs(df, metrics=None):
    allowed = set(metrics) if metrics is not None else None
    return [
        (mk, lk, hk, label)
        for mk, lk, hk, label in METRIC_SPECS
        if (allowed is None or mk in allowed)
        and all(c in df.columns for c in [mk, lk, hk])
    ]


def _build_group_title(group_key, group_cols):
    parts = [
        f"{GROUP_LABELS.get(col, col)}: {_clean_text(val)}"
        for col, val in zip(group_cols, group_key)
    ]
    return " | ".join(parts)


def _chunk_frame(frame, chunk_size):
    if len(frame) <= chunk_size:
        return [frame]
    return [frame.iloc[i: i + chunk_size].copy() for i in range(0, len(frame), chunk_size)]


def _build_lookup_key(row, columns):
    return tuple(row[col] for col in columns)


def _build_baseline_reference_map(df, metric_specs, lookup_cols):
    if df.empty:
        return {}
    value_cols = [col for spec in metric_specs for col in spec[:3]]
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
    return {(): df[value_cols].mean(numeric_only=True)}


def _unique_legend_entries(axes):
    seen = set()
    handles_out, labels_out = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label and label not in seen:
                seen.add(label)
                handles_out.append(handle)
                labels_out.append(label)
    return handles_out, labels_out


# ── per-chunk plot ────────────────────────────────────────────────────────────

def _plot_chunk(group_df, metric_specs, group_title, baseline_row,
                chunk_index, chunk_count, mode, sample_size_ref):
    plt = _get_plt()

    plot_df = group_df.copy()

    if "scale" in plot_df.columns:
        plot_df["_x"] = _transform_scales(plot_df["scale"], mode, sample_size_ref)
    else:
        plot_df["_x"] = np.arange(len(plot_df), dtype=float)

    # domain shift has no numeric scale — fall back to label axis
    use_label_axis = False
    if mode == "default":
        for col, _ in LABEL_AXIS_CANDIDATES:
            if col in plot_df.columns:
                labels_series = plot_df[col].map(_clean_text)
                if (labels_series != "unknown").all():
                    plot_df["_x_label"] = labels_series
                    use_label_axis = True
                    break

    is_numeric = not use_label_axis
    sort_col = "_x" if is_numeric else ("_x_label" if "_x_label" in plot_df.columns else "name")
    plot_df = plot_df.sort_values([sort_col, "name"], kind="stable")

    n_cols = 3
    n_rows = int(math.ceil(len(metric_specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    x_label_str = _axis_label(mode, sample_size_ref)

    for index, (metric_key, low_key, high_key, metric_label) in enumerate(metric_specs):
        ax = axes_flat[index]
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

        if is_numeric:
            x_vals = pd.to_numeric(metric_df["_x"], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(x_vals)
            x_vals, values = x_vals[order], values[order]
            lower_err, upper_err = lower_err[order], upper_err[order]

            ax.errorbar(
                x_vals, values,
                yerr=np.vstack([lower_err, upper_err]),
                fmt="-o",
                color="tab:blue", ecolor="black",
                linewidth=1.4, elinewidth=1.0,
                capsize=3, markersize=4,
                label="Experiment values",
            )
            _apply_axis_ticks(ax, mode, set(x_vals[np.isfinite(x_vals)]))
        else:
            x_labels = (
                metric_df["_x_label"].tolist()
                if "_x_label" in metric_df.columns
                else metric_df["name"].map(_clean_text).tolist()
            )
            x_pos = np.arange(len(x_labels), dtype=float)
            ax.errorbar(
                x_pos, values,
                yerr=np.vstack([lower_err, upper_err]),
                fmt="o",
                color="tab:blue", ecolor="black",
                elinewidth=1.0, capsize=3, markersize=4,
                label="Experiment values",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)

        if baseline_row is not None:
            bv = baseline_row.get(metric_key, np.nan)
            bl = baseline_row.get(low_key, np.nan)
            bh = baseline_row.get(high_key, np.nan)
            if np.isfinite(bv):
                ax.axhline(bv, color="tab:red", linewidth=1.2, linestyle="-",
                           alpha=0.9, label="Baseline")
            if np.isfinite(bl):
                ax.axhline(bl, color="tab:red", linewidth=1.0, linestyle="--",
                           alpha=0.8, label="Baseline CI low")
            if np.isfinite(bh):
                ax.axhline(bh, color="tab:red", linewidth=1.0, linestyle="--",
                           alpha=0.8, label="Baseline CI high")

        ax.set_title(metric_label, fontsize=10)
        ax.set_xlabel(x_label_str, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

    for index in range(len(metric_specs), len(axes_flat)):
        axes_flat[index].axis("off")

    handles, labels = _unique_legend_entries(axes_flat)
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.965),
            ncol=min(4, len(labels)), fontsize=8,
        )

    suffix = f" (part {chunk_index} of {chunk_count})" if chunk_count > 1 else ""
    fig.suptitle(f"Experiment confidence intervals: {group_title}{suffix}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92 if handles else 0.96])
    plt.show()


# ── public entry point ────────────────────────────────────────────────────────

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

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataframe missing required grouping columns: {missing}")
    if "name" not in df.columns:
        raise RuntimeError("Dataframe missing required column: ['name']")

    metric_specs = _available_metric_specs(df, metrics=metrics)
    if not metric_specs:
        raise RuntimeError("No metric CI columns were found in the dataframe.")

    # always sub-group by perturbation_group if available, to get specific titles
    # (e.g. "preprocessing::resize" instead of just "preprocessing")
    has_pg = "perturbation_group" in df.columns
    effective_group_cols = list(group_cols)
    if has_pg and "perturbation_group" not in effective_group_cols:
        effective_group_cols = effective_group_cols + ["perturbation_group"]

    dedupe_cols = effective_group_cols + ["name"]
    plot_df = df.drop_duplicates(subset=dedupe_cols, keep="last").copy()

    baseline_mask = _baseline_mask(plot_df)
    # baseline lookup excludes perturbation_family and perturbation_group
    baseline_lookup_cols = tuple(
        c for c in effective_group_cols
        if c not in ("perturbation_family", "perturbation_group")
    )
    baseline_map = _build_baseline_reference_map(
        plot_df[baseline_mask].copy(),
        metric_specs=metric_specs,
        lookup_cols=baseline_lookup_cols,
    )

    ci_mask = pd.Series(False, index=plot_df.index)
    for mk, lk, hk, _ in metric_specs:
        ci_mask |= plot_df[[mk, lk, hk]].notna().all(axis=1)

    ci_df = plot_df[ci_mask & ~baseline_mask].copy()
    if ci_df.empty:
        print("No non-baseline experiment-level confidence intervals found in the dataframe.")
        return ci_df

    plotted_group_count = 0

    for group_key, group_df in ci_df.groupby(effective_group_cols, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_title = _build_group_title(group_key, effective_group_cols)

        # infer mode directly from perturbation_group values in this group,
        # falling back to perturbation_family if the column is absent
        if has_pg:
            mode = _axis_mode(group_df["perturbation_group"].dropna().unique())
        else:
            families = group_df["perturbation_family"].dropna().astype(str).unique()
            if all(f == "sample_size" for f in families):
                mode = "sample_size"
            elif all(f.startswith("preprocessing") for f in families):
                mode = "preprocessing"
            else:
                mode = "default"

        sample_size_ref = (
            _sample_size_reference(group_df["scale"])
            if mode == "sample_size" and "scale" in group_df.columns
            else 1280.0
        )

        # sample_size: never chunk — log axis shows all points
        if mode == "sample_size":
            chunked_frames = [group_df]
        else:
            sorted_df = group_df.copy()
            if "scale" in sorted_df.columns:
                sorted_df = sorted_df.sort_values(["scale", "name"], kind="stable")
            chunked_frames = _chunk_frame(sorted_df, max_experiments_per_figure)

        baseline_key = (
            _build_lookup_key(group_df.iloc[0], baseline_lookup_cols)
            if baseline_lookup_cols else ()
        )
        baseline_row = baseline_map.get(baseline_key)

        for chunk_index, chunk_df in enumerate(chunked_frames, start=1):
            _plot_chunk(
                chunk_df,
                metric_specs=metric_specs,
                group_title=group_title,
                baseline_row=baseline_row,
                chunk_index=chunk_index,
                chunk_count=len(chunked_frames),
                mode=mode,
                sample_size_ref=sample_size_ref,
            )

        plotted_group_count += 1

    print(
        f"Plotted confidence intervals for {len(ci_df)} non-baseline experiment rows "
        f"across {plotted_group_count} grouped plot set(s)."
    )
    return ci_df
"""
drop-in replacements for the four trustworthiness component plot functions
in full_analysis.py.

paste these functions into full_analysis.py and update the call sites in
run_full_perturbation_analysis() as shown in the comments at the bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Task3.analysis.shared import METRICS, METRIC_LABELS


# ── shared helper ────────────────────────────────────────────────────────────

def _base_family_label(group_label):
    """Return a human-readable family name for a perturbation group string."""
    text = str(group_label)
    if text.startswith("degradation_"):
        return "Degradation"
    if text.startswith("class_imbalance::"):
        return "Class Imbalance"
    if text.startswith("class_removal"):
        return "Class Removal"
    if text == "memoisation":
        return "Memoisation"
    if text == "sample_size":
        return "Sample Size"
    if text.startswith("preprocessing::"):
        return "Preprocessing"
    if text.startswith("domain_shift::"):
        return "Domain Shift"
    return text


FAMILY_ORDER = [
    "Degradation",
    "Memoisation",
    "Class Removal",
    "Class Imbalance",
    "Sample Size",
    "Preprocessing",
    "Domain Shift",
]


def _sort_groups_by_family(groups):
    """Sort perturbation group strings by canonical family order, then alphabetically within."""
    def sort_key(group):
        family = _base_family_label(group)
        family_rank = FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
        return (family_rank, group)
    return sorted(groups, key=sort_key)


# ── grouped heatmap (monotonicity, specificity) ───────────────────────────────

def _plot_grouped_heatmap(
    frame,
    value_column,
    title,
    cmap,
    colorbar_label,
    vmin=None,
    vmax=None,
    annotation_fn=None,
):
    """
    Heatmap of (perturbation_group x metric) with family-level row groupings.

    annotation_fn(row_index, col_index, value, group_name, metric) -> str
        Optional override for cell text. Defaults to 'NA' or '{value:.2f}'.
    """
    groups = _sort_groups_by_family(frame["perturbation_group"].dropna().astype(str).unique())
    pivot = (
        frame.pivot(index="perturbation_group", columns="metric", values=value_column)
        .reindex(index=groups, columns=METRICS)
    )

    # identify family boundaries
    families = [_base_family_label(g) for g in groups]
    boundary_indices = [
        i for i in range(1, len(families)) if families[i] != families[i - 1]
    ]

    fig_height = max(4, 0.38 * len(groups) + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    image = ax.imshow(
        pivot.fillna(0).to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # cell annotations
    for row_i in range(pivot.shape[0]):
        for col_i, metric in enumerate(METRICS):
            value = pivot.iloc[row_i, col_i]
            group_name = pivot.index[row_i]
            if annotation_fn is not None:
                text = annotation_fn(row_i, col_i, value, group_name, metric)
            else:
                text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col_i, row_i, text, ha="center", va="center", fontsize=7)

    # family dividers and left-side family labels
    for boundary in boundary_indices:
        ax.axhline(boundary - 0.5, color="white", linewidth=2.5)

    # build y-tick labels: bold family name on first row of each family block, plain group name elsewhere
    y_labels = []
    prev_family = None
    for group in groups:
        family = _base_family_label(group)
        # strip the family prefix from the group name for brevity
        short = str(group)
        for prefix in [
            "degradation_", "class_imbalance::", "class_removal::",
            "preprocessing::", "domain_shift::",
        ]:
            if short.startswith(prefix):
                short = short[len(prefix):]
                break

        if family != prev_family:
            y_labels.append(f"[{family}]  {short}")
            prev_family = family
        else:
            y_labels.append(f"  {short}")

    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=30, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)

    plt.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label=colorbar_label)
    plt.tight_layout()
    plt.show()


# ── monotonicity (grouped heatmap) ────────────────────────────────────────────

def plot_monotonicity_heatmap(monotonicity):
    # use clipped_rho (signed, clipped to [0,1]) if available,
    # falling back to rho_spearman for backward compatibility
    value_col = "clipped_rho" if "clipped_rho" in monotonicity.columns else "rho_spearman"
    _plot_grouped_heatmap(
        monotonicity,
        value_column=value_col,
        title="Monotonicity: signed Spearman ρ clipped to [0, 1] (higher = correct direction)",
        cmap="YlGn",
        colorbar_label="clipped ρ",
        vmin=0,
        vmax=1,
    )


# ── specificity (grouped heatmap with P marker) ───────────────────────────────

def plot_specificity_heatmap(specificity):
    def _annotate(row_i, col_i, value, group_name, metric):
        kind = specificity.loc[
            (specificity["perturbation_group"] == group_name) & (specificity["metric"] == metric),
            "specificity_kind",
        ]
        kind_str = kind.iloc[0] if len(kind) else ""
        if kind_str == "primary_metric":
            return "P"
        return "NA" if not np.isfinite(value) else f"{value:.2f}"

    _plot_grouped_heatmap(
        specificity,
        value_column="off_target_max_abs_norm",
        title="Specificity: off-target max |normalised change| (lower is better, P = primary metric)",
        cmap="Blues",
        colorbar_label="off-target max |norm change|",
        annotation_fn=_annotate,
    )


# ── sensitivity (heatmap + bar chart) ────────────────────────────────────────

def plot_sensitivity_heatmap(sensitivity):
    """
    Grouped heatmap of mean |norm change| per (perturbation_group, metric).
    On-target cells show the value; off-target cells are shown as blank (nan).
    Mirrors the specificity heatmap structure.
    """
    def _annotate(row_i, col_i, value, group_name, metric):
        kind = sensitivity.loc[
            (sensitivity["perturbation_group"] == group_name) & (sensitivity["metric"] == metric),
            "sensitivity_kind",
        ]
        kind_str = kind.iloc[0] if len(kind) else ""
        if kind_str == "off_target":
            return ""
        return "NA" if not np.isfinite(value) else f"{value:.2f}"

    _plot_grouped_heatmap(
        sensitivity,
        value_column="mean_abs_norm_change",
        title="Sensitivity: mean |normalised change| on-target groups (blank = off-target)",
        cmap="YlOrRd",
        colorbar_label="mean |norm change|",
        annotation_fn=_annotate,
    )


def plot_sensitivity_bars(sensitivity):
    """
    Bar chart of mean |normalised change| per metric aggregated across on-target groups.
    Accepts either the full per-group sensitivity table or the summary table.
    """
    import pandas as pd

    # if we have the full per-group table, aggregate to summary first
    if "perturbation_group" in sensitivity.columns:
        on_target = sensitivity[sensitivity["sensitivity_kind"] == "on_target"]
        summary_rows = []
        for m in METRICS:
            vals = on_target.loc[on_target["metric"] == m, "mean_abs_norm_change"].dropna().to_numpy(dtype=float)
            summary_rows.append({
                "metric": m,
                "mean_abs_norm_change": float(np.mean(vals)) if vals.size else np.nan,
                "max_abs_norm_change": float(np.max(vals)) if vals.size else np.nan,
            })
        summary = pd.DataFrame(summary_rows)
    else:
        summary = sensitivity

    metrics = [m for m in METRICS if m in summary["metric"].values]
    x = np.arange(len(metrics))
    width = 0.38

    mean_vals = [summary.loc[summary["metric"] == m, "mean_abs_norm_change"].iloc[0] for m in metrics]
    max_vals  = [summary.loc[summary["metric"] == m, "max_abs_norm_change"].iloc[0]  for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, mean_vals, width, label="mean |norm change| (on-target)", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, max_vals,  width, label="max |norm change| (on-target)",  color="tomato",    alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylabel("Normalised change", fontsize=9)
    ax.set_title("Sensitivity: on-target normalised response per metric (aggregated)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ── reliability (bar chart, one bar per metric) ───────────────────────────────

def plot_reliability_bars(reliability):
    """
    Bar chart of mean bootstrap CI width per metric.
    Metrics with no valid CI data are shown as zero with a hatched bar.
    """
    metrics = [m for m in METRICS if m in reliability["metric"].values]
    x = np.arange(len(metrics))

    widths = []
    has_data = []
    for m in metrics:
        row = reliability[reliability["metric"] == m]
        w = row["mean_ci_width"].iloc[0] if len(row) else np.nan
        n = row["n_experiments"].iloc[0] if len(row) else 0
        widths.append(float(w) if np.isfinite(w) else 0.0)
        has_data.append(n > 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (w, has) in enumerate(zip(widths, has_data)):
        bar = ax.bar(
            x[i], w,
            color="mediumseagreen" if has else "lightgrey",
            hatch="" if has else "//",
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylabel("Mean 95% CI width", fontsize=9)
    ax.set_title("Reliability: mean bootstrap CI width per metric (lower = more reliable)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    legend_patches = [
        mpatches.Patch(color="mediumseagreen", label="CI data available"),
        mpatches.Patch(facecolor="lightgrey", hatch="//", label="No CI data"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    plt.tight_layout()
    plt.show()


# ── normalised trustworthiness scores (Eq. 9–12 in report) ───────────────────

def _minmax_normalise(values, invert=False, fallback=0.5):
    """
    Min-max normalise an array across metrics.
    invert=True gives (1 - normalised), used for lower-is-better quantities.
    Returns fallback for all metrics if min == max.
    """
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or finite.max() == finite.min():
        return np.where(np.isfinite(arr), fallback, np.nan)
    lo, hi = finite.min(), finite.max()
    normed = (arr - lo) / (hi - lo)
    return (1.0 - normed) if invert else normed


def compute_trustworthiness_scores(sensitivity, specificity, monotonicity, reliability):
    """
    Compute per-metric normalised trustworthiness scores following the report formulas:

      S_m   = minmax(S_raw_m)                          higher is better  (Eq. 9)
      Sp_m  = 1 - minmax(D_raw_m)                      higher is better  (Eq. 10)
      M_m   = mean |rho| over target groups, clipped   already in [0,1]  (Eq. 11)
      R_m   = 1 - minmax(w_bar_m)                      higher is better  (Eq. 12)

    Returns a DataFrame with columns: metric, S, Sp, M, R
    """
    rows = []
    metrics = METRICS

    # S_m — sensitivity: minmax of mean_abs_norm_change aggregated over on-target groups (Eq. 9)
    if "perturbation_group" in sensitivity.columns:
        on_target = sensitivity[sensitivity["sensitivity_kind"] == "on_target"]
        s_raw = np.array([
            on_target.loc[on_target["metric"] == m, "mean_abs_norm_change"].dropna().mean()
            if m in on_target["metric"].values else np.nan
            for m in metrics
        ])
    else:
        s_raw = np.array([
            sensitivity.loc[sensitivity["metric"] == m, "mean_abs_norm_change"].iloc[0]
            if m in sensitivity["metric"].values else np.nan
            for m in metrics
        ])
    s_norm = _minmax_normalise(np.clip(s_raw, 0, None), invert=False, fallback=0.5)

    # Sp_m — specificity: ratio-based formula
    # Sp_m = 1 - d_off / (d_on + d_off + eps)
    # requires both on-target and off-target mean abs norm per metric
    EPSILON = 0.1
    sp_scores = []
    for m in metrics:
        on_rows  = specificity[(specificity["metric"] == m) & (specificity["specificity_kind"] == "primary_metric")]
        off_rows = specificity[(specificity["metric"] == m) & (specificity["specificity_kind"] == "off_target")]
        d_on  = on_rows["on_target_mean_abs_norm"].dropna().mean()  if len(on_rows)  else np.nan
        d_off = off_rows["off_target_mean_abs_norm"].dropna().mean() if len(off_rows) else np.nan
        if np.isfinite(d_on) and np.isfinite(d_off):
            sp_scores.append(float(1.0 - d_off / (d_on + d_off + EPSILON)))
        elif np.isfinite(d_off):
            sp_scores.append(float(1.0 / (1.0 + d_off)))
        else:
            sp_scores.append(1.0)
    sp_norm = np.array(sp_scores, dtype=float)

    # M_m — monotonicity: mean |rho| over target groups, clipped to [0,1] (Eq. 11)
    # target groups are those where the metric is in the primary set, i.e. not off_target
    primary_groups_per_metric = {
        m: specificity.loc[
            (specificity["metric"] == m) & (specificity["specificity_kind"] == "primary_metric"),
            "perturbation_group",
        ].tolist()
        for m in metrics
    }
    m_scores = []
    for m in metrics:
        target_groups = primary_groups_per_metric.get(m, [])
        if not target_groups:
            m_scores.append(0.0)
            continue
        # use clipped_rho (signed, clipped to [0,1]) if available
        rho_col = "clipped_rho" if "clipped_rho" in monotonicity.columns else "abs_rho"
        rhos = monotonicity.loc[
            (monotonicity["metric"] == m) & (monotonicity["perturbation_group"].isin(target_groups)),
            rho_col,
        ].dropna()
        m_scores.append(float(np.clip(rhos.mean(), 0, 1)) if len(rhos) else 0.0)
    m_arr = np.array(m_scores, dtype=float)

    # R_m — reliability: 1 - minmax of mean relative CI width (baseline-normalised)
    # falls back to mean_ci_width if mean_relative_ci_width is not available
    w_col = "mean_relative_ci_width" if "mean_relative_ci_width" in reliability.columns else "mean_ci_width"
    w_bar = np.array([
        reliability.loc[reliability["metric"] == m, w_col].iloc[0]
        if m in reliability["metric"].values else np.nan
        for m in metrics
    ])
    all_missing = not np.isfinite(w_bar).any()
    r_norm = np.full(len(metrics), 0.5) if all_missing else _minmax_normalise(w_bar, invert=True, fallback=0.5)

    for i, m in enumerate(metrics):
        rows.append({
            "metric": m,
            "S":  float(s_norm[i])  if np.isfinite(s_norm[i])  else np.nan,
            "Sp": float(sp_norm[i]) if np.isfinite(sp_norm[i]) else np.nan,
            "M":  float(m_arr[i])   if np.isfinite(m_arr[i])   else np.nan,
            "R":  float(r_norm[i])  if np.isfinite(r_norm[i])  else np.nan,
        })

    import pandas as pd
    return pd.DataFrame(rows)


def plot_trustworthiness_scores(sensitivity, specificity, monotonicity, reliability):
    """
    Grouped bar chart of the four normalised trustworthiness scores per metric.
    All scores are in [0, 1]; higher is always better.
    """
    scores = compute_trustworthiness_scores(sensitivity, specificity, monotonicity, reliability)
    metrics = [m for m in METRICS if m in scores["metric"].values]

    x = np.arange(len(metrics))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    components = [
        ("S",  "Sensitivity",   "steelblue"),
        ("Sp", "Specificity",   "darkorange"),
        ("M",  "Monotonicity",  "mediumseagreen"),
        ("R",  "Reliability",   "mediumpurple"),
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for (col, label, color), offset in zip(components, offsets):
        vals = [
            scores.loc[scores["metric"] == m, col].iloc[0]
            if m in scores["metric"].values else np.nan
            for m in metrics
        ]
        ax.bar(x + offset * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalised score [0, 1]", fontsize=9)
    ax.set_title("Trustworthiness component scores per metric (higher is better)", fontsize=11)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    return scores
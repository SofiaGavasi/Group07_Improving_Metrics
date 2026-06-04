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
    _plot_grouped_heatmap(
        monotonicity,
        value_column="rho_spearman",
        title="Monotonical Association: Spearman ρ (scale vs normalised metric)",
        cmap="coolwarm",
        colorbar_label="ρ",
        vmin=-1,
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


# ── sensitivity (bar chart, one bar per metric) ───────────────────────────────

def plot_sensitivity_bars(sensitivity):
    """
    Bar chart of mean and max |normalised change| per metric,
    computed only over on-target perturbation rows.
    """
    metrics = [m for m in METRICS if m in sensitivity["metric"].values]
    x = np.arange(len(metrics))
    width = 0.38

    mean_vals = [
        sensitivity.loc[sensitivity["metric"] == m, "mean_abs_norm_change"].iloc[0]
        if m in sensitivity["metric"].values else np.nan
        for m in metrics
    ]
    max_vals = [
        sensitivity.loc[sensitivity["metric"] == m, "max_abs_norm_change"].iloc[0]
        if m in sensitivity["metric"].values else np.nan
        for m in metrics
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, mean_vals, width, label="mean |norm change| (on-target)", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, max_vals, width, label="max |norm change| (on-target)", color="tomato", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylabel("Normalised change", fontsize=9)
    ax.set_title("Sensitivity: on-target normalised response per metric", fontsize=11)
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



"""
visualize_perturbation_tests.py
--------------------------------
Reads one or more StyleGAN2 perturbation test JSON files and produces a
multi-page PDF dashboard (one page per JSON file  +  a final comparison page
when multiple files are loaded).

Usage
-----
# Single file
python visualize_perturbation_tests.py results.json

# All JSON files in a directory
python visualize_perturbation_tests.py results_dir/

# Multiple explicit files
python visualize_perturbation_tests.py file1.json file2.json file3.json

# Save to a specific PDF (default: perturbation_dashboard.pdf)
python visualize_perturbation_tests.py results_dir/ -o my_report.pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "baseline":        "#4C72B0",
    "degrade_noise":   "#DD8452",
    "degrade_blur":    "#55A868",
    "degrade_jpeg":    "#C44E52",
    "degrade_all":     "#8172B2",
    "memo":            "#937860",
    "class_removal":   "#DA8BC3",
    "class_imbalance": "#8C8C8C",
    "other":           "#CCB974",
}

METRIC_LABELS = {
    "fid":        "FID ↓",
    "is_mean":    "IS ↑",
    "kid_mean":   "KID ↓",
    "precision":  "Precision ↑",
    "recall":     "Recall ↑",
    "density":    "Density ↑",
    "coverage":   "Coverage ↑",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_files(args_paths: list[str]) -> list[Path]:
    files = []
    for p in args_paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("*.json")))
        elif pp.suffix == ".json" and pp.exists():
            files.append(pp)
        else:
            print(f"[warn] skipping {p} — not a .json file or directory")
    if not files:
        sys.exit("No JSON files found. Provide at least one .json file or a directory.")
    return files


def group_name(exp_name: str) -> str:
    """Map experiment name → colour-group key."""
    n = exp_name.lower()
    if n.startswith("baseline"):           return "baseline"
    if "noise" in n:                       return "degrade_noise"
    if "blur"  in n:                       return "degrade_blur"
    if "jpeg"  in n:                       return "degrade_jpeg"
    if "degrade_all" in n:                 return "degrade_all"
    if n.startswith("memo"):               return "memo"
    if "class_removal" in n:              return "class_removal"
    if "class_imbalance" in n:            return "class_imbalance"
    return "other"


def extract_metrics(data: dict) -> list[dict]:
    """Return a flat list of dicts, one per experiment."""
    rows = []
    for exp in data.get("experiments", []):
        outputs = exp.get("test_outputs") or []
        if not outputs:
            continue
        mr = outputs[0].get("metrics_report")
        if not mr:
            continue
        ov = exp.get("overrides", {})
        row = {
            "name":       exp["name"],
            "group":      group_name(exp["name"]),
            "status":     exp.get("status", "unknown"),
            "fid":        mr.get("fid"),
            "is_mean":    mr["is"][0]   if isinstance(mr.get("is"), list)  else mr.get("is"),
            "is_std":     mr["is"][1]   if isinstance(mr.get("is"), list)  else 0,
            "kid_mean":   mr["kid"][0]  if isinstance(mr.get("kid"), list) else mr.get("kid"),
            "kid_std":    mr["kid"][1]  if isinstance(mr.get("kid"), list) else 0,
            "precision":  (mr.get("precision_recall") or {}).get("precision"),
            "recall":     (mr.get("precision_recall") or {}).get("recall"),
            "density":    (mr.get("density_coverage") or {}).get("density"),
            "coverage":   (mr.get("density_coverage") or {}).get("coverage"),
            "perturb_severity": ov.get("PERTURB_DEGRADE_SEVERITY"),
            "memo_fraction":    ov.get("PERTURB_MEMO_FRACTION"),
            "apply_to":         ov.get("PERTURB_APPLY_TO", ""),
        }
        rows.append(row)
    return rows


def short_label(name: str, max_len: int = 28) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


METRIC_KEYS    = ["fid", "is_mean", "kid_mean", "precision", "recall", "density", "coverage"]
HIGHER_BETTER  = [False, True, False, True, True, True, True]


def divergence_score(row: dict, baseline: dict) -> float:
    """
    Single divergence score (higher = more different from baseline).
    Each metric contributes equally via its relative absolute change,
    then the per-metric values are averaged.
    """
    deltas = []
    for key in METRIC_KEYS:
        b_val = baseline.get(key)
        r_val = row.get(key)
        if b_val is None or r_val is None or b_val == 0:
            continue
        deltas.append(abs(r_val - b_val) / abs(b_val))
    return float(np.mean(deltas)) if deltas else 0.0


def top_n_divergent(rows: list[dict], n: int = 5) -> tuple:
    """
    Returns (baseline_row, top_n_rows_sorted_by_divergence_desc).
    Each row in top_n gets a '_divergence' key attached.
    """
    baseline = next((r for r in rows if r["group"] == "baseline"), None)
    if baseline is None:
        return None, []
    others = [r for r in rows if r["group"] != "baseline"]
    scored = sorted(others, key=lambda r: divergence_score(r, baseline), reverse=True)
    top = scored[:n]
    for r in top:
        r["_divergence"] = divergence_score(r, baseline)
    return baseline, top


def bar_colors(rows: list[dict]) -> list[str]:
    return [GROUP_COLORS.get(r["group"], GROUP_COLORS["other"]) for r in rows]


# ── per-file dashboard ────────────────────────────────────────────────────────

def page_overview(pdf: PdfPages, data: dict, rows: list[dict]):
    """Page 1: title card + experiment status table."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")

    model   = data.get("model_name", "?")
    dataset = data.get("dataset_name", "?")
    created = data.get("created_at_utc", "?")[:19].replace("T", " ")
    n_exp   = len(data.get("experiments", []))

    title_txt = f"{model.upper()}  ×  {dataset.upper()}"
    fig.text(0.5, 0.88, title_txt, ha="center", fontsize=26, fontweight="bold")
    fig.text(0.5, 0.82, f"Created {created}  |  {n_exp} experiments",
             ha="center", fontsize=13, color="#555")

    # Summary table
    col_headers = ["Experiment", "Group", "Status", "FID", "IS", "KID", "Precision", "Recall"]
    table_data = []
    for r in rows:
        table_data.append([
            short_label(r["name"]),
            r["group"],
            r["status"],
            f"{r['fid']:.1f}"        if r["fid"]       is not None else "—",
            f"{r['is_mean']:.3f}"    if r["is_mean"]    is not None else "—",
            f"{r['kid_mean']:.4f}"   if r["kid_mean"]   is not None else "—",
            f"{r['precision']:.3f}"  if r["precision"]  is not None else "—",
            f"{r['recall']:.3f}"     if r["recall"]     is not None else "—",
        ])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.75],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_i % 2 == 0:
            cell.set_facecolor("#F2F2F2")

    fig.suptitle("Experiment Summary", y=0.96, fontsize=14, color="#333")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_main_metrics(pdf: PdfPages, rows: list[dict], file_label: str):
    """Page 2: FID, IS, KID bar charts (one per metric, experiments on x-axis)."""
    metrics = [
        ("fid",      "FID ↓  (lower = better)",      "is_std",   False),
        ("is_mean",  "IS ↑  (higher = better)",       "is_std",   True),
        ("kid_mean", "KID ↓  (lower = better)",       "kid_std",  False),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle(f"Core GAN Metrics — {file_label}", fontsize=15, fontweight="bold", y=1.01)

    labels = [short_label(r["name"], 32) for r in rows]
    x = np.arange(len(rows))
    colors = bar_colors(rows)

    for ax, (key, ylabel, err_key, higher_better) in zip(axes, metrics):
        vals  = [r[key]     if r[key]     is not None else 0 for r in rows]
        errs  = [r[err_key] if r[err_key] is not None else 0 for r in rows]
        bars  = ax.bar(x, vals, color=colors, width=0.65, edgecolor="white", linewidth=0.6)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="#333", capsize=3, linewidth=1)

        # Highlight baseline
        baseline_idx = next((i for i, r in enumerate(rows) if r["group"] == "baseline"), None)
        if baseline_idx is not None:
            ax.axhline(vals[baseline_idx], color="#2C3E50", linestyle="--", linewidth=1,
                       alpha=0.7, label=f"Baseline = {vals[baseline_idx]:.2f}")
            ax.legend(fontsize=8)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_precision_recall_density_coverage(pdf: PdfPages, rows: list[dict], file_label: str):
    """Page 3: Precision/Recall and Density/Coverage side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Precision / Recall / Density / Coverage — {file_label}",
                 fontsize=15, fontweight="bold")

    labels = [short_label(r["name"], 32) for r in rows]
    x = np.arange(len(rows))
    colors = bar_colors(rows)

    pairs = [
        (axes[0, 0], "precision", "Precision ↑"),
        (axes[0, 1], "recall",    "Recall ↑"),
        (axes[1, 0], "density",   "Density ↑"),
        (axes[1, 1], "coverage",  "Coverage ↑"),
    ]
    for ax, key, ylabel in pairs:
        vals = [r[key] if r[key] is not None else 0 for r in rows]
        ax.bar(x, vals, color=colors, width=0.65, edgecolor="white", linewidth=0.6)
        baseline_idx = next((i for i, r in enumerate(rows) if r["group"] == "baseline"), None)
        if baseline_idx is not None:
            ax.axhline(vals[baseline_idx], color="#2C3E50", linestyle="--",
                       linewidth=1, alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # Shared colour legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, list(GROUP_COLORS.keys()), loc="lower center",
               ncol=len(GROUP_COLORS), fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_radar(pdf: PdfPages, rows: list[dict], file_label: str):
    """Page 4: Radar / spider chart normalised across all experiments."""
    metric_keys  = ["fid", "is_mean", "kid_mean", "precision", "recall", "density", "coverage"]
    metric_names = ["FID", "IS", "KID", "Precision", "Recall", "Density", "Coverage"]
    higher_better = [False, True, False, True, True, True, True]

    # Normalise 0-1 (for "lower is better" invert)
    raw = {k: [r[k] for r in rows if r[k] is not None] for k in metric_keys}
    mn  = {k: min(v) for k, v in raw.items() if v}
    mx  = {k: max(v) for k, v in raw.items() if v}

    def norm(k, v):
        if v is None: return 0
        span = mx[k] - mn[k]
        if span == 0: return 0.5
        n = (v - mn[k]) / span
        return n if higher_better[metric_keys.index(k)] else 1 - n

    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Radar Chart (normalised) — {file_label}", fontsize=15, fontweight="bold")

    # Up to 9 experiments per radar; if more, take every nth
    max_traces = 9
    step = max(1, len(rows) // max_traces)
    selected = rows[::step][:max_traces]

    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(selected))
    for i, r in enumerate(selected):
        values = [norm(k, r[k]) for k in metric_keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, color=cmap(i), label=short_label(r["name"], 30))
        ax.fill(angles, values, alpha=0.08, color=cmap(i))

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_heatmap(pdf: PdfPages, rows: list[dict], file_label: str):
    """Page 5: Heatmap of all metrics × all experiments."""
    metric_keys  = ["fid", "is_mean", "kid_mean", "precision", "recall", "density", "coverage"]
    metric_names = list(METRIC_LABELS.values())
    higher_better = [False, True, False, True, True, True, True]

    mat = np.array([[r[k] if r[k] is not None else np.nan for k in metric_keys] for r in rows],
                   dtype=float)

    # Normalise column-wise, then flip "lower is better"
    norm_mat = np.zeros_like(mat)
    for j, hb in enumerate(higher_better):
        col = mat[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        lo, hi = valid.min(), valid.max()
        span = hi - lo
        n = (col - lo) / span if span else np.full_like(col, 0.5)
        norm_mat[:, j] = n if hb else 1 - n

    exp_labels = [short_label(r["name"], 32) for r in rows]

    fig, ax = plt.subplots(figsize=(14, max(6, len(rows) * 0.38 + 2)))
    fig.suptitle(f"Metric Heatmap (normalised, green = better) — {file_label}",
                 fontsize=14, fontweight="bold")

    im = ax.imshow(norm_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(exp_labels, fontsize=8)

    # Annotate with raw values
    for i in range(len(rows)):
        for j, k in enumerate(metric_keys):
            val = mat[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="#111")

    fig.colorbar(im, ax=ax, label="Normalised score (1 = best)", pad=0.01)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_severity_trends(pdf: PdfPages, rows: list[dict], file_label: str):
    """Page 6: Line charts of metrics vs severity for degradation experiments."""
    sev_groups = {
        "noise": [r for r in rows if r["group"] == "degrade_noise"],
        "blur":  [r for r in rows if r["group"] == "degrade_blur"],
        "jpeg":  [r for r in rows if r["group"] == "degrade_jpeg"],
    }
    has_sev = any(g for g in sev_groups.values())
    if not has_sev:
        return

    baseline = next((r for r in rows if r["group"] == "baseline"), None)
    metric_keys  = ["fid", "is_mean", "kid_mean", "precision", "recall", "density", "coverage"]
    metric_names = list(METRIC_LABELS.values())

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(12, 4 * len(metric_keys)), sharex=False)
    fig.suptitle(f"Degradation Severity Trends — {file_label}", fontsize=15, fontweight="bold")

    colors_sev = {"noise": "#DD8452", "blur": "#55A868", "jpeg": "#C44E52"}

    for ax, key, name in zip(axes, metric_keys, metric_names):
        for grp_name, grp_rows in sev_groups.items():
            if not grp_rows: continue
            sev   = [r["perturb_severity"] for r in grp_rows]
            vals  = [r[key] for r in grp_rows]
            pairs = sorted(zip(sev, vals), key=lambda t: (t[0] is None, t[0]))
            sv, vv = zip(*pairs) if pairs else ([], [])
            ax.plot(sv, vv, marker="o", color=colors_sev[grp_name],
                    label=grp_name, linewidth=2)

        if baseline and baseline[key] is not None:
            ax.axhline(baseline[key], color="#2C3E50", linestyle="--",
                       linewidth=1, alpha=0.7, label="baseline")

        ax.set_ylabel(name, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Severity", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── top-5 divergent page ────────────────────────────────────────────────

def page_top_divergent(pdf, rows, file_label, n=5):
    """
    One subplot per metric. Each subplot shows the baseline bar + the top-N
    experiments that deviate most from baseline on THAT specific metric.
    Bars are annotated with the % change. Condensed to one page.
    """
    baseline = next((r for r in rows if r["group"] == "baseline"), None)
    if baseline is None:
        return

    non_baseline = [r for r in rows if r["group"] != "baseline"]
    metric_names = list(METRIC_LABELS.values())
    n_metrics    = len(METRIC_KEYS)

    # 4 columns, enough rows to fit all metrics
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.5),
                             constrained_layout=True)
    fig.suptitle(
        f"Top {n} Most Divergent per Metric vs Baseline — {file_label}",
        fontsize=15, fontweight="bold",
    )
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, key, metric_label in zip(axes_flat, METRIC_KEYS, metric_names):
        bl_val = baseline.get(key)

        # Rank non-baseline experiments by absolute % change on this metric
        scored = []
        for r in non_baseline:
            r_val = r.get(key)
            if bl_val is not None and r_val is not None and bl_val != 0:
                scored.append((abs(r_val - bl_val) / abs(bl_val), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [r for _, r in scored[:n]]

        if not top:
            ax.axis("off")
            continue

        # baseline first, then top-N
        combined = [baseline] + top
        labels   = [short_label(r["name"], 18) for r in combined]
        vals     = [r[key] if r[key] is not None else 0 for r in combined]
        colors   = [GROUP_COLORS["baseline"]] + [
            GROUP_COLORS.get(r["group"], GROUP_COLORS["other"]) for r in top
        ]

        bar_list = ax.bar(range(len(combined)), vals, color=colors,
                          width=0.65, edgecolor="white", linewidth=0.6)
        bar_list[0].set_edgecolor("#2C3E50")
        bar_list[0].set_linewidth(2.5)

        # Annotate % delta on each non-baseline bar
        for b, r in zip(bar_list[1:], top):
            r_val = r.get(key)
            if bl_val is not None and r_val is not None and bl_val != 0:
                pct  = (r_val - bl_val) / abs(bl_val) * 100
                sign = "+" if pct >= 0 else ""
                ypos = max(b.get_height(), bl_val if bl_val else 0) * 0.02 + b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, ypos,
                        f"{sign}{pct:.1f}%",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                        color="#C0392B" if pct < 0 else "#1A5276")

        # Baseline reference line
        if bl_val is not None:
            ax.axhline(bl_val, color="#2C3E50", linestyle="--",
                       linewidth=1.2, alpha=0.6)

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(combined)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    # Hide any unused subplots
    for ax in list(axes_flat)[n_metrics:]:
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── multi-file comparison ─────────────────────────────────────────────────────

def page_cross_file_comparison(pdf: PdfPages, all_data: list[tuple[str, list[dict]]]):
    """Final page: compare baseline metrics across all loaded files."""
    if len(all_data) < 2:
        return

    metric_keys  = ["fid", "is_mean", "kid_mean", "precision", "recall", "density", "coverage"]
    metric_names = list(METRIC_LABELS.values())

    # Extract baseline row from each file
    baselines = []
    for label, rows in all_data:
        bl = next((r for r in rows if r["group"] == "baseline"), None)
        if bl:
            baselines.append((label, bl))

    if len(baselines) < 2:
        return

    fig, axes = plt.subplots(len(metric_keys), 1,
                             figsize=(14, 3.5 * len(metric_keys)))
    fig.suptitle("Cross-File Comparison — Baseline Metrics", fontsize=15, fontweight="bold")

    file_labels = [b[0] for b in baselines]
    x = np.arange(len(baselines))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(baselines))
    colors = [cmap(i) for i in range(len(baselines))]

    for ax, key, name in zip(axes, metric_keys, metric_names):
        vals = [b[1][key] if b[1][key] is not None else 0 for b in baselines]
        ax.bar(x, vals, color=colors, width=0.6, edgecolor="white")
        ax.set_ylabel(name, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(l, 30) for l in file_labels],
                           rotation=30, ha="right", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def build_dashboard(files: list[Path], output_pdf: str):
    print(f"Loading {len(files)} file(s) …")
    all_data = []

    with PdfPages(output_pdf) as pdf:
        for fp in files:
            print(f"  Processing {fp.name} …")
            data  = load_json(fp)
            rows  = extract_metrics(data)
            label = fp.stem          # filename without extension

            if not rows:
                print(f"    [warn] no experiment rows found — skipping")
                continue

            all_data.append((label, rows))

            page_overview(pdf, data, rows)
            page_main_metrics(pdf, rows, label)
            page_precision_recall_density_coverage(pdf, rows, label)
            page_radar(pdf, rows, label)
            page_heatmap(pdf, rows, label)
            page_severity_trends(pdf, rows, label)
            page_top_divergent(pdf, rows, label)

        # Cross-file comparison page
        if len(all_data) >= 2:
            print("  Generating cross-file comparison page …")
            page_cross_file_comparison(pdf, all_data)

        meta = pdf.infodict()
        meta["Title"]   = "StyleGAN2 Perturbation Test Dashboard"
        meta["Author"]  = "visualize_perturbation_tests.py"

    print(f"\n✓ Dashboard saved → {output_pdf}")
    print(f"  {len(all_data)} file(s) processed, "
          f"{sum(len(r) for _, r in all_data)} total experiments.")


def main():
    '''
    parser = argparse.ArgumentParser(
        description="Visualise StyleGAN2 perturbation test JSON files.")
    parser.add_argument("paths", nargs="+",
                        help="JSON files or directories containing JSON files")
    parser.add_argument("-o", "--output", default="perturbation_dashboard.pdf",
                        help="Output PDF path (default: perturbation_dashboard.pdf)")
    args = parser.parse_args()
    '''
    files = collect_files([r"C:\Users\jaime\Documents\Maastricht_work\Master\P5\stylegan2_celeba_perturbation_tests.json"])
    #files = collect_files(args.paths)
    build_dashboard(files, "perturbation_dashboard_shortX.pdf")


if __name__ == "__main__":
    main()

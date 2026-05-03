"""
Metric Scorecard — DCGAN CIFAR-10 perturbation analysis
Evaluates: Monotonicity (Spearman), Sensitivity/Specificity (relative change heatmap),
           Robustness (CV across k-means hyperparameter variants)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from pathlib import Path

# ─── Load data ────────────────────────────────────────────────────────────────

DATA_PATH = Path(r"C:\Users\jaime\Documents\Maastricht_work\Master\P5\Semester project\Group07_Improving_Metrics\dcgan_cifar10_perturbation_tests.json")

with open(DATA_PATH) as f:
    raw = json.load(f)

def extract_metrics(exp):
    """Return flat metric dict or None if experiment has no output."""
    to = exp.get("test_outputs")
    if not to or not to[0].get("metrics_report"):
        return None
    r = to[0]["metrics_report"]
    return {
        "fid":       r["fid"],
        "is":        r["is"][0],
        "kid":       r["kid"][0],
        "precision": r["precision_recall"]["precision"],
        "recall":    r["precision_recall"]["recall"],
        "density":   r["density_coverage"]["density"],
        "coverage":  r["density_coverage"]["coverage"],
    }

# Index all experiments by name
exps = {e["name"]: extract_metrics(e) for e in raw["experiments"]}

METRICS       = ["fid", "is", "kid", "precision", "recall", "density", "coverage"]
METRIC_LABELS = ["FID", "IS", "KID", "Precision", "Recall", "Density", "Coverage"]

BASELINE = exps["dcgan_cifar10_baseline_no_perturbation"]

# ─── Group definitions ────────────────────────────────────────────────────────

# Severity series: list of (severity_level, experiment_name)
SEVERITY_SERIES = {
    "Gaussian noise": [
        (1, "dcgan_cifar10_degrade_noise_sev1"),
        (3, "dcgan_cifar10_degrade_noise_sev3"),
        (5, "dcgan_cifar10_degrade_noise_sev5"),
    ],
    "Gaussian blur": [
        (1, "dcgan_cifar10_degrade_blur_sev1"),
        (3, "dcgan_cifar10_degrade_blur_sev3"),
        (5, "dcgan_cifar10_degrade_blur_sev5"),
    ],
    "JPEG compression": [
        (1, "dcgan_cifar10_degrade_jpeg_sev1"),
        (3, "dcgan_cifar10_degrade_jpeg_sev3"),
        (5, "dcgan_cifar10_degrade_jpeg_sev5"),
    ],
    #use this to add more like sample size I guess 
}

# Sensitivity groups (any series with varying intensity)
SENSITIVITY_GROUPS = {
    "Gaussian noise":   ["dcgan_cifar10_degrade_noise_sev1",  "dcgan_cifar10_degrade_noise_sev3",  "dcgan_cifar10_degrade_noise_sev5"],
    "Gaussian blur":    ["dcgan_cifar10_degrade_blur_sev1",   "dcgan_cifar10_degrade_blur_sev3",   "dcgan_cifar10_degrade_blur_sev5"],
    "JPEG compression": ["dcgan_cifar10_degrade_jpeg_sev1",   "dcgan_cifar10_degrade_jpeg_sev3",   "dcgan_cifar10_degrade_jpeg_sev5"],
    "Memoisation":      ["dcgan_cifar10_memo_frac_05pct",     "dcgan_cifar10_memo_frac_15pct",     "dcgan_cifar10_memo_frac_30pct"],
}

# Robustness: same conceptual perturbation, different k-means K
ROBUSTNESS_VARIANTS = [
    "dcgan_cifar10_class_removal_kmeans_k6_cluster1",
    "dcgan_cifar10_class_removal_kmeans_k8_cluster0",
    "dcgan_cifar10_class_removal_kmeans_k8_cluster3",
]

# ─── Analysis functions ───────────────────────────────────────────────────────

def compute_monotonicity():
    """
    Spearman rho between severity level and metric value for each
    perturbation type × metric combination.
    Returns dict: { perturb_name: { metric: rho } }
    """
    results = {}
    for perturb, series in SEVERITY_SERIES.items():
        sevs   = [s for s, _ in series]
        rhos   = {}
        for m in METRICS:
            vals = [exps[name][m] for _, name in series if exps[name] is not None]
            rho, _ = spearmanr(sevs, vals)
            rhos[m] = rho
        results[perturb] = rhos
    return results

def compute_sensitivity():
    """
    Max relative change from baseline across perturbation levels,
    for each perturbation group × metric.
    Returns dict: { perturb_name: { metric: max_rel_change } }
    """
    results = {}
    for perturb, names in SENSITIVITY_GROUPS.items():
        max_change = {}
        for m in METRICS:
            base = BASELINE[m]
            changes = [abs((exps[n][m] - base) / abs(base)) for n in names if exps[n]]
            max_change[m] = max(changes) if changes else 0.0
        results[perturb] = max_change
    return results

def compute_robustness():
    """
    Coefficient of variation (CV = std/mean) across k-means K variants
    for class removal perturbation.
    Returns dict: { metric: { mean, std, cv } }
    """
    results = {}
    valid = [exps[n] for n in ROBUSTNESS_VARIANTS if exps.get(n)]
    for m in METRICS:
        vals = np.array([e[m] for e in valid])
        mean = vals.mean()
        std  = vals.std()
        cv   = abs(std / mean) if mean != 0 else 0.0
        results[m] = {"mean": mean, "std": std, "cv": cv}
    return results

def flag_findings(mono, sens, robust):
    """
    Produce a list of flag dicts with keys:
      severity ('fail'|'warn'|'pass'), metric, perturb, title, detail
    """
    flags = []
    MONO_PASS = 0.95
    MONO_WARN = 0.70
    SENS_HIGH = 0.30
    CV_PASS   = 0.05
    CV_WARN   = 0.15

    # Monotonicity flags
    for perturb, rhos in mono.items():
        for m, rho in rhos.items():
            av = abs(rho)
            if av < MONO_WARN:
                flags.append({
                    "severity": "fail",
                    "metric": m.upper(),
                    "perturb": perturb,
                    "title": f"{perturb} — {m.upper()} non-monotone",
                    "detail": f"Spearman ρ = {rho:.2f} (|ρ| < {MONO_WARN}). "
                              f"Metric does not track severity consistently."
                })
            elif av < MONO_PASS:
                flags.append({
                    "severity": "warn",
                    "metric": m.upper(),
                    "perturb": perturb,
                    "title": f"{perturb} — {m.upper()} borderline monotonicity",
                    "detail": f"Spearman ρ = {rho:.2f} (borderline {MONO_WARN}–{MONO_PASS})."
                })

    # IS insensitivity (should be sensitive to degradation)
    for perturb in ["Gaussian noise", "Gaussian blur", "JPEG compression"]:
        v = sens[perturb]["is"]
        if v < 0.05:
            flags.append({
                "severity": "fail",
                "metric": "IS",
                "perturb": perturb,
                "title": f"IS insensitive to {perturb}",
                "detail": f"Max relative change = {v*100:.1f}%. IS measures class diversity, "
                          f"not image fidelity — it cannot detect image quality degradation."
            })

    # Robustness flags
    for m, d in robust.items():
        cv = d["cv"]
        if cv > CV_WARN:
            flags.append({
                "severity": "fail",
                "metric": m.upper(),
                "perturb": "Class removal (k-means K)",
                "title": f"{m.upper()} poorly robust to k-means hyperparameter",
                "detail": f"CV = {cv*100:.1f}% (threshold {CV_WARN*100:.0f}%). "
                          f"Mean = {d['mean']:.3f}, std = {d['std']:.3f}."
            })
        elif cv > CV_PASS:
            flags.append({
                "severity": "warn",
                "metric": m.upper(),
                "perturb": "Class removal (k-means K)",
                "title": f"{m.upper()} borderline robust to k-means hyperparameter",
                "detail": f"CV = {cv*100:.1f}% (borderline {CV_PASS*100:.0f}%–{CV_WARN*100:.0f}%)."
            })

    # Passing behaviours
    for perturb, rhos in mono.items():
        for m, rho in rhos.items():
            if abs(rho) >= MONO_PASS:
                flags.append({
                    "severity": "pass",
                    "metric": m.upper(),
                    "perturb": perturb,
                    "title": f"{perturb} — {m.upper()} monotone (pass)",
                    "detail": f"Spearman ρ = {rho:.2f}."
                })

    return flags

# ─── Plotting ─────────────────────────────────────────────────────────────────

def rho_color(rho):
    av = abs(rho)
    if av >= 0.95: return "#EAF3DE"
    if av >= 0.70: return "#FAEEDA"
    return "#FCEBEB"

def rho_textcolor(rho):
    av = abs(rho)
    if av >= 0.95: return "#3B6D11"
    if av >= 0.70: return "#854F0B"
    return "#A32D2D"

def sens_color(v):
    if v >= 0.30: return "#B5D4F4"
    if v >= 0.10: return "#E6F1FB"
    return "#F1EFE8"

def sens_textcolor(v):
    if v >= 0.30: return "#0C447C"
    if v >= 0.10: return "#185FA5"
    return "#5F5E5A"

def cv_color(cv):
    if cv <= 0.05: return "#EAF3DE"
    if cv <= 0.15: return "#FAEEDA"
    return "#FCEBEB"

def cv_textcolor(cv):
    if cv <= 0.05: return "#3B6D11"
    if cv <= 0.15: return "#854F0B"
    return "#A32D2D"


def plot_heatmap(ax, data_matrix, row_labels, col_labels, cell_colors, cell_textcolors, title, fmt=".2f"):
    ax.set_xlim(0, len(col_labels))
    ax.set_ylim(0, len(row_labels))
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels[::-1], fontsize=9)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=11, fontweight="normal", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, row in enumerate(reversed(data_matrix)):
        for j, val in enumerate(row):
            bg   = cell_colors[len(data_matrix)-1-i][j]
            tc   = cell_textcolors[len(data_matrix)-1-i][j]
            rect = plt.Rectangle([j, i], 1, 1, color=bg)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, format(val, fmt),
                    ha="center", va="center", fontsize=9,
                    color=tc, fontweight="normal")


def make_figure(mono, sens, robust, flags):
    fig = plt.figure(figsize=(16, 22), facecolor="white")
    fig.suptitle("Metric Scorecard — DCGAN / CIFAR-10", fontsize=14,
                 fontweight="normal", y=0.98, color="#1a1a18")

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.95, bottom=0.03, left=0.08, right=0.97)

    # ── 1. Monotonicity heatmap ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    perturbs = list(mono.keys())
    mono_vals   = [[mono[p][m] for m in METRICS] for p in perturbs]
    mono_colors = [[rho_color(mono[p][m])     for m in METRICS] for p in perturbs]
    mono_tc     = [[rho_textcolor(mono[p][m]) for m in METRICS] for p in perturbs]
    plot_heatmap(ax1, mono_vals, perturbs, METRIC_LABELS,
                 mono_colors, mono_tc,
                 "Monotonicity — Spearman ρ (severity vs metric value)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EAF3DE", edgecolor="#C0DD97", label="|ρ| ≥ 0.95  pass"),
        Patch(facecolor="#FAEEDA", edgecolor="#FAC775", label="0.70–0.95  borderline"),
        Patch(facecolor="#FCEBEB", edgecolor="#F7C1C1", label="< 0.70  fail"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right",
               fontsize=8, frameon=False, ncol=3)

    # ── 2. Sensitivity heatmap ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    sens_groups = list(sens.keys())
    sens_vals   = [[sens[p][m] for m in METRICS] for p in sens_groups]
    sens_colors = [[sens_color(sens[p][m])     for m in METRICS] for p in sens_groups]
    sens_tc     = [[sens_textcolor(sens[p][m]) for m in METRICS] for p in sens_groups]
    plot_heatmap(ax2, sens_vals, sens_groups, METRIC_LABELS,
                 sens_colors, sens_tc,
                 "Sensitivity — max relative change from baseline",
                 fmt=".0%")
    legend_sens = [
        Patch(facecolor="#B5D4F4", label="≥ 30%  high"),
        Patch(facecolor="#E6F1FB", label="10–30%  moderate"),
        Patch(facecolor="#F1EFE8", label="< 10%  low"),
    ]
    ax2.legend(handles=legend_sens, loc="lower right",
               fontsize=8, frameon=False, ncol=3)

    # ── 3. Robustness CV bars ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    cv_vals   = [robust[m]["cv"] * 100 for m in METRICS]
    bar_colors = [cv_color(robust[m]["cv"]) for m in METRICS]
    bar_ec     = [cv_textcolor(robust[m]["cv"]) for m in METRICS]
    bars = ax3.bar(METRIC_LABELS, cv_vals, color=bar_colors,
                   edgecolor=bar_ec, linewidth=0.8, width=0.6)
    ax3.axhline(5,  color="#3B6D11", linestyle="--", linewidth=0.8, alpha=0.7, label="5% pass threshold")
    ax3.axhline(15, color="#A32D2D", linestyle="--", linewidth=0.8, alpha=0.7, label="15% fail threshold")
    for bar, val in zip(bars, cv_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                 color="#1a1a18")
    ax3.set_title("Robustness — CV across k-means K variants", fontsize=11, fontweight="normal")
    ax3.set_ylabel("CV (%)", fontsize=9)
    ax3.set_ylim(0, max(cv_vals) * 1.25)
    ax3.tick_params(axis="x", labelsize=9)
    ax3.tick_params(axis="y", labelsize=9)
    ax3.legend(fontsize=8, frameon=False)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    ax3.spines["left"].set_color("#d0cfc7")
    ax3.spines["bottom"].set_color("#d0cfc7")

    # ── 4. Blur FID curve (non-monotone case) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    sevs     = [1, 3, 5]
    fid_blur = [exps[n]["fid"] for _, n in SEVERITY_SERIES["Gaussian blur"]]
    fid_noise= [exps[n]["fid"] for _, n in SEVERITY_SERIES["Gaussian noise"]]
    ax4.plot(sevs, fid_blur,  "o-", color="#E24B4A", linewidth=1.8,
             markersize=6, label="Blur FID (non-monotone)")
    ax4.plot(sevs, fid_noise, "s--", color="#378ADD", linewidth=1.5,
             markersize=5, label="Noise FID (monotone)")
    ax4.axhline(BASELINE["fid"], color="#888780", linestyle=":",
                linewidth=1, label=f"Baseline ({BASELINE['fid']:.0f})")
    ax4.set_xticks(sevs)
    ax4.set_xlabel("Severity level", fontsize=9)
    ax4.set_ylabel("FID", fontsize=9)
    ax4.set_title("FID response curves — blur vs noise", fontsize=11, fontweight="normal")
    ax4.legend(fontsize=8, frameon=False)
    ax4.tick_params(labelsize=9)
    for spine in ["top", "right"]:
        ax4.spines[spine].set_visible(False)
    ax4.spines["left"].set_color("#d0cfc7")
    ax4.spines["bottom"].set_color("#d0cfc7")

    # ── 5. Flags summary ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")
    ax5.set_title("Findings summary", fontsize=11, fontweight="normal",
                  loc="left", pad=8, color="#1a1a18")

    sev_order  = ["fail", "warn", "pass"]
    sev_colors = {"fail": "#A32D2D", "warn": "#854F0B", "pass": "#3B6D11"}
    sev_labels = {"fail": "FAIL", "warn": "WARN", "pass": "PASS"}

    y = 0.98
    line_h = 0.115
    for sev in sev_order:
        group = [f for f in flags if f["severity"] == sev]
        for flag in group:
            if y < 0:
                break
            color = sev_colors[sev]
            label = sev_labels[sev]
            ax5.text(0.0, y, f"[{label}]", transform=ax5.transAxes,
                     fontsize=8, color=color, fontweight="normal",
                     va="top", ha="left")
            ax5.text(0.065, y, flag["title"], transform=ax5.transAxes,
                     fontsize=8.5, color="#1a1a18", fontweight="normal",
                     va="top", ha="left")
            ax5.text(0.065, y - 0.055, flag["detail"][:120] + ("…" if len(flag["detail"]) > 120 else ""),
                     transform=ax5.transAxes,
                     fontsize=7.5, color="#5f5e5a", va="top", ha="left",
                     style="italic")
            y -= line_h

    return fig


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mono    = compute_monotonicity()
    sens    = compute_sensitivity()
    robust  = compute_robustness()
    flags   = flag_findings(mono, sens, robust)

    # Print summary to console
    print("=" * 60)
    print("MONOTONICITY (Spearman rho)")
    print("=" * 60)
    for perturb, rhos in mono.items():
        print(f"\n  {perturb}")
        for m, rho in rhos.items():
            status = "PASS" if abs(rho) >= 0.95 else ("WARN" if abs(rho) >= 0.70 else "FAIL")
            print(f"    {m:12s}  rho={rho:+.2f}  [{status}]")

    print("\n" + "=" * 60)
    print("SENSITIVITY (max relative change from baseline)")
    print("=" * 60)
    for perturb, mc in sens.items():
        print(f"\n  {perturb}")
        for m, v in mc.items():
            level = "HIGH" if v >= 0.30 else ("MED" if v >= 0.10 else "LOW")
            print(f"    {m:12s}  {v*100:5.1f}%  [{level}]")

    print("\n" + "=" * 60)
    print("ROBUSTNESS (CV across k-means K variants)")
    print("=" * 60)
    for m, d in robust.items():
        status = "PASS" if d["cv"] <= 0.05 else ("WARN" if d["cv"] <= 0.15 else "FAIL")
        print(f"  {m:12s}  CV={d['cv']*100:.1f}%  mean={d['mean']:.3f}  std={d['std']:.3f}  [{status}]")

    print("\n" + "=" * 60)
    print(f"FLAGS  ({sum(1 for f in flags if f['severity']=='fail')} fail / "
          f"{sum(1 for f in flags if f['severity']=='warn')} warn / "
          f"{sum(1 for f in flags if f['severity']=='pass')} pass)")
    print("=" * 60)
    for sev in ["fail", "warn", "pass"]:
        for f in flags:
            if f["severity"] == sev:
                print(f"  [{sev.upper():4s}]  {f['title']}")

    # Save figure
    out = Path("metric_scorecard.png")
    fig = make_figure(mono, sens, robust, flags)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {out}")
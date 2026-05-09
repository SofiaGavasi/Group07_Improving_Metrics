
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    # helper for display
    def display(x):
        print(x)


METRICS_COLS = ['fid', 'kid_mean', 'is_mean', 'precision', 'recall', 'density', 'coverage']


# helper for summarize by family
def summarize_by_family(df: pd.DataFrame, batch_name: str):
    # 2) High-level overview by perturbation family.
    metrics_cols = ['fid', 'kid_mean', 'is_mean', 'precision', 'recall', 'density', 'coverage']
    
    family_summary = (
        df.groupby('perturbation_family', dropna=False)
          .agg(
              experiments=('name', 'nunique'),
              rows=('name', 'size'),
              successful_rows=('status', lambda s: int((s == 'completed').sum())),
              with_metrics=('has_metrics_report', lambda s: int(s.sum())),
              fid_mean=('fid', 'mean'),
              is_mean=('is_mean', 'mean'),
              precision_mean=('precision', 'mean'),
              recall_mean=('recall', 'mean'),
          )
          .sort_values('experiments', ascending=False)
    )
    
    display(family_summary)
    
    plt.figure(figsize=(10, 4))
    family_counts = df['perturbation_family'].value_counts()
    plt.bar(family_counts.index.astype(str), family_counts.values)
    plt.title(f'Experiment Rows per Perturbation Family ({batch_name})')
    plt.ylabel('Rows')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    return family_summary


# compute baseline deltas
def compute_baseline_deltas(df: pd.DataFrame, metrics_cols: list[str] | None = None):
    metrics_cols = metrics_cols or METRICS_COLS
    # 3) Baseline deltas (per model+dataset): how each experiment moves metrics vs baseline.
    # Baseline detection: either experiment name contains 'baseline' or no active perturbations.
    baseline_mask = (
        df['name'].astype(str).str.contains('baseline', case=False, na=False)
        | (df['perturbation_family'] == 'baseline')
    )
    
    baseline = (
        df[baseline_mask]
        .groupby(['model', 'dataset'], dropna=False)[metrics_cols]
        .mean()
        .rename(columns={c: f'{c}_baseline' for c in metrics_cols})
        .reset_index()
    )
    
    df_delta = df.merge(baseline, on=['model', 'dataset'], how='left')
    for c in metrics_cols:
        df_delta[f'{c}_delta'] = df_delta[c] - df_delta[f'{c}_baseline']
    
    delta_cols = [f'{c}_delta' for c in metrics_cols]
    delta_summary = (
        df_delta.groupby('perturbation_family', dropna=False)[delta_cols]
        .mean()
        .sort_index()
    )
    
    display(delta_summary)

    return df_delta, delta_summary


# helper for show top worst
def show_top_worst(df_delta: pd.DataFrame):
    # 5) Top/worst experiments by key metrics.
    def topn(frame, col, n=10, ascending=True):
        subset = frame.dropna(subset=[col]).sort_values(col, ascending=ascending)
        return subset[['name', 'perturbation_family', 'degradation_variant', 'model', 'dataset', col]].head(n)
    
    print('Best FID (lowest):')
    display(topn(df_delta, 'fid', n=10, ascending=True))
    
    print('Worst FID (highest):')
    display(topn(df_delta, 'fid', n=10, ascending=False))
    
    print('Best Recall:')
    display(topn(df_delta, 'recall', n=10, ascending=False))
    
    print('Largest FID increase vs baseline (delta):')
    display(topn(df_delta, 'fid_delta', n=10, ascending=False))



# plot metric bars
def plot_metric_bars(df_delta: pd.DataFrame):
    # 6) Color-coded bar charts per metric (with CI when available).
    # Bars are experiments; color indicates perturbation family.
    
    bar_specs = [
        ('fid', 'fid_ci_low', 'fid_ci_high', 'FID (lower better)', True),
        ('kid_mean', 'kid_ci_low', 'kid_ci_high', 'KID mean (lower better)', True),
        ('is_mean', 'is_ci_low', 'is_ci_high', 'IS mean (higher better)', False),
        ('precision', 'precision_ci_low', 'precision_ci_high', 'Precision (higher better)', False),
        ('recall', 'recall_ci_low', 'recall_ci_high', 'Recall (higher better)', False),
        ('density', 'density_ci_low', 'density_ci_high', 'Density (higher better)', False),
        ('coverage', 'coverage_ci_low', 'coverage_ci_high', 'Coverage (higher better)', False),
    ]
    
    families = sorted([f for f in df_delta['perturbation_family'].dropna().unique()])
    family_order = ['baseline'] + [f for f in families if f != 'baseline']
    color_map = {fam: plt.cm.tab20(i % 20) for i, fam in enumerate(families)}
    
    MAX_BARS = 60  # keep charts readable
    
    for metric_key, low_key, high_key, title, lower_is_better in bar_specs:
        cols = ['name', 'perturbation_family', metric_key, low_key, high_key]
        tmp = df_delta[cols].dropna(subset=[metric_key]).copy()
        if tmp.empty:
            print(f'Skipping {metric_key}: no valid values.')
            continue
    
        tmp['_fam_order'] = pd.Categorical(tmp['perturbation_family'], categories=family_order, ordered=True)
        tmp = tmp.sort_values(['_fam_order', metric_key], ascending=[True, lower_is_better])
        if len(tmp) > MAX_BARS:
            tmp = tmp.head(MAX_BARS)
    
        values = tmp[metric_key].to_numpy(dtype=float)
        lows = tmp[low_key].to_numpy(dtype=float)
        highs = tmp[high_key].to_numpy(dtype=float)
    
        lower_err = np.where(np.isfinite(lows), np.maximum(0.0, values - lows), np.nan)
        upper_err = np.where(np.isfinite(highs), np.maximum(0.0, highs - values), np.nan)
    
        x = np.arange(len(tmp))
        bar_colors = [color_map.get(fam, 'gray') for fam in tmp['perturbation_family']]
    
        plt.figure(figsize=(14, 7))
        plt.bar(x, values, color=bar_colors, alpha=0.9)
    
        # Draw CI only where both bounds exist.
        has_ci = np.isfinite(lower_err) & np.isfinite(upper_err)
        if has_ci.any():
            plt.errorbar(
                x[has_ci],
                values[has_ci],
                yerr=np.vstack([lower_err[has_ci], upper_err[has_ci]]),
                fmt='none',
                ecolor='black',
                elinewidth=1,
                capsize=2,
                alpha=0.8,
            )
    
        plt.title(f'{title} by experiment, color-coded by perturbation family')
        plt.ylabel(title)
        plt.xlabel('Experiments (sorted)')
        plt.xticks(x, tmp['name'], rotation=45, ha='right', fontsize=7)
    
        # Legend: one entry per perturbation family.
        handles = []
        labels = []
        for fam in families:
            if fam in set(tmp['perturbation_family']):
                handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[fam]))
                labels.append(fam)
        if handles:
            plt.legend(handles, labels, loc='best', fontsize=8, ncols=2)
    
        plt.tight_layout()
        plt.show()


from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from IPython.display import display
except Exception:
    # helper for display
    def display(x):
        print(x)


# run full perturbation analysis
def run_full_perturbation_analysis(df: pd.DataFrame):
    # 7) Full perturbation analysis across families
    # Curves + monotonicity + sensitivity + robustness + specificity
    
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Expected a pandas DataFrame for `df`.")
    
    METRICS = ['fid', 'kid_mean', 'is_mean', 'precision', 'recall', 'density', 'coverage']
    METRIC_LABELS = {
        'fid': 'FID',
        'kid_mean': 'KID',
        'is_mean': 'IS',
        'precision': 'Precision',
        'recall': 'Recall',
        'density': 'Density',
        'coverage': 'Coverage',
    }
    LOWER_BETTER = {'fid', 'kid_mean'}
    
    needed = ['name', 'model', 'dataset', 'perturbation_family'] + METRICS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f'Dataframe missing required columns: {missing}')
    
    # One row per experiment name with valid metrics.
    exps_df = (
        df.dropna(subset=METRICS, how='all')
          .drop_duplicates(subset=['name'], keep='last')
          .copy()
    )
    if exps_df.empty:
        raise RuntimeError('No experiment rows with metrics are available.')
    
    # Baseline per (model,dataset).
    baseline_mask = exps_df['name'].astype(str).str.contains('baseline', case=False, na=False) | (
        exps_df['perturbation_family'].astype(str).str.lower() == 'baseline'
    )
    baseline_df = (
        exps_df[baseline_mask]
        .groupby(['model', 'dataset'], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    if baseline_df.empty:
        raise RuntimeError('Could not identify baseline rows.')
    
    exps_df = exps_df.merge(
        baseline_df.rename(columns={m: f'{m}_baseline' for m in METRICS}),
        on=['model', 'dataset'],
        how='left',
    )
    
    # Direction-aligned normalized change (positive = worse for every metric).
    for m in METRICS:
        base = exps_df[f'{m}_baseline']
        raw_delta = exps_df[m] - base
        rel_delta = np.where(np.isfinite(base) & (base != 0), raw_delta / np.abs(base), np.nan)
        exps_df[f'{m}_norm'] = rel_delta if m in LOWER_BETTER else -rel_delta
    
    
    # parse memo fraction
    def _parse_memo_fraction(name: str) -> float:
        m = re.search(r'memo_frac_(\d+)pct', str(name).lower())
        return float(m.group(1)) / 100.0 if m else np.nan
    
    
    # parse imbalance balance
    def _parse_imbalance_balance(name: str) -> float:
        s = str(name).lower()
        m_dec = re.search(r'_(\d+)p(\d+)$', s)
        if m_dec:
            return float(f"{m_dec.group(1)}.{m_dec.group(2)}")
        m_pct = re.search(r'_(\d+)pct$', s)
        if m_pct:
            return float(m_pct.group(1)) / 100.0
        return np.nan
    
    
    # helper for count target tokens
    def _count_target_tokens(targets: str) -> float:
        tokens = [t for t in re.split(r'[,_]+', str(targets).strip()) if t]
        return float(len(tokens)) if tokens else np.nan


    # parse removal strength
    def _parse_removal_strength(name: str) -> float:
        s = str(name).lower()
        m_label = re.search(r'class_removal_label_(.+)$', s)
        if m_label:
            return _count_target_tokens(m_label.group(1))
        m_kmeans = re.search(r'class_removal_kmeans_k\d+_cluster_(.+)$', s)
        if m_kmeans:
            return _count_target_tokens(m_kmeans.group(1))
        return np.nan


    # parse class imbalance group
    def _parse_class_imbalance_group(name: str) -> str:
        s = str(name).lower()

        m_label = re.search(r'class_imbalance_label_(.+?)_(\d+p\d+|\d+pct)$', s)
        if m_label:
            targets = m_label.group(1).replace('_', ',')
            return f'class_imbalance::label::{targets}'

        m_kmeans = re.search(r'class_imbalance_kmeans_k(\d+)_cluster_(.+?)_(\d+p\d+|\d+pct)$', s)
        if m_kmeans:
            k = m_kmeans.group(1)
            targets = m_kmeans.group(2).replace('_', ',')
            return f'class_imbalance::kmeans_k{k}::{targets}'

        return 'class_imbalance::unknown'


    # parse class removal group
    def _parse_class_removal_group(name: str) -> str:
        s = str(name).lower()
        if re.search(r'class_removal_label_(.+)$', s):
            return 'class_removal::label'
        m_kmeans = re.search(r'class_removal_kmeans_k(\d+)_cluster_(.+)$', s)
        if m_kmeans:
            k = m_kmeans.group(1)
            return f'class_removal::kmeans_k{k}'
        return 'class_removal'
    
    
    # parse preproc scale
    def _parse_preproc_scale(name: str) -> float:
        m = re.search(r'_scale(\d+)p(\d+)$', str(name).lower())
        return float(f"{m.group(1)}.{m.group(2)}") if m else np.nan
    
    
    # parse domain shift dataset
    def _parse_domain_shift_dataset(name: str) -> str:
        m = re.search(r'domain_shift_(.+)$', str(name).lower())
        return m.group(1) if m else 'unknown'
    
    
    # helper for infer group and scale
    def _infer_group_and_scale(row: pd.Series) -> tuple[str, float]:
        fam = str(row.get('perturbation_family', '')).lower()
        name = str(row.get('name', ''))
    
        if fam == 'baseline':
            return ('baseline', np.nan)
    
        if fam.startswith('degradation_'):
            sev = row.get('degradation_severity', np.nan)
            if pd.isna(sev):
                m = re.search(r'sev(\d+)', name.lower())
                sev = float(m.group(1)) if m else np.nan
            return (fam, float(sev) if pd.notna(sev) else np.nan)
    
        if fam.startswith('memo'):
            return ('memoisation', _parse_memo_fraction(name))
    
        if fam.startswith('class_removal'):
            return (_parse_class_removal_group(name), _parse_removal_strength(name))
    
        if fam.startswith('class_imbalance'):
            return (_parse_class_imbalance_group(name), _parse_imbalance_balance(name))
    
        if fam.startswith('sample_size'):
            n = row.get('sample_size_n', np.nan)
            if pd.isna(n):
                m = re.search(r'sample_size_(\d+)$', name.lower())
                n = float(m.group(1)) if m else np.nan
            return ('sample_size', float(n) if pd.notna(n) else np.nan)
    
        if fam.startswith('preprocessing'):
            variant = str(row.get('preprocessing_variant', '') or '').strip()
            if not variant:
                m = re.search(r'preprocessing_([a-z0-9_]+)_scale', name.lower())
                variant = m.group(1) if m else 'unknown'
            scale = row.get('preprocessing_scale', np.nan)
            if pd.isna(scale):
                scale = _parse_preproc_scale(name)
            return (f'preprocessing::{variant}', float(scale) if pd.notna(scale) else np.nan)
    
        if fam.startswith('domain_shift'):
            ds = _parse_domain_shift_dataset(name)
            return (f'domain_shift::{ds}', np.nan)
    
        sev = row.get('severity_from_name', np.nan)
        return (fam, float(sev) if pd.notna(sev) else np.nan)
    
    
    series_rows = exps_df.copy()
    series_rows[['perturbation_group', 'scale']] = series_rows.apply(
        lambda r: pd.Series(_infer_group_and_scale(r)), axis=1
    )
    
    # For curve/mono/sensitivity: use perturbations with defined scales; exclude baseline/domain_shift.
    analysis_df = series_rows[
        ~series_rows['perturbation_group'].isin(['baseline'])
    ].copy()
    
    curve_df = analysis_df[
        (~analysis_df['perturbation_group'].astype(str).str.startswith('domain_shift::'))
        & analysis_df['scale'].notna()
    ].copy()
    if curve_df.empty:
        raise RuntimeError('No perturbation rows with inferred scales found for curve analysis.')
    
    curve_df = curve_df.sort_values(['perturbation_group', 'scale', 'name'])
    curve_agg = (
        curve_df
        .groupby(['perturbation_group', 'scale'], dropna=False)[METRICS + [f'{m}_norm' for m in METRICS]]
        .mean(numeric_only=True)
        .reset_index()
    )
    pert_groups = sorted(curve_agg['perturbation_group'].unique())
    
    
    # ---- Grouped line plots (requested layout)
    def _class_imbalance_disturbed_count(group_label: str) -> float:
        s = str(group_label)
        if not s.startswith('class_imbalance::'):
            return np.nan
        parts = s.split('::')
        if len(parts) < 3:
            return np.nan
        targets = [t for t in parts[-1].split(',') if t]
        return float(len(targets)) if targets else np.nan


    class_imbalance_groups = [g for g in pert_groups if g.startswith('class_imbalance::')]
    class_imbalance_counts = sorted(
        {
            int(cnt)
            for cnt in (_class_imbalance_disturbed_count(g) for g in class_imbalance_groups)
            if pd.notna(cnt)
        }
    )

    PLOT_GROUPS = {
        'Degradation': [g for g in pert_groups if g.startswith('degradation_')],
        'Class Removal': [g for g in pert_groups if g.startswith('class_removal')],
    }

    for disturbed_count in class_imbalance_counts:
        label = 'class' if disturbed_count == 1 else 'classes'
        PLOT_GROUPS[f'Class Imbalance ({disturbed_count} disturbed {label})'] = [
            g for g in class_imbalance_groups
            if _class_imbalance_disturbed_count(g) == disturbed_count
        ]

    class_imbalance_unknown = [
        g for g in class_imbalance_groups
        if pd.isna(_class_imbalance_disturbed_count(g))
    ]
    if class_imbalance_unknown:
        PLOT_GROUPS['Class Imbalance (unknown disturbed classes)'] = class_imbalance_unknown

    PLOT_GROUPS.update({
        'Memoisation': [g for g in pert_groups if g == 'memoisation'],
        'Sample Size': [g for g in pert_groups if g == 'sample_size'],
        'Preprocessing': [g for g in pert_groups if g.startswith('preprocessing::')],
    })
    
    
    # helper for unique legend entries
    def _unique_legend_entries(ax_list):
        seen = set()
        out_h = []
        out_l = []
        for ax in ax_list:
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l in seen:
                    continue
                seen.add(l)
                out_h.append(h)
                out_l.append(l)
        return out_h, out_l
    
    
    # plot group metric panels
    def _plot_group_metric_panels(group_name: str, group_labels: list[str]):
        if not group_labels:
            print(f'No groups to plot for: {group_name}')
            return
    
        n_cols = 4
        n_rows = int(np.ceil(len(METRICS) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False)
        axes_flat = axes.ravel()
    
        for idx, m in enumerate(METRICS):
            ax = axes_flat[idx]
            panel_base = curve_df[curve_df['perturbation_group'].isin(group_labels)][f'{m}_baseline'].mean()
            for g in group_labels:
                sub = curve_df[curve_df['perturbation_group'] == g]
                stats = (
                    sub.groupby('scale', dropna=False)[m]
                    .agg(y_mean='mean', y_std='std', n='count')
                    .reset_index()
                    .sort_values('scale')
                )
                x = stats['scale'].to_numpy(dtype=float)
                y = stats['y_mean'].to_numpy(dtype=float)
                if np.isfinite(x).sum() == 0 or np.isfinite(y).sum() == 0:
                    continue
                ci = np.where(
                    stats['n'].to_numpy(dtype=float) >= 2,
                    1.96 * stats['y_std'].to_numpy(dtype=float) / np.sqrt(stats['n'].to_numpy(dtype=float)),
                    np.nan,
                )
                ax.plot(x, y, marker='o', linewidth=1.4, markersize=3.5, label=g)
                if np.isfinite(ci).any():
                    ax.fill_between(x, y - ci, y + ci, alpha=0.15)
    
            if np.isfinite(panel_base):
                ax.axhline(panel_base, linestyle='--', linewidth=1.0, color='black', alpha=0.8, label='baseline')
            ax.grid(alpha=0.2)
            ax.set_title(METRIC_LABELS[m], fontsize=10)
            ax.set_xlabel('Scale / Severity', fontsize=8)
            ax.set_ylabel('Metric value', fontsize=8)
            ax.tick_params(labelsize=8)
    
        for idx in range(len(METRICS), len(axes_flat)):
            axes_flat[idx].axis('off')
    
        handles, labels = _unique_legend_entries(axes_flat)
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8)
    
        fig.suptitle(f'{group_name}: metric response by perturbation type', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    
    for grp_name, grp_labels in PLOT_GROUPS.items():
        _plot_group_metric_panels(grp_name, grp_labels)
    
    # ---- Domain-shift plot set (dataset-labeled)
    domain_shift_df = analysis_df[
        analysis_df['perturbation_group'].astype(str).str.startswith('domain_shift::')
    ].copy()
    
    if domain_shift_df.empty:
        print('No domain-shift rows found for dedicated plotting.')
    else:
        domain_shift_df['domain_dataset'] = domain_shift_df['perturbation_group'].astype(str).str.replace('domain_shift::', '', regex=False)
        domain_datasets = sorted(domain_shift_df['domain_dataset'].dropna().astype(str).unique())
    
        n_cols = 4
        n_rows = int(np.ceil(len(METRICS) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False)
        axes_flat = axes.ravel()
    
        x = np.arange(len(domain_datasets), dtype=float)
        x_labels = domain_datasets
    
        for idx, m in enumerate(METRICS):
            ax = axes_flat[idx]
            stats = (
                domain_shift_df.groupby('domain_dataset', dropna=False)[m]
                .agg(y_mean='mean', y_std='std', n='count')
                .reset_index()
                .sort_values('domain_dataset')
            )
            stats = stats[stats['domain_dataset'].astype(str).isin(x_labels)].copy()
            y = stats['y_mean'].to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.any():
                ci = np.where(
                    stats['n'].to_numpy(dtype=float) >= 2,
                    1.96 * stats['y_std'].to_numpy(dtype=float) / np.sqrt(stats['n'].to_numpy(dtype=float)),
                    np.nan,
                )
                ax.plot(x[mask], y[mask], marker='o', linewidth=1.4, markersize=4, color='tab:blue', label='domain_shift')
                if np.isfinite(ci).any():
                    ci_mask = mask & np.isfinite(ci)
                    ax.errorbar(
                        x[ci_mask],
                        y[ci_mask],
                        yerr=ci[ci_mask],
                        fmt='none',
                        ecolor='tab:blue',
                        elinewidth=1.0,
                        capsize=3,
                        alpha=0.8,
                    )
                for xi, yi, lab in zip(x[mask], y[mask], np.array(x_labels, dtype=object)[mask]):
                    ax.annotate(lab, (xi, yi), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=7)
    
            panel_base = domain_shift_df[f'{m}_baseline'].mean()
            if np.isfinite(panel_base):
                ax.axhline(panel_base, linestyle='--', linewidth=1.0, color='black', alpha=0.8, label='baseline')
            ax.grid(alpha=0.2)
            ax.set_title(METRIC_LABELS[m], fontsize=10)
            ax.set_xlabel('Domain-shift dataset', fontsize=8)
            ax.set_ylabel('Metric value', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=20, ha='right', fontsize=7)
            ax.tick_params(labelsize=8)
    
        for idx in range(len(METRICS), len(axes_flat)):
            axes_flat[idx].axis('off')
    
        handles, labels = _unique_legend_entries(axes_flat)
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8)
    
        fig.suptitle('Domain Shift: metric response across target datasets', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    
    # ---- 1) Monotonicity: Spearman rho(scale, normalized metric)
    mono_rows = []
    for p in pert_groups:
        sub = curve_agg[curve_agg['perturbation_group'] == p].sort_values('scale')
        x = sub['scale'].to_numpy(dtype=float)
        for m in METRICS:
            y = sub[f'{m}_norm'].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2 and np.unique(x[mask]).size >= 2:
                rho, pval = spearmanr(x[mask], y[mask])
            else:
                rho, pval = np.nan, np.nan
            mono_rows.append({
                'perturbation_group': p,
                'metric': m,
                'rho_spearman': float(rho) if np.isfinite(rho) else np.nan,
                'p_value': float(pval) if np.isfinite(pval) else np.nan,
                'abs_rho': abs(float(rho)) if np.isfinite(rho) else np.nan,
            })
    monotonicity = pd.DataFrame(mono_rows)
    
    # ---- 2) Sensitivity: max |normalized change|
    sens_rows = []
    for p in pert_groups:
        sub = curve_df[curve_df['perturbation_group'] == p]
        for m in METRICS:
            vals = sub[f'{m}_norm'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            sens_rows.append({
                'perturbation_group': p,
                'metric': m,
                'max_abs_norm_change': float(np.max(np.abs(vals))) if vals.size else np.nan,
                'mean_abs_norm_change': float(np.mean(np.abs(vals))) if vals.size else np.nan,
            })
    sensitivity = pd.DataFrame(sens_rows)
    
    # ---- 3) Robustness: nuisance variation span/CV (sample-size + preprocessing + domain shift)
    rob_source = analysis_df[
        analysis_df['perturbation_group'].astype(str).str.startswith('sample_size')
        | analysis_df['perturbation_group'].astype(str).str.startswith('preprocessing::')
        | analysis_df['perturbation_group'].astype(str).str.startswith('domain_shift::')
    ].copy()
    
    rob_rows = []
    if not rob_source.empty:
        rob_source['robustness_group'] = np.where(
            rob_source['perturbation_group'].astype(str).str.startswith('domain_shift::'),
            'domain_shift',
            rob_source['perturbation_group'],
        )
    
        for grp, sub in rob_source.groupby('robustness_group', dropna=False):
            for m in METRICS:
                vals = sub[f'{m}_norm'].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size >= 2:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    cv = abs(std / mean) if mean != 0 else np.nan
                    span = float(np.max(vals) - np.min(vals))
                else:
                    mean, std, cv, span = np.nan, np.nan, np.nan, np.nan
                rob_rows.append({
                    'perturbation_group': grp,
                    'metric': m,
                    'n_points': int(vals.size),
                    'mean_norm': mean,
                    'std_norm': std,
                    'cv_norm': cv,
                    'span_norm': span,
                })
    robustness = pd.DataFrame(rob_rows)
    # Drop robustness rows where statistics cannot be computed.
    # n_points = 1 means std/cv/span are undefined, so they become NaN.
    if not robustness.empty:
        robustness = robustness[robustness["n_points"] >= 2].copy()
    
    # ---- 4) Specificity: off-target drift should stay low
    # PRIMARY_MAP = {
    #     'degradation_noise': {'fid', 'kid_mean'},
    #     'degradation_blur': {'fid', 'kid_mean'},
    #     'degradation_jpeg': {'fid', 'kid_mean'},
    #     'degradation_all': {'fid', 'kid_mean'},
    #     'memoisation': {'fid', 'kid_mean', 'precision', 'density'},
    #     'class_removal': {'recall', 'coverage'},
    #     'class_imbalance': {'recall', 'coverage'},
    #     'sample_size': set(),
    #     'preprocessing': set(),
    # }
    
    # this is the new primary map 
    PRIMARY_MAP = {
        'degradation_noise': {'fid', 'kid_mean','is_mean','precision','density'},
        'degradation_blur': {'fid', 'kid_mean','is_mean','precision','density'},
        'degradation_jpeg': {'fid', 'kid_mean','is_mean','precision','density'},
        'degradation_all': {'fid', 'kid_mean','is_mean','precision','density'},
        'memoisation': {'fid', 'kid_mean', 'precision', 'density','recall','coverage'},
        'class_removal': {'fid','kid_mean','recall', 'coverage'},
        'class_imbalance': {'fid','kid_mean','recall', 'coverage'},
        'sample_size': {'fid','kid_mean','is_mean','precision','recall','density','coverage'},
        'preprocessing': {'fid','kid_mean'},
    }

    
    # helper for base family
    def _base_family(group_label: str) -> str:
        s = str(group_label).lower()
        if s.startswith('degradation_'):
            return s
        if s.startswith('class_imbalance::'):
            return 'class_imbalance'
        if s.startswith('preprocessing::'):
            return 'preprocessing'
        if s.startswith('sample_size'):
            return 'sample_size'
        if s.startswith('class_removal'):
            return 'class_removal'
        if s.startswith('memo'):
            return 'memoisation'
        return s
    
    spec_rows = []
    for p in pert_groups:
        fam = _base_family(p)
        primary = PRIMARY_MAP.get(fam, set())
        sub = curve_df[curve_df['perturbation_group'] == p]
    
        for m in METRICS:
            if m in primary:
                # Primary/target metric for this perturbation: not part of specificity score.
                spec_rows.append({
                    'perturbation_group': p,
                    'metric': m,
                    'off_target_max_abs_norm': 0.0,
                    'off_target_mean_abs_norm': 0.0,
                    'specificity_kind': 'primary_metric',
                })
                continue
    
            vals = sub[f'{m}_norm'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                spec_rows.append({
                    'perturbation_group': p,
                    'metric': m,
                    'off_target_max_abs_norm': float(np.max(np.abs(vals))),
                    'off_target_mean_abs_norm': float(np.mean(np.abs(vals))),
                    'specificity_kind': 'off_target',
                })
            else:
                spec_rows.append({
                    'perturbation_group': p,
                    'metric': m,
                    'off_target_max_abs_norm': np.nan,
                    'off_target_mean_abs_norm': np.nan,
                    'specificity_kind': 'missing_metric_data',
                })
    
    specificity = pd.DataFrame(spec_rows)
    
    # ---- Heatmaps
    mono_piv = monotonicity.pivot(index='perturbation_group', columns='metric', values='rho_spearman').reindex(columns=METRICS)
    fig_m, ax_m = plt.subplots(figsize=(12, max(3, 0.34 * len(mono_piv) + 2)))
    im = ax_m.imshow(mono_piv.fillna(0).to_numpy(), aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax_m.set_title(r'Monotonic Association: Spearman $\rho$ (scale vs normalized metric)')
    ax_m.set_xticks(np.arange(len(METRICS)))
    ax_m.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=30, ha='right')
    ax_m.set_yticks(np.arange(len(mono_piv.index)))
    ax_m.set_yticklabels(mono_piv.index)
    for i in range(mono_piv.shape[0]):
        for j in range(mono_piv.shape[1]):
            v = mono_piv.iloc[i, j]
            ax_m.text(j, i, 'NA' if not np.isfinite(v) else f'{v:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax_m, fraction=0.025, pad=0.02, label='rho')
    plt.tight_layout()
    plt.show()
    
    sens_piv = sensitivity.pivot(index='perturbation_group', columns='metric', values='max_abs_norm_change').reindex(columns=METRICS)
    fig_s, ax_s = plt.subplots(figsize=(12, max(3, 0.34 * len(sens_piv) + 2)))
    im2 = ax_s.imshow(sens_piv.fillna(0).to_numpy(), aspect='auto', cmap='YlOrRd')
    ax_s.set_title('Sensitivity: max |normalized change|')
    ax_s.set_xticks(np.arange(len(METRICS)))
    ax_s.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=30, ha='right')
    ax_s.set_yticks(np.arange(len(sens_piv.index)))
    ax_s.set_yticklabels(sens_piv.index)
    for i in range(sens_piv.shape[0]):
        for j in range(sens_piv.shape[1]):
            v = sens_piv.iloc[i, j]
            ax_s.text(j, i, 'NA' if not np.isfinite(v) else f'{v:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im2, ax=ax_s, fraction=0.025, pad=0.02, label='max |norm change|')
    plt.tight_layout()
    plt.show()
    
    if robustness.empty:
        print('Robustness: no sample-size/preprocessing/domain-shift rows found.')
    else:
        rob_piv = robustness.pivot(index='perturbation_group', columns='metric', values='span_norm').reindex(columns=METRICS)
        fig_r, ax_r = plt.subplots(figsize=(12, max(3, 0.34 * len(rob_piv) + 2)))
        im3 = ax_r.imshow(rob_piv.fillna(0).to_numpy(), aspect='auto', cmap='PuBuGn')
        ax_r.set_title('Robustness: span of normalized response (sample-size, preprocessing, domain-shift)')
        ax_r.set_xticks(np.arange(len(METRICS)))
        ax_r.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=30, ha='right')
        ax_r.set_yticks(np.arange(len(rob_piv.index)))
        ax_r.set_yticklabels(rob_piv.index)
        for i in range(rob_piv.shape[0]):
            for j in range(rob_piv.shape[1]):
                v = rob_piv.iloc[i, j]
                ax_r.text(j, i, 'NA' if not np.isfinite(v) else f'{v:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im3, ax=ax_r, fraction=0.025, pad=0.02, label='max-min norm change')
        plt.tight_layout()
        plt.show()
    
    spec_piv = specificity.pivot(index='perturbation_group', columns='metric', values='off_target_max_abs_norm').reindex(columns=METRICS)
    fig_sp, ax_sp = plt.subplots(figsize=(12, max(3, 0.34 * len(spec_piv) + 2)))
    im4 = ax_sp.imshow(spec_piv.fillna(0).to_numpy(), aspect='auto', cmap='Blues')
    ax_sp.set_title('Specificity: off-target max |normalized change| (lower is better)')
    ax_sp.set_xticks(np.arange(len(METRICS)))
    ax_sp.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=30, ha='right')
    ax_sp.set_yticks(np.arange(len(spec_piv.index)))
    ax_sp.set_yticklabels(spec_piv.index)
    for i in range(spec_piv.shape[0]):
        for j in range(spec_piv.shape[1]):
            v = spec_piv.iloc[i, j]
            kind = specificity[(specificity['perturbation_group'] == spec_piv.index[i]) & (specificity['metric'] == METRICS[j])]['specificity_kind']
            kind = kind.iloc[0] if len(kind) else ''
            if kind == 'primary_metric':
                txt = 'P'
            else:
                txt = 'NA' if not np.isfinite(v) else f'{v:.2f}'
            ax_sp.text(j, i, txt, ha='center', va='center', fontsize=7)
    plt.colorbar(im4, ax=ax_sp, fraction=0.025, pad=0.02, label='off-target max |norm change|')
    plt.tight_layout()
    plt.show()
    
    print('Perturbation groups included (curve analysis; domain_shift has dedicated plots):')
    print(sorted(pert_groups))
    
    na_diag = (
        specificity.assign(is_na=specificity['off_target_max_abs_norm'].isna())
        .groupby(['specificity_kind', 'is_na'], dropna=False)
        .size()
        .reset_index(name='count')
    )
    print('Specificity NA diagnostics (P = primary metric cell):')
    display(na_diag)
    
    print('Monotonicity (head):')
    display(monotonicity.sort_values(['perturbation_group', 'metric']).head(40))
    
    print('Sensitivity (head):')
    display(sensitivity.sort_values(['perturbation_group', 'metric']).head(40))
    
    print('Robustness (head):')
    display(robustness.sort_values(['perturbation_group', 'metric']).head(40) if not robustness.empty else pd.DataFrame())
    
    print('Specificity (head):')
    display(specificity.sort_values(['perturbation_group', 'metric']).head(40))

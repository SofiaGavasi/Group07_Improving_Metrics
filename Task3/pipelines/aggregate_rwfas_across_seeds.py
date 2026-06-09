"""
this file aggregates already computed rwfas scores across repeated seeds

the purpose here is to sit one step after collect_all_scores
first i ask that pipeline for one rwfas score row per batch
then i regroup those rows by model and dataset so i can summarize how stable the composite score and its components are across seeds

the flow of this file is
first i call collect_all_scores without saving intermediate csv files
then i extract seed labels from the batch names
after that i choose the numeric score and component columns that should be aggregated
for each model and dataset pair i compute means std sem confidence bounds and range information across seeds
at the end i optionally save the aggregated table and show a short preview

the main function is aggregate_rwfas_across_seeds
it returns one row per model and dataset pair with cross seed summaries for the rwfas score and all tracked component columns
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

from Task3.pipelines.collect_all_scores import collect_all_scores


def _seed_label(text):
    match = re.search(r'(seed_\d+)', str(text).lower())
    return match.group(1) if match else ''


def _safe_join(values):
    clean = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        clean.append(text)
    return ','.join(sorted(set(clean)))


def _summary(values, alpha=0.05):
    arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    out = {
        'n': int(arr.size),
        'mean': np.nan,
        'std': np.nan,
        'sem': np.nan,
        'ci_low': np.nan,
        'ci_high': np.nan,
        'min': np.nan,
        'max': np.nan,
    }

    if arr.size == 0:
        return out

    out['mean'] = float(np.mean(arr))
    out['min'] = float(np.min(arr))
    out['max'] = float(np.max(arr))

    if arr.size >= 2:
        out['std'] = float(np.std(arr, ddof=1))
        out['sem'] = float(stats.sem(arr, ddof=1))
        if np.isfinite(out['sem']):
            t_crit = float(stats.t.ppf(1.0 - float(alpha) / 2.0, df=arr.size - 1))
            out['ci_low'] = float(out['mean'] - t_crit * out['sem'])
            out['ci_high'] = float(out['mean'] + t_crit * out['sem'])

    return out


def _aggregate_numeric(frame, numeric_cols, alpha=0.05):
    row = {}
    for column in numeric_cols:
        stats_row = _summary(frame[column], alpha=alpha)
        row[f'{column}_n'] = stats_row['n']
        row[f'{column}_mean'] = stats_row['mean']
        row[f'{column}_std'] = stats_row['std']
        row[f'{column}_sem'] = stats_row['sem']
        row[f'{column}_ci_low'] = stats_row['ci_low']
        row[f'{column}_ci_high'] = stats_row['ci_high']
        row[f'{column}_min'] = stats_row['min']
        row[f'{column}_max'] = stats_row['max']
    return row


def aggregate_rwfas_across_seeds(outputs_root='outputs/batch_runs', save_csv=True, csv_path='outputs/task3_rwfas_across_seeds.csv', verbose=True):
    scores = collect_all_scores(
        outputs_root=outputs_root,
        save_csv=False,
        verbose=verbose,
    )
    if scores.empty:
        return pd.DataFrame()

    frame = scores.copy()
    frame['seed_label'] = frame['batch_label'].map(_seed_label)

    numeric_cols = ['rwfas_score']
    numeric_cols.extend(
        column for column in frame.columns
        if column.startswith('score_')
        or column.startswith('norm_')
        or column.startswith('weight_')
        or column.startswith('comp_')
    )

    merged_rows = []
    grouped = frame.groupby(['model', 'dataset'], dropna=False, sort=True)
    for (model, dataset), group in grouped:
        row = {
            'model': model,
            'dataset': dataset,
            'row_count': int(len(group)),
            'seed_count': int(group['seed_label'].replace('', np.nan).dropna().nunique()),
            'batch_count': int(group['batch_label'].dropna().astype(str).nunique()),
            'seed_labels': _safe_join(group['seed_label']),
            'batch_labels': _safe_join(group['batch_label']),
        }
        row.update(_aggregate_numeric(group, numeric_cols=numeric_cols, alpha=0.05))

        ci_half = (pd.to_numeric(group['rwfas_ci_high'], errors='coerce') - pd.to_numeric(group['rwfas_ci_low'], errors='coerce')) / 2.0
        ci_half_stats = _summary(ci_half, alpha=0.05)
        row['rwfas_reported_ci_half_mean'] = ci_half_stats['mean']
        row['rwfas_reported_ci_half_std'] = ci_half_stats['std']
        merged_rows.append(row)

    merged = pd.DataFrame(merged_rows)
    if not merged.empty:
        merged = merged.sort_values(['model', 'dataset'], na_position='last').reset_index(drop=True)

    if save_csv:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(csv_path, index=False)
        if verbose:
            print(f'Saved RWFAS seed aggregation to {csv_path}')

    preview_cols = [
        'model',
        'dataset',
        'seed_count',
        'rwfas_score_mean',
        'rwfas_score_ci_low',
        'rwfas_score_ci_high',
        'rwfas_reported_ci_half_mean',
    ]
    display(merged[preview_cols].head(20) if not merged.empty else pd.DataFrame())
    return merged


if __name__ == '__main__':
    aggregate_rwfas_across_seeds()
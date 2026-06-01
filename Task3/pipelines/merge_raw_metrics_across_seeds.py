"""
this file merges raw metric values across repeated seeds before rwfas scoring

the goal here is different from the score pipeline:
instead of building one composite score i keep the original metric columns and pool repeated runs of the same experiment definition across seeds

the flow of this file is:
- first i discover all report files and load each one with the task3 parser
- then i attach batch and seed labels so i can trace where every row came from
- after that i keep only rows with metric reports and group together rows that
- describe the same model dataset experiment name and perturbation settings
- for each group i compute summary statistics like mean std sem confidence bounds and min max for every raw metric
- at the end i return one merged dataframe and optionally save both the merged table and the per seed table

the main functions are:
_load_per_seed_rows which builds the combined per seed dataframe
merge_raw_metrics_across_seeds which groups equivalent experiment rows and
computes the cross seed summaries
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

from Task3.data.parsing import load_batch_dataframe


METRICS = ['fid', 'kid_mean', 'is_mean', 'precision', 'recall', 'density', 'coverage']

GROUP_COLS = [
    'model',
    'dataset',
    'name',
    'perturbation_family',
    'active_perturbations',
    'apply_to',
    'degradation_variant',
    'degradation_severity',
    'severity_from_name',
    'sample_size_n',
    'preprocessing_variant',
    'preprocessing_scale',
    'domain_shift_dataset',
]


def _report_group_key(path):
    name = path.name
    if name.endswith('_completed_only.json'):
        return name.replace('_completed_only.json', '.json')
    return name


def _discover_report_files(outputs_root='outputs'):
    outputs_root = Path(outputs_root)
    reports_root = outputs_root.parent if outputs_root.name == 'batch_runs' else outputs_root

    candidates = []
    if reports_root.exists():
        candidates.extend(sorted(reports_root.glob('*perturbation_tests*.json')))
    if outputs_root.exists():
        candidates.extend(sorted(outputs_root.rglob('*perturbation_tests*.json')))

    best_by_group = {}
    for path in candidates:
        group_key = _report_group_key(path)
        current = best_by_group.get(group_key)
        if current is None:
            best_by_group[group_key] = path
            continue
        if 'completed_only' in current.name and 'completed_only' not in path.name:
            best_by_group[group_key] = path

    return [best_by_group[key] for key in sorted(best_by_group)]


def _batch_label_from_report(report_path):
    stem = report_path.stem
    return stem.replace('_perturbation_tests', '').replace('_completed_only', '')


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


def _load_per_seed_rows(outputs_root='outputs', verbose=True):
    report_files = _discover_report_files(outputs_root=outputs_root)
    rows = []

    if verbose:
        print(f'Found {len(report_files)} raw report file(s).')

    for report_path in report_files:
        try:
            df = load_batch_dataframe([report_path], verbose=verbose)
        except Exception as exc:
            print(f'WARNING: Failed to parse {report_path.name}: {exc}')
            continue

        if df.empty:
            continue

        batch_label = _batch_label_from_report(report_path)
        seed_label = _seed_label(batch_label)
        frame = df.copy()
        frame['report_name'] = report_path.name
        frame['report_path'] = str(report_path)
        frame['batch_label'] = batch_label
        frame['seed_label'] = seed_label
        rows.append(frame)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    if 'has_metrics_report' in combined.columns:
        combined = combined[combined['has_metrics_report'].fillna(False).astype(bool)].copy()
    return combined


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


def merge_raw_metrics_across_seeds(outputs_root='outputs', save_csv=True, csv_path='outputs/task3_raw_metrics_across_seeds.csv', verbose=True):
    per_seed = _load_per_seed_rows(outputs_root=outputs_root, verbose=verbose)
    if per_seed.empty:
        return pd.DataFrame()

    numeric_cols = [metric for metric in METRICS if metric in per_seed.columns]
    merged_rows = []

    grouped = per_seed.groupby(GROUP_COLS, dropna=False, sort=True)
    for group_key, group in grouped:
        row = {}
        for column, value in zip(GROUP_COLS, group_key):
            row[column] = value

        row['row_count'] = int(len(group))
        row['seed_count'] = int(group['seed_label'].replace('', np.nan).dropna().nunique())
        row['batch_count'] = int(group['batch_label'].dropna().astype(str).nunique())
        row['seed_labels'] = _safe_join(group['seed_label'])
        row['batch_labels'] = _safe_join(group['batch_label'])
        row['report_names'] = _safe_join(group['report_name'])
        row.update(_aggregate_numeric(group, numeric_cols=numeric_cols, alpha=0.05))
        merged_rows.append(row)

    merged = pd.DataFrame(merged_rows)
    if not merged.empty:
        merged = merged.sort_values(['model', 'dataset', 'name'], na_position='last').reset_index(drop=True)

    if save_csv:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(csv_path, index=False)
        per_seed_path = csv_path.with_name(f'{csv_path.stem}_per_seed.csv')
        per_seed.to_csv(per_seed_path, index=False)
        if verbose:
            print(f'Saved merged raw metrics to {csv_path}')
            print(f'Saved per-seed raw metrics to {per_seed_path}')

    preview_cols = ['model', 'dataset', 'name', 'seed_count']
    preview_cols.extend([f'{metric}_mean' for metric in numeric_cols[:3]])
    display(merged[preview_cols].head(20) if not merged.empty else pd.DataFrame())
    return merged


if __name__ == '__main__':
    merge_raw_metrics_across_seeds()

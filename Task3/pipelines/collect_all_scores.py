"""
this file runs the main task3 scoring pipeline across many batch report files

the main goal here is to discover all perturbation test reports load them one by 
one fit rwfas weights inside each batch context and collect the final scores in one table

the flow of this file is:
- first i find report files and choose the best version when both full and completed_only reports exist
- then i load each report into a dataframe and build baseline deltas and normalized metric columns
- after that i split by model and dataset pair so each batch keeps its own baseline and does not mix unrelated experiments
- for each pair i run the full perturbation analysis to get monotonicity sensitivity reliability and specificity tables
- then i fit rwfas on those tables and score the pair at the end i combine all score rows save optional csv outputs and print a short summary

the main functions are
prepare_full_analysis_input which removes baseline columns that could collide with the analysis step
collect_all_scores which drives the full end to end pipeline and returns the final scores dataframe
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

from Task3.scoring.rwfas import METRICS, RWFAS, ensure_norm_columns
import Task3.analysis.full_analysis as full_analysis_module
from Task3.data.parsing import load_batch_dataframe
from Task3.analysis.summaries import compute_baseline_deltas


def _report_group_key(path):
    name = path.name
    if name.endswith('_completed_only.json'):
        return name.replace('_completed_only.json', '.json')
    return name


def _discover_report_files(outputs_root):
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


def prepare_full_analysis_input(df):
    baseline_cols = [f'{metric}_baseline' for metric in METRICS]
    return df.drop(columns=baseline_cols, errors='ignore').copy()


def _run_full_analysis_silently(full_analysis_module, df):
    return full_analysis_module.run_full_perturbation_analysis(
        df,
        show_plots=False,
        show_tables=False,
        verbose=False,
    )


def collect_all_scores(outputs_root='outputs/batch_runs', save_csv=True, csv_path='outputs/task3_all_scores.csv', verbose=True):
    outputs_root = Path(outputs_root)
    report_files = _discover_report_files(outputs_root)

    if verbose:
        print(f'Found {len(report_files)} batch report file(s).')

    if not report_files:
        return pd.DataFrame()

    all_batch_dfs = []
    batch_overview_rows = []

    for report_path in report_files:
        batch_label = _batch_label_from_report(report_path)

        try:
            df = load_batch_dataframe([report_path], verbose=verbose)
            if df.empty:
                print(f'WARNING: Skipping {batch_label}. Loaded dataframe is empty.')
                continue

            has_metrics = df['has_metrics_report'].fillna(False).astype(bool) if 'has_metrics_report' in df.columns else pd.Series(False, index=df.index)
            if not has_metrics.any():
                print(f'WARNING: Skipping {batch_label}. No rows with has_metrics_report == True.')
                continue

            df_delta, _ = compute_baseline_deltas(df, show_table=verbose)
            df_delta = ensure_norm_columns(df_delta)

            norm_cols = [f'{metric}_norm' for metric in METRICS if f'{metric}_norm' in df_delta.columns]
            if not norm_cols:
                print(f'WARNING: Skipping {batch_label}. No normalized metric columns were created.')
                continue

            norm_values = df_delta[norm_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            if not pd.notna(norm_values).any():
                print(f'WARNING: Skipping {batch_label}. Baseline normalization produced only NaN values.')
                continue

            unique_pairs = df_delta[['model', 'dataset']].drop_duplicates()
            if len(unique_pairs) > 1:
                if verbose:
                    print(
                        f'WARNING: {batch_label} contains multiple (model, dataset) pairs. '
                        'Processing each pair independently instead of mixing them.'
                    )

            for pair in unique_pairs.itertuples(index=False):
                pair_model = pair.model
                pair_dataset = pair.dataset
                pair_df = df_delta[
                    (df_delta['model'] == pair_model)
                    & (df_delta['dataset'] == pair_dataset)
                ].copy()

                if pair_df.empty:
                    continue

                pair_batch_label = batch_label
                if len(unique_pairs) > 1:
                    pair_batch_label = f'{batch_label}__{pair_model}__{pair_dataset}'

                analysis_input = prepare_full_analysis_input(pair_df)
                tables = _run_full_analysis_silently(full_analysis_module, analysis_input)
                monotonicity = tables.get('monotonicity', pd.DataFrame())
                sensitivity = tables.get('sensitivity', pd.DataFrame())
                robustness = tables.get('robustness', pd.DataFrame())
                specificity = tables.get('specificity', pd.DataFrame())

                if monotonicity.empty or sensitivity.empty or robustness.empty or specificity.empty:
                    print(f'WARNING: Skipping {pair_batch_label}. At least one analysis table is empty.')
                    continue

                rwfas = RWFAS()
                rwfas.fit(
                    pair_df,
                    monotonicity,
                    sensitivity,
                    specificity,
                    robustness,
                    batch_label=pair_batch_label,
                )

                dataset_filter = pair_dataset if pd.notna(pair_dataset) else None
                scores_df = rwfas.score_all(pair_df, dataset_id=dataset_filter)
                scores_df['batch_label'] = pair_batch_label

                for metric in METRICS:
                    comps = rwfas.components_.get(metric, {})
                    scores_df[f'comp_sensitivity_{metric}'] = comps.get('sensitivity')
                    scores_df[f'comp_specificity_{metric}'] = comps.get('specificity')
                    scores_df[f'comp_monotonicity_{metric}'] = comps.get('monotonicity')
                    scores_df[f'comp_reliability_{metric}'] = comps.get('reliability')
                    scores_df[f'weight_{metric}'] = comps.get('final_weight')

                all_batch_dfs.append(scores_df)
                batch_overview_rows.append({
                    'batch_label': pair_batch_label,
                    'report_file': str(report_path),
                    'model': pair_model,
                    'dataset': pair_dataset,
                    'rows_loaded': int(len(pair_df)),
                    'rows_with_metrics': int(pair_df['has_metrics_report'].fillna(False).astype(bool).sum()),
                    'score_rows': int(len(scores_df)),
                })
        except Exception as exc:
            print(f'WARNING: Failed to process {batch_label}: {exc}')
            continue

    if not all_batch_dfs:
        return pd.DataFrame()

    all_scores = pd.concat(all_batch_dfs, ignore_index=True)

    if save_csv:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        all_scores.to_csv(csv_path, index=False)
        if verbose:
            print(f'Saved master scores to {csv_path}')

        overview_df = pd.DataFrame(batch_overview_rows)
        overview_path = csv_path.with_name(f'{csv_path.stem}_overview.csv')
        overview_df.to_csv(overview_path, index=False)
        if verbose:
            print(f'Saved batch overview to {overview_path}')

    summary_cols = ['batch_label', 'model', 'dataset', 'rwfas_score', 'rwfas_ci_low', 'rwfas_ci_high']
    print(all_scores[summary_cols].to_string(index=False))
    return all_scores


if __name__ == '__main__':
    collect_all_scores()

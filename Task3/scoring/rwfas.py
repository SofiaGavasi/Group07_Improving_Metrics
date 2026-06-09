"""
this file builds the rwfas scoring logic used by task3


the main job here is to take the parsed task3 tables and turn them into one composite score per model and dataset pair

the flow of this file is:
- first i define the metric list and a few small helpers for safe numeric handling
- then i define how normalized metric columns are rebuilt from baseline rows
- after that the rwfas class fits metric weights from four ideas
- sensitivity tells me whether a metric reacts strongly when the target perturbation changes
- specificity tells me whether a metric stays quiet when other perturbations are the main focus
- monotonicity tells me whether the metric moves in a clean direction as the perturbation gets stronger
- reliability is estimated here from confidence interval width so noisier metrics get less trust

the main function-like steps are:
- ensure_norm_columns rebuilds baseline and normalized columns when they are missing or incomplete
- RWFAS.fit reads the analysis tables and turns them into final metric weights
- RWFAS.score uses those weights to score one model and dataset pair
- RWFAS.score_all applies the same scoring logic to every valid pair in the input

this file does not read reports directly
it expects dataframes that were already prepared by the task3 parsing and analysis modules
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)


METRICS = ['fid', 'kid_mean', 'is_mean', 'precision', 'recall', 'density', 'coverage']

TARGET_PREFIXES = {
    'fid':       ['degradation_', 'memoisation'],
    'kid_mean':  ['degradation_', 'memoisation'],
    'is_mean':   ['class_removal', 'class_imbalance'],
    'precision': ['degradation_', 'memoisation'],
    'density':   ['degradation_', 'memoisation'],
    'recall':    ['class_removal', 'class_imbalance', 'memoisation'],
    'coverage':  ['class_removal', 'class_imbalance', 'memoisation'],
}

LOWER_BETTER = {'fid', 'kid_mean'}


def _starts_with_any(value, prefixes):
    text = str(value)
    return any(text.startswith(prefix) for prefix in prefixes)


def _to_numeric(series):
    return pd.to_numeric(series, errors='coerce')


def _mean_finite(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _minmax_normalize(score_map, equal_value=0.5):
    normalized = {metric: np.nan for metric in score_map}
    values = np.array([score_map[metric] for metric in score_map], dtype=float)
    finite = np.isfinite(values)

    if not finite.any():
        return normalized

    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))

    if high == low:
        for metric, value in score_map.items():
            if np.isfinite(value):
                normalized[metric] = float(equal_value)
        return normalized

    span = high - low
    for metric, value in score_map.items():
        if np.isfinite(value):
            normalized[metric] = float((value - low) / span)

    return normalized


def _mean_ci_width(df, metric):
    low_col = f'{metric}_ci_low'
    high_col = f'{metric}_ci_high'

    if low_col not in df.columns or high_col not in df.columns:
        return np.nan

    low_vals = _to_numeric(df[low_col]).to_numpy(dtype=float)
    high_vals = _to_numeric(df[high_col]).to_numpy(dtype=float)
    widths = high_vals - low_vals
    widths = widths[np.isfinite(widths)]
    if widths.size == 0:
        return np.nan
    return float(np.mean(widths))


def _baseline_mask(df):
    # Prefer rows whose experiment name explicitly contains "baseline".
    # Some failed perturbation rows currently arrive with missing configs and
    # get parsed as perturbation_family == "baseline", so we exclude those here.
    name_mask = df['name'].astype(str).str.contains('baseline', case=False, na=False)
    family_mask = df['perturbation_family'].astype(str).str.lower().eq('baseline')
    metric_mask = df[METRICS].apply(pd.to_numeric, errors='coerce').notna().any(axis=1)

    if name_mask.any():
        return name_mask & metric_mask

    return family_mask & metric_mask


def ensure_norm_columns(df_delta):
    if not isinstance(df_delta, pd.DataFrame):
        raise RuntimeError('Expected a pandas DataFrame for `df_delta`.')

    frame = df_delta.copy()
    required = ['name', 'model', 'dataset', 'perturbation_family'] + METRICS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f'Dataframe missing required columns: {missing}')

    baseline_mask = _baseline_mask(frame)

    if not baseline_mask.any():
        return frame

    baseline = (
        frame[baseline_mask]
        .groupby(['model', 'dataset'], dropna=False)[METRICS]
        .mean()
        .rename(columns={metric: f'{metric}_baseline' for metric in METRICS})
        .reset_index()
    )

    baseline_cols = [f'{metric}_baseline' for metric in METRICS]
    for column in baseline_cols:
        if column not in frame.columns:
            frame[column] = np.nan

    merged = frame[['model', 'dataset']].merge(
        baseline,
        on=['model', 'dataset'],
        how='left',
    )

    for metric in METRICS:
        base_col = f'{metric}_baseline'
        delta_col = f'{metric}_delta'
        norm_col = f'{metric}_norm'

        merged_baseline = _to_numeric(merged[base_col]).to_numpy(dtype=float)
        existing_baseline = _to_numeric(frame[base_col]).to_numpy(dtype=float)
        frame[base_col] = np.where(
            np.isfinite(existing_baseline),
            existing_baseline,
            merged_baseline,
        )

        metric_vals = _to_numeric(frame[metric]).to_numpy(dtype=float)
        raw_delta = metric_vals - _to_numeric(frame[base_col]).to_numpy(dtype=float)

        if delta_col not in frame.columns:
            frame[delta_col] = raw_delta
        else:
            existing_delta = _to_numeric(frame[delta_col]).to_numpy(dtype=float)
            frame[delta_col] = np.where(np.isfinite(existing_delta), existing_delta, raw_delta)

        if norm_col not in frame.columns:
            base_vals = _to_numeric(frame[base_col]).to_numpy(dtype=float)
            # epsilon=0.1 matches the floor used in raw weight computation
            # and prevents instability for near-zero baselines (e.g. KID)
            EPSILON = 0.1
            rel_delta = np.where(
                np.isfinite(base_vals),
                raw_delta / (np.abs(base_vals) + EPSILON),
                np.nan,
            )
            frame[norm_col] = rel_delta if metric in LOWER_BETTER else -rel_delta

    return frame


class RWFAS:
    def __init__(self):
        self.weights_ = {}
        self.components_ = {}
        self.batch_label_ = 'unknown_batch'
        self._df_delta_ref = None

    def fit(self, df_delta, monotonicity, sensitivity, specificity, robustness, batch_label='unknown_batch'):
        reliability_table = robustness  # robustness is the reliability table

        prepared = ensure_norm_columns(df_delta)

        sensitivity_raw = {}
        for metric in METRICS:
            prefixes = TARGET_PREFIXES.get(metric, [])
            subset = sensitivity[
                (sensitivity['metric'] == metric)
                & sensitivity['perturbation_group'].astype(str).map(lambda value: _starts_with_any(value, prefixes))
            ]
            values = _to_numeric(subset['mean_abs_norm_change']).to_numpy(dtype=float) if not subset.empty else np.array([], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                sensitivity_raw[metric] = np.nan
            else:
                sensitivity_raw[metric] = float(np.clip(np.mean(values), 0.0, np.inf))
        sensitivity_norm = _minmax_normalize(sensitivity_raw, equal_value=0.5)

        specificity_scores = {}
        EPSILON = 0.1
        for metric in METRICS:
            prefixes = TARGET_PREFIXES.get(metric, [])

            # on-target mean abs drift for this metric
            on_target_rows = specificity[
                (specificity['metric'] == metric)
                & (specificity['specificity_kind'] == 'primary_metric')
                & specificity['perturbation_group'].astype(str).map(lambda value: _starts_with_any(value, prefixes))
            ]
            d_on = _mean_finite(_to_numeric(on_target_rows.get('on_target_mean_abs_norm', pd.Series(dtype=float))).to_numpy(dtype=float))

            # off-target mean abs drift for this metric
            off_target_rows = specificity[
                (specificity['metric'] == metric)
                & (specificity['specificity_kind'] == 'off_target')
                & ~specificity['perturbation_group'].astype(str).map(lambda value: _starts_with_any(value, prefixes))
            ]
            d_off = _mean_finite(_to_numeric(off_target_rows['off_target_mean_abs_norm']).to_numpy(dtype=float))

            if np.isfinite(d_on) and np.isfinite(d_off):
                # ratio-based: specificity = 1 - d_off / (d_on + d_off + eps)
                # a metric must respond more on-target than off-target to score well
                specificity_scores[metric] = float(1.0 - d_off / (d_on + d_off + EPSILON))
            elif np.isfinite(d_off):
                # no on-target data: fall back to penalising off-target drift alone
                specificity_scores[metric] = float(1.0 / (1.0 + d_off))
            else:
                # no off-target rows: default to perfect specificity
                specificity_scores[metric] = 1.0

        monotonicity_scores = {}
        for metric in METRICS:
            prefixes = TARGET_PREFIXES.get(metric, [])
            subset = monotonicity[
                (monotonicity['metric'] == metric)
                & monotonicity['perturbation_group'].astype(str).map(lambda value: _starts_with_any(value, prefixes))
            ]
            # use clipped_rho (signed, clipped to [0,1]) if available,
            # otherwise fall back to abs_rho for backward compatibility
            if 'clipped_rho' in subset.columns:
                value = _mean_finite(_to_numeric(subset['clipped_rho']).to_numpy(dtype=float))
            else:
                value = _mean_finite(_to_numeric(subset['abs_rho']).to_numpy(dtype=float))
            monotonicity_scores[metric] = float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0

        # read relative CI widths directly from the reliability table to ensure
        # consistency with plot_components.py, which also reads from this table.
        # this avoids recomputing baseline normalisation here with a potentially
        # different baseline row set.
        reliability_raw = {}
        reliability_scores = {}
        w_col = 'mean_relative_ci_width' if 'mean_relative_ci_width' in reliability_table.columns else 'mean_ci_width'
        for metric in METRICS:
            row = reliability_table[reliability_table['metric'] == metric]
            if not row.empty:
                val = pd.to_numeric(row[w_col].iloc[0], errors='coerce')
                reliability_raw[metric] = float(val) if np.isfinite(val) else np.nan
            else:
                reliability_raw[metric] = np.nan
        reliability_norm = _minmax_normalize(reliability_raw, equal_value=0.5)
        for metric in METRICS:
            if np.isfinite(reliability_raw.get(metric, np.nan)) and np.isfinite(reliability_norm.get(metric, np.nan)):
                reliability_scores[metric] = float(1.0 - reliability_norm[metric])
            else:
                reliability_scores[metric] = 0.5

        EPSILON = 0.1
        raw_weights = {}
        for metric in METRICS:
            values = [
                sensitivity_norm.get(metric, np.nan),
                specificity_scores.get(metric, np.nan),
                monotonicity_scores.get(metric, np.nan),
                reliability_scores.get(metric, np.nan),
            ]
            if all(np.isfinite(value) for value in values):
                floored = [max(float(v), EPSILON) for v in values]
                raw_weights[metric] = float(np.prod(floored))
            else:
                raw_weights[metric] = np.nan

        norm_cols = [f'{metric}_norm' for metric in METRICS]
        corr_matrix = np.full((len(METRICS), len(METRICS)), np.nan, dtype=float)
        if all(column in prepared.columns for column in norm_cols):
            norm_matrix = prepared[norm_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            if norm_matrix.ndim == 2 and norm_matrix.shape[0] >= 2:
                complete_rows = np.isfinite(norm_matrix).all(axis=1)
                clean_matrix = norm_matrix[complete_rows]
                if clean_matrix.shape[0] >= 2:
                    corr_matrix = np.corrcoef(clean_matrix, rowvar=False)

        redundancy_penalties = {}
        unnormalized_final = {}
        for index, metric in enumerate(METRICS):
            corr_row = corr_matrix[index] if corr_matrix.ndim == 2 else np.array([], dtype=float)
            correlated = []
            for other_index, other_metric in enumerate(METRICS):
                if other_metric == metric or other_index >= len(corr_row):
                    continue
                corr_value = corr_row[other_index]
                if np.isfinite(corr_value) and corr_value > 0.8:
                    correlated.append(float(corr_value))

            penalty = 1.0 - float(np.mean(correlated)) if correlated else 1.0
            redundancy_penalties[metric] = penalty

            raw_value = raw_weights.get(metric, np.nan)
            if np.isfinite(raw_value):
                unnormalized_final[metric] = float(raw_value * penalty)
            else:
                unnormalized_final[metric] = np.nan

        finite_weights = [
            value for value in unnormalized_final.values()
            if np.isfinite(value) and value > 0
        ]
        weight_sum = float(np.sum(finite_weights)) if finite_weights else 0.0

        if weight_sum <= 0.0:
            warnings.warn(
                'All RWFAS weights are zero or NaN. Falling back to uniform weights.',
                RuntimeWarning,
            )
            final_weights = {metric: 1.0 / len(METRICS) for metric in METRICS}
        else:
            final_weights = {}
            for metric in METRICS:
                value = unnormalized_final.get(metric, np.nan)
                if np.isfinite(value) and value > 0:
                    final_weights[metric] = float(value / weight_sum)
                else:
                    final_weights[metric] = 0.0

        self.weights_ = final_weights
        self.components_ = {}
        EPSILON = 0.1
        for metric in METRICS:
            self.components_[metric] = {
                'sensitivity':        max(sensitivity_norm.get(metric, np.nan), EPSILON),
                'specificity':        max(specificity_scores.get(metric, np.nan), EPSILON),
                'monotonicity':       max(monotonicity_scores.get(metric, np.nan), EPSILON),
                'reliability':        max(reliability_scores.get(metric, np.nan), EPSILON),
                'raw_weight':         raw_weights.get(metric, np.nan),
                'redundancy_penalty': redundancy_penalties.get(metric, np.nan),
                'final_weight':       final_weights.get(metric, np.nan),
            }

        self.batch_label_ = str(batch_label)
        self._df_delta_ref = prepared
        return self

    def score(self, df_delta, model_id, dataset_id=None):
        if not self.weights_:
            raise RuntimeError('RWFAS.fit() must be called before score().')

        prepared = ensure_norm_columns(df_delta)
        subset = prepared[prepared['model'].astype(str) == str(model_id)].copy()
        if dataset_id is not None:
            subset = subset[subset['dataset'].astype(str) == str(dataset_id)].copy()

        if subset.empty:
            raise ValueError('No rows found for the requested model and dataset.')

        component_norm = {}
        component_scores = {}
        ci_widths = {}

        MEMOISATION_PREFIX = 'memoisation'

        for metric in METRICS:
            norm_col = f'{metric}_norm'

            if norm_col in subset.columns:
                metric_subset = subset.copy()
                # excluding memoisation rows from goodness score: memoisation arificially improves fidelity metrics, which would reward
                # a degenerate generator. memoisation is interpreted diagnostically via the failure mode chart instead.
                memo_mask = metric_subset['perturbation_group'].astype(str).str.startswith(MEMOISATION_PREFIX)
                metric_subset = metric_subset[~memo_mask]

                # two-level averaging: first average within each perturbation family, then average across families. 
                # this prevents families with more severity levels (e.g. class_imbalance) from dominating.
                if 'perturbation_group' in metric_subset.columns and not metric_subset.empty:
                    family_means = (
                        metric_subset
                        .groupby('perturbation_group', dropna=True)[norm_col]
                        .apply(lambda x: pd.to_numeric(x, errors='coerce').dropna().mean())
                        .dropna()
                    )
                    values = family_means.to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                else:
                    values = _to_numeric(metric_subset[norm_col]).to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
            else:
                values = np.array([], dtype=float)

            if values.size:
                clipped = np.clip(float(np.mean(values)), -1.0, 1.0)
                component_norm[metric] = clipped
                component_scores[metric] = float(0.5 - 0.5 * clipped)
            else:
                component_norm[metric] = np.nan
                component_scores[metric] = np.nan

            ci_widths[metric] = _mean_ci_width(subset, metric)

        weighted_terms = []
        ci_terms = []
        for metric in METRICS:
            weight = self.weights_.get(metric, np.nan)
            goodness = component_scores.get(metric, np.nan)
            ci_width = ci_widths.get(metric, np.nan)

            if np.isfinite(weight) and np.isfinite(goodness):
                weighted_terms.append(weight * goodness)

            if np.isfinite(weight) and np.isfinite(ci_width):
                ci_terms.append(weight * ci_width / 2.0)

        rwfas_score = float(np.sum(weighted_terms)) if weighted_terms else np.nan
        rwfas_ci_half = float(np.sum(ci_terms)) if ci_terms else np.nan
        rwfas_ci_low = rwfas_score - rwfas_ci_half if np.isfinite(rwfas_score) and np.isfinite(rwfas_ci_half) else np.nan
        rwfas_ci_high = rwfas_score + rwfas_ci_half if np.isfinite(rwfas_score) and np.isfinite(rwfas_ci_half) else np.nan

        dataset_value = dataset_id
        if dataset_value is None:
            datasets = subset['dataset'].dropna().astype(str).unique().tolist()
            dataset_value = datasets[0] if len(datasets) == 1 else None

        return {
            'rwfas_score': rwfas_score,
            'rwfas_ci_low': rwfas_ci_low,
            'rwfas_ci_high': rwfas_ci_high,
            'component_scores': component_scores,
            'component_norm': component_norm,
            'weights': dict(self.weights_),
            'batch_label': self.batch_label_,
            'model': str(model_id),
            'dataset': dataset_value,
        }

    def score_all(self, df_delta, dataset_id=None):
        if not self.weights_:
            raise RuntimeError('RWFAS.fit() must be called before score_all().')

        prepared = ensure_norm_columns(df_delta)
        if dataset_id is not None:
            prepared = prepared[prepared['dataset'].astype(str) == str(dataset_id)].copy()

        if prepared.empty:
            raise ValueError('score_all() received no rows after filtering.')

        pairs = prepared[['model', 'dataset']].drop_duplicates()
        if len(pairs) > 1:
            raise ValueError(
                'score_all() received data from multiple (model, dataset) pairs.\n'
            )

        models = prepared['model'].dropna().astype(str).unique().tolist()
        if not models:
            raise ValueError('score_all() could not find any valid model values.')

        

        rows = []
        for model_id in models:
            scored = self.score(prepared, model_id, dataset_id=dataset_id)
            row = {
                'batch_label': scored['batch_label'],
                'model': scored['model'],
                'dataset': scored['dataset'],
                'rwfas_score': scored['rwfas_score'],
                'rwfas_ci_low': scored['rwfas_ci_low'],
                'rwfas_ci_high': scored['rwfas_ci_high'],
            }
            for metric in METRICS:
                row[f'score_{metric}'] = scored['component_scores'].get(metric, np.nan)
                row[f'norm_{metric}'] = scored['component_norm'].get(metric, np.nan)
                row[f'weight_{metric}'] = scored['weights'].get(metric, np.nan)
            rows.append(row)

        scores_df = pd.DataFrame(rows)
        if not scores_df.empty:
            ordered_cols = [
                'batch_label', 'model', 'dataset', 'rwfas_score', 'rwfas_ci_low', 'rwfas_ci_high',
            ]
            ordered_cols.extend([f'score_{metric}' for metric in METRICS])
            ordered_cols.extend([f'norm_{metric}' for metric in METRICS])
            ordered_cols.extend([f'weight_{metric}' for metric in METRICS])
            scores_df = scores_df[ordered_cols].sort_values('rwfas_score', ascending=False, na_position='last').reset_index(drop=True)

        return scores_df
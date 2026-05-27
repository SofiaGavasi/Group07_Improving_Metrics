
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd


# helper for to float or nan
def _to_float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def metric_point(metric_obj, subkey=None):
    obj = metric_obj

    if subkey is not None:
        if not isinstance(obj, dict):
            return np.nan
        obj = obj.get(subkey)

    if obj is None:
        return np.nan

    # Direct scalar, e.g. "fid": 79.43
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return float(obj)

    # List/tuple format, e.g. "kid": [mean, std], "is": [mean, std]
    if isinstance(obj, (list, tuple)):
        return _to_float_or_nan(obj[0]) if len(obj) >= 1 else np.nan

    # Dict format, e.g. {"value": ...} or {"mean": ..., "std": ..., "ci": ...}
    if isinstance(obj, dict):
        if "error" in obj:
            return np.nan
        if "value" in obj:
            return _to_float_or_nan(obj.get("value"))
        if "mean" in obj:
            return _to_float_or_nan(obj.get("mean"))

    return np.nan


def metric_std(metric_obj, subkey=None):
    obj = metric_obj

    if subkey is not None:
        if not isinstance(obj, dict):
            return np.nan
        obj = obj.get(subkey)

    if obj is None:
        return np.nan

    # List/tuple format, e.g. "kid": [mean, std]
    if isinstance(obj, (list, tuple)):
        return _to_float_or_nan(obj[1]) if len(obj) >= 2 else np.nan

    # Dict format, e.g. {"mean": ..., "std": ...}
    if isinstance(obj, dict):
        if "error" in obj:
            return np.nan
        if "std" in obj:
            return _to_float_or_nan(obj.get("std"))

    return np.nan


# helper for metric error
def metric_error(metric_obj):
    if isinstance(metric_obj, dict) and 'error' in metric_obj:
        return str(metric_obj.get('error'))
    return ''


# helper for ci bounds
def ci_bounds(metric_obj, subkey=None):
    obj = metric_obj
    if subkey is not None:
        if not isinstance(obj, dict):
            return (np.nan, np.nan)
        obj = obj.get(subkey)

    if not isinstance(obj, dict):
        return (np.nan, np.nan)

    ci = obj.get('ci')
    if not isinstance(ci, dict):
        return (np.nan, np.nan)

    return (_to_float_or_nan(ci.get('low')), _to_float_or_nan(ci.get('high')))


# helper for degradation variant
def degradation_variant(pert_cfg):
    if not isinstance(pert_cfg, dict):
        return ''
    deg = pert_cfg.get('degradation')
    if not isinstance(deg, dict) or not deg.get('enabled'):
        return ''

    enabled = []
    if deg.get('gaussian_noise'):
        enabled.append('noise')
    if deg.get('gaussian_blur'):
        enabled.append('blur')
    if deg.get('jpeg_compression'):
        enabled.append('jpeg')
    if len(enabled) == 3:
        return 'all'
    return '+'.join(enabled)


# helper for perturbation family
def perturbation_family(pert_cfg):
    if not isinstance(pert_cfg, dict):
        return 'unknown'

    active = [str(x) for x in (pert_cfg.get('active_perturbations') or [])]
    if not active:
        return 'baseline'

    # Split degradation into explicit variants instead of one shared bucket.
    if len(active) == 1 and active[0] == 'degradation':
        variant = degradation_variant(pert_cfg)
        if variant:
            return f'degradation_{variant}'
        return 'degradation_unknown'

    if len(active) == 1:
        return active[0].replace('preprocessing_variation', 'preprocessing')

    return 'combined'


# parse severity from name
def parse_severity_from_name(name):
    if not isinstance(name, str):
        return np.nan
    m = re.search(r'sev(\d+)', name)
    return float(m.group(1)) if m else np.nan



# load batch dataframe
def load_batch_dataframe(report_files: list) -> pd.DataFrame:
    # Flatten batch JSON into one row per test output.
    rows = []
    for report_path in report_files:
        payload = json.loads(report_path.read_text(encoding='utf-8'))
        for exp in payload.get('experiments', []):
            outputs = exp.get('test_outputs') or []
            if not outputs:
                outputs = [{}]
    
            for out in outputs:
                out = out if isinstance(out, dict) else {}
                metrics = out.get('metrics_report') if isinstance(out.get('metrics_report'), dict) else {}
                pert_cfg = out.get('perturbation_config') if isinstance(out.get('perturbation_config'), dict) else {}
    
                fid_obj = metrics.get('fid')
                kid_obj = metrics.get('kid')
                is_obj = metrics.get('is')
                pr_obj = metrics.get('precision_recall')
                dc_obj = metrics.get('density_coverage')
    
                fid_low, fid_high = ci_bounds(fid_obj)
                kid_low, kid_high = ci_bounds(kid_obj)
                is_low, is_high = ci_bounds(is_obj)
                p_low, p_high = ci_bounds(pr_obj, 'precision')
                r_low, r_high = ci_bounds(pr_obj, 'recall')
                d_low, d_high = ci_bounds(dc_obj, 'density')
                c_low, c_high = ci_bounds(dc_obj, 'coverage')
    
                row = {
                    'report_file': report_path.name,
                    'experiment_id': exp.get('experiment_id'),
                    'name': exp.get('name'),
                    'model': exp.get('model_name'),
                    'dataset': exp.get('dataset_name'),
                    'status': exp.get('status'),
                    'exit_code': exp.get('exit_code'),
                    'metrics_expected': bool(exp.get('metrics_expected')),
                    'metrics_available': bool(exp.get('metrics_available')),
                    'step_name': out.get('step_name'),
                    'output_dir': out.get('output_dir'),
                    'metrics_path': out.get('metrics_path'),
                    'has_metrics_report': bool(metrics),
                    'perturbation_family': perturbation_family(pert_cfg),
                    'active_perturbations': ','.join((pert_cfg.get('active_perturbations') or [])),
                    'apply_to': pert_cfg.get('apply_to', ''),
                    'degradation_variant': degradation_variant(pert_cfg),
                    'degradation_severity': _to_float_or_nan((pert_cfg.get('degradation') or {}).get('severity')),
                    'severity_from_name': parse_severity_from_name(exp.get('name')),
                    'sample_size_n': _to_float_or_nan((pert_cfg.get('sample_size') or {}).get('n')),
                    'preprocessing_variant': (pert_cfg.get('preprocessing') or {}).get('variant', ''),
                    'preprocessing_scale': _to_float_or_nan((pert_cfg.get('preprocessing') or {}).get('scale')),
                    'domain_shift_dataset': (pert_cfg.get('domain_shift') or {}).get('dataset', ''),
                    'fid': metric_point(fid_obj),
                    'kid_mean': metric_point(kid_obj),
                    'kid_std': metric_std(kid_obj),
                    'is_mean': metric_point(is_obj),
                    'is_std': metric_std(is_obj),
                    'precision': metric_point(pr_obj, 'precision'),
                    'recall': metric_point(pr_obj, 'recall'),
                    'density': metric_point(dc_obj, 'density'),
                    'coverage': metric_point(dc_obj, 'coverage'),
                    'fid_ci_low': fid_low,
                    'fid_ci_high': fid_high,
                    'kid_ci_low': kid_low,
                    'kid_ci_high': kid_high,
                    'is_ci_low': is_low,
                    'is_ci_high': is_high,
                    'precision_ci_low': p_low,
                    'precision_ci_high': p_high,
                    'recall_ci_low': r_low,
                    'recall_ci_high': r_high,
                    'density_ci_low': d_low,
                    'density_ci_high': d_high,
                    'coverage_ci_low': c_low,
                    'coverage_ci_high': c_high,
                    'fid_error': metric_error(fid_obj),
                    'kid_error': metric_error(kid_obj),
                    'is_error': metric_error(is_obj),
                    'pr_error': metric_error(pr_obj),
                    'dc_error': metric_error(dc_obj),
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    print('Rows loaded:', len(df))
    print('Columns:', len(df.columns))
    df[['name', 'model', 'dataset', 'perturbation_family', 'status', 'metrics_available']].head(8)
    return df

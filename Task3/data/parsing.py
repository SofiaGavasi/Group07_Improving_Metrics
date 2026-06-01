"""
this file reads batch report json files and turns them into a dataframe

it contains small helpers to unpack metric values and one main function that builds one row per test output
"""

import json
import re

import numpy as np
import pandas as pd


def _to_float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _dict_or_empty(value):
    return value if isinstance(value, dict) else {}


def _nested_dict(root, *path):
    current = root
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _nested_value(root, *path, default=None):
    current = root
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def metric_point(metric_obj, subkey=None):
    value = metric_obj

    if subkey is not None:
        if not isinstance(value, dict):
            return np.nan
        value = value.get(subkey)

    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, (list, tuple)):
        return _to_float_or_nan(value[0]) if len(value) >= 1 else np.nan

    if isinstance(value, dict):
        if "error" in value:
            return np.nan
        if "value" in value:
            return _to_float_or_nan(value.get("value"))
        if "mean" in value:
            return _to_float_or_nan(value.get("mean"))

    return np.nan


def metric_std(metric_obj, subkey=None):
    value = metric_obj

    if subkey is not None:
        if not isinstance(value, dict):
            return np.nan
        value = value.get(subkey)

    if value is None:
        return np.nan

    if isinstance(value, (list, tuple)):
        return _to_float_or_nan(value[1]) if len(value) >= 2 else np.nan

    if isinstance(value, dict):
        if "error" in value:
            return np.nan
        if "std" in value:
            return _to_float_or_nan(value.get("std"))

    return np.nan


def metric_error(metric_obj):
    if isinstance(metric_obj, dict) and "error" in metric_obj:
        return str(metric_obj.get("error"))
    return ""


def ci_bounds(metric_obj, subkey=None):
    value = metric_obj

    if subkey is not None:
        if not isinstance(value, dict):
            return (np.nan, np.nan)
        value = value.get(subkey)

    if not isinstance(value, dict):
        return (np.nan, np.nan)

    ci = value.get("ci")
    if not isinstance(ci, dict):
        return (np.nan, np.nan)

    return (_to_float_or_nan(ci.get("low")), _to_float_or_nan(ci.get("high")))


def degradation_variant(perturbation_config):
    if not isinstance(perturbation_config, dict):
        return ""

    degradation = perturbation_config.get("degradation")
    if not isinstance(degradation, dict) or not degradation.get("enabled"):
        return ""

    enabled = []
    if degradation.get("gaussian_noise"):
        enabled.append("noise")
    if degradation.get("gaussian_blur"):
        enabled.append("blur")
    if degradation.get("jpeg_compression"):
        enabled.append("jpeg")

    if len(enabled) == 3:
        return "all"

    return "+".join(enabled)


def perturbation_family(perturbation_config):
    if not isinstance(perturbation_config, dict):
        return "unknown"

    active = [str(value) for value in (perturbation_config.get("active_perturbations") or [])]
    if not active:
        return "baseline"

    # i keep degradation split by variant because task3 uses those groups later
    if len(active) == 1 and active[0] == "degradation":
        variant = degradation_variant(perturbation_config)
        return f"degradation_{variant}" if variant else "degradation_unknown"

    if len(active) == 1:
        return active[0].replace("preprocessing_variation", "preprocessing")

    return "combined"


def parse_severity_from_name(name):
    if not isinstance(name, str):
        return np.nan

    match = re.search(r"sev(\d+)", name)
    return float(match.group(1)) if match else np.nan


def load_batch_dataframe(report_files, verbose=True):
    rows = []

    for report_path in report_files:
        payload = json.loads(report_path.read_text(encoding="utf-8"))

        for experiment in payload.get("experiments", []):
            outputs = experiment.get("test_outputs") or [{}]

            for output in outputs:
                output = output if isinstance(output, dict) else {}
                metrics = output.get("metrics_report") if isinstance(output.get("metrics_report"), dict) else {}
                perturbation_config = output.get("perturbation_config") if isinstance(output.get("perturbation_config"), dict) else {}
                class_removal_result = _nested_dict(perturbation_config, "class_removal", "result")
                class_imbalance_result = _nested_dict(perturbation_config, "class_imbalance", "result")

                fid_obj = metrics.get("fid")
                kid_obj = metrics.get("kid")
                is_obj = metrics.get("is")
                pr_obj = metrics.get("precision_recall")
                dc_obj = metrics.get("density_coverage")

                fid_low, fid_high = ci_bounds(fid_obj)
                kid_low, kid_high = ci_bounds(kid_obj)
                is_low, is_high = ci_bounds(is_obj)
                precision_low, precision_high = ci_bounds(pr_obj, "precision")
                recall_low, recall_high = ci_bounds(pr_obj, "recall")
                density_low, density_high = ci_bounds(dc_obj, "density")
                coverage_low, coverage_high = ci_bounds(dc_obj, "coverage")

                rows.append(
                    {
                        "report_file": report_path.name,
                        "experiment_id": experiment.get("experiment_id"),
                        "name": experiment.get("name"),
                        "model": experiment.get("model_name"),
                        "dataset": experiment.get("dataset_name"),
                        "status": experiment.get("status"),
                        "exit_code": experiment.get("exit_code"),
                        "metrics_expected": bool(experiment.get("metrics_expected")),
                        "metrics_available": bool(experiment.get("metrics_available")),
                        "step_name": output.get("step_name"),
                        "output_dir": output.get("output_dir"),
                        "metrics_path": output.get("metrics_path"),
                        "has_metrics_report": bool(metrics),
                        "perturbation_family": perturbation_family(perturbation_config),
                        "active_perturbations": ",".join((perturbation_config.get("active_perturbations") or [])),
                        "apply_to": perturbation_config.get("apply_to", ""),
                        "degradation_variant": degradation_variant(perturbation_config),
                        "degradation_severity": _to_float_or_nan((perturbation_config.get("degradation") or {}).get("severity")),
                        "severity_from_name": parse_severity_from_name(experiment.get("name")),
                        "sample_size_n": _to_float_or_nan((perturbation_config.get("sample_size") or {}).get("n")),
                        "preprocessing_variant": (perturbation_config.get("preprocessing") or {}).get("variant", ""),
                        "preprocessing_scale": _to_float_or_nan((perturbation_config.get("preprocessing") or {}).get("scale")),
                        "domain_shift_dataset": (perturbation_config.get("domain_shift") or {}).get("dataset", ""),
                        "class_removal_strategy": _nested_value(perturbation_config, "class_removal", "strategy", default=""),
                        "class_removal_label_mode": _nested_value(class_removal_result, "label_mode", default=""),
                        "class_removal_removed_count": _to_float_or_nan(class_removal_result.get("removed_count")),
                        "class_removal_kept_count": _to_float_or_nan(class_removal_result.get("kept_count")),
                        "class_removal_survivor_count": _to_float_or_nan(class_removal_result.get("survivor_count")),
                        "class_removal_evaluation_count": _to_float_or_nan(class_removal_result.get("evaluation_count")),
                        "class_removal_returned_count": _to_float_or_nan(class_removal_result.get("returned_count")),
                        "class_removal_pool_count": _to_float_or_nan(class_removal_result.get("pool_count")),
                        "class_removal_drop_classes": class_removal_result.get("drop_classes", []),
                        "class_removal_drop_class_names": class_removal_result.get("drop_class_names", []),
                        "class_removal_predicted_label_histogram_fake": class_removal_result.get("predicted_label_histogram_fake", {}),
                        "class_removal_predicted_positive_counts": class_removal_result.get("predicted_positive_counts", {}),
                        "class_imbalance_strategy": _nested_value(perturbation_config, "class_imbalance", "strategy", default=""),
                        "class_imbalance_label_mode": _nested_value(class_imbalance_result, "label_mode", default=""),
                        "class_imbalance_removed_count": _to_float_or_nan(class_imbalance_result.get("removed_count")),
                        "class_imbalance_kept_count": _to_float_or_nan(class_imbalance_result.get("kept_count")),
                        "class_imbalance_survivor_count": _to_float_or_nan(class_imbalance_result.get("survivor_count")),
                        "class_imbalance_evaluation_count": _to_float_or_nan(class_imbalance_result.get("evaluation_count")),
                        "class_imbalance_returned_count": _to_float_or_nan(class_imbalance_result.get("returned_count")),
                        "class_imbalance_pool_count": _to_float_or_nan(class_imbalance_result.get("pool_count")),
                        "class_imbalance_drop_fraction": class_imbalance_result.get("drop_fraction"),
                        "class_imbalance_drop_classes": class_imbalance_result.get("drop_classes", []),
                        "class_imbalance_drop_class_names": class_imbalance_result.get("drop_class_names", []),
                        "class_imbalance_predicted_label_histogram_fake": class_imbalance_result.get("predicted_label_histogram_fake", {}),
                        "class_imbalance_predicted_positive_counts": class_imbalance_result.get("predicted_positive_counts", {}),
                        "fid": metric_point(fid_obj),
                        "kid_mean": metric_point(kid_obj),
                        "kid_std": metric_std(kid_obj),
                        "is_mean": metric_point(is_obj),
                        "is_std": metric_std(is_obj),
                        "precision": metric_point(pr_obj, "precision"),
                        "recall": metric_point(pr_obj, "recall"),
                        "density": metric_point(dc_obj, "density"),
                        "coverage": metric_point(dc_obj, "coverage"),
                        "fid_ci_low": fid_low,
                        "fid_ci_high": fid_high,
                        "kid_ci_low": kid_low,
                        "kid_ci_high": kid_high,
                        "is_ci_low": is_low,
                        "is_ci_high": is_high,
                        "precision_ci_low": precision_low,
                        "precision_ci_high": precision_high,
                        "recall_ci_low": recall_low,
                        "recall_ci_high": recall_high,
                        "density_ci_low": density_low,
                        "density_ci_high": density_high,
                        "coverage_ci_low": coverage_low,
                        "coverage_ci_high": coverage_high,
                        "fid_error": metric_error(fid_obj),
                        "kid_error": metric_error(kid_obj),
                        "is_error": metric_error(is_obj),
                        "pr_error": metric_error(pr_obj),
                        "dc_error": metric_error(dc_obj),
                    }
                )

    frame = pd.DataFrame(rows)
    if verbose:
        print("Rows loaded:", len(frame))
        print("Columns:", len(frame.columns))

    return frame

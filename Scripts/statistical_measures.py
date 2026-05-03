#### Statistical testing ####

import json
import numpy as np
from scipy.stats import norm
from pathlib import Path
from collections import defaultdict



def statistical_measures(data,original_value, a=0.05):
    # calculating CIs and bias
    n = len(data)
    mean = np.mean(data)
    variance = np.var(data)

    z = norm.ppf(1-a/2)
    ci = np.sqrt(variance/n)*z

    bias = None
    if original_value is not None:
        bias = original_value - mean

    return {
        'original_value': original_value,
        'mean': mean,
        'bias': bias,
        'variance': variance,
        'ci_lower': mean-ci,
        'ci_upper': mean+ci
    }


def flatten_metrics(metrics_dict):
    # this was needed to properly read the json file
    flat = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            flat[key] = value
            
        elif isinstance(value, list) and len(value) > 0:
            flat[f"{key}_mean"] = value[0]
            
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    flat[f"{key}_{sub_key}"] = sub_val
    return flat


def analyze_json(report_path):
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    experiments = data.get('experiments', [])
    final_results = {}

    for exp in experiments:
        name = exp.get('name', 'unknown_exp')
        bootstrap_entries = exp.get('all_bootstrap_outputs', [])
        original_entries = exp.get('test_outputs', []) 
        
        metrics_accumulator = defaultdict(list)
        original_metrics_map = {}

        if original_entries:
            raw_orig = original_entries[0].get('metrics_report', {})
            original_metrics_map = flatten_metrics(raw_orig)

        for run in bootstrap_entries:
            if run:
                raw_boot = run[0].get('metrics_report', {})
                flat_boot = flatten_metrics(raw_boot)
                for k, v in flat_boot.items():
                    metrics_accumulator[k].append(v)

        exp_stats = {}
        for metric_name, values in metrics_accumulator.items():
            orig_val = original_metrics_map.get(metric_name)
            exp_stats[metric_name] = statistical_measures(values, orig_val)
            
        final_results[name] = exp_stats

    return final_results

if __name__ == "__main__":
    ####  change to local path
    input_json = Path(r"C:\Users\ilyad\Downloads\application\Group07_Improving_Metrics\outputs\dcgan_cifar10_perturbation_tests.json")
    output_json = Path(r"outputs\analyzed_results.json")

    if input_json.exists():
        processed_data = analyze_json(input_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_json, 'w') as f:
            json.dump(processed_data, f, indent=4)
        
        print(f"Successfully run {len(processed_data)} experiments.")
        print(f"Results saved to: {output_json}")
    else:
        print(f"Could not find file")



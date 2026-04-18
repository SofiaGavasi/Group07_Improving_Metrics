from __future__ import annotations

from typing import Any, Dict

from .density_coverage import compute_density_coverage
from .fid import compute_fid_from_features
from .is_score import compute_inception_score
from .kid import compute_kid
from .precision_recall import compute_precision_recall


def compute_all_metrics(real_samples: Any, fake_samples: Any):

    #runs all requested metrics and returns a single results dictionary.
    results: Dict[str, Any] = {}

    try:
        results["fid"] = compute_fid_from_features(real_samples, fake_samples)
    except Exception as exc:
        results["fid"] = {"error": str(exc), "todo": "validate feature extraction pipeline"}

    # IS is based on generated samples only, while the others compare real vs fake.
    metric_calls = [
        ("is", lambda real, fake: compute_inception_score(fake)),
        ("kid", compute_kid),
        ("precision_recall", compute_precision_recall),
        ("density_coverage", compute_density_coverage),
    ]

    for key, fn in metric_calls:
        try:
            results[key] = fn(real_samples, fake_samples)
        except NotImplementedError as exc:
            results[key] = {"todo": str(exc)}
        except Exception as exc:  
            results[key] = {"error": str(exc)}

    return results

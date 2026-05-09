from __future__ import annotations

from typing import Any, Callable

import numpy as np


# helper for bootstrap percentile interval
def bootstrap_percentile_interval(
    samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return lo, hi


# helper for bootstrap metric distribution
def bootstrap_metric_distribution(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    real_count = int(real_features.shape[0])
    fake_count = int(fake_features.shape[0])
    if real_count < 2 or fake_count < 2:
        return np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(max(0, int(bootstrap_samples))):
        real_idx = rng.choice(real_count, size=real_count, replace=True)
        fake_idx = rng.choice(fake_count, size=fake_count, replace=True)
        value = metric_fn(real_features[real_idx], fake_features[fake_idx])
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


# helper for with bootstrap summary
def with_bootstrap_summary(
    point_estimate: Any,
    bootstrap_distribution: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    lo, hi = bootstrap_percentile_interval(bootstrap_distribution, alpha=alpha)
    return {
        "value": point_estimate,
        "ci": {
            "method": "bootstrap_percentile",
            "alpha": float(alpha),
            "low": lo,
            "high": hi,
            "bootstrap_samples": int(bootstrap_distribution.size),
        },
    }


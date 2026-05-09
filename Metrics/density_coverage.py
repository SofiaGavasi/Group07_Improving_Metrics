from __future__ import annotations

from typing import Any

import numpy as np

from .prdc_utils import nearest_neighbour_radii, pairwise_distance, to_numpy_float32


# compute density coverage
def compute_density_coverage(real_samples: Any, fake_samples: Any, k: int = 5, **kwargs: Any):
    """
    Compute PRDC density and coverage in feature space.

    Reference implementation:
    https://github.com/clovaai/generative-evaluation-prdc
    """
    real = to_numpy_float32(real_samples)
    fake = to_numpy_float32(fake_samples)

    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("density/coverage expects 2D feature arrays.")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake feature dimensions must match.")
    if int(real.shape[0]) <= int(k):
        raise ValueError(f"Need more than k={k} real samples for density/coverage.")

    real_radii = nearest_neighbour_radii(real, nearest_k=int(k))
    dist_real_fake = pairwise_distance(real, fake)

    density = (1.0 / float(k)) * (
        dist_real_fake < np.expand_dims(real_radii, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        dist_real_fake.min(axis=1) < real_radii
    ).mean()

    return {
        "density": float(density),
        "coverage": float(coverage),
    }

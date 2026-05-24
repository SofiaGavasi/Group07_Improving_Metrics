from __future__ import annotations

from typing import Any

import numpy as np

from .prdc_utils import nearest_neighbour_radii, pairwise_distance, to_numpy_float32


# compute precision recall
def compute_precision_recall(real_samples: Any, fake_samples: Any, k: int = 5, **kwargs: Any):
    """
    Compute PRDC-style precision and recall in feature space.

    Reference implementation:
    https://github.com/clovaai/generative-evaluation-prdc
    """
    real = to_numpy_float32(real_samples)
    fake = to_numpy_float32(fake_samples)

    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("precision/recall expects 2D feature arrays.")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake feature dimensions must match.")
    if int(real.shape[0]) <= int(k) or int(fake.shape[0]) <= int(k):
        raise ValueError(f"Need more than k={k} samples per set for precision/recall.")

    real_radii = nearest_neighbour_radii(real, nearest_k=int(k))
    fake_radii = nearest_neighbour_radii(fake, nearest_k=int(k))
    dist_real_fake = pairwise_distance(real, fake)

    # Precision: fraction of fake samples that lie in at least one real neighborhood.
    precision = (
        dist_real_fake < np.expand_dims(real_radii, axis=1)
    ).any(axis=0).mean()

    # Recall: fraction of real samples that lie in at least one fake neighborhood.
    recall = (
        dist_real_fake < np.expand_dims(fake_radii, axis=0)
    ).any(axis=1).mean()

    return {
        "precision": float(precision),
        "recall": float(recall),
    }

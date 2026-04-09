from __future__ import annotations

from typing import Any
import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if y is None:
        y = x
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def compute_density_coverage(real_samples: Any, fake_samples: Any, k: int = 5, **kwargs: Any):
    """
    Compute Density and Coverage from feature vectors.

    Args:
        real_samples: array-like of shape (N, D)
        fake_samples: array-like of shape (M, D)
        k: number of nearest neighbors for defining real neighborhoods

    Returns:
        {"density": float, "coverage": float}
    """
    real = _to_numpy(real_samples)
    fake = _to_numpy(fake_samples)

    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("real_samples and fake_samples must be 2D arrays of shape (num_samples, feature_dim)")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real_samples and fake_samples must have the same feature dimension")

    n_real = real.shape[0]
    n_fake = fake.shape[0]

    if n_real < 2:
        raise ValueError("Need at least 2 real samples")
    if n_fake < 1:
        raise ValueError("Need at least 1 fake sample")
    if not (1 <= k < n_real):
        raise ValueError(f"k must satisfy 1 <= k < {n_real}")

    # Real-real distances
    real_real_d2 = _pairwise_squared_distances(real)
    np.fill_diagonal(real_real_d2, np.inf)

    # Radius of each real sample = distance to its k-th nearest real neighbor
    radii_d2 = np.partition(real_real_d2, kth=k - 1, axis=1)[:, k - 1]

    # Fake-real distances
    fake_real_d2 = _pairwise_squared_distances(fake, real)

    # inside[j, i] = whether fake_j lies in the neighborhood of real_i
    inside = fake_real_d2 <= radii_d2[None, :]

    density = inside.sum() / (k * n_fake)
    coverage = inside.any(axis=0).mean()

    return {
        "density": float(density),
        "coverage": float(coverage),
    }


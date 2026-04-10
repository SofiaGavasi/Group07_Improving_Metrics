from __future__ import annotations

from typing import Any
import numpy as np
import sklearn.metrics
""""
This implementation is adapted from the PRDC repository:
https://github.com/clovaai/generative-evaluation-prdc

Original PRDC code authors:
Copyright (c) 2020-present NAVER Corp.

The implementation is based on the paper:

Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., and Yoo, J.
"Reliable Fidelity and Diversity Metrics for Generative Models."
ICML 2020.
"""

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if hasattr(data_x, "detach"):
        data_x = data_x.detach().cpu().numpy()
    if data_y is not None and hasattr(data_y, "detach"):
        data_y = data_y.detach().cpu().numpy()

    data_x = np.asarray(data_x, dtype=np.float32)
    if data_y is None:
        data_y = data_x
    else:
        data_y = np.asarray(data_y, dtype=np.float32)

    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=8
    )
    return dists

def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_density_coverage(real_samples: Any, fake_samples: Any, k: int = 5, **kwargs: Any):
    """
    Computes density and coverage given two manifolds.

    Args:
        real_samples: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_samples: numpy.ndarray([N, feature_dim], dtype=np.float32)
        k: int
    Returns:
        dict with density and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_samples, k
    )
    distance_real_fake = compute_pairwise_distance(real_samples, fake_samples)

    density = (1.0 / float(k)) * (
        distance_real_fake <
        np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) <
        real_nearest_neighbour_distances
    ).mean()

    return {
        "density": float(density),
        "coverage": float(coverage),
    }

''' Our own implementation
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
    '''
#some tests



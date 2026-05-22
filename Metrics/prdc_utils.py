from __future__ import annotations

import numpy as np
import sklearn.metrics

PAIRWISE_DISTANCE_JOBS = 1


# helper for to numpy float32
def to_numpy_float32(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


# helper for pairwise distance
def pairwise_distance(data_x, data_y=None) -> np.ndarray:
    data_x = to_numpy_float32(data_x)
    data_y = data_x if data_y is None else to_numpy_float32(data_y)
    return sklearn.metrics.pairwise_distances(
        data_x,
        data_y,
        metric="euclidean",
        n_jobs=int(PAIRWISE_DISTANCE_JOBS),
    )


# helper for kth value
def kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


# helper for nearest neighbour radii
def nearest_neighbour_radii(features, nearest_k: int) -> np.ndarray:
    distances = pairwise_distance(features)
    return kth_value(distances, k=nearest_k + 1, axis=-1)


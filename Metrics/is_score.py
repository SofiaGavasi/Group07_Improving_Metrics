from __future__ import annotations

import numpy as np


# compute inception score
def compute_inception_score(probabilities: np.ndarray, num_splits: int = 10):
    """
    Compute Inception Score from class probabilities.

    IS requires probabilities p(y|x) from an ImageNet-pretrained Inception model.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("Inception Score expects a 2D probability array [N, num_classes].")
    if probs.shape[0] == 0:
        raise ValueError("Inception Score received zero samples.")

    row_sums = probs.sum(axis=1)
    if np.any(probs < 0.0) or not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Inception Score input must be class probabilities that sum to 1.")

    num_splits = max(1, min(int(num_splits), int(probs.shape[0])))
    split_size = max(1, int(probs.shape[0]) // num_splits)

    scores: list[float] = []
    for split_index in range(num_splits):
        start = split_index * split_size
        end = min((split_index + 1) * split_size, int(probs.shape[0]))
        split_probs = probs[start:end, :]
        if split_probs.size == 0:
            continue

        marginal = np.mean(split_probs, axis=0)
        kl = split_probs * (np.log(split_probs + 1e-10) - np.log(marginal + 1e-10))
        kl = np.sum(kl, axis=1)
        scores.append(float(np.exp(np.mean(kl))))

    if not scores:
        raise ValueError("Inception Score could not compute any valid split.")

    return float(np.mean(scores)), float(np.std(scores))

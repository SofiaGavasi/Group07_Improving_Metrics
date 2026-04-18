from __future__ import annotations

from typing import Any, Dict
import torch
from torchvision import models
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
def compute_inception_score(preds, num_splits=10):
    if isinstance(preds, np.ndarray):
        preds = np.asarray(preds)
    else:
        preds = np.concatenate(preds, axis=0)

    if preds.ndim != 2:
        raise ValueError("Inception Score expects a 2D array of predictions/features.")

    # if caller gives raw features (not class probabilities), convert safely with softmax.
    row_sums = np.sum(preds, axis=1)
    looks_like_probs = np.all(preds >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-3)
    if not looks_like_probs:
        shifted = preds - np.max(preds, axis=1, keepdims=True)
        exp_preds = np.exp(shifted)
        denom = np.sum(exp_preds, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        preds = exp_preds / denom

    scores = []
    N = preds.shape[0]
    if N == 0:
        raise ValueError("Inception Score received zero samples.")

    num_splits = max(1, min(int(num_splits), int(N)))
    split_size = max(1, N // num_splits)

    for k in range(num_splits):
        start = k * split_size
        end = min((k + 1) * split_size, N)
        part = preds[start:end, :]
        if part.size == 0:
            continue
        py = np.mean(part, axis=0)
        kl_div = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_div = np.sum(kl_div, axis=1)
        scores.append(np.exp(np.mean(kl_div)))

    if not scores:
        raise ValueError("Inception Score could not compute any valid split.")
    return np.mean(scores), np.std(scores)

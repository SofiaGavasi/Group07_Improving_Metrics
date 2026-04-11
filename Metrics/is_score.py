from __future__ import annotations

from typing import Any, Dict
import torch
from torchvision import models
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
def compute_inception_score(preds, num_splits=10):
    if isinstance(preds, np.ndarray):
        preds = preds  # already a single array, use as-is
    else:
        preds = np.concatenate(preds, axis=0)  # list of arrays
    
    scores = []
    N = preds.shape[0]

    for k in range(num_splits):
        part = preds[k * (N // num_splits): (k + 1) * (N // num_splits), :]
        py = np.mean(part, axis=0)
        kl_div = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_div = np.sum(kl_div, axis=1)
        scores.append(np.exp(np.mean(kl_div)))

    return np.mean(scores), np.std(scores)
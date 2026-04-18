from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch
from torchmetrics.image.kid import KernelInceptionDistance


def polynomial_kernel(x: np.ndarray,y: np.ndarray):
    d = x.shape[1] # kernel on feature dimension
    return ((1/d)*(x @ y.T)+ 1 )**3

def kid(real_features: np.ndarray, fake_features: np.ndarray):
    m= real_features.shape[0] # comparing based on samples
    n= fake_features.shape[0]

    Kxx = polynomial_kernel(real_features,real_features)
    sum_Kxx=  np.sum(Kxx) - np.trace(Kxx) # we need this so it is not biased
    Kyy = polynomial_kernel(fake_features,fake_features)
    sum_Kyy=  np.sum(Kyy) - np.trace(Kyy)

    sum_Kxy = np.sum(polynomial_kernel(real_features,fake_features))

    kid = (1/(m*(m-1)))*sum_Kxx +(1/(n*(n-1)))*sum_Kyy  - 2*(1/(n*m))*sum_Kxy

    return float(kid)

def compute_kid(real_features: np.ndarray, fake_features: np.ndarray):
    # casting to float32 cuts memory pressure in half for large feature vectors.
    real_features = np.asarray(real_features, dtype=np.float32)
    fake_features = np.asarray(fake_features, dtype=np.float32)

    m = int(real_features.shape[0])
    n = int(fake_features.shape[0])
    if m < 2 or n < 2:
        raise ValueError("KID needs at least 2 real and 2 fake samples.")

    feature_dim = int(real_features.shape[1])

    # this keeps sampling stable while avoiding huge temporary arrays.
    # rough budget: two sampled matrices (real+fake), float32.
    memory_budget_bytes = 256 * 1024 * 1024
    max_subset_by_memory = max(
        2,
        int(memory_budget_bytes // max(1, feature_dim * 4 * 2)),
    )
    subset_size = min(1000, m, n, max_subset_by_memory)

    # if subset is tiny there is no value in many bootstrap rounds.
    num_rounds = 100 if subset_size >= 16 else 20

    KIDs = []
    for _ in range(num_rounds):
        replace_real = m < subset_size
        replace_fake = n < subset_size
        random_real = np.random.choice(m, subset_size, replace=replace_real)
        random_fake = np.random.choice(n, subset_size, replace=replace_fake)
        sampled_kid = kid(real_features[random_real], fake_features[random_fake])
        KIDs.append(sampled_kid)

    return np.mean(KIDs) , np.std(KIDs)

def kid_clean(real_images: np.ndarray, fake_images: np.ndarray):

    kid = KernelInceptionDistance(subset_size=100)
    kid.update(real_images, real=True)
    kid.update(fake_images, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean, kid_std

    ## Warning: 
    #   - kid_clean uses images and not features 
    # from documentation : RGB images of shape (3 x H x W) with dtype uint8. 
    # All images will be resized to 299 x 299 which is the size of the original training data.


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
    KIDs= []
    for _ in range(100):
        random_real = np.random.choice(real_features.shape[0], 1000)
        random_fake = np.random.choice(fake_features.shape[0], 1000)
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


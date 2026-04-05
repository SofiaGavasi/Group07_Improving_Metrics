from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) :

    diff = mu1 - mu2
    sum_sq_diff = diff.dot(diff)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(sum_sq_diff + tr_covmean)


def compute_fid_from_features(real_features: np.ndarray, fake_features: np.ndarray) :
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)
    fake_mu = np.mean(fake_features, axis=0)
    fake_sigma = np.cov(fake_features, rowvar=False)
    return calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)


def compute_fid_with_clean_fid(real_dir: str, fake_dir: str):
    """
    This is fid using the library clean-fid, just as a reference for debugging
    """
    # TODO: add optional mode/settings args once eval protocol is finalized

    from cleanfid import fid

    return float(fid.compute_fid(real_dir, fake_dir))

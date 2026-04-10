from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray):

    diff = mu1 - mu2
    sum_sq_diff = diff.dot(diff)
    cov_prod = sigma1.dot(sigma2)
    covmean, _ = sqrtm(cov_prod, disp=False)

    if not np.isfinite(covmean).all():
        # added tiny diagonal jitter for numerical stability
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(sum_sq_diff + tr_covmean)


def _to_2d_float64(features: np.ndarray) :
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("features must be a 2D array of shape [num_samples, feature_dim]")
    return arr


def _project_high_dim_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    max_cov_dim: int = 2048,
) :
    # for very large D ( 3*256*256), computing DxD covariance is not feasible
    # project both sets to a shared low-dimensional subspace before FID
    real = _to_2d_float64(real_features)
    fake = _to_2d_float64(fake_features)
    feature_dim = int(real.shape[1])

    if feature_dim <= max_cov_dim:
        return real, fake

    stacked = np.concatenate([real, fake], axis=0)
    shared_mean = np.mean(stacked, axis=0, keepdims=True)
    centered = stacked - shared_mean

    # compact SVD in sample space; effective rank <= num_samples - 1
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj_dim = max(1, min(int(vt.shape[0]), int(max_cov_dim)))
    basis = vt[:proj_dim].T

    real_proj = (real - shared_mean) @ basis
    fake_proj = (fake - shared_mean) @ basis
    return real_proj, fake_proj


def compute_fid_from_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    max_cov_dim: int = 2048,
):
    real_features, fake_features = _project_high_dim_features(
        real_features=real_features,
        fake_features=fake_features,
        max_cov_dim=max_cov_dim,
    )

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

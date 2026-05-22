from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .density_coverage import compute_density_coverage
from .fid import compute_fid_from_features, prepare_features_for_fid
from .inception_features import InceptionFeatureConfig, InceptionFeatureExtractor
from .is_score import compute_inception_score
from .kid import compute_kid
from .precision_recall import compute_precision_recall
from .statistics import (
    bootstrap_metric_distribution,
    with_bootstrap_summary,
)

FID_MAX_COV_DIM = 512



@dataclass
class MetricComputationConfig:
    """Configuration used by the metrics orchestration layer."""

    feature_space: str = "inception_v3"
    feature_batch_size: int = 64
    feature_device: str = "cpu"
    bootstrap_samples: int = 0
    requested_bootstrap_samples: int | None = None
    bootstrap_seed: int = 0
    bootstrap_alpha: float = 0.05
    bootstrap_policy: str = "full"
    pr_k: int = 3
    dc_k: int = 5
    is_splits: int = 10
    verbose: bool = False


# helper for as tensor
def _as_tensor(samples: Any) -> torch.Tensor:
    if isinstance(samples, torch.Tensor):
        return samples
    arr = np.asarray(samples)
    return torch.as_tensor(arr)


# helper for safe metric call
def _safe_metric_call(metric_fn):
    try:
        return metric_fn()
    except Exception as exc:
        return {"error": str(exc)}


# check whether error payload
def _is_error_payload(value: Any) -> bool:
    return isinstance(value, dict) and "error" in value


# helper for verbose print
def _verbose_print(cfg: MetricComputationConfig, message: str) -> None:
    if bool(cfg.verbose):
        print(f"[metrics] {message}", flush=True)


def _compute_metrics_from_extracted(
    *,
    real_features,
    fake_features,
    fake_probs,
    config,
):
    # the math below is the same no matter where the features came from
    cfg = config
    fid_real_features, fid_fake_features = prepare_features_for_fid(
        real_features=real_features,
        fake_features=fake_features,
        max_cov_dim=int(FID_MAX_COV_DIM),
    )

    results: dict[str, Any] = {
        "metadata": {
            "feature_space": cfg.feature_space,
            "paired_samples": int(real_features.shape[0]),
            "feature_batch_size": int(cfg.feature_batch_size),
            "fid_max_cov_dim": int(FID_MAX_COV_DIM),
            "bootstrap_samples": int(cfg.bootstrap_samples),
            "requested_bootstrap_samples": int(
                cfg.requested_bootstrap_samples
                if cfg.requested_bootstrap_samples is not None
                else cfg.bootstrap_samples
            ),
            "bootstrap_seed": int(cfg.bootstrap_seed),
            "bootstrap_alpha": float(cfg.bootstrap_alpha),
            "bootstrap_policy": str(cfg.bootstrap_policy),
            "pr_k": int(cfg.pr_k),
            "dc_k": int(cfg.dc_k),
            "is_splits": int(cfg.is_splits),
            "bootstrap_errors": {},
        }
    }

    fid_value = _safe_metric_call(
        lambda: compute_fid_from_features(
            fid_real_features,
            fid_fake_features,
            max_cov_dim=None,
        )
    )
    kid_value = _safe_metric_call(lambda: compute_kid(real_features, fake_features))
    is_value = _safe_metric_call(lambda: compute_inception_score(fake_probs, num_splits=int(cfg.is_splits)))
    pr_value = _safe_metric_call(
        lambda: compute_precision_recall(real_features, fake_features, k=int(cfg.pr_k))
    )
    dc_value = _safe_metric_call(
        lambda: compute_density_coverage(real_features, fake_features, k=int(cfg.dc_k))
    )

    results["fid"] = fid_value
    results["kid"] = kid_value
    results["is"] = is_value
    results["precision_recall"] = pr_value
    results["density_coverage"] = dc_value
    _verbose_print(cfg, "base metric estimates computed")

    if int(cfg.bootstrap_samples) > 0 and not _is_error_payload(fid_value):
        try:
            _verbose_print(cfg, f"bootstrapping FID with B={cfg.bootstrap_samples}")
            fid_boot = bootstrap_metric_distribution(
                real_features=fid_real_features,
                fake_features=fid_fake_features,
                metric_fn=lambda a, b: compute_fid_from_features(
                    a,
                    b,
                    max_cov_dim=None,
                ),
                bootstrap_samples=int(cfg.bootstrap_samples),
                seed=int(cfg.bootstrap_seed) + 11,
            )
            results["fid"] = with_bootstrap_summary(
                point_estimate=float(fid_value),
                bootstrap_distribution=fid_boot,
                alpha=float(cfg.bootstrap_alpha),
            )
        except Exception as exc:
            results["metadata"]["bootstrap_errors"]["fid"] = str(exc)

    if int(cfg.bootstrap_samples) > 0 and not _is_error_payload(kid_value):
        try:
            _verbose_print(cfg, f"bootstrapping KID mean with B={cfg.bootstrap_samples}")
            # compute_kid returns (mean, std), we bootstrap the mean estimate only.
            kid_boot = bootstrap_metric_distribution(
                real_features=real_features,
                fake_features=fake_features,
                metric_fn=lambda a, b: float(compute_kid(a, b)[0]),
                bootstrap_samples=int(cfg.bootstrap_samples),
                seed=int(cfg.bootstrap_seed) + 23,
            )
            results["kid"] = {
                "mean": float(kid_value[0]),
                "std": float(kid_value[1]),
                "ci": with_bootstrap_summary(
                    point_estimate=float(kid_value[0]),
                    bootstrap_distribution=kid_boot,
                    alpha=float(cfg.bootstrap_alpha),
                )["ci"],
            }
        except Exception as exc:
            results["metadata"]["bootstrap_errors"]["kid"] = str(exc)

    if int(cfg.bootstrap_samples) > 0 and not _is_error_payload(is_value):
        try:
            _verbose_print(cfg, f"bootstrapping IS mean with B={cfg.bootstrap_samples}")
            # IS bootstrap is done from fake probabilities only.
            rng = np.random.default_rng(int(cfg.bootstrap_seed) + 37)
            is_boot: list[float] = []
            n_fake = int(fake_probs.shape[0])
            for _ in range(int(cfg.bootstrap_samples)):
                idx = rng.choice(n_fake, size=n_fake, replace=True)
                boot_mean, _ = compute_inception_score(fake_probs[idx], num_splits=int(cfg.is_splits))
                is_boot.append(float(boot_mean))
            results["is"] = {
                "mean": float(is_value[0]),
                "std": float(is_value[1]),
                "ci": with_bootstrap_summary(
                    point_estimate=float(is_value[0]),
                    bootstrap_distribution=np.asarray(is_boot, dtype=np.float64),
                    alpha=float(cfg.bootstrap_alpha),
                )["ci"],
            }
        except Exception as exc:
            results["metadata"]["bootstrap_errors"]["is"] = str(exc)

    if int(cfg.bootstrap_samples) > 0 and not _is_error_payload(pr_value):
        try:
            _verbose_print(cfg, f"bootstrapping precision/recall with B={cfg.bootstrap_samples}")
            pr_prec_boot = bootstrap_metric_distribution(
                real_features=real_features,
                fake_features=fake_features,
                metric_fn=lambda a, b: float(compute_precision_recall(a, b, k=int(cfg.pr_k))["precision"]),
                bootstrap_samples=int(cfg.bootstrap_samples),
                seed=int(cfg.bootstrap_seed) + 41,
            )
            pr_rec_boot = bootstrap_metric_distribution(
                real_features=real_features,
                fake_features=fake_features,
                metric_fn=lambda a, b: float(compute_precision_recall(a, b, k=int(cfg.pr_k))["recall"]),
                bootstrap_samples=int(cfg.bootstrap_samples),
                seed=int(cfg.bootstrap_seed) + 43,
            )
            results["precision_recall"] = {
                "precision": with_bootstrap_summary(
                    point_estimate=float(pr_value["precision"]),
                    bootstrap_distribution=pr_prec_boot,
                    alpha=float(cfg.bootstrap_alpha),
                ),
                "recall": with_bootstrap_summary(
                    point_estimate=float(pr_value["recall"]),
                    bootstrap_distribution=pr_rec_boot,
                    alpha=float(cfg.bootstrap_alpha),
                ),
            }
        except Exception as exc:
            results["metadata"]["bootstrap_errors"]["precision_recall"] = str(exc)

    if int(cfg.bootstrap_samples) > 0 and not _is_error_payload(dc_value):
        try:
            _verbose_print(cfg, f"bootstrapping density/coverage with B={cfg.bootstrap_samples}")
            dc_density_boot = bootstrap_metric_distribution(
                real_features=real_features,
                fake_features=fake_features,
                metric_fn=lambda a, b: float(compute_density_coverage(a, b, k=int(cfg.dc_k))["density"]),
                bootstrap_samples=int(cfg.bootstrap_samples),
                seed=int(cfg.bootstrap_seed) + 47,
            )
            dc_coverage_boot = bootstrap_metric_distribution(
                real_features=real_features,
                fake_features=fake_features,
                metric_fn=lambda a, b: float(compute_density_coverage(a, b, k=int(cfg.dc_k))["coverage"]),
                bootstrap_samples=int(cfg.bootstrap_samples),
                seed=int(cfg.bootstrap_seed) + 53,
            )
            results["density_coverage"] = {
                "density": with_bootstrap_summary(
                    point_estimate=float(dc_value["density"]),
                    bootstrap_distribution=dc_density_boot,
                    alpha=float(cfg.bootstrap_alpha),
                ),
                "coverage": with_bootstrap_summary(
                    point_estimate=float(dc_value["coverage"]),
                    bootstrap_distribution=dc_coverage_boot,
                    alpha=float(cfg.bootstrap_alpha),
                ),
            }
        except Exception as exc:
            results["metadata"]["bootstrap_errors"]["density_coverage"] = str(exc)

    _verbose_print(cfg, "metric computation completed")
    return results


def compute_all_metrics_from_extracted(
    *,
    real_features,
    fake_features,
    fake_probs,
    config = None,
):
    # this is the new entrypoint the cache layer uses after it skips extraction
    cfg = config or MetricComputationConfig()
    _verbose_print(cfg, "starting metric computation from cached features")

    real_arr = np.asarray(real_features, dtype=np.float64)
    fake_arr = np.asarray(fake_features, dtype=np.float64)
    probs_arr = np.asarray(fake_probs, dtype=np.float64)

    if real_arr.ndim != 2 or fake_arr.ndim != 2:
        raise ValueError("Expected 2D feature arrays for metric computation.")
    if probs_arr.ndim != 2:
        raise ValueError("Expected 2D probability array for metric computation.")

    paired_count = min(int(real_arr.shape[0]), int(fake_arr.shape[0]), int(probs_arr.shape[0]))
    if paired_count < 4:
        raise ValueError("Need at least 4 paired real/fake samples for metric computation.")

    real_arr = real_arr[:paired_count]
    fake_arr = fake_arr[:paired_count]
    probs_arr = probs_arr[:paired_count]

    if cfg.feature_space != "inception_v3":
        raise ValueError(f"Unsupported metrics feature space: {cfg.feature_space}")
    _verbose_print(
        cfg,
        f"feature_space={cfg.feature_space} paired_count={paired_count} "
        f"feature_batch_size={cfg.feature_batch_size} feature_device={cfg.feature_device}",
    )

    return _compute_metrics_from_extracted(
        real_features=real_arr,
        fake_features=fake_arr,
        fake_probs=probs_arr,
        config=cfg,
    )


def compute_all_metrics(
    real_samples: Any,
    fake_samples: Any,
    config: MetricComputationConfig | None = None,
):
    """
    Compute all metrics from image tensors using pretrained Inception features.

    References:
    - FID/KID/IS protocol: torch-fidelity / clean-fid conventions.
    - PR/DC protocol: PRDC reference implementation.
    """
    cfg = config or MetricComputationConfig()
    _verbose_print(cfg, "starting metric computation")

    real_tensor = _as_tensor(real_samples).detach().cpu().float()
    fake_tensor = _as_tensor(fake_samples).detach().cpu().float()

    paired_count = min(int(real_tensor.shape[0]), int(fake_tensor.shape[0]))
    if paired_count < 4:
        raise ValueError("Need at least 4 paired real/fake samples for metric computation.")

    real_tensor = real_tensor[:paired_count]
    fake_tensor = fake_tensor[:paired_count]

    if cfg.feature_space != "inception_v3":
        raise ValueError(f"Unsupported metrics feature space: {cfg.feature_space}")
    _verbose_print(
        cfg,
        f"feature_space={cfg.feature_space} paired_count={paired_count} "
        f"feature_batch_size={cfg.feature_batch_size} feature_device={cfg.feature_device}",
    )

    extractor = InceptionFeatureExtractor(
        config=InceptionFeatureConfig(
            batch_size=int(cfg.feature_batch_size),
            device=str(cfg.feature_device),
            num_workers=0,
        )
    )
    try:
        _verbose_print(cfg, "extracting real features/probabilities with Inception-v3")
        real_features, _, _ = extractor.extract(real_tensor)
        _verbose_print(cfg, "extracting fake features/probabilities with Inception-v3")
        fake_features, _, fake_probs = extractor.extract(fake_tensor)
    finally:
        extractor.close()
    _verbose_print(
        cfg,
        f"extraction done: real_features={real_features.shape} fake_features={fake_features.shape} "
        f"fake_probs={fake_probs.shape}",
    )
    return _compute_metrics_from_extracted(
        real_features=real_features,
        fake_features=fake_features,
        fake_probs=fake_probs,
        config=cfg,
    )

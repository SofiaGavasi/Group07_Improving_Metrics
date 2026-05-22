from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from .class_imbalance import apply_class_imbalance
from .class_removal import apply_class_removal


def add_perturbation_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use-perturbations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable configured perturbations before saving samples and computing metrics.",
    )
    parser.add_argument(
        "--perturb-apply-to",
        choices=["fake", "real", "both"],
        default="fake",
        help="Target sample split for perturbations that support it.",
    )
    parser.add_argument(
        "--perturb-degrade",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DegradedDataset perturbation.",
    )
    parser.add_argument("--perturb-degrade-severity", type=int, default=1)
    parser.add_argument(
        "--perturb-degrade-gaussian-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--perturb-degrade-gaussian-blur",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--perturb-degrade-jpeg-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--perturb-memoisation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable MemoisationDataset perturbation.",
    )
    parser.add_argument("--perturb-memo-fraction", type=float, default=0.1)
    parser.add_argument("--perturb-memo-seed", type=int, default=10)
    parser.add_argument(
        "--perturb-class-removal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable class-removal perturbation (mode dropping).",
    )
    parser.add_argument(
        "--perturb-class-removal-strategy",
        choices=["label", "kmeans"],
        default="label",
        help="Class-removal strategy: direct labels or kmeans over label co-occurrence.",
    )
    parser.add_argument(
        "--perturb-class-removal-targets",
        type=str,
        default="",
        help="Comma-separated labels/indices (or kmeans label-cluster ids) to drop.",
    )
    parser.add_argument("--perturb-class-removal-kmeans-k", type=int, default=8)
    parser.add_argument(
        "--perturb-class-removal-kmeans-cache-path",
        type=str,
        default="",
        help="Optional path to save/reuse kmeans label-cluster assignments.",
    )
    parser.add_argument(
        "--perturb-class-removal-kmeans-recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, ignore existing cache and rebuild kmeans label clusters.",
    )
    parser.add_argument("--perturb-class-removal-seed", type=int, default=10)
    parser.add_argument(
        "--perturb-class-removal-label-threshold",
        type=float,
        default=0.0,
        help="Margin for multi-label prototype assignment. Higher means stricter positives.",
    )
    parser.add_argument("--perturb-class-removal-min-kept", type=int, default=4)
    parser.add_argument(
        "--perturb-class-fixed-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep class-removal and class-imbalance metrics on a fixed fake sample count.",
    )
    parser.add_argument(
        "--perturb-class-eval-count",
        type=int,
        default=0,
        help="Fixed fake sample count used after class perturbations. 0 falls back to metrics_samples.",
    )
    parser.add_argument(
        "--perturb-class-pool-size",
        type=int,
        default=0,
        help="Optional explicit fake pool size for class perturbation sweeps. 0 uses the multiplier.",
    )
    parser.add_argument(
        "--perturb-class-pool-multiplier",
        type=float,
        default=3.0,
        help="Multiplier used to size the fake pool for class perturbation sweeps.",
    )
    parser.add_argument(
        "--perturb-class-imbalance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable class-imbalance perturbation (partial class dropping).",
    )
    parser.add_argument(
        "--perturb-class-imbalance-strategy",
        choices=["label", "kmeans"],
        default="label",
        help="Class-imbalance strategy: direct labels or kmeans over label co-occurrence.",
    )
    parser.add_argument(
        "--perturb-class-imbalance-targets",
        type=str,
        default="",
        help="Comma-separated labels/indices (or kmeans label-cluster ids) to skew.",
    )
    parser.add_argument(
        "--perturb-class-imbalance-balance",
        type=str,
        default="0.5",
        help="Drop ratio for selected targets. Use a single float or comma-separated per target.",
    )
    parser.add_argument("--perturb-class-imbalance-kmeans-k", type=int, default=8)
    parser.add_argument(
        "--perturb-class-imbalance-kmeans-cache-path",
        type=str,
        default="",
        help="Optional path to save/reuse kmeans label-cluster assignments.",
    )
    parser.add_argument(
        "--perturb-class-imbalance-kmeans-recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, ignore existing cache and rebuild kmeans label clusters.",
    )
    parser.add_argument("--perturb-class-imbalance-seed", type=int, default=10)
    parser.add_argument(
        "--perturb-class-imbalance-label-threshold",
        type=float,
        default=0.0,
        help="Margin for multi-label prototype assignment. Higher means stricter positives.",
    )
    parser.add_argument("--perturb-class-imbalance-min-kept", type=int, default=4)

    parser.add_argument(
        "--perturb-sample-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sample-size variation perturbation.",
    )
    parser.add_argument("--perturb-sample-size-n", type=int, default=1000)
    parser.add_argument("--perturb-sample-size-seed", type=int, default=10)
    parser.add_argument(
        "--perturb-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable preprocessing-variation perturbation.",
    )
    parser.add_argument(
        "--perturb-preprocessing-variant",
        type=str,
        default="downsample_bilinear",
        choices=[
            "downsample_nearest",
            "downsample_bilinear",
            "downsample_bicubic",
            "center_crop_pad",
            "grayscale_triplicate",
        ],
    )
    parser.add_argument("--perturb-preprocessing-scale", type=float, default=0.75)
    parser.add_argument(
        "--perturb-domain-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable domain-shift evaluation (uses alternate real reference dataset).",
    )
    parser.add_argument("--perturb-domain-shift-dataset", type=str, default="")
    parser.add_argument("--perturb-domain-shift-data-root", type=str, default="")
    parser.add_argument("--perturb-domain-shift-image-size", type=int, default=0)

def perturbations_enabled(args: argparse.Namespace):
    return bool(
        getattr(args, "use_perturbations", False)
        or getattr(args, "perturb_degrade", False)
        or getattr(args, "perturb_memoisation", False)
        or getattr(args, "perturb_class_removal", False)
        or getattr(args, "perturb_class_imbalance", False)
        or getattr(args, "perturb_sample_size", False)
        or getattr(args, "perturb_preprocessing", False)
        or getattr(args, "perturb_domain_shift", False)
    )


def perturbation_needs_reference_targets(args: argparse.Namespace):
    # both class-removal strategies need real labels:
    # - label: direct drop list
    # - kmeans: build label co-occurrence clusters
    return bool(
        getattr(args, "perturb_class_removal", False)
        or getattr(args, "perturb_class_imbalance", False)
    )


def perturbation_needs_real_reference(args: argparse.Namespace):
    if getattr(args, "perturb_memoisation", False):
        return True
    if getattr(args, "perturb_apply_to", "fake") in {"real", "both"}:
        return True
    if perturbation_needs_reference_targets(args):
        return True
    return False


def get_perturbation_config_dict(args: argparse.Namespace) :
    enabled = perturbations_enabled(args)
    active = []
    if getattr(args, "perturb_degrade", False):
        active.append("degradation")
    if getattr(args, "perturb_memoisation", False):
        active.append("memoisation")
    if getattr(args, "perturb_class_removal", False):
        active.append("class_removal")
    if getattr(args, "perturb_class_imbalance", False):
        active.append("class_imbalance")
    if getattr(args, "perturb_sample_size", False):
        active.append("sample_size")
    if getattr(args, "perturb_preprocessing", False):
        active.append("preprocessing_variation")
    if getattr(args, "perturb_domain_shift", False):
        active.append("domain_shift")

    return {
        "enabled": enabled,
        "apply_to": getattr(args, "perturb_apply_to", "fake"),
        "active_perturbations": active,
        "degradation": {
            "enabled": bool(getattr(args, "perturb_degrade", False)),
            "severity": int(getattr(args, "perturb_degrade_severity", 1)),
            "gaussian_noise": bool(getattr(args, "perturb_degrade_gaussian_noise", False)),
            "gaussian_blur": bool(getattr(args, "perturb_degrade_gaussian_blur", False)),
            "jpeg_compression": bool(getattr(args, "perturb_degrade_jpeg_compression", False)),
        },
        "memoisation": {
            "enabled": bool(getattr(args, "perturb_memoisation", False)),
            "fraction": float(getattr(args, "perturb_memo_fraction", 0.1)),
            "seed": int(getattr(args, "perturb_memo_seed", 10)),
        },
        "class_removal": {
            "enabled": bool(getattr(args, "perturb_class_removal", False)),
            "strategy": str(getattr(args, "perturb_class_removal_strategy", "label")),
            "targets_raw": str(getattr(args, "perturb_class_removal_targets", "")),
            "kmeans_k": int(getattr(args, "perturb_class_removal_kmeans_k", 8)),
            "kmeans_cache_path": str(getattr(args, "perturb_class_removal_kmeans_cache_path", "")),
            "kmeans_recreate": bool(getattr(args, "perturb_class_removal_kmeans_recreate", False)),
            "seed": int(getattr(args, "perturb_class_removal_seed", 10)),
            "label_threshold": float(getattr(args, "perturb_class_removal_label_threshold", 0.0)),
            "min_kept": int(getattr(args, "perturb_class_removal_min_kept", 4)),
            "out_dir": str(getattr(args, "out_dir", "")),
        },
        "class_fixed_eval": {
            "enabled": bool(
                getattr(args, "perturb_class_fixed_eval", True)
                and (
                    getattr(args, "perturb_class_removal", False)
                    or getattr(args, "perturb_class_imbalance", False)
                )
            ),
            "evaluation_count": int(getattr(args, "perturb_class_eval_count", 0)),
            "pool_size": int(getattr(args, "perturb_class_pool_size", 0)),
            "pool_multiplier": float(getattr(args, "perturb_class_pool_multiplier", 3.0)),
        },
        "class_imbalance": {
            "enabled": bool(getattr(args, "perturb_class_imbalance", False)),
            "strategy": str(getattr(args, "perturb_class_imbalance_strategy", "label")),
            "targets_raw": str(getattr(args, "perturb_class_imbalance_targets", "")),
            "balance": _parse_balance_value(
                str(getattr(args, "perturb_class_imbalance_balance", "0.5"))
            ),
            "kmeans_k": int(getattr(args, "perturb_class_imbalance_kmeans_k", 8)),
            "kmeans_cache_path": str(
                getattr(args, "perturb_class_imbalance_kmeans_cache_path", "")
            ),
            "kmeans_recreate": bool(
                getattr(args, "perturb_class_imbalance_kmeans_recreate", False)
            ),
            "seed": int(getattr(args, "perturb_class_imbalance_seed", 10)),
            "label_threshold": float(
                getattr(args, "perturb_class_imbalance_label_threshold", 0.0)
            ),
            "min_kept": int(getattr(args, "perturb_class_imbalance_min_kept", 4)),
            "out_dir": str(getattr(args, "out_dir", "")),
        },
        "sample_size": {
            "enabled": bool(getattr(args, "perturb_sample_size", False)),
            "n": int(getattr(args, "perturb_sample_size_n", 1000)),
            "seed": int(getattr(args, "perturb_sample_size_seed", 10)),
        },
        "preprocessing": {
            "enabled": bool(getattr(args, "perturb_preprocessing", False)),
            "variant": str(getattr(args, "perturb_preprocessing_variant", "downsample_bilinear")),
            "scale": float(getattr(args, "perturb_preprocessing_scale", 0.75)),
        },
        "domain_shift": {
            "enabled": bool(getattr(args, "perturb_domain_shift", False)),
            "dataset": str(getattr(args, "perturb_domain_shift_dataset", "")),
            "data_root": str(getattr(args, "perturb_domain_shift_data_root", "")),
            "image_size": int(getattr(args, "perturb_domain_shift_image_size", 0)),
        },
    }


def _parse_balance_value(raw: str) -> float | list[float]:
    tokens = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    if not tokens:
        return 0.5
    if len(tokens) == 1:
        return float(tokens[0])
    return [float(token) for token in tokens]


def _as_labeled_tensor_dataset(samples: torch.Tensor):
    count = int(samples.shape[0])
    labels = torch.zeros(count, dtype=torch.long)
    return TensorDataset(samples, labels)


def _dataset_to_tensor(dataset, expected_count: int):
    images: list[torch.Tensor] = []
    for idx in range(expected_count):
        image, _ = dataset[idx]
        images.append(image)
    if not images:
        return torch.empty(0)
    return torch.stack(images, dim=0)


def _apply_degradation(
    samples: torch.Tensor,
    severity: int,
    gaussian_noise: bool,
    gaussian_blur: bool,
    jpeg_compression: bool,
) :
    from .degrade_dataset import DegradationConfig, DegradedDataset

    config = DegradationConfig(
        severity=int(severity),
        gaussian_noise=bool(gaussian_noise),
        gaussian_blur=bool(gaussian_blur),
        jpeg_compression=bool(jpeg_compression),
    )
    wrapped = DegradedDataset(_as_labeled_tensor_dataset(samples), config)
    return _dataset_to_tensor(wrapped, expected_count=int(samples.shape[0]))


def _apply_memoisation(
    fake_samples: torch.Tensor,
    real_samples: torch.Tensor,
    fraction: float,
    seed: int,
):
    from .memorization_dataset import MemoisationConfig, MemoisationDataset

    config = MemoisationConfig(
        fraction=float(fraction),
        seed=int(seed),
    )
    expected_injected = int(int(fake_samples.shape[0]) * float(config.fraction))
    if expected_injected > int(real_samples.shape[0]):
        raise ValueError(
            "Memoisation perturbation requires enough real samples for replacement. "
            f"Need at least {expected_injected}, got {int(real_samples.shape[0])}."
        )

    wrapped = MemoisationDataset(
        generated_ds=_as_labeled_tensor_dataset(fake_samples),
        real_ds=_as_labeled_tensor_dataset(real_samples),
        config=config,
    )
    # i keep both lists because later we can rebuild fake feature rows without extracting again
    injected_positions = sorted(int(idx) for idx in wrapped.injected.keys())
    injected_real_indices = [int(wrapped.injected[idx]) for idx in injected_positions]
    details = {
        "fraction": float(config.fraction),
        "seed": int(config.seed),
        "total_fake_samples": int(fake_samples.shape[0]),
        "total_real_samples": int(real_samples.shape[0]),
        "expected_injected": int(expected_injected),
        "injected_count": int(len(injected_positions)),
        "injected_positions": injected_positions,
        "injected_real_indices": injected_real_indices,
    }
    return _dataset_to_tensor(wrapped, expected_count=int(fake_samples.shape[0])), details

def _apply_sample_size_variation(
    samples: torch.Tensor,
    n: int,
    seed: int,
) :
    total = int(samples.shape[0])

    if total == 0:
        raise ValueError("Sample-size perturbation received an empty tensor.")

    if int(n) <= 0:
        raise ValueError("Sample-size n must be > 0.")

    if int(n) > total:
        raise ValueError(
            f"Sample-size n cannot exceed available samples. "
            f"Requested n={int(n)}, available={total}."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    indices = torch.randperm(total, generator=generator)[: int(n)]
    selected_indices = [int(idx) for idx in indices.tolist()]
    return samples[indices], selected_indices


def _apply_preprocessing_variation(
    samples: torch.Tensor,
    variant: str,
    scale: float,
) :
    if samples.ndim != 4:
        raise ValueError("Preprocessing variation expects image tensors shaped [N, C, H, W].")

    output = samples.clone()
    height = int(samples.shape[-2])
    width = int(samples.shape[-1])
    down_h = max(1, int(round(height * float(scale))))
    down_w = max(1, int(round(width * float(scale))))

    if variant == "downsample_nearest":
        low = F.interpolate(output, size=(down_h, down_w), mode="nearest")
        return F.interpolate(low, size=(height, width), mode="nearest")

    if variant == "downsample_bilinear":
        low = F.interpolate(output, size=(down_h, down_w), mode="bilinear", align_corners=False)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)

    if variant == "downsample_bicubic":
        low = F.interpolate(output, size=(down_h, down_w), mode="bicubic", align_corners=False)
        return F.interpolate(low, size=(height, width), mode="bicubic", align_corners=False)

    if variant == "center_crop_pad":
        crop_h = max(1, int(round(height * float(scale))))
        crop_w = max(1, int(round(width * float(scale))))
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
        cropped = output[:, :, top : top + crop_h, left : left + crop_w]
        restored = torch.zeros_like(output)
        pad_top = max(0, (height - crop_h) // 2)
        pad_left = max(0, (width - crop_w) // 2)
        restored[:, :, pad_top : pad_top + crop_h, pad_left : pad_left + crop_w] = cropped
        return restored

    if variant == "grayscale_triplicate":
        if int(output.shape[1]) == 1:
            return output
        gray = output.mean(dim=1, keepdim=True)
        return gray.repeat(1, int(output.shape[1]), 1, 1)

    raise ValueError(f"Unknown preprocessing variation variant: {variant}")


def get_domain_shift_override(args: argparse.Namespace) -> dict[str, Any] | None:
    """
    Build domain-shift override for test scripts.

    The actual data-source switch happens in the test script because it owns
    real-reference dataset loading.
    """
    if not bool(getattr(args, "perturb_domain_shift", False)):
        return None

    dataset = str(getattr(args, "perturb_domain_shift_dataset", "")).strip()
    data_root = str(getattr(args, "perturb_domain_shift_data_root", "")).strip()
    image_size = int(getattr(args, "perturb_domain_shift_image_size", 0))
    if not dataset or not data_root:
        raise ValueError("Domain-shift perturbation needs dataset and data_root.")
    return {
        "dataset": dataset,
        "data_root": data_root,
        "image_size": image_size,
    }

def apply_configured_perturbations(
    fake_samples: torch.Tensor,
    args: argparse.Namespace,
    real_samples: torch.Tensor | None = None,
    reference_targets: torch.Tensor | None = None,
    reference_class_names: list[str] | None = None,
    dataset_name: str = "",
    runtime_context= None,
) :
    verbose = bool(getattr(args, "verbose", False))

    def _v(message: str) -> None:
        if verbose:
            print(f"[perturbation] {message}", flush=True)

    config = get_perturbation_config_dict(args)
    config["applied"] = []
    config["skipped"] = []

    if not config["enabled"]:
        _v("no perturbations enabled")
        return fake_samples, real_samples, config

    fake_out = fake_samples.detach().cpu()
    real_out = real_samples.detach().cpu() if real_samples is not None else None
    apply_to = str(config["apply_to"])
    apply_to_fake = apply_to in {"fake", "both"}
    apply_to_real = apply_to in {"real", "both"}

    _v(
        f"enabled={config['active_perturbations']} apply_to={apply_to} "
        f"fake_in={tuple(fake_out.shape)} real_in={tuple(real_out.shape) if real_out is not None else None}"
    )

    if config["degradation"]["enabled"]:
        if apply_to_fake:
            fake_out = _apply_degradation(
                samples=fake_out,
                severity=int(config["degradation"]["severity"]),
                gaussian_noise=bool(config["degradation"]["gaussian_noise"]),
                gaussian_blur=bool(config["degradation"]["gaussian_blur"]),
                jpeg_compression=bool(config["degradation"]["jpeg_compression"]),
            )
            config["applied"].append("degradation:fake")
            _v(f"applied degradation to fake -> shape={tuple(fake_out.shape)}")
        if apply_to_real:
            if real_out is None:
                raise ValueError("Real samples are required for perturb_apply_to='real' or 'both'.")
            real_out = _apply_degradation(
                samples=real_out,
                severity=int(config["degradation"]["severity"]),
                gaussian_noise=bool(config["degradation"]["gaussian_noise"]),
                gaussian_blur=bool(config["degradation"]["gaussian_blur"]),
                jpeg_compression=bool(config["degradation"]["jpeg_compression"]),
            )
            config["applied"].append("degradation:real")
            _v(f"applied degradation to real -> shape={tuple(real_out.shape)}")

    if config["memoisation"]["enabled"]:
        if not apply_to_fake:
            config["skipped"].append("memoisation skipped because perturb_apply_to excludes fake samples")
        else:
            if real_out is None:
                raise ValueError("Memoisation perturbation requires real samples.")
            fake_out, memoisation_details = _apply_memoisation(
                fake_samples=fake_out,
                real_samples=real_out,
                fraction=float(config["memoisation"]["fraction"]),
                seed=int(config["memoisation"]["seed"]),
            )
            config["memoisation"]["result"] = memoisation_details
            config["applied"].append("memoisation:fake")
            _v(f"applied memoisation to fake -> shape={tuple(fake_out.shape)}")

    if config["class_removal"]["enabled"]:
        if not apply_to_fake:
            config["skipped"].append("class_removal skipped because perturb_apply_to excludes fake samples")
        else:
            fake_out, class_removal_details = apply_class_removal(
                fake_samples=fake_out,
                config=config["class_removal"],
                real_samples=real_out,
                reference_targets=reference_targets,
                reference_class_names=reference_class_names,
                dataset_name=dataset_name,
                runtime_context=runtime_context,
            )
            config["class_removal"]["result"] = class_removal_details
            config["applied"].append("class_removal:fake")
            _v(
                "applied class_removal to fake -> "
                f"removed={class_removal_details.get('removed_count')} "
                f"survivors={class_removal_details.get('survivor_count', class_removal_details.get('kept_count'))} "
                f"eval={class_removal_details.get('evaluation_count', class_removal_details.get('returned_count'))}"
            )

    if config["class_imbalance"]["enabled"]:
        if not apply_to_fake:
            config["skipped"].append("class_imbalance skipped because perturb_apply_to excludes fake samples")
        else:
            fake_out, class_imbalance_details = apply_class_imbalance(
                fake_samples=fake_out,
                config=config["class_imbalance"],
                real_samples=real_out,
                reference_targets=reference_targets,
                reference_class_names=reference_class_names,
                dataset_name=dataset_name,
                runtime_context=runtime_context,
            )
            config["class_imbalance"]["result"] = class_imbalance_details
            config["applied"].append("class_imbalance:fake")
            _v(
                "applied class_imbalance to fake -> "
                f"removed={class_imbalance_details.get('removed_count')} "
                f"survivors={class_imbalance_details.get('survivor_count', class_imbalance_details.get('kept_count'))} "
                f"eval={class_imbalance_details.get('evaluation_count', class_imbalance_details.get('returned_count'))}"
            )

    if config["sample_size"]["enabled"]:
        n = int(config["sample_size"]["n"])
        seed = int(config["sample_size"]["seed"])
        # i store the exact picks so the metric cache can reuse baseline features safely
        sample_size_result: dict[str, Any] = {
            "n": n,
            "seed": seed,
        }

        if apply_to_fake:
            fake_out, selected_fake_indices = _apply_sample_size_variation(
                samples=fake_out,
                n=n,
                seed=seed,
            )
            sample_size_result["selected_indices_fake"] = selected_fake_indices
            config["applied"].append("sample_size:fake")
            _v(f"applied sample_size to fake n={n} -> shape={tuple(fake_out.shape)}")

        if apply_to_real:
            if real_out is None:
                raise ValueError("Real samples are required for sample-size perturbation on real data.")

            real_out, selected_real_indices = _apply_sample_size_variation(
                samples=real_out,
                n=n,
                seed=seed + 1,
            )
            sample_size_result["selected_indices_real"] = selected_real_indices
            config["applied"].append("sample_size:real")
            _v(f"applied sample_size to real n={n} -> shape={tuple(real_out.shape)}")

        config["sample_size"]["result"] = sample_size_result

    if config["preprocessing"]["enabled"]:
        variant = str(config["preprocessing"]["variant"])
        scale = float(config["preprocessing"]["scale"])
        if apply_to_fake:
            fake_out = _apply_preprocessing_variation(
                samples=fake_out,
                variant=variant,
                scale=scale,
            )
            config["applied"].append("preprocessing_variation:fake")
            _v(
                "applied preprocessing_variation to fake -> "
                f"variant={variant} scale={scale} shape={tuple(fake_out.shape)}"
            )
        if apply_to_real:
            if real_out is None:
                raise ValueError("Real samples are required for preprocessing variation on real data.")
            real_out = _apply_preprocessing_variation(
                samples=real_out,
                variant=variant,
                scale=scale,
            )
            config["applied"].append("preprocessing_variation:real")
            _v(
                "applied preprocessing_variation to real -> "
                f"variant={variant} scale={scale} shape={tuple(real_out.shape)}"
            )

    _v(
        f"done applied={config['applied']} skipped={config['skipped']} "
        f"fake_out={tuple(fake_out.shape)} real_out={tuple(real_out.shape) if real_out is not None else None}"
    )
    return fake_out, real_out, config

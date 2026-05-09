from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.pretrained_wrappers import StudioGANWrapper
from Perturbation.pipeline_perturbations import (
    add_perturbation_args,
    apply_configured_perturbations,
    get_domain_shift_override,
    perturbation_needs_real_reference,
    perturbation_needs_reference_targets,
    perturbations_enabled,
)
from Scripts.test_runtime_utils import (
    annotate_memoisation_effective_count,
    make_torch_generator,
    set_deterministic_seed,
)


# parse args
def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a pretrained StudioGAN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--repo-path", type=str, default="")
    parser.add_argument("--config-name", type=str, default="SNGAN.yaml")
    parser.add_argument("--out-dir", type=str, default="outputs/studiogan_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, compute metrics with Metrics/compute_all.py and save JSON report.",
    )
    parser.add_argument(
        "--metrics-dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10", "celeba", "chestxray14"],
        help="Dataset used as real reference distribution during metric evaluation.",
    )
    parser.add_argument(
        "--metrics-data-root",
        type=str,
        default="data/CIFAR10",
        help="Data root for metrics real samples.",
    )
    parser.add_argument("--metrics-image-size", type=int, default=32)
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument("--metrics-feature-space", type=str, default="inception_v3")
    parser.add_argument("--metrics-feature-batch-size", type=int, default=64)
    parser.add_argument("--metrics-feature-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--metrics-bootstrap-samples", type=int, default=0)
    parser.add_argument("--metrics-bootstrap-seed", type=int, default=10)
    parser.add_argument("--metrics-bootstrap-alpha", type=float, default=0.05)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow dataset download/setup fallback during metric evaluation if files are missing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors instead of skipping.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging for generation, perturbations, and metrics.",
    )
    add_perturbation_args(parser)
    return parser.parse_args()


# helper for dataset class names
def _dataset_class_names(dataset) -> list[str]:
    # This keeps label/attribute handling consistent for perturbation modules.
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return [str(name) for name in list(dataset.classes)]
    if hasattr(dataset, "attr_names") and dataset.attr_names is not None:
        return [str(name) for name in list(dataset.attr_names)]
    if hasattr(dataset, "finding_classes") and dataset.finding_classes is not None:
        return [str(name) for name in list(dataset.finding_classes)]
    return []


# load real reference
def _load_real_reference(
    dataset_name: str,
    data_root: str,
    image_size: int,
    sample_count: int,
    download_if_missing: bool,
    include_targets: bool,
    seed: int,
):
    from Datasets.unified_dataset_loader import make_default_loader

    loader = make_default_loader(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
    )

    try:
        dataset = loader.get_dataset(train=False, download=False)
    except (FileNotFoundError, RuntimeError):
        if not download_if_missing:
            raise
        dataset = loader.get_dataset(train=False, download=True)

    dataloader = DataLoader(
        dataset,
        batch_size=min(128, sample_count),
        shuffle=True,
        num_workers=0,
        generator=make_torch_generator(seed),
    )
    image_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    total = 0
    for images, targets in dataloader:
        image_batches.append(images)
        if include_targets:
            target_batches.append(torch.as_tensor(targets))
        total += int(images.shape[0])
        if total >= sample_count:
            break

    if not image_batches:
        raise ValueError("Could not load real samples for metrics.")
    result = {
        "samples": torch.cat(image_batches, dim=0)[:sample_count],
        "targets": None,
        "class_names": _dataset_class_names(dataset),
    }
    if include_targets:
        if not target_batches:
            raise ValueError("Could not load real sample labels for perturbation.")
        result["targets"] = torch.cat(target_batches, dim=0)[:sample_count]
    return result


# load real samples
def _load_real_samples(
    dataset_name: str,
    data_root: str,
    image_size: int,
    sample_count: int,
    download_if_missing: bool,
    seed: int,
):
    return _load_real_reference(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        sample_count=sample_count,
        download_if_missing=download_if_missing,
        include_targets=False,
        seed=seed,
    )["samples"]


# helper for resolve real reference request
def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    # Domain-shift perturbation swaps the real-reference domain used for metrics.
    override = get_domain_shift_override(args)
    if override is None:
        return args.metrics_dataset, args.metrics_data_root, default_image_size
    image_size = int(override["image_size"]) if int(override["image_size"]) > 0 else int(default_image_size)
    return str(override["dataset"]), str(override["data_root"]), image_size


# compute and save metrics
def _evaluate_and_save_metrics(
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    out_dir: Path,
    args: argparse.Namespace,
):
    from Metrics.compute_all import MetricComputationConfig, compute_all_metrics

    paired_count = min(int(real_samples.shape[0]), int(fake_samples.shape[0]))
    if paired_count < 4:
        raise ValueError("Need at least 4 paired real/fake samples to compute metrics robustly.")

    metric_device = args.metrics_feature_device
    if metric_device == "cuda" and not torch.cuda.is_available():
        metric_device = "cpu"

    # StudioGAN uses the same metrics backend as the other model test scripts.
    if args.verbose:
        print(
            f"[test_studiogan] evaluating metrics with paired_count={paired_count} "
            f"feature_space={args.metrics_feature_space} feature_device={metric_device}",
            flush=True,
        )
    results = compute_all_metrics(
        real_samples=real_samples[:paired_count],
        fake_samples=fake_samples[:paired_count],
        config=MetricComputationConfig(
            feature_space=args.metrics_feature_space,
            feature_batch_size=int(args.metrics_feature_batch_size),
            feature_device=metric_device,
            bootstrap_samples=int(args.metrics_bootstrap_samples),
            bootstrap_seed=int(args.metrics_bootstrap_seed),
            bootstrap_alpha=float(args.metrics_bootstrap_alpha),
            verbose=bool(args.verbose),
        ),
    )

    metrics_path = out_dir / "metrics_report.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metric report to {metrics_path}")
    return results


# build logger
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("studiogan")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# helper for generate in batches
def _generate_in_batches(
    wrapper: StudioGANWrapper,
    total_samples: int,
    batch_size: int,
    seed: int | None,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    remaining = total_samples
    offset = 0

    while remaining > 0:
        this_batch = min(batch_size, remaining)
        if seed is not None:
            torch.manual_seed(int(seed + offset))
        # StudioGAN wrapper handles internal class sampling and returns [-1,1] tensors.
        outputs.append(wrapper.sample(this_batch).detach().cpu())
        remaining -= this_batch
        offset += this_batch

    return torch.cat(outputs, dim=0)


# entry point when running this script
def main():
    args = parse_args()
    set_deterministic_seed(seed=int(args.seed), verbose=bool(args.verbose), context="test_studiogan")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        message = f"StudioGAN test skipped: checkpoint not found at {checkpoint}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return

    repo_path = Path(args.repo_path) if args.repo_path else checkpoint.parent / "studioGAN_src"

    try:
        wrapper = StudioGANWrapper(
            repo_path=repo_path,
            ckpt_path=checkpoint,
            config_name=args.config_name,
            device=("cuda" if args.cuda and torch.cuda.is_available() else "cpu"),
            logger=_build_logger(),
        )
        samples = _generate_in_batches(
            wrapper=wrapper,
            total_samples=max(1, int(args.num_samples)),
            batch_size=max(1, int(args.batch_size)),
            seed=args.seed,
        )
    except Exception as exc:
        print(f"StudioGAN test failed: {exc}")
        if args.strict:
            raise SystemExit(1)
        return

    if args.verbose:
        print(f"[test_studiogan] generated sample tensor shape={tuple(samples.shape)}", flush=True)

    perturbed_real_samples: torch.Tensor | None = None
    perturb_reference_targets: torch.Tensor | None = None
    perturb_reference_class_names: list[str] | None = None
    perturbation_info: dict | None = None
    if perturbations_enabled(args):
        needs_real_reference = perturbation_needs_real_reference(args)
        needs_reference_targets = perturbation_needs_reference_targets(args)
        if needs_real_reference:
            metric_image_size = args.metrics_image_size
            if int(samples.shape[-1]) == int(samples.shape[-2]):
                metric_image_size = int(samples.shape[-1])
            ref_dataset, ref_root, ref_image_size = _resolve_real_reference_request(args, metric_image_size)
            reference_bundle = _load_real_reference(
                dataset_name=ref_dataset,
                data_root=ref_root,
                image_size=ref_image_size,
                sample_count=max(int(samples.shape[0]), int(args.metrics_samples)),
                download_if_missing=args.metrics_download_if_missing,
                include_targets=needs_reference_targets,
                seed=int(args.seed),
            )
            perturbed_real_samples = reference_bundle["samples"]
            perturb_reference_targets = reference_bundle["targets"]
            perturb_reference_class_names = reference_bundle["class_names"]

        samples, perturbed_real_samples, perturbation_info = apply_configured_perturbations(
            fake_samples=samples,
            args=args,
            real_samples=perturbed_real_samples,
            reference_targets=perturb_reference_targets,
            reference_class_names=perturb_reference_class_names,
            dataset_name=args.metrics_dataset,
        )
        perturbation_path = out_dir / "perturbation_config.json"
        perturbation_path.write_text(json.dumps(perturbation_info, indent=2), encoding="utf-8")
        print(f"Saved perturbation config to {perturbation_path}")

    save_image(samples, out_dir / "generated_samples.png", nrow=8, normalize=True)
    print(f"Saved {samples.shape[0]} samples to {out_dir / 'generated_samples.png'}")

    if args.eval_metrics:
        try:
            metric_image_size = args.metrics_image_size
            if int(samples.shape[-1]) == int(samples.shape[-2]):
                metric_image_size = int(samples.shape[-1])

            real_samples = perturbed_real_samples
            if real_samples is None or int(real_samples.shape[0]) < int(args.metrics_samples):
                ref_dataset, ref_root, ref_image_size = _resolve_real_reference_request(args, metric_image_size)
                real_samples = _load_real_samples(
                    dataset_name=ref_dataset,
                    data_root=ref_root,
                    image_size=ref_image_size,
                    sample_count=args.metrics_samples,
                    download_if_missing=args.metrics_download_if_missing,
                    seed=int(args.seed),
                )
            else:
                real_samples = real_samples[: int(args.metrics_samples)]
            if perturbation_info is not None:
                evaluation_subset_size = min(int(real_samples.shape[0]), int(samples.shape[0]))
                perturbation_info = annotate_memoisation_effective_count(
                    perturbation_info=perturbation_info,
                    evaluation_subset_size=evaluation_subset_size,
                    verbose=bool(args.verbose),
                    context="test_studiogan",
                )
                perturbation_path = out_dir / "perturbation_config.json"
                perturbation_path.write_text(json.dumps(perturbation_info, indent=2), encoding="utf-8")
            metrics = _evaluate_and_save_metrics(
                real_samples=real_samples,
                fake_samples=samples,
                out_dir=out_dir,
                args=args,
            )
            print(f"Metric summary keys: {list(metrics.keys())}")
        except Exception as exc:
            if args.strict:
                raise
            print(f"Metric evaluation skipped/failed: {exc}")


if __name__ == "__main__":
    main()

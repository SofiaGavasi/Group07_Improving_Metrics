# example: py Scripts/test_stylegan2.py --checkpoint checkpoints/StyleGAN/CelebA/stylegan2_generator.pkl --out-dir outputs/stylegan2_test
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Models.pretrained_wrappers import StyleGAN2Wrapper
from Perturbation.pipeline_perturbations import (
    add_perturbation_args,
    apply_configured_perturbations,
    perturbation_needs_real_reference,
    perturbation_needs_reference_targets,
    perturbations_enabled,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a pretrained StyleGAN2 checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="outputs/stylegan2_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--noise-mode", type=str, default="const", choices=["const", "random", "none"])
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
        default="celeba",
        choices=["mnist", "cifar10", "celeba", "chestxray14"],
        help="Dataset used as real reference distribution during metric evaluation.",
    )
    parser.add_argument(
        "--metrics-data-root",
        type=str,
        default="data/CelebA",
        help="Data root for metrics real samples.",
    )
    parser.add_argument("--metrics-image-size", type=int, default=256)
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow dataset download/setup fallback during metric evaluation if files are missing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors (missing checkpoint/loading/sampling) instead of skipping.",
    )
    add_perturbation_args(parser)
    return parser.parse_args()


def _load_real_samples(
    dataset_name: str,
    data_root: str,
    image_size: int,
    sample_count: int,
    download_if_missing: bool,
):
    return _load_real_reference(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        sample_count=sample_count,
        download_if_missing=download_if_missing,
        include_targets=False,
    )["samples"]


def _dataset_class_names(dataset) -> list[str]:
    # checking common class-name fields used by our loaders/datasets
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return [str(name) for name in list(dataset.classes)]
    if hasattr(dataset, "attr_names") and dataset.attr_names is not None:
        return [str(name) for name in list(dataset.attr_names)]
    if hasattr(dataset, "finding_classes") and dataset.finding_classes is not None:
        return [str(name) for name in list(dataset.finding_classes)]
    return []


def _load_real_reference(
    dataset_name: str,
    data_root: str,
    image_size: int,
    sample_count: int,
    download_if_missing: bool,
    include_targets: bool,
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


def _to_feature_matrix(samples: torch.Tensor) -> np.ndarray:
    return (
        samples.detach()
        .cpu()
        .float()
        .reshape(samples.shape[0], -1)
        .numpy()
        .astype(np.float64, copy=False)
    )


def _evaluate_and_save_metrics(
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    out_dir: Path,
):
    from Metrics.compute_all import compute_all_metrics

    paired_count = min(int(real_samples.shape[0]), int(fake_samples.shape[0]))
    if paired_count < 4:
        raise ValueError("Need at least 4 paired real/fake samples to compute metrics robustly.")

    real_features = _to_feature_matrix(real_samples[:paired_count])
    fake_features = _to_feature_matrix(fake_samples[:paired_count])

    results = compute_all_metrics(real_features, fake_features)
    metrics_path = out_dir / "metrics_report.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metric report to {metrics_path}")
    return results


def _generate_in_batches(
    wrapper: StyleGAN2Wrapper,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    class_idx: int | None,
    seed: int | None,
) :
    samples: list[torch.Tensor] = []
    remaining = total_samples
    offset = 0

    while remaining > 0:
        this_batch = min(batch_size, remaining)
        batch_seed = None if seed is None else int(seed + offset)
        generated = wrapper.sample(
            this_batch,
            device=device,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            class_idx=class_idx,
            seed=batch_seed,
        )
        samples.append(generated)
        remaining -= this_batch
        offset += this_batch

    return torch.cat(samples, dim=0)


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        message = f"StyleGAN2 test skipped: checkpoint not found at {checkpoint}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return

    try:
        wrapper = StyleGAN2Wrapper(checkpoint_path=str(checkpoint))
        samples = _generate_in_batches(
            wrapper=wrapper,
            total_samples=args.num_samples,
            batch_size=max(1, args.batch_size),
            device=device,
            truncation_psi=args.truncation_psi,
            noise_mode=args.noise_mode,
            class_idx=args.class_idx,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"StyleGAN2 test failed: {exc}")
        if args.strict:
            raise SystemExit(1)
        return

    output_path = Path(args.out_dir) / "generated_samples.png"

    perturbed_real_samples: torch.Tensor | None = None
    perturb_reference_targets: torch.Tensor | None = None
    perturb_reference_class_names: list[str] | None = None
    if perturbations_enabled(args):
        needs_real_reference = perturbation_needs_real_reference(args)
        needs_reference_targets = perturbation_needs_reference_targets(args)
        if needs_real_reference:
            metric_image_size = args.metrics_image_size
            if int(samples.shape[-1]) == int(samples.shape[-2]):
                metric_image_size = int(samples.shape[-1])
            reference_bundle = _load_real_reference(
                dataset_name=args.metrics_dataset,
                data_root=args.metrics_data_root,
                image_size=metric_image_size,
                sample_count=max(int(samples.shape[0]), int(args.metrics_samples)),
                download_if_missing=args.metrics_download_if_missing,
                include_targets=needs_reference_targets,
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
        perturbation_path = Path(args.out_dir) / "perturbation_config.json"
        perturbation_path.write_text(json.dumps(perturbation_info, indent=2), encoding="utf-8")
        print(f"Saved perturbation config to {perturbation_path}")

    save_image(samples, output_path, nrow=8, normalize=True)
    print(f"Saved {samples.shape[0]} samples to {output_path}")

    if args.eval_metrics:
        try:
            metric_image_size = args.metrics_image_size
            if int(samples.shape[-1]) == int(samples.shape[-2]):
                metric_image_size = int(samples.shape[-1])

            real_samples = perturbed_real_samples
            if real_samples is None or int(real_samples.shape[0]) < int(args.metrics_samples):
                real_samples = _load_real_samples(
                    dataset_name=args.metrics_dataset,
                    data_root=args.metrics_data_root,
                    image_size=metric_image_size,
                    sample_count=args.metrics_samples,
                    download_if_missing=args.metrics_download_if_missing,
                )
            else:
                real_samples = real_samples[: int(args.metrics_samples)]
            metrics = _evaluate_and_save_metrics(
                real_samples=real_samples,
                fake_samples=samples,
                out_dir=Path(args.out_dir),
            )
            print(f"Metric summary keys: {list(metrics.keys())}")
        except Exception as exc:
            if args.strict:
                raise
            print(f"Metric evaluation skipped/failed: {exc}")


if __name__ == "__main__":
    main()

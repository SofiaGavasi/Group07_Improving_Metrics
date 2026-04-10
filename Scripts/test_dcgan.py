# example: py Scripts/test_dcgan.py --netG outputs/dcgan_cifar10/netG_latest.pth --out-dir outputs/dcgan_test --num-samples 64 --image-size 32 --channels 3
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

from Datasets.unified_dataset_loader import make_default_loader
from Models.dcgan import DCGANGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained DCGAN generator.")
    parser.add_argument("--netG", type=str, required=True, help="Path to trained generator checkpoint.")
    parser.add_argument("--out-dir", type=str, default="outputs/dcgan_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true"
    )
    parser.add_argument(
        "--eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--metrics-dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10"]
    )
    parser.add_argument(
        "--metrics-data-root",
        type=str,
        default="data/CIFAR10"
    )
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    return parser.parse_args()


def _load_real_samples(
    dataset_name: str,
    data_root: str,
    image_size: int,
    sample_count: int,
    download_if_missing: bool,
):
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
    batches: list[torch.Tensor] = []
    total = 0
    for images, _ in dataloader:
        batches.append(images)
        total += int(images.shape[0])
        if total >= sample_count:
            break

    if not batches:
        raise ValueError("Could not load real samples for metrics.")
    return torch.cat(batches, dim=0)[:sample_count]


def _to_feature_matrix(samples: torch.Tensor) -> np.ndarray:
    # current metrics API expects 2D features, so flatten image tensors to vectors.
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


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    netg_path = Path(args.netG)

    # for pipeline runs we keep non-strict as default, so missing checkpoints skip cleanly
    if not netg_path.exists():
        message = f"DCGAN test skipped: checkpoint not found at {netg_path}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    # architecture args must match the checkpoint training config
    generator = DCGANGenerator(
        ngpu=0,
        nc=args.channels,
        nz=args.latent_dim,
        ngf=args.ngf,
        image_size=args.image_size,
    ).to(device)
    generator.load_state_dict(torch.load(netg_path, map_location=device))
    generator.eval()

    generated = []
    remaining = args.num_samples
    with torch.no_grad():
        # generate in chunks to control memory usage for large num_samples
        while remaining > 0:
            this_batch = min(args.batch_size, remaining)
            z = torch.randn(this_batch, args.latent_dim, 1, 1, device=device)
            generated.append(generator(z).detach().cpu())
            remaining -= this_batch

    samples = torch.cat(generated, dim=0)
    # quick visual sanity check grid
    save_image(samples, out_dir / "generated_samples.png", nrow=8, normalize=True)
    print(f"Saved {samples.shape[0]} samples to {out_dir / 'generated_samples.png'}")

    if args.eval_metrics:
        try:
            real_samples = _load_real_samples(
                dataset_name=args.metrics_dataset,
                data_root=args.metrics_data_root,
                image_size=args.image_size,
                sample_count=args.metrics_samples,
                download_if_missing=args.metrics_download_if_missing,
            )
            metrics = _evaluate_and_save_metrics(
                real_samples=real_samples,
                fake_samples=samples,
                out_dir=out_dir,
            )
            print(f"Metric summary keys: {list(metrics.keys())}")
        except Exception as exc:
            if args.strict:
                raise
            print(f"Metric evaluation skipped/failed: {exc}")


if __name__ == "__main__":
    main()

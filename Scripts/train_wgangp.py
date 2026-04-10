# Example: py Scripts/train_wgangp.py --dataset cifar10 --data-root data/CIFAR10 --epochs 1 --batch-size 64 --image-size 32 --out-dir outputs/wgangp_cifar10 --subset-fraction 0.2 --subset-strategy random
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP on MNIST/CIFAR-10.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="outputs/wgangp")
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--subset-max-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--subset-strategy",
        type=str,
        default="random",
        choices=["random", "class_balanced"],
    )
    parser.add_argument("--subset-include-classes", type=str, default="")
    parser.add_argument("--subset-drop-classes", type=str, default="")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # TODO: instantiate WGANGPGenerator/WGANGPCritic from Models/wgangp.py
    # TODO: build dataloaders with Datasets/unified_dataset_loader.py (subset args already parsed)
    # TODO: implement full training loop (critic iterations, gradient penalty, checkpointing)
    # TODO: add evaluation hooks to call Metrics/compute_all.py on generated samples
    print(
        "Subset settings:",
        {
            "fraction": args.subset_fraction,
            "max_samples": args.subset_max_samples,
            "seed": args.subset_seed,
            "strategy": args.subset_strategy,
            "include_classes": args.subset_include_classes,
            "drop_classes": args.subset_drop_classes,
        },
    )
    print("WGAN-GP training skeleton is in place; TODO sections remain.")


if __name__ == "__main__":
    main()

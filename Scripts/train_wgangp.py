# Example: py Scripts/train_wgangp.py --dataset cifar10 --data-root data/CIFAR10 --epochs 1 --batch-size 64 --image-size 32 --out-dir outputs/wgangp_cifar10
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
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # TODO: instantiate WGANGPGenerator/WGANGPCritic from Models/wgangp.py
    # TODO: build dataloaders with Datasets/unified_dataset_loader.py
    # TODO: implement full training loop (critic iterations, gradient penalty, checkpointing)
    # TODO: add evaluation hooks to call Metrics/compute_all.py on generated samples
    print("WGAN-GP training skeleton is in place; TODO sections remain.")


if __name__ == "__main__":
    main()

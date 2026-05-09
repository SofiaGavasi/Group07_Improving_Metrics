# example: py Scripts/download_preprocess_mnist_cifar10.py --data-root data

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


# entry point when running this script
def main():
    parser = argparse.ArgumentParser(description="Download MNIST + CIFAR-10.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root folder.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    datasets.MNIST(root=str(data_root / "MNIST"), train=True, download=True)
    datasets.MNIST(root=str(data_root / "MNIST"), train=False, download=True)

    datasets.CIFAR10(root=str(data_root / "CIFAR10"), train=True, download=True)
    datasets.CIFAR10(root=str(data_root / "CIFAR10"), train=False, download=True)

    # TODO: add preprocessing/export steps (cached tensors, stats, splits)

    print("MNIST and CIFAR-10 download/setup completed.")


if __name__ == "__main__":
    main()

# example: py Scripts/train_dcgan.py --dataset cifar10 --data-root data/CIFAR10 --epochs 1 --batch-size 64 --image-size 32 --outf outputs/dcgan_cifar10
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST/CIFAR-10.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--outf", type=str, default="outputs/dcgan")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    Path(args.outf).mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    if args.dataset == "mnist":
        raise NotImplementedError(
            "TODO: update Models/dcgan.py dataset pipeline for MNIST training."
        )

    cmd = [
        "python",
        str(repo_root / "Models" / "dcgan.py"),
        "--dataset",
        args.dataset,
        "--dataroot",
        args.data_root,
        "--niter",
        str(args.epochs),
        "--batchSize",
        str(args.batch_size),
        "--imageSize",
        str(args.image_size),
        "--outf",
        args.outf,
    ]
    if args.cuda:
        cmd.append("--cuda")

    subprocess.run(cmd, check=True)
    # TODO: add logging hooks and standardized checkpoint names.


if __name__ == "__main__":
    main()

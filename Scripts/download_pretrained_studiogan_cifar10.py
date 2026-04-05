# example: py Scripts/download_pretrained_studiogan_cifar10.py --output-dir checkpoints/StudioGAN/CIFAR10

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stage StudioGAN checkpoints for CIFAR-10.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/StudioGAN/CIFAR10")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint directory ready: {output_dir}")
    


if __name__ == "__main__":
    main()

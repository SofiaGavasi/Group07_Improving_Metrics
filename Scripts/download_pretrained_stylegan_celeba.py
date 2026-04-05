# example: py Scripts/download_pretrained_stylegan_celeba.py --output-dir checkpoints/StyleGAN/CelebA
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stage StyleGAN/StyleGAN2 checkpoints for CelebA.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/StyleGAN/CelebA")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint directory ready: {output_dir}")


if __name__ == "__main__":
    main()

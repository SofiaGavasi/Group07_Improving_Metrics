# example: py Scripts/download_preprocess_celeba.py --data-root data


from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description="Download CelebA.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root folder.")
    args = parser.parse_args()

    data_root = Path(args.data_root) / "CelebA"
    data_root.mkdir(parents=True, exist_ok=True)

    # TODO: decide whether to use aligned/unaligned variants and exact split policy.

    datasets.CelebA(root=str(data_root), split="train", target_type="attr", download=True)
    datasets.CelebA(root=str(data_root), split="valid", target_type="attr", download=True)
    datasets.CelebA(root=str(data_root), split="test", target_type="attr", download=True)
    print("CelebA download/setup completed.")


if __name__ == "__main__":
    main()

# example: py Scripts/download_preprocess_chestxray14.py --data-root data

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare ChestX-ray14 directory structure.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root folder.")
    args = parser.parse_args()

    # TODO: implement actual download/preprocessing steps. 
    
    chest_root = Path(args.data_root) / "ChestXray14"
    chest_root.mkdir(parents=True, exist_ok=True)
    (chest_root / "images").mkdir(parents=True, exist_ok=True)
    (chest_root / "metadata").mkdir(parents=True, exist_ok=True)

    print(f"Created placeholder structure under: {chest_root}")


if __name__ == "__main__":
    main()

# example: py Scripts/download_preprocess_chestxray14.py --data-root data

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datasets.chestxray14_dataset import prepare_chestxray14_dataset


# entry point when running this script
def main():
    parser = argparse.ArgumentParser(description="Download and index ChestX-ray14.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--no-download",
        action="store_true"
    )
    args = parser.parse_args()

    index_path = prepare_chestxray14_dataset(
        data_root=args.data_root,
        download=not args.no_download,
    )

    print(f"ChestX-ray14 setup completed. Index created at: {index_path}")


if __name__ == "__main__":
    main()

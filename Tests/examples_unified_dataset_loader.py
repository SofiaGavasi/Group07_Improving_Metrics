"""
examples for calling UnifiedDatasetLoader.
See at the bottom of this file to comment/uncomment different dataset examples
Run manually from repo root:
    py Tests/examples_unified_dataset_loader.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datasets.unified_dataset_loader import DatasetConfig, UnifiedDatasetLoader, make_default_loader


def example_mnist():
    loader = make_default_loader(
        dataset_name="mnist",
        data_root="data/MNIST",
        image_size=32,
    )
    train_ds = loader.get_dataset(train=True, download=True)
    test_ds = loader.get_dataset(train=False, download=True)
    print("MNIST:", len(train_ds), len(test_ds))


def example_cifar10():
    config = DatasetConfig(
        name="cifar10",
        data_root="data/CIFAR10",
        image_size=32,
        normalize_to_neg_one_one=True,
    )
    loader = UnifiedDatasetLoader(config)
    train_ds = loader.get_dataset(train=True, download=True)
    test_ds = loader.get_dataset(train=False, download=True)
    print("CIFAR10:", len(train_ds), len(test_ds))


def example_celeba():
    loader = make_default_loader(
        dataset_name="celeba",
        data_root="data/CelebA",
        image_size=64,
    )
    train_ds = loader.get_dataset(train=True, download=True)
    test_ds = loader.get_dataset(train=False, download=True)
    print("CelebA:", len(train_ds), len(test_ds))


def example_chestxray14():
    # this will not work yet, see TODO
    loader = make_default_loader(
        dataset_name="chestxray14",
        data_root="data/ChestXray14",
        image_size=64,
    )
    ds = loader.get_dataset(train=True, download=False)
    print("ChestX-ray14:", len(ds))


if __name__ == "__main__":
    
     example_mnist()
    # example_cifar10()
    # example_celeba()
    # example_chestxray14()


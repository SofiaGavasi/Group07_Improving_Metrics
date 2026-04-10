from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torchvision import datasets, transforms

from .dataset_subset import DatasetSubsetConfig, apply_dataset_subset

# here we define the loader settings. name will be the name string (see below). data_root is the path to the dataset. 
# image_size is the size to which all images will be resized. normalize_to_neg_one_one indicates whether to normalize pixel values to [-1, 1] range.
@dataclass
class DatasetConfig: 
    name: str
    data_root: str
    image_size: int = 32
    normalize_to_neg_one_one: bool = True
    subset_config: Optional[DatasetSubsetConfig] = None


class UnifiedDatasetLoader:
    SUPPORTED_DATASETS = {"mnist", "cifar10", "celeba", "chestxray14"}

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_name = config.name.lower()
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError("Unsupported dataset")
    

    def get_dataset(self, train = True, download= False):
        # here we build the transform pipeline based on the configuration.
        transform = self.build_transform()
        dataset = None

        if self.dataset_name == "mnist":
            dataset = datasets.MNIST(
                root=self.config.data_root,
                train=train,
                transform=transform,
                download=download,
            )

        elif self.dataset_name == "cifar10":
            dataset = datasets.CIFAR10(
                root=self.config.data_root,
                train=train,
                transform=transform,
                download=download,
            )

        elif self.dataset_name == "celeba":
            split = "train" if train else "test"
            dataset = datasets.CelebA(
                root=self.config.data_root,
                split=split,
                target_type="attr",
                transform=transform,
                download=download,
            )
        
        elif self.dataset_name == "chestxray14":
            from .chestxray14_dataset import ChestXray14Dataset

            split = "train" if train else "test"
            dataset = ChestXray14Dataset(
                root=self.config.data_root,
                split=split,
                transform=transform,
                download=download,
            )

        if dataset is None:
            raise ValueError("Unsupported dataset")

        return apply_dataset_subset(
            dataset,
            config=self.config.subset_config,
            dataset_name=self.dataset_name,
        )






    def build_transform(self): 
        # SHARED PREPROCESSING STEPS: resize and convert to tensor. if normalize_to_neg_one_one is true, also add normalization to [-1, 1] range.
        base = [
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
        ]

        if not self.config.normalize_to_neg_one_one:
            return transforms.Compose(base)

        if self.dataset_name == "mnist": #1 channel
            base.append(transforms.Normalize((0.5,), (0.5,)))
        else:#3 channels
            base.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(base)






def make_default_loader(dataset_name, data_root, image_size=None, subset_config=None):
    size = 32 if image_size is None else image_size
    config = DatasetConfig(
        name=dataset_name,
        data_root=data_root,
        image_size=size,
        subset_config=subset_config,
    )
    return UnifiedDatasetLoader(config)

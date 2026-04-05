from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from torchvision import datasets, transforms

# here we define the loader settings. name will be the name string (see below). data_root is the path to the dataset. 
# image_size is the size to which all images will be resized. normalize_to_neg_one_one indicates whether to normalize pixel values to [-1, 1] range.
@dataclass
class DatasetConfig: 
    name: str
    data_root: str
    image_size: int = 32
    normalize_to_neg_one_one: bool = True


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

        if self.dataset_name == "mnist":
            return datasets.MNIST(
                root=self.config.data_root,
                train=train,
                transform=transform,
                download=download,
            )

        if self.dataset_name == "cifar10":
            return datasets.CIFAR10(
                root=self.config.data_root,
                train=train,
                transform=transform,
                download=download,
            )

        if self.dataset_name == "celeba":
            split = "train" if train else "test"
            return datasets.CelebA(
                root=self.config.data_root,
                split=split,
                target_type="attr",
                transform=transform,
                download=download,
            )
        
        if self.dataset_name == "chestxray14":
            #TODO: this is a placeholder, we should replace it with the actual path to the chest x-ray dataset
            #  the dataset should be organized in a way that ImageFolder can read 
            chest_root = Path(self.config.data_root) / "chestxray14"






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






def make_default_loader(  dataset_name, data_root, image_size= None,):
    size = 32 if image_size is None else image_size
    config = DatasetConfig(name=dataset_name, data_root=data_root, image_size=size)
    return UnifiedDatasetLoader(config)

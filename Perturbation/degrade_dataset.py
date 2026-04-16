# Perturbation/degraded_dataset.py

from __future__ import annotations

import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class DegradationConfig:
    severity: int = 1              # 1 (mild) → 5 (heavy)
    gaussian_noise: bool = False
    gaussian_blur: bool = False
    jpeg_compression: bool = False


class DegradedDataset(Dataset):
    """
    A transparent wrapper around any existing PyTorch Dataset.
    Applies controlled image degradations without touching the base dataset.

    Expects the base dataset to return (tensor, label) where tensor is
    in [-1, 1] range with shape (C, H, W).

    Usage:
        train_ds = unified_loader.get_dataset(train=True)
        train_ds = DegradedDataset(train_ds, DegradationConfig(severity=3, gaussian_noise=True))
    """

    _NOISE_SIGMAS   = [2,  15, 25, 40, 60  ]
    _BLUR_KERNELS   = [2,  5,  7,  9,  11  ]
    _BLUR_SIGMAS    = [0.5,1.0,2.0,3.0,5.0 ]
    _JPEG_QUALITIES = [80, 60, 40, 20, 5   ]

    def __init__(self, dataset: Dataset, config: DegradationConfig):
        self.dataset  = dataset
        self.config   = config
        self._transform = self._build_transform()

    def _build_transform(self) -> A.Compose | None:
        s   = max(0, min(self.config.severity - 1, 4))  # clamp to 0-4
        aug = []

        if self.config.gaussian_noise:
            sigma = self._NOISE_SIGMAS[s]
            aug.append(A.GaussNoise(var_limit=(sigma ** 2, sigma ** 2), p=1.0))

        if self.config.gaussian_blur:
            k = self._BLUR_KERNELS[s]
            aug.append(A.GaussianBlur(blur_limit=(k, k), sigma_limit=self._BLUR_SIGMAS[s], p=1.0))

        if self.config.jpeg_compression:
            q = self._JPEG_QUALITIES[s]
            aug.append(A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0))

        return A.Compose(aug) if aug else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]

        if self._transform is None:
            return image, label

        # [-1,1] tensor (C,H,W) → [0,255] numpy (H,W,C)
        img_np = ((image.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        # apply degradations
        img_np = self._transform(image=img_np)["image"]

        # [0,255] numpy (H,W,C) → [-1,1] tensor (C,H,W)
        image = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0

        return image, label




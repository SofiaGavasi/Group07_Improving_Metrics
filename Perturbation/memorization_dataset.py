# Perturbation/memorization_dataset.py

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class MemoisationConfig:
    fraction: float = 0.1   # fraction of generated samples to replace with real ones
    seed: int = 10


class MemoisationDataset(Dataset):
    """
    Replaces a fraction of generated samples with real training images.
    Simulates memorisation / overfitting behaviour.

    Usage:
        mem_ds = MemoisationDataset(generated_ds, real_ds, MemoisationConfig(fraction=0.2))
    """

    def __init__(
        self,
        generated_ds: Dataset,   # GAN outputs
        real_ds: Dataset,        # real training images
        config: MemoisationConfig,
    ):
        self.generated_ds = generated_ds
        self.real_ds      = real_ds
        self.config       = config
        self.injected     = self._build_index()

    def _build_index(self):
        rng        = np.random.default_rng(self.config.seed)
        n_total    = len(self.generated_ds)
        n_inject   = int(n_total * self.config.fraction)

        # which positions in generated_ds get replaced
        inject_pos = set(rng.choice(n_total, size=n_inject, replace=False).tolist())

        # which real samples to inject
        real_indices = rng.choice(len(self.real_ds), size=n_inject, replace=False).tolist()

        # map generated_idx → real_idx for injected positions
        inject_map = {}
        for pos, real_idx in zip(sorted(inject_pos), real_indices):
            inject_map[pos] = real_idx

        return inject_map

    def __len__(self):
        return len(self.generated_ds)

    def __getitem__(self, idx):
        if idx in self.injected:
            return self.real_ds[self.injected[idx]]
        return self.generated_ds[idx]
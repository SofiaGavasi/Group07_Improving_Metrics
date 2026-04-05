from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class StudioGANWrapper:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        # TODO: load StudioGAN generator from checkpoint

    def sample(self, n: int, device: torch.device, **kwargs: Any):
        # TODO: implement StudioGAN sampling
        raise NotImplementedError("TODO: StudioGANWrapper.sample")


class DDPMWrapper:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        # TODO: load DDPM model from checkpoint
    def sample(self, n: int, device: torch.device, **kwargs: Any):
        # TODO: add DDIM sampling path
        raise NotImplementedError("TODO: DDPMWrapper.sample")


class StyleGAN2Wrapper:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        # TODO: load StyleGAN/StyleGAN2 generator from checkpoint

    def sample(self, n: int, device: torch.device, **kwargs: Any) :
        # TODO: implement StyleGAN/StyleGAN2 sampling
        raise NotImplementedError("TODO: StyleGAN2Wrapper.sample")

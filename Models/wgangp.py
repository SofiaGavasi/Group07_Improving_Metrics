from __future__ import annotations

import torch
import torch.nn as nn


class WGANGPGenerator(nn.Module):
    """
    TODO: Replace with final WGAN-GP generator architecture
    """

    def __init__(self, latent_dim: int = 100, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.main = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: implement generator forward pass.
        raise NotImplementedError("TODO: implement WGANGPGenerator.forward")


class WGANGPCritic(nn.Module):
    """
    TODO: Replace with final WGAN-GP critic architecture
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.main = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement critic forward pass
        raise NotImplementedError("TODO: implement WGANGPCritic.forward")


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    TODO: implement WGAN-GP gradient penalty
    """
    raise NotImplementedError("TODO: implement gradient_penalty")

from __future__ import annotations

import torch
import torch.nn as nn


class WGANGPGenerator(nn.Module):
    """DCGAN-style generator used for WGAN-GP experiments."""

    def __init__(self, latent_dim: int = 100, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError("WGANGPGenerator expects latent tensors shaped [N, Z, 1, 1].")
        return self.main(z)


class WGANGPCritic(nn.Module):
    """Convolutional critic used for WGAN-GP experiments."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("WGANGPCritic expects image tensors shaped [N, C, H, W].")
        return self.main(x).view(x.size(0), -1).mean(dim=1)


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty from interpolated samples."""
    batch_size = real.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)

    critic_scores = critic(interpolated)
    grad_outputs = torch.ones_like(critic_scores, device=device)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1.0) ** 2).mean()

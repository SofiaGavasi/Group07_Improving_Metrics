from __future__ import annotations

import math

import torch
import torch.nn as nn


def _validate_image_size(image_size):
    # this architecture upsamples/downsampes by factor 2 repeatedly
    # so image size must be a power of 2 and at least 32
    if image_size < 32 or (image_size & (image_size - 1)) != 0:
        raise ValueError("image_size must be a power of 2 and >= 32.")


class DCGANGenerator(nn.Module):
    def __init__(self, ngpu: int = 1, nc: int = 3, nz: int = 100, ngf: int = 64, image_size: int = 32):
        super().__init__()
        _validate_image_size(image_size)
        self.ngpu = ngpu
        self.nz = nz
        self.image_size = image_size

        layers: list[nn.Module] = [
            # start from latent vector z and project to a small spatial feature map
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        ]

        current_channels = ngf * 8
        # number of x2 upsampling blocks needed to reach target image size
        upsample_blocks = int(math.log2(image_size)) - 2
        for _ in range(upsample_blocks):
            next_channels = max(ngf, current_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True),
                ]
            )
            current_channels = next_channels

        layers.extend(
            [
                # final conv maps features to image channels (mnist=1, cifar=3, etc.)
                nn.Conv2d(current_channels, nc, kernel_size=3, stride=1, padding=1, bias=False),
                # tanh keeps outputs in [-1, 1], matching standard GAN preprocessing
                nn.Tanh(),
            ]
        )
        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, ngpu: int = 1, nc: int = 3, ndf: int = 64, image_size: int = 32):
        super().__init__()
        _validate_image_size(image_size)
        self.ngpu = ngpu

        layers: list[nn.Module] = [
            # first block does not use batchnorm by DCGAN convention
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        current_channels = ndf
        current_size = image_size // 2
        while current_size > 4:
            # progressively downsample until we reach 4x4 spatial map
            next_channels = min(current_channels * 2, ndf * 8)
            layers.extend(
                [
                    nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_channels = next_channels
            current_size //= 2

        layers.extend(
            [
                # collapse final 4x4 features to a single real/fake probability
                nn.Conv2d(current_channels, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            ]
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        # return shape [batch] so BCELoss can compare directly with label vector
        return logits.view(-1, 1).squeeze(1)


def dcgan_weights_init(module):
    # standard DCGAN init: conv ~ N(0, 0.02), bn gamma ~ N(1, 0.02), bn beta = 0
    class_name = module.__class__.__name__
    if "Conv" in class_name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


# Backward-compatible aliases used by existing imports/tests
Generator = DCGANGenerator
Discriminator = DCGANDiscriminator

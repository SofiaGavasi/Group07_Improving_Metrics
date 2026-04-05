from __future__ import annotations

from typing import Any, Optional

import torch


def generate_samples(
    model: Any,
    n: int,
    device: Optional[torch.device] = None,
    latent_dim: int = 100,
    return_01_range: bool = False,
    **kwargs: Any,
):
    """
    Unified interface to sample from GANs/diffusion wrappers
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(model, "sample"):
        samples = model.sample(n, device=device, **kwargs)
        return _postprocess(samples, return_01_range=return_01_range)

    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
        with torch.no_grad():
            z = torch.randn(n, latent_dim, 1, 1, device=device)
            samples = model(z)
        return _postprocess(samples, return_01_range=return_01_range)

    raise TypeError("Unsupported model type for generate_samples")


def _postprocess(samples: torch.Tensor, return_01_range: bool):
    if not return_01_range:
        return samples
    #  GAN output range conversion: [-1, 1] -> [0, 1].
    return (samples + 1.0) / 2.0

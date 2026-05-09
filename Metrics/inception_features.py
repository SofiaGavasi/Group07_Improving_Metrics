from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import Inception_V3_Weights, inception_v3


# class for InceptionFeatureConfig
@dataclass
class InceptionFeatureConfig:
    """Configuration for Inception-v3 feature/probability extraction."""

    batch_size: int = 64
    device: str = "cpu"
    num_workers: int = 0


# helper for normalize to unit range
def _normalize_to_unit_range(images: torch.Tensor) -> torch.Tensor:
    # The pipeline stores tensors in [-1, 1], but metric backbones expect [0, 1].
    if images.numel() == 0:
        return images
    min_value = float(images.min().item())
    max_value = float(images.max().item())
    if min_value < 0.0 or max_value > 1.0:
        images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


# helper for ensure three channels
def _ensure_three_channels(images: torch.Tensor) -> torch.Tensor:
    channels = int(images.shape[1])
    if channels == 3:
        return images
    if channels == 1:
        return images.repeat(1, 3, 1, 1)
    raise ValueError(f"Unsupported channel count for Inception features: {channels}")


# helper for preprocess for inception
def _preprocess_for_inception(images: torch.Tensor) -> torch.Tensor:
    # This mirrors common FID/IS preprocessing: RGB, 299x299, ImageNet normalization.
    images = _normalize_to_unit_range(images)
    images = _ensure_three_channels(images)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# class for InceptionFeatureExtractor
class InceptionFeatureExtractor:
    """Extract pool3 features and class probabilities with pretrained Inception-v3."""

    # helper for init
    def __init__(self, config: InceptionFeatureConfig | None = None):
        self.config = config or InceptionFeatureConfig()
        self.device = torch.device(self.config.device)
        # Redirect torchvision weight cache to workspace-writable directory.
        torch.hub.set_dir(str((Path.cwd() / ".torch_cache").resolve()))
        try:
            self.model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                aux_logits=True,
                transform_input=False,
            ).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained Inception-v3 weights. "
                "Ensure internet access for first download or pre-stage weights in .torch_cache/checkpoints."
            ) from exc
        self.model.eval()
        self._features_buffer: torch.Tensor | None = None

        # Hook avgpool to collect pool3 vectors (2048D).
        def _capture_avgpool(_: Any, __: Any, output: torch.Tensor):
            self._features_buffer = output.detach()

        self._hook = self.model.avgpool.register_forward_hook(_capture_avgpool)

    # helper for close
    def close(self) -> None:
        self._hook.remove()

    # helper for extract
    def extract(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = TensorDataset(images.detach().cpu())
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.config.batch_size)),
            shuffle=False,
            num_workers=max(0, int(self.config.num_workers)),
        )

        all_features: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=True).float()
                batch = _preprocess_for_inception(batch)
                raw_output = self.model(batch)
                # torchvision may return InceptionOutputs(logits, aux_logits).
                logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output
                if self._features_buffer is None:
                    raise RuntimeError("Inception avgpool hook did not capture features.")
                features = self._features_buffer.view(self._features_buffer.shape[0], -1)
                probs = torch.softmax(logits, dim=1)

                all_features.append(features.cpu())
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())

        if not all_features:
            raise ValueError("Cannot extract Inception features from an empty image tensor.")

        features_np = torch.cat(all_features, dim=0).numpy().astype(np.float64, copy=False)
        logits_np = torch.cat(all_logits, dim=0).numpy().astype(np.float64, copy=False)
        probs_np = torch.cat(all_probs, dim=0).numpy().astype(np.float64, copy=False)
        return features_np, logits_np, probs_np

from __future__ import annotations

import io
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass
class DegradationConfig:
    severity: int = 1
    gaussian_noise: bool = False
    gaussian_blur: bool = False
    jpeg_compression: bool = False


class DegradedDataset(Dataset):
    """Dataset wrapper that applies controlled degradations on-the-fly.

    Expects base dataset items as (image, label) where image is a tensor in
    [-1, 1] with shape (C, H, W).
    """

    _NOISE_SIGMAS = [2, 15, 25, 40, 60]
    _BLUR_KERNELS = [3, 5, 7, 9, 11]
    _BLUR_SIGMAS = [0.5, 1.0, 2.0, 3.0, 5.0]
    _JPEG_QUALITIES = [80, 60, 40, 20, 5]

    def __init__(self, dataset: Dataset, config: DegradationConfig):
        self.dataset = dataset
        self.config = config
        self._severity_idx = max(0, min(int(config.severity) - 1, 4))

    def __len__(self) -> int:
        return len(self.dataset)

    def _apply_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        sigma_255 = float(self._NOISE_SIGMAS[self._severity_idx])
        std = sigma_255 * 2.0 / 255.0
        noised = image + torch.randn_like(image) * std
        return noised.clamp(-1.0, 1.0)

    def _apply_gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        kernel_size = int(self._BLUR_KERNELS[self._severity_idx])
        sigma = float(self._BLUR_SIGMAS[self._severity_idx])
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def _apply_jpeg_compression(self, image: torch.Tensor) -> torch.Tensor:
        quality = int(self._JPEG_QUALITIES[self._severity_idx])
        device = image.device
        dtype = image.dtype
        channels = int(image.shape[0])

        image_01 = ((image.clamp(-1.0, 1.0) + 1.0) / 2.0).detach().cpu()
        pil_img = TF.to_pil_image(image_01)

        with io.BytesIO() as buffer:
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            decoded = Image.open(buffer)
            decoded.load()

        decoded_tensor = TF.to_tensor(decoded).to(dtype=dtype)

        if channels == 1 and int(decoded_tensor.shape[0]) == 3:
            decoded_tensor = decoded_tensor.mean(dim=0, keepdim=True)
        elif channels == 3 and int(decoded_tensor.shape[0]) == 1:
            decoded_tensor = decoded_tensor.repeat(3, 1, 1)

        return (decoded_tensor.to(device=device) * 2.0 - 1.0).clamp(-1.0, 1.0)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]

        if self.config.gaussian_noise:
            image = self._apply_gaussian_noise(image)

        if self.config.gaussian_blur:
            image = self._apply_gaussian_blur(image)

        if self.config.jpeg_compression:
            image = self._apply_jpeg_compression(image)

        return image, label

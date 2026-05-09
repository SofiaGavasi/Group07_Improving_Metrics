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

    # Base settings are calibrated around 32x32 images and then scaled by resolution.
    # Severity=5 is intentionally very strong.
    _NOISE_SIGMAS = [6, 18, 32, 56, 96]  # sigma in [0..255] space
    _BLUR_KERNELS = [3, 7, 11, 17, 25]
    _BLUR_SIGMAS = [0.8, 1.8, 3.0, 5.0, 8.0]
    _JPEG_QUALITIES = [75, 45, 25, 10, 3]

    def __init__(self, dataset: Dataset, config: DegradationConfig):
        self.dataset = dataset
        self.config = config
        self._severity_idx = max(0, min(int(config.severity) - 1, 4))

    def __len__(self) -> int:
        return len(self.dataset)

    def _resolution_factor(self, image: torch.Tensor) -> float:
        # Scale corruption strength with spatial resolution so severity levels
        # are comparably disruptive across datasets.
        short_side = max(1, min(int(image.shape[-2]), int(image.shape[-1])))
        return max(1.0, float(short_side) / 32.0)

    @staticmethod
    def _to_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    def _apply_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        # Use sqrt scaling to avoid over-amplifying high-resolution noise too early.
        resolution_scale = self._resolution_factor(image) ** 0.5
        sigma_255 = float(self._NOISE_SIGMAS[self._severity_idx]) * resolution_scale
        std = sigma_255 * 2.0 / 255.0
        noised = image + torch.randn_like(image) * std
        return noised.clamp(-1.0, 1.0)

    def _apply_gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        resolution_scale = self._resolution_factor(image)
        base_kernel = int(self._BLUR_KERNELS[self._severity_idx])
        kernel_size = self._to_odd(int(round(base_kernel * resolution_scale)))
        # Keep kernel bounded and valid for current image size.
        max_kernel = self._to_odd(max(3, min(int(image.shape[-2]), int(image.shape[-1])) - 1))
        kernel_size = max(3, min(kernel_size, max_kernel))
        sigma = float(self._BLUR_SIGMAS[self._severity_idx]) * (resolution_scale ** 0.5)
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def _apply_jpeg_compression(self, image: torch.Tensor) -> torch.Tensor:
        # Larger images preserve more detail at a given JPEG quality, so push
        # quality lower as resolution increases.
        resolution_scale = self._resolution_factor(image)
        quality_drop = int(round((resolution_scale - 1.0) * 6.0))
        quality = max(1, int(self._JPEG_QUALITIES[self._severity_idx]) - quality_drop)
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

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn






class StudioGANWrapper:
    def __init__(self, repo_path, ckpt_path, config_name="SNGAN.yaml", device="cpu", logger=None):
        self.repo_path = Path(repo_path)
        self.ckpt_path = Path(ckpt_path)
        
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            self.device = torch.device("cuda:0")
            model_device = 0 
        else:
            self.device = torch.device("cpu")
            model_device = self.device

        self.logger = logger

        # i keep the source lookup flexible because the staged archive can unpack
        # with slightly different top level folder shapes
        src_path = self._resolve_studiogan_src_path(self.repo_path)
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from config import Configurations
        from models.model import load_generator_discriminator

        # more about paths
        cfg_file_path = src_path / "configs" / "CIFAR10" / config_name
        if not cfg_file_path.exists():
            raise FileNotFoundError(f"Config not found at: {cfg_file_path}")
            
        self.studiogan_cfg = Configurations(str(cfg_file_path))

        # i keep the config width from studiogan here
        # the staged cifar10 checkpoint was trained with the normal config width
        # and forcing 96 makes the generator too wide for the saved weights
        
        # attributes handling
        if not hasattr(self.studiogan_cfg.RUN, "mixed_precision"):
            self.studiogan_cfg.RUN.mixed_precision = False
        if not hasattr(self.studiogan_cfg.RUN, "train"):
            self.studiogan_cfg.RUN.train = False

        # loading the discriminator
        self.G, _, _, _, _, _, _, _ = load_generator_discriminator(
            self.studiogan_cfg.DATA,
            self.studiogan_cfg.OPTIMIZATION,
            self.studiogan_cfg.MODEL,
            self.studiogan_cfg.STYLEGAN,
            self.studiogan_cfg.MODULES,
            self.studiogan_cfg.RUN,
            model_device,
            self.logger
        )

        # loading weights
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        
        # choosing ema weights
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("generator_ema", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        self.G.load_state_dict(state_dict, strict=False)
        self.G.to(self.device)
        self.G.eval()

    @staticmethod
    def _resolve_studiogan_src_path(repo_path: Path):
        candidates = [
            repo_path / "src",
            repo_path,
        ]

        for candidate in candidates:
            if (candidate / "config.py").exists() and (candidate / "models" / "model.py").exists():
                return candidate

        nested_hits = list(repo_path.rglob("config.py"))
        for config_path in nested_hits:
            candidate = config_path.parent
            if (candidate / "models" / "model.py").exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find StudioGAN source files under {repo_path}. "
            "i expected a folder containing config.py and models/model.py"
        )

    @torch.no_grad()
    def sample(self, n):
        z_dim = self.studiogan_cfg.MODEL.z_dim
        z = torch.randn(n, z_dim, device=self.device)
        # generates random class labels if needed
        y = torch.randint(0, self.studiogan_cfg.DATA.num_classes, (n,), device=self.device)
        
        # since SNGAN with cBN expects (z, y), both images and labels
        return self.G(z, y)





####################################################################################################################

class DDPMWrapper:
    """
    Wrapper for unconditional CIFAR-10 diffusion checkpoints stored in diffusers format.

    checkpoint_path should point to a local directory like:
        checkpoints/DDPM/CIFAR10
    """

    def __init__(
        self,
        checkpoint_path: str,
        pipeline_type: str = "ddpm",
        torch_dtype: torch.dtype | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.pipeline_type = pipeline_type.lower()
        self.torch_dtype = torch_dtype
        self.pipeline = self._load_pipeline()
        self._current_device = torch.device("cpu")

    def _load_pipeline(self):
        try:
            from diffusers import DDPMPipeline, DDIMPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for DDPM/DDIM wrappers. "
                "Install with: pip install diffusers"
            ) from exc

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint directory not found: {self.checkpoint_path}"
            )

        load_kwargs: dict[str, Any] = {}
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype

        if self.pipeline_type == "ddpm":
            return DDPMPipeline.from_pretrained(str(self.checkpoint_path), **load_kwargs)

        if self.pipeline_type == "ddim":
            return DDIMPipeline.from_pretrained(str(self.checkpoint_path), **load_kwargs)

        raise ValueError("pipeline_type must be 'ddpm' or 'ddim'")

    @staticmethod
    def _output_to_tensor(images: Any) -> torch.Tensor:
        import numpy as np

        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(
                    f"Expected tensor output with shape [N,C,H,W], got {tuple(images.shape)}"
                )
            return images.clamp(-1.0, 1.0).detach().cpu()

        if isinstance(images, list):
            arr = np.stack(
                [np.asarray(img, dtype=np.float32) / 255.0 for img in images],
                axis=0,
            )
        else:
            arr = np.asarray(images, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(
                    f"Expected array output with shape [N,H,W,C], got {arr.shape}"
                )
            if arr.max() > 1.0:
                arr = arr / 255.0

        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        tensor = tensor * 2.0 - 1.0
        return tensor.clamp(-1.0, 1.0)

    def sample(self, n: int, device: torch.device, **kwargs: Any):
        """
        kwargs:
            num_inference_steps: int
            seed: int
            output_type: "np" or "pil"
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        if self._current_device != device:
            self.pipeline.to(device)
            self._current_device = device

        seed = kwargs.get("seed", None)
        output_type = str(kwargs.get("output_type", "np"))
        default_steps = 1000 if self.pipeline_type == "ddpm" else 50
        num_inference_steps = int(kwargs.get("num_inference_steps", default_steps))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        with torch.no_grad():
            result = self.pipeline(
                batch_size=n,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type=output_type,
            )

        if hasattr(result, "images"):
            images = result.images
        elif isinstance(result, dict) and "images" in result:
            images = result["images"]
        elif isinstance(result, dict) and "sample" in result:
            images = result["sample"]
        else:
            raise ValueError("Unexpected diffusers pipeline output format.")

        return self._output_to_tensor(images)


class DDIMWrapper(DDPMWrapper):
    def __init__(
        self,
        checkpoint_path: str,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            pipeline_type="ddim",
            torch_dtype=torch_dtype,
        )











####################################################################################################################

class StyleGAN2Wrapper:

    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"StyleGAN2 checkpoint not found: {self.checkpoint_path}")

        self.generator = self._load_generator(self.checkpoint_path)
        self.generator.eval()
        self._current_device = torch.device("cpu")
        self.latent_dim = self._infer_latent_dim(self.generator)


    def _load_generator(self, checkpoint_path):
        checkpoint_obj: Any
        # stylegan checkpoints are usually .pkl, but we also support .pth/.pt exports
        if checkpoint_path.suffix.lower() in {".pkl", ".pickle"}:
            try:
                with checkpoint_path.open("rb") as handle:
                    checkpoint_obj = pickle.load(handle)
            except ModuleNotFoundError:
                # official stylegan2-ada pickles may require loading through legacy.py
                checkpoint_obj = self._load_with_stylegan2_ada_legacy(checkpoint_path)
        else:
            try:
                # torch>=2.6 defaults to weights_only=True; set false for module-style checkpoints
                checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                # fallback
                checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

        generator = self._extract_generator(checkpoint_obj)
        if generator is None:
            raise ValueError(
                "Could not find a generator module in checkpoint. "
            )
        return generator



    def _load_with_stylegan2_ada_legacy(self, checkpoint_path):

        # check common staging locations for the stylegan2-ada source tree
        for source_dir in self._candidate_stylegan2_source_dirs(checkpoint_path):
            source_dir_str = str(source_dir)
            remove_after = False
            if source_dir_str not in sys.path:
                sys.path.insert(0, source_dir_str)
                remove_after = True

            try:
                if "legacy" in sys.modules:
                    del sys.modules["legacy"]
                import legacy  # type: ignore

                with checkpoint_path.open("rb") as handle:
                    return legacy.load_network_pkl(handle)
            except ModuleNotFoundError:
                continue
            finally:
                if remove_after and source_dir_str in sys.path:
                    sys.path.remove(source_dir_str)

        raise ModuleNotFoundError(
            "This checkpoint requires StyleGAN2-ADA source modules for unpickling. "
            "Run Scripts/download_pretrained_stylegan_celeba.py to stage both checkpoint and source code"
        )

    @staticmethod
    def _candidate_stylegan2_source_dirs(checkpoint_path):
        repo_root = Path(__file__).resolve().parents[1]
        return [
            checkpoint_path.parent / "stylegan2_ada_src",
            repo_root / "third_party" / "stylegan2_ada_src",
        ]

    def _extract_generator(self, value):
        # simplest case: checkpoint is already an nn.Module generator
        if isinstance(value, nn.Module):
            return value

        if isinstance(value, dict):
            # based on the staged stylegan2 checkpoint, these are the relevant keys
            for key in ("G_ema", "G"):
                candidate = value.get(key)
                if isinstance(candidate, nn.Module):
                    return candidate

        return None

    @staticmethod
    def _infer_latent_dim(generator):
        # common explicit attributes first
        for attr in ("z_dim", "latent_dim", "nz", "style_dim"):
            value = getattr(generator, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        mapping = getattr(generator, "mapping", None)
        if mapping is not None:
            # stylegan variant store z_dim on mapping
            for attr in ("z_dim", "latent_dim", "in_features"):
                value = getattr(mapping, attr, None)
                if isinstance(value, int) and value > 0:
                    return value

        # fallback: first linear layer input size
        for module in generator.modules():
            if isinstance(module, nn.Linear) and module.in_features > 0:
                return int(module.in_features)

        return 512

    def _build_conditioning(self, n: int, device: torch.device, class_idx):
        # unconditional models have c_dim=0, so this becomes an empty conditioning tensor
        c_dim = int(getattr(self.generator, "c_dim", 0) or 0)
        conditioning = torch.zeros(n, c_dim, device=device)
        if c_dim > 0 and class_idx is not None:
            # for conditional checkpoints, allow one-hot class override
            if not 0 <= class_idx < c_dim:
                raise ValueError(f"class_idx must be in [0, {c_dim - 1}] for this checkpoint.")
            conditioning[:, class_idx] = 1.0
        return conditioning


    def sample(self, n: int, device: torch.device, **kwargs: Any):
        truncation_psi = float(kwargs.get("truncation_psi", 0.7))
        noise_mode = str(kwargs.get("noise_mode", "const"))
        class_idx = kwargs.get("class_idx", None)
        seed = kwargs.get("seed", None)

        if n <= 0:
            raise ValueError("n must be a positive integer.")

        # move model only when target device changes
        if self._current_device != device:
            self.generator.to(device)
            self._current_device = device

        random_generator = None
        if seed is not None:
            # optional seeded generator for deterministic sampling
            random_generator = torch.Generator(device=device)
            random_generator.manual_seed(int(seed))

        with torch.no_grad():
            # sample latent z + optional conditioning c
            z = torch.randn(n, self.latent_dim, device=device, generator=random_generator)
            c = self._build_conditioning(n=n, device=device, class_idx=class_idx)

            # known stylegan2 call signatures
            attempts: list[Any] = []
            try:
                attempts.append(
                    self.generator(
                        z,
                        c,
                        truncation_psi=truncation_psi,
                        noise_mode=noise_mode,
                    )
                )
            except TypeError:
                pass

            try:
                attempts.append(
                    self.generator(
                        [z],
                        truncation=truncation_psi,
                        randomize_noise=(noise_mode == "random"),
                    )
                )
            except TypeError:
                pass

            if not attempts:
                attempts.append(self.generator(z))

            images = attempts[0]
            if isinstance(images, (list, tuple)):
                if not images:
                    raise ValueError("StyleGAN2 generator returned an empty tuple/list.")
                images = images[0]

            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError(
                    "StyleGAN2 generator output must be a tensor with shape [N, C, H, W]."
                )

            return images.clamp(-1.0, 1.0).detach().cpu()

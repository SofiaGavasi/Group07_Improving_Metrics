# example: py Scripts/download_pretrained_ddpm_cifar10.py --output-dir checkpoints/DDPM/CIFAR10

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stage pretrained DDPM/DDIM checkpoint for CIFAR-10.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/DDPM/CIFAR10")
    parser.add_argument("--model-id", type=str, default="google/ddpm-cifar10-32")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from diffusers import DDPMPipeline
    except ImportError as exc:
        raise ImportError(
            "diffusers is required. Install with: pip install diffusers"
        ) from exc

    print(f"Downloading model: {args.model_id}")
    pipe = DDPMPipeline.from_pretrained(args.model_id)
    pipe.save_pretrained(str(output_dir))

    print(f"Checkpoint directory ready: {output_dir}")
    print("Saved files:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(" -", path.relative_to(output_dir))


if __name__ == "__main__":
    main()
# example: py Scripts/test_stylegan2.py --checkpoint checkpoints/StyleGAN/CelebA/stylegan2_generator.pkl --out-dir outputs/stylegan2_test
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.pretrained_wrappers import StyleGAN2Wrapper
from Perturbation.pipeline_perturbations import add_perturbation_args, get_domain_shift_override
from Scripts.evaluation_runtime import file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import set_deterministic_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a pretrained StyleGAN2 checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="outputs/stylegan2_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--noise-mode", type=str, default="const", choices=["const", "random", "none"])
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, compute metrics with Metrics/compute_all.py and save JSON report.",
    )
    parser.add_argument(
        "--metrics-dataset",
        type=str,
        default="celeba",
        choices=["mnist", "cifar10", "celeba", "chestxray14"],
        help="Dataset used as real reference distribution during metric evaluation.",
    )
    parser.add_argument(
        "--metrics-data-root",
        type=str,
        default="data/CelebA",
        help="Data root for metrics real samples.",
    )
    parser.add_argument("--metrics-image-size", type=int, default=256)
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument("--metrics-feature-space", type=str, default="inception_v3")
    parser.add_argument("--metrics-feature-batch-size", type=int, default=64)
    parser.add_argument("--metrics-feature-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--metrics-bootstrap-samples", type=int, default=0)
    parser.add_argument("--metrics-bootstrap-seed", type=int, default=10)
    parser.add_argument("--metrics-bootstrap-alpha", type=float, default=0.05)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow dataset download/setup fallback during metric evaluation if files are missing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors (missing checkpoint/loading/sampling) instead of skipping.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging for generation, perturbations, and metrics.",
    )
    add_perturbation_args(parser)
    return parser.parse_args()


def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    override = get_domain_shift_override(args)
    if override is None:
        return args.metrics_dataset, args.metrics_data_root, int(default_image_size)
    image_size = int(override["image_size"]) if int(override["image_size"]) > 0 else int(default_image_size)
    return str(override["dataset"]), str(override["data_root"]), image_size


def _build_generation_payload(args: argparse.Namespace, checkpoint):
    return {
        "model_name": "stylegan2",
        "checkpoint": file_signature(checkpoint),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "truncation_psi": float(args.truncation_psi),
        "noise_mode": str(args.noise_mode),
        "class_idx": None if args.class_idx is None else int(args.class_idx),
        "seed": int(args.seed),
    }


def _generate_in_batches(
    *,
    wrapper: StyleGAN2Wrapper,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    class_idx: int | None,
    seed: int | None,
):
    samples: list[torch.Tensor] = []
    remaining = int(total_samples)
    offset = 0

    while remaining > 0:
        this_batch = min(int(batch_size), remaining)
        batch_seed = None if seed is None else int(seed + offset)
        generated = wrapper.sample(
            this_batch,
            device=device,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            class_idx=class_idx,
            seed=batch_seed,
        )
        samples.append(generated)
        remaining -= this_batch
        offset += this_batch

    return torch.cat(samples, dim=0)


def main():
    args = parse_args()
    set_deterministic_seed(seed=int(args.seed), verbose=bool(args.verbose), context="test_stylegan2")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        message = f"StyleGAN2 test skipped: checkpoint not found at {checkpoint}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    try:
        wrapper = StyleGAN2Wrapper(checkpoint_path=str(checkpoint))
        run_cached_evaluation(
            args=args,
            model_name="stylegan2",
            generation_payload=_build_generation_payload(args, checkpoint),
            generate_samples=lambda: _generate_in_batches(
                wrapper=wrapper,
                total_samples=int(args.num_samples),
                batch_size=max(1, int(args.batch_size)),
                device=device,
                truncation_psi=float(args.truncation_psi),
                noise_mode=str(args.noise_mode),
                class_idx=args.class_idx,
                seed=args.seed,
            ),
            resolve_reference_request=_resolve_real_reference_request,
        )
    except Exception as exc:
        print(f"StyleGAN2 test failed: {exc}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.wgangp import WGANGPCritic, WGANGPGenerator
from Perturbation.pipeline_perturbations import add_perturbation_args, get_domain_shift_override
from Scripts.evaluation_runtime import file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import set_deterministic_seed



def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained WGAN-GP generator.")
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--critic-checkpoint", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="outputs/wgangp_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--g-base", type=int, default=64)
    parser.add_argument("--d-base", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, compute metrics with Metrics/compute_all.py and save JSON report.",
    )
    parser.add_argument(
        "--metrics-dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10", "celeba", "chestxray14"],
        help="Dataset used as real reference distribution during metric evaluation.",
    )
    parser.add_argument("--metrics-data-root", type=str, default="data/CIFAR10")
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument("--metrics-feature-space", type=str, default="inception_v3")
    parser.add_argument("--metrics-feature-batch-size", type=int, default=64)
    parser.add_argument("--metrics-feature-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--metrics-bootstrap-samples", type=int, default=0)
    parser.add_argument("--metrics-bootstrap-seed", type=int, default=10)
    parser.add_argument("--metrics-bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        return args.metrics_dataset, args.metrics_data_root, int(args.image_size or default_image_size)
    image_size = int(override["image_size"]) if int(override["image_size"]) > 0 else int(args.image_size or default_image_size)
    return str(override["dataset"]), str(override["data_root"]), image_size


def _build_generation_payload(args: argparse.Namespace, generator_path, critic_path):
    payload: dict[str, object] = {
        "model_name": "wgangp",
        "generator_checkpoint": file_signature(generator_path),
        "num_samples": int(args.num_samples),
        "latent_dim": int(args.latent_dim),
        "channels": int(args.channels),
        "g_base": int(args.g_base),
        "d_base": int(args.d_base),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
    }
    if critic_path is not None and critic_path.exists():
        payload["critic_checkpoint"] = file_signature(critic_path)
    return payload


def _generate_samples(
    *,
    args: argparse.Namespace,
    generator: WGANGPGenerator,
    device: torch.device,
) :
    generated: list[torch.Tensor] = []
    remaining = int(args.num_samples)
    latent_rng = torch.Generator(device=device)
    latent_rng.manual_seed(int(args.seed))
    with torch.no_grad():
        while remaining > 0:
            this_batch = min(int(args.batch_size), remaining)
            noise = torch.randn(this_batch, int(args.latent_dim), 1, 1, device=device, generator=latent_rng)
            generated.append(generator(noise).detach().cpu())
            remaining -= this_batch
    return torch.cat(generated, dim=0)


def main():
    args = parse_args()
    set_deterministic_seed(seed=int(args.seed), verbose=bool(args.verbose), context="test_wgangp")

    generator_path = Path(args.generator_checkpoint)
    if not generator_path.exists():
        message = f"WGAN-GP test skipped: generator checkpoint not found at {generator_path}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    generator = WGANGPGenerator(
        latent_dim=int(args.latent_dim),
        out_channels=int(args.channels),
        base_channels=int(args.g_base),
    ).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    critic_path = Path(args.critic_checkpoint) if args.critic_checkpoint.strip() else None
    if critic_path is not None and critic_path.exists():
        # i still load the critic when it is present because it catches checkpoint mismatches early
        critic = WGANGPCritic(in_channels=int(args.channels), base_channels=int(args.d_base)).to(device)
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        critic.eval()

    run_cached_evaluation(
        args=args,
        model_name="wgangp",
        generation_payload=_build_generation_payload(args, generator_path, critic_path),
        generate_samples=lambda: _generate_samples(args=args, generator=generator, device=device),
        resolve_reference_request=_resolve_real_reference_request,
    )


if __name__ == "__main__":
    main()

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
from Scripts.evaluation_runtime import EvaluationArtifacts, EvaluationReuseSession, file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import (
    PreparedTestRun,
    add_common_test_args,
    close_prepared_test_run,
    get_generation_seed,
    initialize_test_run,
    resolve_reference_request,
)



def parse_args(argv = None):
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
    add_common_test_args(
        parser,
        metrics_dataset_default="cifar10",
        metrics_dataset_choices=["mnist", "cifar10", "celeba", "chestxray14"],
        metrics_data_root_default="data/CIFAR10",
    )
    add_perturbation_args(parser)
    return parser.parse_args(argv)


def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    return resolve_reference_request(args, default_image_size, get_domain_shift_override(args))


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
        "generation_seed": get_generation_seed(args),
    }
    if critic_path is not None and critic_path.exists():
        payload["critic_checkpoint"] = file_signature(critic_path)
    return payload


def _generate_samples(
    *,
    args: argparse.Namespace,
    generator: WGANGPGenerator,
    device: torch.device,
    total_samples: int | None = None,
) :
    generated: list[torch.Tensor] = []
    remaining = int(args.num_samples) if total_samples is None else int(total_samples)
    latent_rng = torch.Generator(device=device)
    latent_rng.manual_seed(get_generation_seed(args))
    with torch.no_grad():
        while remaining > 0:
            this_batch = min(int(args.batch_size), remaining)
            noise = torch.randn(this_batch, int(args.latent_dim), 1, 1, device=device, generator=latent_rng)
            generated.append(generator(noise).detach().cpu())
            remaining -= this_batch
    return torch.cat(generated, dim=0)


def prepare_run(args):
    initialize_test_run(args, context="test_wgangp")

    generator_path = Path(args.generator_checkpoint)
    if not generator_path.exists():
        message = f"WGAN-GP test skipped: generator checkpoint not found at {generator_path}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return None

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

    return PreparedTestRun(
        model_name="wgangp",
        generation_payload=_build_generation_payload(args, generator_path, critic_path),
        generate_samples=lambda total_samples=None: _generate_samples(
            args=args,
            generator=generator,
            device=device,
            total_samples=total_samples,
        ),
        resolve_reference_request=_resolve_real_reference_request,
        cleanup=None,
    )


def run_with_args(
    args: argparse.Namespace,
    *,
    prepared = None,
    session= None,
    write_preview= True,
    persist_derived_feature_artifacts = True,
    bootstrap_samples_override= None,
    bootstrap_policy = "full",
) :
    owned_prepared = prepared is None
    prepared_run = prepared or prepare_run(args)
    if prepared_run is None:
        return None
    try:
        return run_cached_evaluation(
            args=args,
            model_name=prepared_run.model_name,
            generation_payload=prepared_run.generation_payload,
            generate_samples=prepared_run.generate_samples,
            resolve_reference_request=prepared_run.resolve_reference_request,
            session=session,
            write_preview=write_preview,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            bootstrap_samples_override=bootstrap_samples_override,
            bootstrap_policy=bootstrap_policy,
        )
    finally:
        if owned_prepared:
            close_prepared_test_run(prepared_run)


def main():
    args = parse_args()
    run_with_args(args)


if __name__ == "__main__":
    main()

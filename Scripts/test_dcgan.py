# example: py Scripts/test_dcgan.py --netG outputs/dcgan_cifar10/netG_latest.pth --out-dir outputs/dcgan_test --num-samples 64 --image-size 32 --channels 3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.dcgan import DCGANGenerator
from Perturbation.pipeline_perturbations import add_perturbation_args, get_domain_shift_override
from Scripts.evaluation_runtime import EvaluationArtifacts, EvaluationReuseSession, file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import (
    PreparedTestRun,
    close_prepared_test_run,
    set_deterministic_seed,
)


def parse_args(argv= None):
    parser = argparse.ArgumentParser(description="Generate samples from a trained DCGAN generator.")
    parser.add_argument("--netG", type=str, required=True, help="Path to trained generator checkpoint.")
    parser.add_argument("--out-dir", type=str, default="outputs/dcgan_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--metrics-dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10"]
    )
    parser.add_argument(
        "--metrics-data-root",
        type=str,
        default="data/CIFAR10"
    )
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging for generation, perturbations, and metrics.",
    )
    parser.add_argument("--metrics-feature-space", type=str, default="inception_v3")
    parser.add_argument("--metrics-feature-batch-size", type=int, default=64)
    parser.add_argument("--metrics-feature-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--metrics-bootstrap-samples", type=int, default=0)
    parser.add_argument("--metrics-bootstrap-seed", type=int, default=10)
    parser.add_argument("--metrics-bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--reference-seed", type=int, default=None)
    add_perturbation_args(parser)
    return parser.parse_args(argv)


# this keeps the domain shift logic in the script, but the heavy lifting now lives elsewhere


def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    override = get_domain_shift_override(args)
    if override is None:
        return args.metrics_dataset, args.metrics_data_root, int(args.image_size or default_image_size)
    image_size = int(override["image_size"]) if int(override["image_size"]) > 0 else int(args.image_size or default_image_size)
    return str(override["dataset"]), str(override["data_root"]), image_size


def _generation_seed(args):
    return int(args.seed if args.generation_seed is None else args.generation_seed)


def _reference_seed(args):
    return int(_generation_seed(args) if args.reference_seed is None else args.reference_seed)


def _build_generation_payload(args: argparse.Namespace, checkpoint_path: Path):
    return {
        "model_name": "dcgan",
        "checkpoint": file_signature(checkpoint_path),
        "num_samples": int(args.num_samples),
        "latent_dim": int(args.latent_dim),
        "ngf": int(args.ngf),
        "channels": int(args.channels),
        "image_size": int(args.image_size),
        "generation_seed": _generation_seed(args),
    }


def _generate_samples(
    *,
    args: argparse.Namespace,
    generator: DCGANGenerator,
    device: torch.device,
    total_samples: int | None = None,
):
    generated: list[torch.Tensor] = []
    remaining = int(args.num_samples) if total_samples is None else int(total_samples)
    latent_rng = torch.Generator(device=device)
    latent_rng.manual_seed(_generation_seed(args))
    with torch.no_grad():
        while remaining > 0:
            this_batch = min(int(args.batch_size), remaining)
            noise = torch.randn(this_batch, int(args.latent_dim), 1, 1, device=device, generator=latent_rng)
            generated.append(generator(noise).detach().cpu())
            remaining -= this_batch
    return torch.cat(generated, dim=0)


def prepare_run(args):
    set_deterministic_seed(seed=_generation_seed(args), verbose=bool(args.verbose), context="test_dcgan")
    args.reference_seed = _reference_seed(args)

    netg_path = Path(args.netG)
    if not netg_path.exists():
        message = f"DCGAN test skipped: checkpoint not found at {netg_path}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    generator = DCGANGenerator(
        ngpu=0,
        nc=int(args.channels),
        nz=int(args.latent_dim),
        ngf=int(args.ngf),
        image_size=int(args.image_size),
    ).to(device)
    generator.load_state_dict(torch.load(netg_path, map_location=device))
    generator.eval()

    return PreparedTestRun(
        model_name="dcgan",
        generation_payload=_build_generation_payload(args, netg_path),
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
    session = None,
    write_preview = True,
    persist_derived_feature_artifacts = True,
    bootstrap_samples_override = None,
    bootstrap_policy = "full",
):
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

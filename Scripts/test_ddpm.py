# example:
# py Scripts/test_ddpm.py --checkpoint checkpoints/DDPM/CIFAR10 --mode ddpm --out-dir outputs/ddpm_test
# py Scripts/test_ddpm.py --checkpoint checkpoints/DDPM/CIFAR10 --mode ddim --out-dir outputs/ddim_test

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.pretrained_wrappers import DDIMWrapper, DDPMWrapper
from Perturbation.pipeline_perturbations import add_perturbation_args, get_domain_shift_override
from Scripts.evaluation_runtime import EvaluationArtifacts, EvaluationReuseSession, file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import (
    PreparedTestRun,
    close_prepared_test_run,
    set_deterministic_seed,
)


# i kept the public flags intact so old commands still work


def parse_args(argv= None):
    parser = argparse.ArgumentParser(description="Generate samples from a pretrained DDPM/DDIM checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/DDPM/CIFAR10")
    parser.add_argument("--mode", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--out-dir", type=str, default="outputs/ddpm_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--reference-seed", type=int, default=None)
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
        default="cifar10",
        choices=["mnist", "cifar10", "celeba", "chestxray14"],
        help="Dataset used as real reference distribution during metric evaluation.",
    )
    parser.add_argument(
        "--metrics-data-root",
        type=str,
        default="data/CIFAR10",
        help="Data root for metrics real samples.",
    )
    parser.add_argument("--metrics-image-size", type=int, default=32)
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
        help="Fail on errors instead of skipping.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging for generation, perturbations, and metrics.",
    )
    add_perturbation_args(parser)
    return parser.parse_args(argv)


def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    override = get_domain_shift_override(args)
    if override is None:
        return args.metrics_dataset, args.metrics_data_root, int(default_image_size)
    image_size = int(override["image_size"]) if int(override["image_size"]) > 0 else int(default_image_size)
    return str(override["dataset"]), str(override["data_root"]), image_size


def _generation_seed(args):
    return int(args.seed if args.generation_seed is None else args.generation_seed)


def _reference_seed(args):
    return int(_generation_seed(args) if args.reference_seed is None else args.reference_seed)


def _build_generation_payload(args: argparse.Namespace, checkpoint: Path):
    return {
        "model_name": str(args.mode),
        "checkpoint": file_signature(checkpoint),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "num_inference_steps": None if args.num_inference_steps is None else int(args.num_inference_steps),
        "generation_seed": _generation_seed(args),
    }


def _generate_in_batches(
    *,
    wrapper: DDPMWrapper | DDIMWrapper,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    num_inference_steps: int | None,
    seed: int | None,
):
    samples: list[torch.Tensor] = []
    remaining = int(total_samples)
    offset = 0

    while remaining > 0:
        this_batch = min(int(batch_size), remaining)
        batch_seed = None if seed is None else int(seed + offset)
        kwargs: dict[str, object] = {"seed": batch_seed}
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = int(num_inference_steps)
        generated = wrapper.sample(
            this_batch,
            device=device,
            **kwargs,
        )
        samples.append(generated)
        remaining -= this_batch
        offset += this_batch

    return torch.cat(samples, dim=0)


def prepare_run(args):
    set_deterministic_seed(seed=_generation_seed(args), verbose=bool(args.verbose), context="test_ddpm")
    args.reference_seed = _reference_seed(args)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        message = f"{args.mode.upper()} test skipped: checkpoint not found at {checkpoint}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    wrapper_cls = DDPMWrapper if args.mode == "ddpm" else DDIMWrapper
    wrapper = wrapper_cls(checkpoint_path=str(checkpoint))
    return PreparedTestRun(
        model_name=str(args.mode),
        generation_payload=_build_generation_payload(args, checkpoint),
        generate_samples=lambda total_samples=None: _generate_in_batches(
            wrapper=wrapper,
            total_samples=int(args.num_samples) if total_samples is None else int(total_samples),
            batch_size=max(1, int(args.batch_size)),
            device=device,
            num_inference_steps=args.num_inference_steps,
            seed=_generation_seed(args),
        ),
        resolve_reference_request=_resolve_real_reference_request,
        cleanup=None,
    )


def run_with_args(
    args: argparse.Namespace,
    *,
    prepared = None,
    session= None,
    write_preview = True,
    persist_derived_feature_artifacts= True,
    bootstrap_samples_override= None,
    bootstrap_policy = "full",
) :
    owned_prepared = prepared is None
    try:
        prepared_run = prepared or prepare_run(args)
        if prepared_run is None:
            return None
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
    except Exception as exc:
        print(f"{args.mode.upper()} test failed: {exc}")
        if args.strict:
            raise SystemExit(1)
        return None
    finally:
        if owned_prepared:
            close_prepared_test_run(prepared if prepared is not None else locals().get("prepared_run"))


def main():
    args = parse_args()
    run_with_args(args)


if __name__ == "__main__":
    main()

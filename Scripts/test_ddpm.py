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
    add_common_test_args,
    close_prepared_test_run,
    get_generation_seed,
    initialize_test_run,
    resolve_reference_request,
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
    parser.add_argument("--cuda", action="store_true")
    add_common_test_args(
        parser,
        metrics_dataset_default="cifar10",
        metrics_dataset_choices=["mnist", "cifar10", "celeba", "chestxray14"],
        metrics_data_root_default="data/CIFAR10",
        metrics_image_size_default=32,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors instead of skipping.",
    )
    add_perturbation_args(parser)
    return parser.parse_args(argv)


def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    return resolve_reference_request(args, default_image_size, get_domain_shift_override(args))


def _build_generation_payload(args: argparse.Namespace, checkpoint: Path):
    return {
        "model_name": str(args.mode),
        "checkpoint": file_signature(checkpoint),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "num_inference_steps": None if args.num_inference_steps is None else int(args.num_inference_steps),
        "generation_seed": get_generation_seed(args),
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
    initialize_test_run(args, context="test_ddpm")

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
            seed=get_generation_seed(args),
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

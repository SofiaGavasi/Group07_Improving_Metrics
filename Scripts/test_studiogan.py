from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.pretrained_wrappers import StudioGANWrapper
from Perturbation.pipeline_perturbations import add_perturbation_args, get_domain_shift_override
from Scripts.evaluation_runtime import EvaluationArtifacts, EvaluationReuseSession, file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import (
    PreparedTestRun,
    close_prepared_test_run,
    set_deterministic_seed,
)
def parse_args(argv= None):
    parser = argparse.ArgumentParser(description="Generate samples from a pretrained StudioGAN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--repo-path", type=str, default="")
    parser.add_argument("--config-name", type=str, default="SNGAN.yaml")
    parser.add_argument("--out-dir", type=str, default="outputs/studiogan_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
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


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("studiogan")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _build_generation_payload(args: argparse.Namespace, checkpoint: Path, repo_path: Path):
    return {
        "model_name": "studiogan",
        "checkpoint": file_signature(checkpoint),
        "repo_path": file_signature(repo_path) if repo_path.exists() else {"path": str(repo_path.resolve()), "exists": False},
        "config_name": str(args.config_name),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }


def _generate_in_batches(
    *,
    wrapper: StudioGANWrapper,
    total_samples: int,
    batch_size: int,
    seed: int | None,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    remaining = int(total_samples)
    offset = 0

    while remaining > 0:
        this_batch = min(int(batch_size), remaining)
        if seed is not None:
            torch.manual_seed(int(seed + offset))
        outputs.append(wrapper.sample(this_batch).detach().cpu())
        remaining -= this_batch
        offset += this_batch

    return torch.cat(outputs, dim=0)


def prepare_run(args) :
    set_deterministic_seed(seed=int(args.seed), verbose=bool(args.verbose), context="test_studiogan")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        message = f"StudioGAN test skipped: checkpoint not found at {checkpoint}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    repo_path = Path(args.repo_path) if args.repo_path else checkpoint.parent / "studioGAN_src"
    wrapper = StudioGANWrapper(
        repo_path=repo_path,
        ckpt_path=checkpoint,
        config_name=args.config_name,
        device=("cuda" if args.cuda and torch.cuda.is_available() else "cpu"),
        logger=_build_logger(),
    )
    return PreparedTestRun(
        model_name="studiogan",
        generation_payload=_build_generation_payload(args, checkpoint, repo_path),
        generate_samples=lambda total_samples=None: _generate_in_batches(
            wrapper=wrapper,
            total_samples=max(1, int(args.num_samples) if total_samples is None else int(total_samples)),
            batch_size=max(1, int(args.batch_size)),
            seed=args.seed,
        ),
        resolve_reference_request=_resolve_real_reference_request,
        cleanup=None,
    )


def run_with_args(
    args: argparse.Namespace,
    *,
    prepared = None,
    session= None,
    write_preview: bool = True,
    persist_derived_feature_artifacts = True,
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
        print(f"StudioGAN test failed: {exc}")
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

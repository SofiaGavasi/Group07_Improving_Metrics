from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments import build_experiments_for_suite, default_experiment_base_overrides
from Scripts.test_runtime_utils import close_prepared_test_run


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ============================================================================
#  CONFIGURATIONS
# edit these values, then run: `py main.py`
# ============================================================================


# Pipeline mode:
# - "setup": download/prepare datasets + stage pretrained checkpoints (no training)
# - "train": run training steps only
# - "test": run model test entrypoints (implemented + skeleton test scripts)
# - "full": setup + train + test + smoke checks
# - "smoke": run lightweight model/module smoke checks only
PROFILE = "test"  # Which pipeline profile to run when CUSTOM_STEPS is empty.


# optional custom ordered steps. If non-empty, this overrides PROFILE.
# example: ["prep_mnist_cifar10", "train_dcgan_cifar10", "smoke_models"]
CUSTOM_STEPS = ["prep_mnist_cifar10"]  # Exact step list override; non-empty ignores PROFILE.


# execution behavior flags:
# - RUN=False prints commands only (safe dry-run)
# - RUN=True actually executes each planned command
RUN = True  # True executes commands; False only prints the generated commands.

# if True, pipeline keeps going after a failed step
CONTINUE_ON_ERROR = False  # True keeps running next steps/experiments after failures.

# if True, training commands include --cuda (when supported by the target script)
CUDA = _cuda_available()  # Adds --cuda to child scripts where supported.
VERBOSE = True  # Enables detailed logs across pipeline/test/metric code paths.


# shared roots used by pipeline scripts
DATA_ROOT = "data"  # Root folder that contains datasets (MNIST/CIFAR10/CelebA/ChestXray14).
CHECKPOINTS_ROOT = "checkpoints"  # Root folder for pretrained/downloaded model checkpoints.
OUTPUTS_ROOT = "outputs"  # Root folder where run artifacts and reports are written.


# training stuff:
IMAGE_SIZE = 32  # Training/test image size expected by model/data transforms.
DCGAN_EPOCHS = 1  # Epochs for DCGAN training steps.
DCGAN_BATCH_SIZE = 64  # Batch size for DCGAN training steps.
WGANGP_EPOCHS = 1  # Epochs for WGAN-GP training steps.
WGANGP_BATCH_SIZE = 64  # Batch size for WGAN-GP training steps.


# test stuff:
TEST_NUM_SAMPLES = 1280  # Number of fake samples generated per test script run.
TEST_BATCH_SIZE = 64  # Generation batch size used in test scripts.
#GENERATION_SEED = 1  # RNG seed for fake sample generation in model test scripts  (I moved it down so its easier to change)
REFERENCE_SEED = 10  # RNG seed for choosing the real reference subset during metric evaluation


# metric evaluation during test stage:
EVAL_METRICS = True  # Compute metrics after sample generation.
METRICS_SAMPLES = 1280  # Real-reference sample count loaded for metric evaluation.
METRICS_DOWNLOAD_IF_MISSING = False  # Allow dataset auto-download during metric eval if missing.
METRICS_FEATURE_SPACE = "inception_v3"  # Feature backbone used for FID/KID/PRDC/IS.
METRICS_FEATURE_BATCH_SIZE = 64  # Batch size for feature extraction.
METRICS_FEATURE_DEVICE = "cuda" if CUDA else "cpu"  # Device for feature extraction: "cpu" or "cuda".
METRICS_BOOTSTRAP_SAMPLES = 100  # Number of bootstrap resamples for confidence intervals when the policy enables them.
METRICS_BOOTSTRAP_SEED = 10  # RNG seed for bootstrap reproducibility.
METRICS_BOOTSTRAP_ALPHA = 0.05  # CI significance level (0.05 -> 95% CI).


# perturbation controls:
USE_PERTURBATIONS = False  # Master toggle for all perturbation logic.
PERTURB_APPLY_TO = "fake"  # "fake", "real", "both"

# degradation perturbation:
PERTURB_DEGRADE = False  # Enables image-quality degradation perturbation.
PERTURB_DEGRADE_SEVERITY = 1  # Corruption severity level in [1..5].
PERTURB_DEGRADE_GAUSSIAN_NOISE = False  # Include gaussian-noise corruption.
PERTURB_DEGRADE_GAUSSIAN_BLUR = False  # Include gaussian-blur corruption.
PERTURB_DEGRADE_JPEG_COMPRESSION = False  # Include JPEG-compression corruption.

# memoisation perturbation:
PERTURB_MEMOISATION = False  # Enables memoisation perturbation (replace fake with real samples).
PERTURB_MEMO_FRACTION = 0.1  # Fraction of fake samples to replace with real samples.
PERTURB_MEMO_SEED = 10  # RNG seed for memoisation sample selection.

# class-removal perturbation:
PERTURB_CLASS_REMOVAL = True  # Enables class-removal perturbation (simulate mode dropping).
PERTURB_CLASS_REMOVAL_STRATEGY = "label"  # Selection strategy: "label" or "kmeans".
PERTURB_CLASS_REMOVAL_TARGETS = "Smiling"  # Target labels/clusters to remove (comma-separated).
PERTURB_CLASS_REMOVAL_KMEANS_K = 8  # Number of clusters when strategy="kmeans".
PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH = ""  # Optional path to cached KMeans assignments.
PERTURB_CLASS_REMOVAL_KMEANS_RECREATE = False  # Recompute KMeans cache even if cache exists.
PERTURB_CLASS_REMOVAL_SEED = 10  # RNG seed for stochastic selection operations.
PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD = 0.0  # Multi-label positive threshold.
PERTURB_CLASS_REMOVAL_MIN_KEPT = 4  # Safety floor: minimum samples kept after filtering.
PERTURB_CLASS_FIXED_EVAL = True  # Keep class-removal and class-imbalance metrics on a fixed fake count.
PERTURB_CLASS_EVAL_COUNT = 0  # 0 falls back to METRICS_SAMPLES for class-removal and class-imbalance runs.
PERTURB_CLASS_POOL_SIZE = 0  # 0 lets the runtime choose a larger fake pool from the multiplier.
PERTURB_CLASS_POOL_MULTIPLIER = 3.0  # Starting fake-pool multiplier for class-removal and class-imbalance runs.

# class-imbalance perturbation:
PERTURB_CLASS_IMBALANCE = False  # Enables class-imbalance perturbation.
PERTURB_CLASS_IMBALANCE_STRATEGY = "label"  # Selection strategy: "label" or "kmeans".
PERTURB_CLASS_IMBALANCE_TARGETS = ""  # Target labels/clusters to downsample (comma-separated).
# single ratio "0.3" or per-target list "0.2,0.6"
PERTURB_CLASS_IMBALANCE_BALANCE = "0.5"  # Drop fraction for target classes (or per-target fractions list).
PERTURB_CLASS_IMBALANCE_KMEANS_K = 8  # Number of clusters when strategy="kmeans".
PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH = ""  # Optional path to cached KMeans assignments.
PERTURB_CLASS_IMBALANCE_KMEANS_RECREATE = False  # Recompute KMeans cache even if cache exists.
PERTURB_CLASS_IMBALANCE_SEED = 10  # RNG seed for imbalance sampling.
PERTURB_CLASS_IMBALANCE_LABEL_THRESHOLD = 0.0  # Multi-label positive threshold.
PERTURB_CLASS_IMBALANCE_MIN_KEPT = 4  # Safety floor: minimum samples kept after downsampling.
# sample-size perturbation:
PERTURB_SAMPLE_SIZE = False  # Enables sample-size perturbation.
PERTURB_SAMPLE_SIZE_N = 30  # Number of samples kept after sample-size perturbation.
PERTURB_SAMPLE_SIZE_SEED = 10  # RNG seed for sample-size selection.
PERTURB_PREPROCESSING = False  # Enables preprocessing-variation perturbation.
PERTURB_PREPROCESSING_VARIANT = "downsample_bilinear"  # Preprocessing variant name to apply.
PERTURB_PREPROCESSING_SCALE = 0.75  # Scale factor used by preprocessing variants.
PERTURB_DOMAIN_SHIFT = False  # Enables domain-shift perturbation (swap real-reference dataset).
PERTURB_DOMAIN_SHIFT_DATASET = ""  # Domain-shift real-reference dataset name.
PERTURB_DOMAIN_SHIFT_DATA_ROOT = ""  # Domain-shift dataset root path.
PERTURB_DOMAIN_SHIFT_IMAGE_SIZE = 0  # Override image size for domain-shift reference (0 keeps default).

# optional explicit checkpoint overrides for model test scripts
DCGAN_TEST_NETG = ""  # Optional explicit DCGAN generator checkpoint path.
WGANGP_TEST_GENERATOR = ""  # Optional explicit WGAN-GP generator checkpoint path.
WGANGP_TEST_CRITIC = ""  # Optional explicit WGAN-GP critic checkpoint path.
STUDIOGAN_TEST_CHECKPOINT = ""  # Optional explicit StudioGAN checkpoint path.
DDPM_TEST_CHECKPOINT = ""  # Optional explicit DDPM checkpoint path.
STYLEGAN2_TEST_CHECKPOINT = ""  # Optional explicit StyleGAN2 checkpoint path.


# strict test mode:
STRICT_TESTS = False  # True fails immediately on missing checkpoints/metric errors in test scripts.


# dataset subset controls
SUBSET_FRACTION: float | None = None  # Keep this fraction of dataset (None disables).
SUBSET_MAX_SAMPLES: int | None = None  # Hard cap for dataset size after subsetting (None disables).
SUBSET_SEED = 10  # RNG seed for subset selection.
SUBSET_STRATEGY = "random"  # Subset strategy: "random" or "class_balanced".
SUBSET_INCLUDE_CLASSES = ""  # Optional class filter include list (comma-separated).
SUBSET_DROP_CLASSES = ""  # Optional class filter drop list (comma-separated).




#______________________________________________________________________________
# batch naming to keep large experiment campaigns easy to identify.
BATCH_NAME = "seed_01"  # Logical campaign name used in output/report paths
GENERATION_SEED = 1 

# active batch suite:
# - "dcgan_pretrained_both" (default): runs both user-provided DCGAN checkpoints
# - "dcgan_cifar10_pretrained": only CIFAR-10 checkpoint sweep
# - "dcgan_mnist_pretrained": only MNIST checkpoint sweep
# - "stylegan2_celeba": existing StyleGAN2/CelebA sweep
# - "wgangp_cifar10": WGAN-GP/CIFAR-10 sweep
# - "wgangp_chestxray14": WGAN-GP/ChestXray14 sweep
# - "ddpm_cifar10": DDPM/CIFAR-10 sweep
# - "studiogan_cifar10": StudioGAN/CIFAR-10 sweep
EXPERIMENT_SUITE = "dcgan_cifar10_pretrained"

# explicit DCGAN checkpoints 
DCGAN_CIFAR10_PRETRAINED_NETG = "Models/saved_weights/netG_best.pth"
DCGAN_MNIST_PRETRAINED_NETG = "Models/saved_weights/netG_epoch_30.pth"


# batch experiments:
# leave empty to run single execution using settings above.
# for resume behavior:
# - each experiment gets a deterministic id
# - completed experiments are skipped on re-run if report exists
# - reports are written incrementally after each experiment
SKIP_COMPLETED_EXPERIMENTS = True  # Skip experiments already marked completed in report JSON.
ENFORCE_TEST_ONLY_EXPERIMENTS = True  # Prevent non-test steps in EXPERIMENTS entries.
REPORT_SUFFIX = "perturbation_tests"  # Report filename suffix for batch summary JSON.
IN_PROCESS_BATCH_RUNNER = True  # Reuse one in-process test session for large perturbation campaigns
GROUPED_BOOTSTRAP_THRESHOLD = 128  # Kept for compatibility, but the policy below now decides which experiments bootstrap
# Baseline overrides are now defined in experiments.py for easier suite maintenance.
EXPERIMENT_BASE_OVERRIDES: dict[str, Any] = default_experiment_base_overrides()

EXPERIMENTS: list[dict[str, Any]] = build_experiments_for_suite(
    experiment_suite=EXPERIMENT_SUITE,
    dcgan_cifar10_pretrained_netg=DCGAN_CIFAR10_PRETRAINED_NETG,
    dcgan_mnist_pretrained_netg=DCGAN_MNIST_PRETRAINED_NETG,
    experiment_base_overrides=EXPERIMENT_BASE_OVERRIDES,
)



TEST_STEP_METADATA: dict[str, tuple[str, str, str]] = {
    "test_dcgan_cifar10": ("dcgan", "cifar10", "dcgan_cifar10_test"),
    "test_dcgan_mnist": ("dcgan", "mnist", "dcgan_mnist_test"),
    "test_wgangp_cifar10": ("wgangp", "cifar10", "wgangp_cifar10_test"),
    "test_wgangp_chestxray14": ("wcgan", "chestxray14", "wgangp_chestxray14_test"),
    "test_studiogan_cifar10": ("studiogan", "cifar10", "studiogan_cifar10_test"),
    "test_ddpm_cifar10": ("ddpm", "cifar10", "ddpm_cifar10_test"),
    "test_stylegan2_celeba": ("stylegan2", "celeba", "stylegan2_celeba_test"),
}

DATASET_SUBDIRS: dict[str, str] = {
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "celeba": "CelebA",
    "chestxray14": "ChestXray14",
}

DATASET_DEFAULT_IMAGE_SIZES: dict[str, int] = {
    "mnist": 32,
    "cifar10": 32,
    "celeba": 256,
    "chestxray14": 32,
}

DATASET_PREP_STEPS: dict[str, str] = {
    "mnist": "prep_mnist_cifar10",
    "cifar10": "prep_mnist_cifar10",
    "celeba": "prep_celeba",
    "chestxray14": "prep_chestxray14",
}

MODEL_SETUP_STEPS: dict[str, str] = {
    "test_wgangp_cifar10": "train_wgangp_cifar10",
    "test_wgangp_chestxray14": "train_wgangp_chestxray14",
    "test_ddpm_cifar10": "stage_ddpm",
    "test_studiogan_cifar10": "stage_studiogan",
    "test_stylegan2_celeba": "stage_stylegan",
}


def _default_settings() -> dict[str, Any]:
    return {
        "PROFILE": PROFILE,
        "CUSTOM_STEPS": list(CUSTOM_STEPS),
        "RUN": RUN,
        "CONTINUE_ON_ERROR": CONTINUE_ON_ERROR,
        "CUDA": CUDA,
        "VERBOSE": VERBOSE,
        "DATA_ROOT": DATA_ROOT,
        "CHECKPOINTS_ROOT": CHECKPOINTS_ROOT,
        "OUTPUTS_ROOT": OUTPUTS_ROOT,
        "IMAGE_SIZE": IMAGE_SIZE,
        "DCGAN_EPOCHS": DCGAN_EPOCHS,
        "DCGAN_BATCH_SIZE": DCGAN_BATCH_SIZE,
        "WGANGP_EPOCHS": WGANGP_EPOCHS,
        "WGANGP_BATCH_SIZE": WGANGP_BATCH_SIZE,
        "TEST_NUM_SAMPLES": TEST_NUM_SAMPLES,
        "TEST_BATCH_SIZE": TEST_BATCH_SIZE,
        "GENERATION_SEED": GENERATION_SEED,
        "REFERENCE_SEED": REFERENCE_SEED,
        "EVAL_METRICS": EVAL_METRICS,
        "METRICS_SAMPLES": METRICS_SAMPLES,
        "METRICS_DOWNLOAD_IF_MISSING": METRICS_DOWNLOAD_IF_MISSING,
        "METRICS_FEATURE_SPACE": METRICS_FEATURE_SPACE,
        "METRICS_FEATURE_BATCH_SIZE": METRICS_FEATURE_BATCH_SIZE,
        "METRICS_FEATURE_DEVICE": METRICS_FEATURE_DEVICE,
        "METRICS_BOOTSTRAP_SAMPLES": METRICS_BOOTSTRAP_SAMPLES,
        "METRICS_BOOTSTRAP_SEED": METRICS_BOOTSTRAP_SEED,
        "METRICS_BOOTSTRAP_ALPHA": METRICS_BOOTSTRAP_ALPHA,
        "USE_PERTURBATIONS": USE_PERTURBATIONS,
        "PERTURB_APPLY_TO": PERTURB_APPLY_TO,
        "PERTURB_DEGRADE": PERTURB_DEGRADE,
        "PERTURB_DEGRADE_SEVERITY": PERTURB_DEGRADE_SEVERITY,
        "PERTURB_DEGRADE_GAUSSIAN_NOISE": PERTURB_DEGRADE_GAUSSIAN_NOISE,
        "PERTURB_DEGRADE_GAUSSIAN_BLUR": PERTURB_DEGRADE_GAUSSIAN_BLUR,
        "PERTURB_DEGRADE_JPEG_COMPRESSION": PERTURB_DEGRADE_JPEG_COMPRESSION,
        "PERTURB_MEMOISATION": PERTURB_MEMOISATION,
        "PERTURB_MEMO_FRACTION": PERTURB_MEMO_FRACTION,
        "PERTURB_MEMO_SEED": PERTURB_MEMO_SEED,
        "PERTURB_CLASS_REMOVAL": PERTURB_CLASS_REMOVAL,
        "PERTURB_CLASS_REMOVAL_STRATEGY": PERTURB_CLASS_REMOVAL_STRATEGY,
        "PERTURB_CLASS_REMOVAL_TARGETS": PERTURB_CLASS_REMOVAL_TARGETS,
        "PERTURB_CLASS_REMOVAL_KMEANS_K": PERTURB_CLASS_REMOVAL_KMEANS_K,
        "PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH": PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH,
        "PERTURB_CLASS_REMOVAL_KMEANS_RECREATE": PERTURB_CLASS_REMOVAL_KMEANS_RECREATE,
        "PERTURB_CLASS_REMOVAL_SEED": PERTURB_CLASS_REMOVAL_SEED,
        "PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD": PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD,
        "PERTURB_CLASS_REMOVAL_MIN_KEPT": PERTURB_CLASS_REMOVAL_MIN_KEPT,
        "PERTURB_CLASS_FIXED_EVAL": PERTURB_CLASS_FIXED_EVAL,
        "PERTURB_CLASS_EVAL_COUNT": PERTURB_CLASS_EVAL_COUNT,
        "PERTURB_CLASS_POOL_SIZE": PERTURB_CLASS_POOL_SIZE,
        "PERTURB_CLASS_POOL_MULTIPLIER": PERTURB_CLASS_POOL_MULTIPLIER,
        "PERTURB_CLASS_IMBALANCE": PERTURB_CLASS_IMBALANCE,
        "PERTURB_CLASS_IMBALANCE_STRATEGY": PERTURB_CLASS_IMBALANCE_STRATEGY,
        "PERTURB_CLASS_IMBALANCE_TARGETS": PERTURB_CLASS_IMBALANCE_TARGETS,
        "PERTURB_CLASS_IMBALANCE_BALANCE": PERTURB_CLASS_IMBALANCE_BALANCE,
        "PERTURB_CLASS_IMBALANCE_KMEANS_K": PERTURB_CLASS_IMBALANCE_KMEANS_K,
        "PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH": PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH,
        "PERTURB_CLASS_IMBALANCE_KMEANS_RECREATE": PERTURB_CLASS_IMBALANCE_KMEANS_RECREATE,
        "PERTURB_CLASS_IMBALANCE_SEED": PERTURB_CLASS_IMBALANCE_SEED,
        "PERTURB_CLASS_IMBALANCE_LABEL_THRESHOLD": PERTURB_CLASS_IMBALANCE_LABEL_THRESHOLD,
        "PERTURB_CLASS_IMBALANCE_MIN_KEPT": PERTURB_CLASS_IMBALANCE_MIN_KEPT,
        "PERTURB_SAMPLE_SIZE": PERTURB_SAMPLE_SIZE,
        "PERTURB_SAMPLE_SIZE_N": PERTURB_SAMPLE_SIZE_N,
        "PERTURB_SAMPLE_SIZE_SEED": PERTURB_SAMPLE_SIZE_SEED,
        "PERTURB_PREPROCESSING": PERTURB_PREPROCESSING,
        "PERTURB_PREPROCESSING_VARIANT": PERTURB_PREPROCESSING_VARIANT,
        "PERTURB_PREPROCESSING_SCALE": PERTURB_PREPROCESSING_SCALE,
        "PERTURB_DOMAIN_SHIFT": PERTURB_DOMAIN_SHIFT,
        "PERTURB_DOMAIN_SHIFT_DATASET": PERTURB_DOMAIN_SHIFT_DATASET,
        "PERTURB_DOMAIN_SHIFT_DATA_ROOT": PERTURB_DOMAIN_SHIFT_DATA_ROOT,
        "PERTURB_DOMAIN_SHIFT_IMAGE_SIZE": PERTURB_DOMAIN_SHIFT_IMAGE_SIZE,
        "DCGAN_TEST_NETG": DCGAN_TEST_NETG,
        "WGANGP_TEST_GENERATOR": WGANGP_TEST_GENERATOR,
        "WGANGP_TEST_CRITIC": WGANGP_TEST_CRITIC,
        "STUDIOGAN_TEST_CHECKPOINT": STUDIOGAN_TEST_CHECKPOINT,
        "DDPM_TEST_CHECKPOINT": DDPM_TEST_CHECKPOINT,
        "STYLEGAN2_TEST_CHECKPOINT": STYLEGAN2_TEST_CHECKPOINT,
        "STRICT_TESTS": STRICT_TESTS,
        "SUBSET_FRACTION": SUBSET_FRACTION,
        "SUBSET_MAX_SAMPLES": SUBSET_MAX_SAMPLES,
        "SUBSET_SEED": SUBSET_SEED,
        "SUBSET_STRATEGY": SUBSET_STRATEGY,
        "SUBSET_INCLUDE_CLASSES": SUBSET_INCLUDE_CLASSES,
        "SUBSET_DROP_CLASSES": SUBSET_DROP_CLASSES,
        "BATCH_NAME": BATCH_NAME,
    }


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _safe_slug(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if not value:
        return "unnamed"
    slug = "".join(ch if ch.isalnum() else "_" for ch in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unnamed"


def _json_dumps_sorted(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _perturbations_enabled(settings: dict[str, Any]) -> bool:
    return bool(
        settings["USE_PERTURBATIONS"]
        or settings["PERTURB_DEGRADE"]
        or settings["PERTURB_MEMOISATION"]
        or settings["PERTURB_CLASS_REMOVAL"]
        or settings["PERTURB_CLASS_IMBALANCE"]
        or settings["PERTURB_SAMPLE_SIZE"]
        or settings["PERTURB_PREPROCESSING"]
        or settings["PERTURB_DOMAIN_SHIFT"]
    )


def _append_perturbation_args(cmd: list[str], settings: dict[str, Any]) -> None:
    if not _perturbations_enabled(settings):
        return

    cmd.append("--use-perturbations")
    cmd.extend(["--perturb-apply-to", str(settings["PERTURB_APPLY_TO"])])

    if settings["PERTURB_DEGRADE"]:
        cmd.append("--perturb-degrade")
        cmd.extend(["--perturb-degrade-severity", str(settings["PERTURB_DEGRADE_SEVERITY"])])
        if settings["PERTURB_DEGRADE_GAUSSIAN_NOISE"]:
            cmd.append("--perturb-degrade-gaussian-noise")
        if settings["PERTURB_DEGRADE_GAUSSIAN_BLUR"]:
            cmd.append("--perturb-degrade-gaussian-blur")
        if settings["PERTURB_DEGRADE_JPEG_COMPRESSION"]:
            cmd.append("--perturb-degrade-jpeg-compression")

    if settings["PERTURB_MEMOISATION"]:
        cmd.append("--perturb-memoisation")
        cmd.extend(["--perturb-memo-fraction", str(settings["PERTURB_MEMO_FRACTION"])])
        cmd.extend(["--perturb-memo-seed", str(settings["PERTURB_MEMO_SEED"])])

    if settings["PERTURB_CLASS_REMOVAL"]:
        cmd.append("--perturb-class-removal")
        cmd.extend(["--perturb-class-removal-strategy", str(settings["PERTURB_CLASS_REMOVAL_STRATEGY"])])
        cmd.extend(["--perturb-class-removal-targets", str(settings["PERTURB_CLASS_REMOVAL_TARGETS"])])
        cmd.extend(["--perturb-class-removal-kmeans-k", str(settings["PERTURB_CLASS_REMOVAL_KMEANS_K"])])
        if str(settings["PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH"]).strip():
            cmd.extend(
                [
                    "--perturb-class-removal-kmeans-cache-path",
                    str(settings["PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH"]).strip(),
                ]
            )
        if settings["PERTURB_CLASS_REMOVAL_KMEANS_RECREATE"]:
            cmd.append("--perturb-class-removal-kmeans-recreate")
        cmd.extend(["--perturb-class-removal-seed", str(settings["PERTURB_CLASS_REMOVAL_SEED"])])
        cmd.extend(
            [
                "--perturb-class-removal-label-threshold",
                str(settings["PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD"]),
            ]
        )
        cmd.extend(["--perturb-class-removal-min-kept", str(settings["PERTURB_CLASS_REMOVAL_MIN_KEPT"])])
    if settings["PERTURB_CLASS_IMBALANCE"]:
        cmd.append("--perturb-class-imbalance")
        cmd.extend(
            ["--perturb-class-imbalance-strategy", str(settings["PERTURB_CLASS_IMBALANCE_STRATEGY"])]
        )
        cmd.extend(["--perturb-class-imbalance-targets", str(settings["PERTURB_CLASS_IMBALANCE_TARGETS"])])
        cmd.extend(["--perturb-class-imbalance-balance", str(settings["PERTURB_CLASS_IMBALANCE_BALANCE"])])
        cmd.extend(["--perturb-class-imbalance-kmeans-k", str(settings["PERTURB_CLASS_IMBALANCE_KMEANS_K"])])
        if str(settings["PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH"]).strip():
            cmd.extend(
                [
                    "--perturb-class-imbalance-kmeans-cache-path",
                    str(settings["PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH"]).strip(),
                ]
            )
        if settings["PERTURB_CLASS_IMBALANCE_KMEANS_RECREATE"]:
            cmd.append("--perturb-class-imbalance-kmeans-recreate")
        cmd.extend(["--perturb-class-imbalance-seed", str(settings["PERTURB_CLASS_IMBALANCE_SEED"])])
        cmd.extend(
            [
                "--perturb-class-imbalance-label-threshold",
                str(settings["PERTURB_CLASS_IMBALANCE_LABEL_THRESHOLD"]),
            ]
        )
        cmd.extend(["--perturb-class-imbalance-min-kept", str(settings["PERTURB_CLASS_IMBALANCE_MIN_KEPT"])])

    if settings["PERTURB_CLASS_REMOVAL"] or settings["PERTURB_CLASS_IMBALANCE"]:
        if settings["PERTURB_CLASS_FIXED_EVAL"]:
            cmd.append("--perturb-class-fixed-eval")
        else:
            cmd.append("--no-perturb-class-fixed-eval")
        cmd.extend(["--perturb-class-eval-count", str(settings["PERTURB_CLASS_EVAL_COUNT"])])
        cmd.extend(["--perturb-class-pool-size", str(settings["PERTURB_CLASS_POOL_SIZE"])])
        cmd.extend(["--perturb-class-pool-multiplier", str(settings["PERTURB_CLASS_POOL_MULTIPLIER"])])
    if settings["PERTURB_SAMPLE_SIZE"]:
        cmd.append("--perturb-sample-size")
        cmd.extend(["--perturb-sample-size-n", str(settings["PERTURB_SAMPLE_SIZE_N"])])
        cmd.extend(["--perturb-sample-size-seed", str(settings["PERTURB_SAMPLE_SIZE_SEED"])])
    if settings["PERTURB_PREPROCESSING"]:
        cmd.append("--perturb-preprocessing")
        cmd.extend(["--perturb-preprocessing-variant", str(settings["PERTURB_PREPROCESSING_VARIANT"])])
        cmd.extend(["--perturb-preprocessing-scale", str(settings["PERTURB_PREPROCESSING_SCALE"])])
    if settings["PERTURB_DOMAIN_SHIFT"]:
        cmd.append("--perturb-domain-shift")
        cmd.extend(["--perturb-domain-shift-dataset", str(settings["PERTURB_DOMAIN_SHIFT_DATASET"])])
        cmd.extend(["--perturb-domain-shift-data-root", str(settings["PERTURB_DOMAIN_SHIFT_DATA_ROOT"])])
        cmd.extend(["--perturb-domain-shift-image-size", str(settings["PERTURB_DOMAIN_SHIFT_IMAGE_SIZE"])])


def _build_pipeline_command(pipeline_script: Path, settings: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--profile",
        str(settings["PROFILE"]),
        "--data-root",
        str(settings["DATA_ROOT"]),
        "--checkpoints-root",
        str(settings["CHECKPOINTS_ROOT"]),
        "--outputs-root",
        str(settings["OUTPUTS_ROOT"]),
        "--image-size",
        str(settings["IMAGE_SIZE"]),
        "--dcgan-epochs",
        str(settings["DCGAN_EPOCHS"]),
        "--dcgan-batch-size",
        str(settings["DCGAN_BATCH_SIZE"]),
        "--wgangp-epochs",
        str(settings["WGANGP_EPOCHS"]),
        "--wgangp-batch-size",
        str(settings["WGANGP_BATCH_SIZE"]),
        "--test-num-samples",
        str(settings["TEST_NUM_SAMPLES"]),
        "--test-batch-size",
        str(settings["TEST_BATCH_SIZE"]),
        "--generation-seed",
        str(settings["GENERATION_SEED"]),
        "--reference-seed",
        str(settings["REFERENCE_SEED"]),
        "--metrics-samples",
        str(settings["METRICS_SAMPLES"]),
        "--metrics-feature-space",
        str(settings["METRICS_FEATURE_SPACE"]),
        "--metrics-feature-batch-size",
        str(settings["METRICS_FEATURE_BATCH_SIZE"]),
        "--metrics-feature-device",
        str(settings["METRICS_FEATURE_DEVICE"]),
        "--metrics-bootstrap-samples",
        str(settings["METRICS_BOOTSTRAP_SAMPLES"]),
        "--metrics-bootstrap-seed",
        str(settings["METRICS_BOOTSTRAP_SEED"]),
        "--metrics-bootstrap-alpha",
        str(settings["METRICS_BOOTSTRAP_ALPHA"]),
        "--subset-seed",
        str(settings["SUBSET_SEED"]),
        "--subset-strategy",
        str(settings["SUBSET_STRATEGY"]),
    ]

    custom_steps = _to_str_list(settings["CUSTOM_STEPS"])
    if custom_steps:
        cmd.extend(["--steps", *custom_steps])

    if settings["SUBSET_FRACTION"] is not None:
        cmd.extend(["--subset-fraction", str(settings["SUBSET_FRACTION"])])
    if settings["SUBSET_MAX_SAMPLES"] is not None:
        cmd.extend(["--subset-max-samples", str(settings["SUBSET_MAX_SAMPLES"])])

    if str(settings["SUBSET_INCLUDE_CLASSES"]).strip():
        cmd.extend(["--subset-include-classes", str(settings["SUBSET_INCLUDE_CLASSES"]).strip()])
    if str(settings["SUBSET_DROP_CLASSES"]).strip():
        cmd.extend(["--subset-drop-classes", str(settings["SUBSET_DROP_CLASSES"]).strip()])

    if str(settings["DCGAN_TEST_NETG"]).strip():
        cmd.extend(["--dcgan-test-netg", str(settings["DCGAN_TEST_NETG"]).strip()])
    if str(settings["WGANGP_TEST_GENERATOR"]).strip():
        cmd.extend(["--wgangp-test-generator", str(settings["WGANGP_TEST_GENERATOR"]).strip()])
    if str(settings["WGANGP_TEST_CRITIC"]).strip():
        cmd.extend(["--wgangp-test-critic", str(settings["WGANGP_TEST_CRITIC"]).strip()])
    if str(settings["STUDIOGAN_TEST_CHECKPOINT"]).strip():
        cmd.extend(["--studiogan-test-checkpoint", str(settings["STUDIOGAN_TEST_CHECKPOINT"]).strip()])
    if str(settings["DDPM_TEST_CHECKPOINT"]).strip():
        cmd.extend(["--ddpm-test-checkpoint", str(settings["DDPM_TEST_CHECKPOINT"]).strip()])
    if str(settings["STYLEGAN2_TEST_CHECKPOINT"]).strip():
        cmd.extend(["--stylegan2-test-checkpoint", str(settings["STYLEGAN2_TEST_CHECKPOINT"]).strip()])

    if settings["EVAL_METRICS"]:
        cmd.append("--eval-metrics")
    else:
        cmd.append("--no-eval-metrics")
    if settings["METRICS_DOWNLOAD_IF_MISSING"]:
        cmd.append("--metrics-download-if-missing")

    _append_perturbation_args(cmd, settings)

    if settings["CUDA"]:
        cmd.append("--cuda")
    if settings["VERBOSE"]:
        cmd.append("--verbose")
    if settings["STRICT_TESTS"]:
        cmd.append("--strict-tests")
    if settings["RUN"]:
        cmd.append("--run")
    if settings["CONTINUE_ON_ERROR"]:
        cmd.append("--continue-on-error")
    return cmd


def _dataset_is_ready(*, dataset_name: str, data_root: str, image_size: int):
    from Datasets.unified_dataset_loader import make_default_loader

    try:
        loader = make_default_loader(
            dataset_name=str(dataset_name),
            data_root=str(data_root),
            image_size=int(image_size),
        )
        dataset = loader.get_dataset(train=False, download=False)
        return len(dataset) > 0
    except Exception:
        return False


def _run_prerequisite_steps(
    *,
    repo_root: Path,
    pipeline_script: Path,
    base_settings,
    step_names,
    reason: str,
) :
    unique_steps: list[str] = []
    for step_name in step_names:
        if step_name not in unique_steps:
            unique_steps.append(step_name)

    if not unique_steps:
        return

    print(f"\nPreflight: {reason}", flush=True)
    for step_name in unique_steps:
        settings = dict(base_settings)
        settings["CUSTOM_STEPS"] = [step_name]
        cmd = _build_pipeline_command(pipeline_script=pipeline_script, settings=settings)
        if step_name in {"stage_studiogan", "stage_stylegan"}:
            cmd.append("--force")
        print(" ".join(cmd), flush=True)
        if not bool(settings["RUN"]):
            continue
        subprocess.run(cmd, cwd=str(repo_root), check=True)


def _required_datasets_for_batch(resolved_records, base_settings) :
    required: dict[str, dict[str, Any]] = {}

    def _add_dataset(dataset_name: str, data_root = None, image_size = None):
        normalized = str(dataset_name).strip().lower()
        if not normalized:
            return
        if normalized in required:
            return
        required[normalized] = {
            "dataset_name": normalized,
            "data_root": str(
                data_root
                or Path(str(base_settings["DATA_ROOT"])) / DATASET_SUBDIRS.get(normalized, normalized)
            ),
            "image_size": int(image_size or DATASET_DEFAULT_IMAGE_SIZES.get(normalized, IMAGE_SIZE)),
            "prep_step": DATASET_PREP_STEPS.get(normalized),
        }

    for record in resolved_records:
        _add_dataset(str(record["dataset_name"]))
        overrides = record["overrides"]
        if bool(overrides.get("PERTURB_DOMAIN_SHIFT", False)):
            _add_dataset(
                str(overrides.get("PERTURB_DOMAIN_SHIFT_DATASET", "")),
                str(overrides.get("PERTURB_DOMAIN_SHIFT_DATA_ROOT", "")) or None,
                int(overrides.get("PERTURB_DOMAIN_SHIFT_IMAGE_SIZE", 0) or 0) or None,
            )

    return required


def _ensure_datasets_ready(
    *,
    repo_root: Path,
    pipeline_script: Path,
    base_settings,
    resolved_records,
) :
    required = _required_datasets_for_batch(resolved_records, base_settings)
    missing_steps: list[str] = []
    missing_labels: list[str] = []

    for payload in required.values():
        if _dataset_is_ready(
            dataset_name=str(payload["dataset_name"]),
            data_root=str(payload["data_root"]),
            image_size=int(payload["image_size"]),
        ):
            continue
        prep_step = payload.get("prep_step")
        if prep_step is None:
            raise FileNotFoundError(
                f"No prep step is configured for dataset '{payload['dataset_name']}'."
            )
        missing_steps.append(str(prep_step))
        missing_labels.append(str(payload["dataset_name"]))

    if missing_steps:
        _run_prerequisite_steps(
            repo_root=repo_root,
            pipeline_script=pipeline_script,
            base_settings=base_settings,
            step_names=missing_steps,
            reason=f"loading missing datasets: {', '.join(sorted(set(missing_labels)))}",
        )

    for payload in required.values():
        if not _dataset_is_ready(
            dataset_name=str(payload["dataset_name"]),
            data_root=str(payload["data_root"]),
            image_size=int(payload["image_size"]),
        ):
            raise FileNotFoundError(
                f"Dataset preflight failed for '{payload['dataset_name']}' at {payload['data_root']}."
            )


def _build_candidate_from_command(repo_root, step_name, cmd) :
    module_name = _module_name_from_step_command(repo_root, cmd)
    module = importlib.import_module(module_name)
    parsed_args = module.parse_args(cmd[2:])
    return {
        "step_name": step_name,
        "command": cmd,
        "module_name": module_name,
        "module": module,
        "args": parsed_args,
        "session_key": f"{step_name}:{_in_process_session_key(repo_root, cmd)}",
    }


def _checkpoint_paths_for_step(step_name: str, args) -> list[Path]:
    paths: list[Path] = []
    if step_name.startswith("test_dcgan"):
        paths.append(Path(str(getattr(args, "netG", ""))))
        return [path for path in paths if str(path)]
    if step_name.startswith("test_wgangp"):
        generator_path = Path(str(getattr(args, "generator_checkpoint", "")))
        critic_path = Path(str(getattr(args, "critic_checkpoint", "")))
        paths.append(generator_path)
        if str(getattr(args, "critic_checkpoint", "")).strip():
            paths.append(critic_path)
        return [path for path in paths if str(path)]
    if step_name.startswith(("test_studiogan", "test_ddpm", "test_stylegan2")):
        paths.append(Path(str(getattr(args, "checkpoint", ""))))
        return [path for path in paths if str(path)]
    return []


def _extra_required_paths_for_step(step_name: str, args) -> list[Path]:
    if step_name.startswith("test_studiogan"):
        checkpoint = Path(str(getattr(args, "checkpoint", "")))
        repo_path_raw = str(getattr(args, "repo_path", "")).strip()
        repo_path = Path(repo_path_raw) if repo_path_raw else checkpoint.parent / "studioGAN_src"
        return [repo_path]
    return []


def _artifacts_missing_for_step(step_name: str, args) -> bool:
    checkpoint_paths = _checkpoint_paths_for_step(step_name, args)
    extra_paths = _extra_required_paths_for_step(step_name, args)
    if not checkpoint_paths and not extra_paths:
        return False
    return any(not path.exists() for path in checkpoint_paths + extra_paths)


def _ensure_models_ready(
    *,
    repo_root: Path,
    pipeline_script: Path,
    base_settings,
    resolved_records,
):
    checked_keys: set[str] = set()

    for record in resolved_records:
        settings = record["settings"]
        step_name = _to_str_list(settings["CUSTOM_STEPS"])[0]
        cmd = record["command"]
        candidate = record["candidate"] or _build_candidate_from_command(repo_root, step_name, cmd)
        session_key = str(candidate["session_key"])
        if session_key in checked_keys:
            continue

        setup_step = MODEL_SETUP_STEPS.get(step_name)
        should_attempt_setup = bool(setup_step) and _artifacts_missing_for_step(step_name, candidate["args"])

        for attempt in range(2):
            probe_args = copy.deepcopy(candidate["args"])
            setattr(probe_args, "strict", True)
            prepared = None
            try:
                prepared = candidate["module"].prepare_run(probe_args)
                if prepared is None:
                    raise FileNotFoundError(f"prepare_run returned None for {step_name}.")
                close_prepared_test_run(prepared)
                checked_keys.add(session_key)
                break
            except Exception as exc:
                close_prepared_test_run(prepared)
                if attempt == 0 and should_attempt_setup and setup_step is not None:
                    try:
                        _run_prerequisite_steps(
                            repo_root=repo_root,
                            pipeline_script=pipeline_script,
                            base_settings=base_settings,
                            step_names=[setup_step],
                            reason=f"loading missing model for {step_name}",
                        )
                        continue
                    except subprocess.CalledProcessError as setup_exc:
                        raise RuntimeError(
                            f"Model preflight failed for {step_name}. "
                            f"the checkpoint looked missing, so i tried setup step '{setup_step}', "
                            f"but that setup step failed with exit code {setup_exc.returncode}. "
                            f"original prepare_run error: {exc}"
                        ) from setup_exc
                raise RuntimeError(f"Model preflight failed for {step_name}: {exc}") from exc


def _ensure_batch_prerequisites(
    *,
    repo_root: Path,
    pipeline_script: Path,
    base_settings,
    resolved_records,
):
    _ensure_datasets_ready(
        repo_root=repo_root,
        pipeline_script=pipeline_script,
        base_settings=base_settings,
        resolved_records=resolved_records,
    )
    _ensure_models_ready(
        repo_root=repo_root,
        pipeline_script=pipeline_script,
        base_settings=base_settings,
        resolved_records=resolved_records,
    )


def _step_metadata(step_name: str) -> tuple[str, str, str]:
    if step_name in TEST_STEP_METADATA:
        return TEST_STEP_METADATA[step_name]
    if step_name.startswith("test_"):
        fallback = step_name.removeprefix("test_")
        return fallback or "unknown", "unknown", f"{fallback}_test"
    return "unknown", "unknown", ""


def _infer_model_dataset_from_steps(steps: list[str]) -> tuple[str, str]:
    combos: set[tuple[str, str]] = set()
    for step_name in steps:
        if not str(step_name).startswith("test_"):
            continue
        model_name, dataset_name, _ = _step_metadata(step_name)
        combos.add((model_name, dataset_name))

    if len(combos) == 1:
        return next(iter(combos))
    if len(combos) > 1:
        return "mixed", "mixed"
    return "unknown", "unknown"


def _resolve_experiment_settings(
    base_settings: dict[str, Any],
    experiment: dict[str, Any],
    index: int,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    settings = copy.deepcopy(base_settings)
    overrides_raw = experiment.get("overrides", {})
    if not isinstance(overrides_raw, dict):
        raise TypeError(f"Experiment #{index} has invalid 'overrides' (expected dict).")

    unknown_keys = sorted(key for key in overrides_raw.keys() if key not in settings)
    if unknown_keys:
        raise KeyError(f"Experiment #{index} has unknown override keys: {', '.join(unknown_keys)}")

    overrides = dict(overrides_raw)
    settings.update(overrides)

    if "profile" in experiment:
        settings["PROFILE"] = str(experiment["profile"])
    if "steps" in experiment:
        settings["CUSTOM_STEPS"] = _to_str_list(experiment["steps"])
    if "outputs_root" in experiment:
        settings["OUTPUTS_ROOT"] = str(experiment["outputs_root"])

    steps = _to_str_list(settings["CUSTOM_STEPS"])
    if ENFORCE_TEST_ONLY_EXPERIMENTS and any(not step.startswith("test_") for step in steps):
        raise ValueError(
            f"Experiment #{index} includes non-test steps while ENFORCE_TEST_ONLY_EXPERIMENTS=True: {steps}"
        )

    inferred_model, inferred_dataset = _infer_model_dataset_from_steps(steps)
    model_name = str(experiment.get("model_name", "")).strip() or inferred_model
    dataset_name = str(experiment.get("dataset_name", "")).strip() or inferred_dataset

    experiment_name = str(experiment.get("name", "")).strip() or f"experiment_{index:03d}"
    explicit_output_root = ("outputs_root" in experiment) or ("OUTPUTS_ROOT" in overrides)
    if not explicit_output_root:
        batch_slug = _safe_slug(str(settings.get("BATCH_NAME", "default_batch")))
        settings["OUTPUTS_ROOT"] = str(
            Path(str(base_settings["OUTPUTS_ROOT"]))
            / "batch_runs"
            / batch_slug
            / f"{_safe_slug(model_name)}_{_safe_slug(dataset_name)}"
            / _safe_slug(experiment_name)
        )

    return experiment_name, model_name, dataset_name, settings, overrides


def _build_experiment_id(
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    settings: dict[str, Any],
    overrides: dict[str, Any],
) -> str:
    payload = {
        "name": experiment_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "profile": settings["PROFILE"],
        "steps": _to_str_list(settings["CUSTOM_STEPS"]),
        "overrides": overrides,
    }
    digest = hashlib.sha1(_json_dumps_sorted(payload).encode("utf-8")).hexdigest()[:10]
    return f"{_safe_slug(experiment_name)}__{digest}"


def _report_path(base_output_root: Path, model_name: str, dataset_name: str) -> Path:
    filename = (
        f"{_safe_slug(str(BATCH_NAME))}_"
        f"{_safe_slug(model_name)}_{_safe_slug(dataset_name)}_{REPORT_SUFFIX}.json"
    )
    return base_output_root / filename


def _load_report(path: Path, model_name: str, dataset_name: str) -> dict[str, Any]:
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                loaded.setdefault("model_name", model_name)
                loaded.setdefault("dataset_name", dataset_name)
                loaded.setdefault("report_file", str(path))
                loaded.setdefault("experiments", [])
                return loaded
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "report_file": str(path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": [],
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    # Lock-tolerant write for Windows/OneDrive.
    for _ in range(5):
        try:
            path.write_text(payload, encoding="utf-8")
            return
        except PermissionError:
            time.sleep(0.5)
    raise PermissionError(
        f"Could not write report file after retries (file may be locked): {path}"
    )


def _index_by_experiment_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in report.get("experiments", []):
        exp_id = str(entry.get("experiment_id", "")).strip()
        if exp_id:
            indexed[exp_id] = entry
    return indexed


def _upsert_report_entry(report: dict[str, Any], entry: dict[str, Any]) -> None:
    experiments = report.setdefault("experiments", [])
    entry_id = str(entry.get("experiment_id", "")).strip()
    if not entry_id:
        experiments.append(entry)
        return

    for idx, current in enumerate(experiments):
        if str(current.get("experiment_id", "")).strip() == entry_id:
            experiments[idx] = entry
            return
    experiments.append(entry)


def _entry_expects_metrics(entry: dict[str, Any]) -> bool:
    if "metrics_expected" in entry:
        return bool(entry.get("metrics_expected"))
    command = str(entry.get("command", ""))
    return "--eval-metrics" in command and "--no-eval-metrics" not in command


def _entry_has_metrics(entry: dict[str, Any]) -> bool:
    test_outputs = entry.get("test_outputs")
    if not isinstance(test_outputs, list) or not test_outputs:
        return False
    for output in test_outputs:
        if output.get("metrics_report") is None:
            return False
    return True


def _is_completed_entry(entry: dict[str, Any]) -> bool:
    if str(entry.get("status", "")) != "completed":
        return False
    if int(entry.get("exit_code", 1)) != 0:
        return False
    if not _entry_expects_metrics(entry):
        return True
    return _entry_has_metrics(entry)


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_test_outputs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for step_name in _to_str_list(settings["CUSTOM_STEPS"]):
        if not step_name.startswith("test_"):
            continue

        _, _, out_subdir = _step_metadata(step_name)
        if not out_subdir:
            continue

        out_dir = Path(str(settings["OUTPUTS_ROOT"])) / out_subdir
        metrics_path = out_dir / "metrics_report.json"
        perturb_path = out_dir / "perturbation_config.json"
        cache_path = out_dir / "cache_report.json"

        outputs.append(
            {
                "step_name": step_name,
                "output_dir": str(out_dir),
                "metrics_path": str(metrics_path),
                "metrics_report": _read_json_if_exists(metrics_path),
                "perturbation_config_path": str(perturb_path),
                "perturbation_config": _read_json_if_exists(perturb_path),
                "cache_report_path": str(cache_path),
                "cache_report": _read_json_if_exists(cache_path),
            }
        )
    return outputs


def _settings_to_pipeline_namespace(settings: dict[str, Any]) -> argparse.Namespace:
    payload = {str(key).lower(): value for key, value in settings.items()}
    payload["steps"] = _to_str_list(settings.get("CUSTOM_STEPS"))
    return argparse.Namespace(**payload)


def _load_pipeline_runtime_module():
    return importlib.import_module("Tests.run_operations_pipeline")


def _module_name_from_step_command(repo_root: Path, cmd: list[str]) -> str:
    script_path = Path(cmd[1]).resolve()
    relative = script_path.relative_to(repo_root)
    return relative.with_suffix("").as_posix().replace("/", ".")


def _in_process_session_key(repo_root: Path, cmd: list[str]) -> str:
    skip_flags_with_value = {
        "--out-dir",
        "--metrics-dataset",
        "--metrics-data-root",
        "--metrics-samples",
        "--metrics-feature-space",
        "--metrics-feature-batch-size",
        "--metrics-feature-device",
        "--metrics-bootstrap-samples",
        "--metrics-bootstrap-seed",
        "--metrics-bootstrap-alpha",
        "--perturb-apply-to",
        "--perturb-degrade-severity",
        "--perturb-memo-fraction",
        "--perturb-memo-seed",
        "--perturb-class-removal-strategy",
        "--perturb-class-removal-targets",
        "--perturb-class-removal-kmeans-k",
        "--perturb-class-removal-kmeans-cache-path",
        "--perturb-class-removal-seed",
        "--perturb-class-removal-label-threshold",
        "--perturb-class-removal-min-kept",
        "--perturb-class-eval-count",
        "--perturb-class-pool-size",
        "--perturb-class-pool-multiplier",
        "--perturb-class-imbalance-strategy",
        "--perturb-class-imbalance-targets",
        "--perturb-class-imbalance-balance",
        "--perturb-class-imbalance-kmeans-k",
        "--perturb-class-imbalance-kmeans-cache-path",
        "--perturb-class-imbalance-seed",
        "--perturb-class-imbalance-label-threshold",
        "--perturb-class-imbalance-min-kept",
        "--perturb-sample-size-n",
        "--perturb-sample-size-seed",
        "--perturb-preprocessing-variant",
        "--perturb-preprocessing-scale",
        "--perturb-domain-shift-dataset",
        "--perturb-domain-shift-data-root",
        "--perturb-domain-shift-image-size",
    }
    skip_flags_without_value = {
        "--eval-metrics",
        "--no-eval-metrics",
        "--metrics-download-if-missing",
        "--use-perturbations",
        "--perturb-degrade",
        "--perturb-degrade-gaussian-noise",
        "--perturb-degrade-gaussian-blur",
        "--perturb-degrade-jpeg-compression",
        "--perturb-memoisation",
        "--perturb-class-removal",
        "--perturb-class-removal-kmeans-recreate",
        "--perturb-class-fixed-eval",
        "--no-perturb-class-fixed-eval",
        "--perturb-class-imbalance",
        "--perturb-class-imbalance-kmeans-recreate",
        "--perturb-sample-size",
        "--perturb-preprocessing",
        "--perturb-domain-shift",
        "--verbose",
        "--strict",
        "--cuda",
    }

    normalized: list[str] = [_module_name_from_step_command(repo_root, cmd)]
    idx = 2
    while idx < len(cmd):
        token = str(cmd[idx])
        if token in skip_flags_with_value:
            idx += 2
            continue
        if token in skip_flags_without_value:
            idx += 1
            continue
        normalized.append(token)
        if token.startswith("--") and idx + 1 < len(cmd) and not str(cmd[idx + 1]).startswith("--"):
            normalized.append(str(cmd[idx + 1]))
            idx += 2
            continue
        idx += 1

    return hashlib.sha1(_json_dumps_sorted(normalized).encode("utf-8")).hexdigest()


def _build_in_process_candidate(repo_root: Path, settings: dict[str, Any]) -> dict[str, Any] | None:
    steps = _to_str_list(settings["CUSTOM_STEPS"])
    if len(steps) != 1:
        return None
    step_name = steps[0]
    if not step_name.startswith("test_"):
        return None

    pipeline_module = _load_pipeline_runtime_module()
    step_builder = pipeline_module.STEP_BUILDERS.get(step_name)
    if step_builder is None:
        return None

    namespace = _settings_to_pipeline_namespace(settings)
    cmd = step_builder(namespace)
    module_name = _module_name_from_step_command(repo_root, cmd)
    module = importlib.import_module(module_name)
    parsed_args = module.parse_args(cmd[2:])
    return {
        "step_name": step_name,
        "command": cmd,
        "module_name": module_name,
        "module": module,
        "args": parsed_args,
        "session_key": f"{step_name}:{_in_process_session_key(repo_root, cmd)}",
    }


def _bootstrap_bucket(name: str, overrides: dict[str, Any]) -> tuple[Any, ...] | None:
    if str(name).strip() == "baseline_no_perturbation":
        return ("baseline",)

    if bool(overrides.get("PERTURB_SAMPLE_SIZE", False)):
        return (
            "sample_size",
            int(overrides.get("PERTURB_SAMPLE_SIZE_N", 0)),
            str(overrides.get("PERTURB_APPLY_TO", "fake")),
        )

    return None


def _effective_bootstrap_samples(
    *,
    group_size: int,
    name: str,
    overrides: dict[str, Any],
    requested_samples: int,
    seen_buckets: set[tuple[Any, ...]],
) -> tuple[int, str]:
    # i only keep bootstrap on the baseline and sample-size runs.
    requested = int(requested_samples)
    if requested <= 0:
        return 0, "disabled"

    bucket = _bootstrap_bucket(name=name, overrides=overrides)
    if bucket is None:
        return 0, "baseline_and_sample_size_only"
    seen_buckets.add(bucket)
    return requested, "baseline_and_sample_size_only"


def _run_single(repo_root: Path, pipeline_script: Path, settings: dict[str, Any]) -> None:
    cmd = _build_pipeline_command(pipeline_script=pipeline_script, settings=settings)
    print("Calling pipeline with command:", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _run_batch(repo_root: Path, pipeline_script: Path, base_settings: dict[str, Any]) -> None:
    from Scripts.evaluation_runtime import create_evaluation_reuse_session

    reports_cache: dict[Path, dict[str, Any]] = {}
    failed_experiments: list[str] = []
    skipped_count = 0
    resolved_records: list[dict[str, Any]] = []
    group_sizes: dict[str, int] = {}
    group_states: dict[str, dict[str, Any]] = {}

    for idx, experiment in enumerate(EXPERIMENTS, start=1):
        name, model_name, dataset_name, settings, overrides = _resolve_experiment_settings(
            base_settings=base_settings,
            experiment=experiment,
            index=idx,
        )
        cmd = _build_pipeline_command(pipeline_script=pipeline_script, settings=settings)
        candidate = None
        if IN_PROCESS_BATCH_RUNNER and bool(settings["RUN"]):
            candidate = _build_in_process_candidate(repo_root, settings)
            if candidate is not None:
                group_sizes[candidate["session_key"]] = group_sizes.get(candidate["session_key"], 0) + 1
        resolved_records.append(
            {
                "idx": idx,
                "experiment": experiment,
                "name": name,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "settings": settings,
                "overrides": overrides,
                "command": cmd,
                "candidate": candidate,
            }
        )

    _ensure_batch_prerequisites(
        repo_root=repo_root,
        pipeline_script=pipeline_script,
        base_settings=base_settings,
        resolved_records=resolved_records,
    )

    print(
        f"Running batch '{base_settings.get('BATCH_NAME', BATCH_NAME)}' with {len(EXPERIMENTS)} experiments.",
        flush=True,
    )

    try:
        for record in resolved_records:
            idx = int(record["idx"])
            name = str(record["name"])
            model_name = str(record["model_name"])
            dataset_name = str(record["dataset_name"])
            settings = record["settings"]
            overrides = record["overrides"]
            cmd = record["command"]
            candidate = record["candidate"]
            started_at = datetime.now(timezone.utc)

            exp_id = _build_experiment_id(
                experiment_name=name,
                model_name=model_name,
                dataset_name=dataset_name,
                settings=settings,
                overrides=overrides,
            )

            report_path = _report_path(repo_root / str(base_settings["OUTPUTS_ROOT"]), model_name, dataset_name)
            report = reports_cache.get(report_path)
            if report is None:
                report = _load_report(report_path, model_name=model_name, dataset_name=dataset_name)
                reports_cache[report_path] = report

            existing = _index_by_experiment_id(report).get(exp_id)
            if settings["RUN"] and SKIP_COMPLETED_EXPERIMENTS and existing and _is_completed_entry(existing):
                print(f"Skipping completed experiment {idx}/{len(EXPERIMENTS)}: {name} ({exp_id})", flush=True)
                skipped_count += 1
                continue

            print(f"\nExperiment {idx}/{len(EXPERIMENTS)}: {name}", flush=True)
            print(" ".join(cmd), flush=True)

            exit_code = 0
            test_outputs: list[dict[str, Any]] = []
            if settings["RUN"]:
                if candidate is not None:
                    session_key = str(candidate["session_key"])
                    state = group_states.get(session_key)
                    if state is None:
                        prepared = candidate["module"].prepare_run(candidate["args"])
                        state = {
                            "prepared": prepared,
                            "session": create_evaluation_reuse_session(),
                            "group_size": int(group_sizes.get(session_key, 1)),
                            "seen_bootstrap_buckets": set(),
                        }
                        group_states[session_key] = state

                    try:
                        effective_bootstrap, bootstrap_policy = _effective_bootstrap_samples(
                            group_size=int(state["group_size"]),
                            name=name,
                            overrides=overrides,
                            requested_samples=int(settings["METRICS_BOOTSTRAP_SAMPLES"]),
                            seen_buckets=state["seen_bootstrap_buckets"],
                        )
                        artifacts = candidate["module"].run_with_args(
                            candidate["args"],
                            prepared=state["prepared"],
                            session=state["session"],
                            write_preview=(name == "baseline_no_perturbation"),
                            persist_derived_feature_artifacts=False,
                            bootstrap_samples_override=effective_bootstrap,
                            bootstrap_policy=bootstrap_policy,
                        )
                        if artifacts is not None:
                            test_outputs = [artifacts.to_test_output(candidate["step_name"])]
                    except Exception as exc:
                        exit_code = 1
                        print(f"In-process experiment failed: {exc}", flush=True)
                else:
                    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
                    exit_code = int(completed.returncode)
                    test_outputs = _collect_test_outputs(settings)

            status = "completed" if exit_code == 0 else "failed"
            entry = {
                "experiment_id": exp_id,
                "name": name,
                "status": status,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "batch_name": str(base_settings.get("BATCH_NAME", BATCH_NAME)),
                "profile": str(settings["PROFILE"]),
                "steps": _to_str_list(settings["CUSTOM_STEPS"]),
                "output_root": str(settings["OUTPUTS_ROOT"]),
                "exit_code": exit_code,
                "command": " ".join(cmd),
                "overrides": overrides,
                "test_outputs": test_outputs if settings["RUN"] else [],
                "metrics_expected": bool(settings["EVAL_METRICS"]),
            }
            entry["metrics_available"] = _entry_has_metrics(entry) if settings["RUN"] else False

            _upsert_report_entry(report, entry)
            _write_report(report_path, report)

            if exit_code != 0:
                failed_experiments.append(f"{name} ({exp_id})")
                if not settings["CONTINUE_ON_ERROR"]:
                    break
    finally:
        for state in group_states.values():
            close_prepared_test_run(state.get("prepared"))
            state["session"].close()

    print("\nBatch summary:", flush=True)
    print(f"- total configured experiments: {len(EXPERIMENTS)}", flush=True)
    print(f"- skipped completed experiments: {skipped_count}", flush=True)
    print(f"- report files updated: {len(reports_cache)}", flush=True)
    for path in reports_cache.keys():
        print(str(path), flush=True)

    if failed_experiments:
        print("\nFailed experiments:", flush=True)
        for item in failed_experiments:
            print(f"- {item}", flush=True)
        raise SystemExit(1)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    pipeline_script = repo_root / "Tests" / "run_operations_pipeline.py"
    base_settings = _default_settings()

    if EXPERIMENTS:
        _run_batch(repo_root=repo_root, pipeline_script=pipeline_script, base_settings=base_settings)
        return

    _run_single(repo_root=repo_root, pipeline_script=pipeline_script, settings=base_settings)


if __name__ == "__main__":
    main()

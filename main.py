from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
PROFILE = "test"


# optional custom ordered steps. If non-empty, this overrides PROFILE.
# example: ["prep_mnist_cifar10", "train_dcgan_cifar10", "smoke_models"]
CUSTOM_STEPS = ["prep_mnist_cifar10"]


# execution behavior flags:
# - RUN=False prints commands only (safe dry-run)
# - RUN=True actually executes each planned command
RUN = True

# if True, pipeline keeps going after a failed step
CONTINUE_ON_ERROR = False

# if True, training commands include --cuda (when supported by the target script)
CUDA = False


# shared roots used by pipeline scripts
DATA_ROOT = "data"
CHECKPOINTS_ROOT = "checkpoints"
OUTPUTS_ROOT = "outputs"


# training stuff:
IMAGE_SIZE = 32
DCGAN_EPOCHS = 1
DCGAN_BATCH_SIZE = 64
WGANGP_EPOCHS = 1
WGANGP_BATCH_SIZE = 64


# test stuff:
TEST_NUM_SAMPLES = 64
TEST_BATCH_SIZE = 64


# metric evaluation during test stage:
EVAL_METRICS = True
METRICS_SAMPLES = 64
METRICS_DOWNLOAD_IF_MISSING = False


# perturbation controls:
USE_PERTURBATIONS = False
PERTURB_APPLY_TO = "fake"  # "fake", "real", "both"

# degradation perturbation:
PERTURB_DEGRADE = False
PERTURB_DEGRADE_SEVERITY = 1  # 1..5
PERTURB_DEGRADE_GAUSSIAN_NOISE = False
PERTURB_DEGRADE_GAUSSIAN_BLUR = False
PERTURB_DEGRADE_JPEG_COMPRESSION = False

# memoisation perturbation:
PERTURB_MEMOISATION = False
PERTURB_MEMO_FRACTION = 0.1
PERTURB_MEMO_SEED = 10

# class-removal perturbation:
PERTURB_CLASS_REMOVAL = True
PERTURB_CLASS_REMOVAL_STRATEGY = "label"  # "label" or "kmeans"
PERTURB_CLASS_REMOVAL_TARGETS = "Smiling"
PERTURB_CLASS_REMOVAL_KMEANS_K = 8
PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH = ""
PERTURB_CLASS_REMOVAL_KMEANS_RECREATE = False
PERTURB_CLASS_REMOVAL_SEED = 10
PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD = 0.0
PERTURB_CLASS_REMOVAL_MIN_KEPT = 4

# class-imbalance perturbation:
PERTURB_CLASS_IMBALANCE = False
PERTURB_CLASS_IMBALANCE_STRATEGY = "label"  # "label" or "kmeans"
PERTURB_CLASS_IMBALANCE_TARGETS = ""
# single ratio "0.3" or per-target list "0.2,0.6"
PERTURB_CLASS_IMBALANCE_BALANCE = "0.5"
PERTURB_CLASS_IMBALANCE_KMEANS_K = 8
PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH = ""
PERTURB_CLASS_IMBALANCE_KMEANS_RECREATE = False
PERTURB_CLASS_IMBALANCE_SEED = 10
PERTURB_CLASS_IMBALANCE_LABEL_THRESHOLD = 0.0
PERTURB_CLASS_IMBALANCE_MIN_KEPT = 4


# optional explicit checkpoint overrides for model test scripts
DCGAN_TEST_NETG = ""
WGANGP_TEST_GENERATOR = ""
WGANGP_TEST_CRITIC = ""
STUDIOGAN_TEST_CHECKPOINT = ""
DDPM_TEST_CHECKPOINT = ""
STYLEGAN2_TEST_CHECKPOINT = ""


# strict test mode:
STRICT_TESTS = False


# dataset subset controls
SUBSET_FRACTION: float | None = None
SUBSET_MAX_SAMPLES: int | None = None
SUBSET_SEED = 10
SUBSET_STRATEGY = "random"
SUBSET_INCLUDE_CLASSES = ""
SUBSET_DROP_CLASSES = ""


# batch experiments:
# leave empty to run single execution using settings above.
# for resume behavior:
# - each experiment gets a deterministic id
# - completed experiments are skipped on re-run if report exists
# - reports are written incrementally after each experiment
SKIP_COMPLETED_EXPERIMENTS = True
ENFORCE_TEST_ONLY_EXPERIMENTS = True
REPORT_SUFFIX = "perturbation_tests"
EXPERIMENT_BASE_OVERRIDES: dict[str, Any] = {
    "USE_PERTURBATIONS": True,
    "PERTURB_APPLY_TO": "fake",
    "PERTURB_DEGRADE": False,
    "PERTURB_DEGRADE_SEVERITY": 1,
    "PERTURB_DEGRADE_GAUSSIAN_NOISE": False,
    "PERTURB_DEGRADE_GAUSSIAN_BLUR": False,
    "PERTURB_DEGRADE_JPEG_COMPRESSION": False,
    "PERTURB_MEMOISATION": False,
    "PERTURB_MEMO_FRACTION": 0.1,
    "PERTURB_MEMO_SEED": 10,
    "PERTURB_CLASS_REMOVAL": False,
    "PERTURB_CLASS_REMOVAL_STRATEGY": "label",
    "PERTURB_CLASS_REMOVAL_TARGETS": "",
    "PERTURB_CLASS_REMOVAL_KMEANS_K": 8,
    "PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH": "",
    "PERTURB_CLASS_REMOVAL_KMEANS_RECREATE": False,
    "PERTURB_CLASS_REMOVAL_SEED": 10,
    "PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD": 0.0,
    "PERTURB_CLASS_REMOVAL_MIN_KEPT": 4,
    "PERTURB_CLASS_IMBALANCE": False,
    "PERTURB_CLASS_IMBALANCE_STRATEGY": "label",
    "PERTURB_CLASS_IMBALANCE_TARGETS": "",
    "PERTURB_CLASS_IMBALANCE_BALANCE": "0.5",
    "PERTURB_CLASS_IMBALANCE_KMEANS_K": 8,
    "PERTURB_CLASS_IMBALANCE_KMEANS_CACHE_PATH": "",
    "PERTURB_CLASS_IMBALANCE_KMEANS_RECREATE": False,
    "PERTURB_CLASS_IMBALANCE_SEED": 10,
    "PERTURB_CLASS_IMBALANCE_LABEL_THRESHOLD": 0.0,
    "PERTURB_CLASS_IMBALANCE_MIN_KEPT": 4,
}


# full StyleGAN2/CelebA perturbation experiments
EXPERIMENTS: list[dict[str, Any]] = [
    # baseline:
    {
        "name": "baseline_no_perturbation",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "USE_PERTURBATIONS": False,
        },
    },
    # degradation-only sweeps:
    {
        "name": "degrade_noise_sev1",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 1,
            "PERTURB_DEGRADE_GAUSSIAN_NOISE": True,
        },
    },
    {
        "name": "degrade_noise_sev3",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 3,
            "PERTURB_DEGRADE_GAUSSIAN_NOISE": True,
        },
    },
    {
        "name": "degrade_noise_sev5",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 5,
            "PERTURB_DEGRADE_GAUSSIAN_NOISE": True,
        },
    },
    {
        "name": "degrade_blur_sev1",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 1,
            "PERTURB_DEGRADE_GAUSSIAN_BLUR": True,
        },
    },
    {
        "name": "degrade_blur_sev3",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 3,
            "PERTURB_DEGRADE_GAUSSIAN_BLUR": True,
        },
    },
    {
        "name": "degrade_blur_sev5",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 5,
            "PERTURB_DEGRADE_GAUSSIAN_BLUR": True,
        },
    },
    {
        "name": "degrade_jpeg_sev1",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 1,
            "PERTURB_DEGRADE_JPEG_COMPRESSION": True,
        },
    },
    {
        "name": "degrade_jpeg_sev3",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 3,
            "PERTURB_DEGRADE_JPEG_COMPRESSION": True,
        },
    },
    {
        "name": "degrade_jpeg_sev5",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 5,
            "PERTURB_DEGRADE_JPEG_COMPRESSION": True,
        },
    },
    {
        "name": "degrade_all_sev3",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_DEGRADE": True,
            "PERTURB_DEGRADE_SEVERITY": 3,
            "PERTURB_DEGRADE_GAUSSIAN_NOISE": True,
            "PERTURB_DEGRADE_GAUSSIAN_BLUR": True,
            "PERTURB_DEGRADE_JPEG_COMPRESSION": True,
        },
    },
    # memoisation-only sweeps:
    {
        "name": "memo_frac_05pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_MEMOISATION": True,
            "PERTURB_MEMO_FRACTION": 0.05,
            "PERTURB_MEMO_SEED": 10,
        },
    },
    {
        "name": "memo_frac_15pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_MEMOISATION": True,
            "PERTURB_MEMO_FRACTION": 0.15,
            "PERTURB_MEMO_SEED": 10,
        },
    },
    {
        "name": "memo_frac_30pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_MEMOISATION": True,
            "PERTURB_MEMO_FRACTION": 0.30,
            "PERTURB_MEMO_SEED": 10,
        },
    },
    # class-removal-only sweeps (CelebA is multi-label):
    {
        "name": "class_removal_label_smiling",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_REMOVAL": True,
            "PERTURB_CLASS_REMOVAL_STRATEGY": "label",
            "PERTURB_CLASS_REMOVAL_TARGETS": "Smiling",
            "PERTURB_CLASS_REMOVAL_SEED": 10,
            "PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD": 0.0,
        },
    },
    {
        "name": "class_removal_label_male",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_REMOVAL": True,
            "PERTURB_CLASS_REMOVAL_STRATEGY": "label",
            "PERTURB_CLASS_REMOVAL_TARGETS": "Male",
            "PERTURB_CLASS_REMOVAL_SEED": 10,
            "PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD": 0.0,
        },
    },
    {
        "name": "class_removal_kmeans_k8_cluster0",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_REMOVAL": True,
            "PERTURB_CLASS_REMOVAL_STRATEGY": "kmeans",
            "PERTURB_CLASS_REMOVAL_KMEANS_K": 8,
            "PERTURB_CLASS_REMOVAL_TARGETS": "0",
            "PERTURB_CLASS_REMOVAL_SEED": 10,
        },
    },
    {
        "name": "class_removal_kmeans_k8_cluster3",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_REMOVAL": True,
            "PERTURB_CLASS_REMOVAL_STRATEGY": "kmeans",
            "PERTURB_CLASS_REMOVAL_KMEANS_K": 8,
            "PERTURB_CLASS_REMOVAL_TARGETS": "3",
            "PERTURB_CLASS_REMOVAL_SEED": 10,
        },
    },
    {
        "name": "class_removal_kmeans_k12_cluster1",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_REMOVAL": True,
            "PERTURB_CLASS_REMOVAL_STRATEGY": "kmeans",
            "PERTURB_CLASS_REMOVAL_KMEANS_K": 12,
            "PERTURB_CLASS_REMOVAL_TARGETS": "1",
            "PERTURB_CLASS_REMOVAL_SEED": 10,
        },
    },
    # class-imbalance-only sweeps (CelebA multi-label + kmeans variants):
    {
        "name": "class_imbalance_label_smiling_20pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_IMBALANCE": True,
            "PERTURB_CLASS_IMBALANCE_STRATEGY": "label",
            "PERTURB_CLASS_IMBALANCE_TARGETS": "Smiling",
            "PERTURB_CLASS_IMBALANCE_BALANCE": "0.2",
            "PERTURB_CLASS_IMBALANCE_SEED": 10,
        },
    },
    {
        "name": "class_imbalance_label_smiling_40pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_IMBALANCE": True,
            "PERTURB_CLASS_IMBALANCE_STRATEGY": "label",
            "PERTURB_CLASS_IMBALANCE_TARGETS": "Smiling",
            "PERTURB_CLASS_IMBALANCE_BALANCE": "0.4",
            "PERTURB_CLASS_IMBALANCE_SEED": 10,
        },
    },
    {
        "name": "class_imbalance_label_smiling_60pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_IMBALANCE": True,
            "PERTURB_CLASS_IMBALANCE_STRATEGY": "label",
            "PERTURB_CLASS_IMBALANCE_TARGETS": "Smiling",
            "PERTURB_CLASS_IMBALANCE_BALANCE": "0.6",
            "PERTURB_CLASS_IMBALANCE_SEED": 10,
        },
    },
    {
        "name": "class_imbalance_kmeans_k8_cluster0_30pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_IMBALANCE": True,
            "PERTURB_CLASS_IMBALANCE_STRATEGY": "kmeans",
            "PERTURB_CLASS_IMBALANCE_KMEANS_K": 8,
            "PERTURB_CLASS_IMBALANCE_TARGETS": "0",
            "PERTURB_CLASS_IMBALANCE_BALANCE": "0.3",
            "PERTURB_CLASS_IMBALANCE_SEED": 10,
        },
    },
    {
        "name": "class_imbalance_kmeans_k8_cluster3_50pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_IMBALANCE": True,
            "PERTURB_CLASS_IMBALANCE_STRATEGY": "kmeans",
            "PERTURB_CLASS_IMBALANCE_KMEANS_K": 8,
            "PERTURB_CLASS_IMBALANCE_TARGETS": "3",
            "PERTURB_CLASS_IMBALANCE_BALANCE": "0.5",
            "PERTURB_CLASS_IMBALANCE_SEED": 10,
        },
    },
    {
        "name": "class_imbalance_kmeans_k12_cluster1_40pct",
        "steps": ["test_stylegan2_celeba"],
        "model_name": "stylegan2",
        "dataset_name": "celeba",
        "overrides": {
            **EXPERIMENT_BASE_OVERRIDES,
            "PERTURB_CLASS_IMBALANCE": True,
            "PERTURB_CLASS_IMBALANCE_STRATEGY": "kmeans",
            "PERTURB_CLASS_IMBALANCE_KMEANS_K": 12,
            "PERTURB_CLASS_IMBALANCE_TARGETS": "1",
            "PERTURB_CLASS_IMBALANCE_BALANCE": "0.4",
            "PERTURB_CLASS_IMBALANCE_SEED": 10,
        },
    },
]



TEST_STEP_METADATA: dict[str, tuple[str, str, str]] = {
    "test_dcgan_cifar10": ("dcgan", "cifar10", "dcgan_cifar10_test"),
    "test_dcgan_mnist": ("dcgan", "mnist", "dcgan_mnist_test"),
    "test_wgangp_cifar10": ("wgangp", "cifar10", "wgangp_cifar10_test"),
    "test_studiogan_cifar10": ("studiogan", "cifar10", "studiogan_cifar10_test"),
    "test_ddpm_cifar10": ("ddpm", "cifar10", "ddpm_cifar10_test"),
    "test_stylegan2_celeba": ("stylegan2", "celeba", "stylegan2_celeba_test"),
}


def _default_settings() -> dict[str, Any]:
    return {
        "PROFILE": PROFILE,
        "CUSTOM_STEPS": list(CUSTOM_STEPS),
        "RUN": RUN,
        "CONTINUE_ON_ERROR": CONTINUE_ON_ERROR,
        "CUDA": CUDA,
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
        "EVAL_METRICS": EVAL_METRICS,
        "METRICS_SAMPLES": METRICS_SAMPLES,
        "METRICS_DOWNLOAD_IF_MISSING": METRICS_DOWNLOAD_IF_MISSING,
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
        "--metrics-samples",
        str(settings["METRICS_SAMPLES"]),
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
    if settings["STRICT_TESTS"]:
        cmd.append("--strict-tests")
    if settings["RUN"]:
        cmd.append("--run")
    if settings["CONTINUE_ON_ERROR"]:
        cmd.append("--continue-on-error")
    return cmd


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
        settings["OUTPUTS_ROOT"] = str(
            Path(str(base_settings["OUTPUTS_ROOT"]))
            / "batch_runs"
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
    filename = f"{_safe_slug(model_name)}_{_safe_slug(dataset_name)}_{REPORT_SUFFIX}.json"
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

        outputs.append(
            {
                "step_name": step_name,
                "output_dir": str(out_dir),
                "metrics_path": str(metrics_path),
                "metrics_report": _read_json_if_exists(metrics_path),
                "perturbation_config_path": str(perturb_path),
                "perturbation_config": _read_json_if_exists(perturb_path),
            }
        )
    return outputs


def _run_single(repo_root: Path, pipeline_script: Path, settings: dict[str, Any]) -> None:
    cmd = _build_pipeline_command(pipeline_script=pipeline_script, settings=settings)
    print("Calling pipeline with command:", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _run_batch(repo_root: Path, pipeline_script: Path, base_settings: dict[str, Any]) -> None:
    reports_cache: dict[Path, dict[str, Any]] = {}
    failed_experiments: list[str] = []
    skipped_count = 0

    print(f"Running batch with {len(EXPERIMENTS)} experiments.", flush=True)

    for idx, experiment in enumerate(EXPERIMENTS, start=1):
        name, model_name, dataset_name, settings, overrides = _resolve_experiment_settings(
            base_settings=base_settings,
            experiment=experiment,
            index=idx,
        )
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

        cmd = _build_pipeline_command(pipeline_script=pipeline_script, settings=settings)
        started_at = datetime.now(timezone.utc)

        print(f"\nExperiment {idx}/{len(EXPERIMENTS)}: {name}", flush=True)
        print(f"Experiment ID: {exp_id}", flush=True)
        print("Calling pipeline with command:", flush=True)
        print(" ".join(cmd), flush=True)

        exit_code = 0
        status = "planned"
        if settings["RUN"]:
            completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
            exit_code = int(completed.returncode)
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
            "profile": str(settings["PROFILE"]),
            "steps": _to_str_list(settings["CUSTOM_STEPS"]),
            "output_root": str(settings["OUTPUTS_ROOT"]),
            "exit_code": exit_code,
            "command": " ".join(cmd),
            "overrides": overrides,
            "test_outputs": _collect_test_outputs(settings) if settings["RUN"] else [],
            "metrics_expected": bool(settings["EVAL_METRICS"]),
        }
        entry["metrics_available"] = _entry_has_metrics(entry) if settings["RUN"] else False
        _upsert_report_entry(report, entry)
        _write_report(report_path, report)

        if exit_code != 0:
            failed_experiments.append(f"{name} ({exp_id})")
            if not settings["CONTINUE_ON_ERROR"]:
                print(
                    f"Stopping batch early because experiment failed with exit code {exit_code}: {name}",
                    flush=True,
                )
                break

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

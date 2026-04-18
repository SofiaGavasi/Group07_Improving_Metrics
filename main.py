from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
# use this when you want exact step-level control for debugging
#CUSTOM_STEPS: list[str] = ["test_stylegan2_celeba"]
CUSTOM_STEPS = ["test_stylegan2_celeba"]


"""
these are all of the existing steps:

PROFILES: dict[str, list[str]] = {
    "setup": [
        "prep_mnist_cifar10",
        "prep_celeba",
        "prep_chestxray14",
        "stage_studiogan",
        "stage_ddpm",
        "stage_stylegan",
    ],
    "train": [
        "train_dcgan_cifar10",
        "train_wgangp_cifar10",
    ],
    "test": [
        "test_dcgan_cifar10",
        "test_wgangp_cifar10",
        "test_studiogan_cifar10",
        "test_ddpm_cifar10",
        "test_stylegan2_celeba",
    ],
    "full": [
        "prep_mnist_cifar10",
        "prep_celeba",
        "prep_chestxray14",
        "stage_studiogan",
        "stage_ddpm",
        "stage_stylegan",
        "train_dcgan_cifar10",
        "train_wgangp_cifar10",
        "test_dcgan_cifar10",
        "test_wgangp_cifar10",
        "test_studiogan_cifar10",
        "test_ddpm_cifar10",
        "test_stylegan2_celeba",
        "smoke_models",
    ],
    "smoke": [
        "smoke_models",
    ],
}
"""



# execution behavior flags:

# - RUN=False prints commands only (safe dry-run)
# - RUN=True actually executes each planned command
RUN = True

# if True, pipeline keeps going after a failed step
CONTINUE_ON_ERROR = False

# if True, training commands include --cuda (when supported by the target script)
CUDA = False



# shared roots used by pipeline scripts (shouldn't need to change these)
DATA_ROOT = "data"
CHECKPOINTS_ROOT = "checkpoints"
OUTPUTS_ROOT = "outputs"



# training stuff:
IMAGE_SIZE = 32
DCGAN_EPOCHS = 1
DCGAN_BATCH_SIZE = 64
WGANGP_EPOCHS = 1
WGANGP_BATCH_SIZE = 64



# Test stuff:
# TEST_NUM_SAMPLES controls how many synthetic samples each test script should generate
# TEST_BATCH_SIZE is currently used by DCGAN testing for chunked generation
TEST_NUM_SAMPLES = 64
TEST_BATCH_SIZE = 64


# Metric evaluation during test stage:
# - EVAL_METRICS enables compute_all.py for compatible tests (currently DCGAN test script)
# - METRICS_SAMPLES controls how many real/fake samples are compared
# - METRICS_DOWNLOAD_IF_MISSING enables dataset download fallback during metric evaluation
EVAL_METRICS = True
METRICS_SAMPLES = 64
METRICS_DOWNLOAD_IF_MISSING = False



# Perturbation controls (used in compatible test scripts):
# - USE_PERTURBATIONS or any specific perturbation toggle enables perturbation support
# - PERTURB_APPLY_TO: "fake", "real", or "both"
USE_PERTURBATIONS = False
PERTURB_APPLY_TO = "fake"

# Degradation perturbation:
PERTURB_DEGRADE = False
PERTURB_DEGRADE_SEVERITY = 1   # 1..5
PERTURB_DEGRADE_GAUSSIAN_NOISE = False
PERTURB_DEGRADE_GAUSSIAN_BLUR = False
PERTURB_DEGRADE_JPEG_COMPRESSION = False

# Memoisation perturbation:
PERTURB_MEMOISATION = False
PERTURB_MEMO_FRACTION = 0.1
PERTURB_MEMO_SEED = 10

# Class-removal perturbation (mode dropping):
# - label strategy: drop targets are class names/indices
# - kmeans strategy: build label co-occurrence clusters, then drop cluster ids
PERTURB_CLASS_REMOVAL = True
PERTURB_CLASS_REMOVAL_STRATEGY = "label"   # "label" or "kmeans"
PERTURB_CLASS_REMOVAL_TARGETS = "Smiling"         # example single-label: "0,1" ; multi-label: "Smiling,Pleural_Thickening" ; kmeans: "2,5"
PERTURB_CLASS_REMOVAL_KMEANS_K = 8
# Optional cache path for kmeans label-clusters. Keep empty for default auto-path.
PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH = ""
# if True, existing kmeans cache is ignored and label clusters are rebuilt
PERTURB_CLASS_REMOVAL_KMEANS_RECREATE = False
PERTURB_CLASS_REMOVAL_SEED = 10
PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD = 0.0
PERTURB_CLASS_REMOVAL_MIN_KEPT = 4





# optional explicit checkpoint overrides for model test scripts
# keep empty strings to use pipeline defaults based on CHECKPOINTS_ROOT/OUTPUTS_ROOT
DCGAN_TEST_NETG = ""
WGANGP_TEST_GENERATOR = ""
WGANGP_TEST_CRITIC = ""
STUDIOGAN_TEST_CHECKPOINT = ""
DDPM_TEST_CHECKPOINT = ""
STYLEGAN2_TEST_CHECKPOINT = ""



# strict test mode:
# - False: TODO skeleton tests report placeholders and continue (non-breaking).
# - True: TODO skeleton tests return non-zero exit code and fail pipeline.
STRICT_TESTS = False



# Dataset subset controls:
# - SUBSET_FRACTION: keep this fraction of data (0,1]. None disables fraction cap
# - SUBSET_MAX_SAMPLES: hard cap on samples. None disables cap
# - SUBSET_STRATEGY:
#     "random" = random sample after filters,
#     "class_balanced" = attempts balanced counts per class (single-label datasets)
# - SUBSET_INCLUDE_CLASSES: comma-separated class names/indices to keep
# - SUBSET_DROP_CLASSES: comma-separated class names/indices to remove
# You can combine include/drop with fraction/max-samples
SUBSET_FRACTION: float | None = 0.5
SUBSET_MAX_SAMPLES: int | None = None
SUBSET_SEED = 10
SUBSET_STRATEGY = "random"
SUBSET_INCLUDE_CLASSES = "" #  "1,3"  "cat,dog"
SUBSET_DROP_CLASSES = ""  # "0"




def main():
    repo_root = Path(__file__).resolve().parent
    pipeline_script = repo_root / "Tests" / "run_operations_pipeline.py"

    # base command with arguments:
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--profile",
        PROFILE,
        "--data-root",
        DATA_ROOT,
        "--checkpoints-root",
        CHECKPOINTS_ROOT,
        "--outputs-root",
        OUTPUTS_ROOT,
        "--image-size",
        str(IMAGE_SIZE),
        "--dcgan-epochs",
        str(DCGAN_EPOCHS),
        "--dcgan-batch-size",
        str(DCGAN_BATCH_SIZE),
        "--wgangp-epochs",
        str(WGANGP_EPOCHS),
        "--wgangp-batch-size",
        str(WGANGP_BATCH_SIZE),
        "--test-num-samples",
        str(TEST_NUM_SAMPLES),
        "--test-batch-size",
        str(TEST_BATCH_SIZE),
        "--metrics-samples",
        str(METRICS_SAMPLES),
        "--subset-seed",
        str(SUBSET_SEED),
        "--subset-strategy",
        SUBSET_STRATEGY,
    ]

    # optional arguments:
    if CUSTOM_STEPS:
        cmd.extend(["--steps", *CUSTOM_STEPS])
    if SUBSET_FRACTION is not None:
        cmd.extend(["--subset-fraction", str(SUBSET_FRACTION)])
    if SUBSET_MAX_SAMPLES is not None:
        cmd.extend(["--subset-max-samples", str(SUBSET_MAX_SAMPLES)])
    if SUBSET_INCLUDE_CLASSES.strip():
        cmd.extend(["--subset-include-classes", SUBSET_INCLUDE_CLASSES.strip()])
    if SUBSET_DROP_CLASSES.strip():
        cmd.extend(["--subset-drop-classes", SUBSET_DROP_CLASSES.strip()])
    if DCGAN_TEST_NETG.strip():
        cmd.extend(["--dcgan-test-netg", DCGAN_TEST_NETG.strip()])
    if WGANGP_TEST_GENERATOR.strip():
        cmd.extend(["--wgangp-test-generator", WGANGP_TEST_GENERATOR.strip()])
    if WGANGP_TEST_CRITIC.strip():
        cmd.extend(["--wgangp-test-critic", WGANGP_TEST_CRITIC.strip()])
    if STUDIOGAN_TEST_CHECKPOINT.strip():
        cmd.extend(["--studiogan-test-checkpoint", STUDIOGAN_TEST_CHECKPOINT.strip()])
    if DDPM_TEST_CHECKPOINT.strip():
        cmd.extend(["--ddpm-test-checkpoint", DDPM_TEST_CHECKPOINT.strip()])
    if STYLEGAN2_TEST_CHECKPOINT.strip():
        cmd.extend(["--stylegan2-test-checkpoint", STYLEGAN2_TEST_CHECKPOINT.strip()])
    if EVAL_METRICS:
        cmd.append("--eval-metrics")
    else:
        cmd.append("--no-eval-metrics")
    if METRICS_DOWNLOAD_IF_MISSING:
        cmd.append("--metrics-download-if-missing")

    perturbations_enabled = bool(
        USE_PERTURBATIONS
        or PERTURB_DEGRADE
        or PERTURB_MEMOISATION
        or PERTURB_CLASS_REMOVAL
    )
    if perturbations_enabled:
        cmd.append("--use-perturbations")
        cmd.extend(["--perturb-apply-to", PERTURB_APPLY_TO])
        if PERTURB_DEGRADE:
            cmd.append("--perturb-degrade")
            cmd.extend(["--perturb-degrade-severity", str(PERTURB_DEGRADE_SEVERITY)])
            if PERTURB_DEGRADE_GAUSSIAN_NOISE:
                cmd.append("--perturb-degrade-gaussian-noise")
            if PERTURB_DEGRADE_GAUSSIAN_BLUR:
                cmd.append("--perturb-degrade-gaussian-blur")
            if PERTURB_DEGRADE_JPEG_COMPRESSION:
                cmd.append("--perturb-degrade-jpeg-compression")
        if PERTURB_MEMOISATION:
            cmd.append("--perturb-memoisation")
            cmd.extend(["--perturb-memo-fraction", str(PERTURB_MEMO_FRACTION)])
            cmd.extend(["--perturb-memo-seed", str(PERTURB_MEMO_SEED)])
        if PERTURB_CLASS_REMOVAL:
            cmd.append("--perturb-class-removal")
            cmd.extend(["--perturb-class-removal-strategy", PERTURB_CLASS_REMOVAL_STRATEGY])
            cmd.extend(["--perturb-class-removal-targets", PERTURB_CLASS_REMOVAL_TARGETS])
            cmd.extend(["--perturb-class-removal-kmeans-k", str(PERTURB_CLASS_REMOVAL_KMEANS_K)])
            if PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH.strip():
                cmd.extend(
                    [
                        "--perturb-class-removal-kmeans-cache-path",
                        PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH.strip(),
                    ]
                )
            if PERTURB_CLASS_REMOVAL_KMEANS_RECREATE:
                cmd.append("--perturb-class-removal-kmeans-recreate")
            cmd.extend(["--perturb-class-removal-seed", str(PERTURB_CLASS_REMOVAL_SEED)])
            cmd.extend(
                [
                    "--perturb-class-removal-label-threshold",
                    str(PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD),
                ]
            )
            cmd.extend(["--perturb-class-removal-min-kept", str(PERTURB_CLASS_REMOVAL_MIN_KEPT)])

    if CUDA:
        cmd.append("--cuda")
    if STRICT_TESTS:
        cmd.append("--strict-tests")
    if RUN:
        cmd.append("--run")
    if CONTINUE_ON_ERROR:
        cmd.append("--continue-on-error")


    print("Calling pipeline with command:", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()

"""
dynamic operations runner
 to execute common workflows by calling existing Scripts/*.py files.

examples (from repo root):

    to get information about available steps and profiles:
    py Tests/run_operations_pipeline.py --list

    running predefined profiles (setup, train, test, full, smoke):
    py Tests/run_operations_pipeline.py --profile setup --run
        setup runs all data preparation and staging steps for all datasets and pretrained models, but no training.
        train runs just the training steps (DCGAN and WGAN-GP on CIFAR-10).
        test runs model test entrypoints (DCGAN + model test skeletons).
        full runs everything (setup + train + test + smoke test).
        smoke runs just the smoke test that calls all model files to check for import/runtime errors.

    running custom ordered step lists:
    py Tests/run_operations_pipeline.py --profile full --dcgan-epochs 1 --run
    py Tests/run_operations_pipeline.py --steps prep_mnist_cifar10 train_dcgan_cifar10 smoke_models --run

    running training with subset controls:
    py Tests/run_operations_pipeline.py --profile train --subset-fraction 0.2 --subset-strategy class_balanced --subset-drop-classes 0 --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


StepBuilder = Callable[[argparse.Namespace], list[str]]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Scripts"
TESTS_DIR = REPO_ROOT / "Tests"
PYTHON_EXE = sys.executable


def step_prep_mnist_cifar10(args: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "download_preprocess_mnist_cifar10.py"),
        "--data-root",
        args.data_root,
    ]


def step_prep_celeba(args: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "download_preprocess_celeba.py"),
        "--data-root",
        args.data_root,
    ]


def step_prep_chestxray14(args: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "download_preprocess_chestxray14.py"),
        "--data-root",
        args.data_root,
    ]


def step_stage_studiogan(args: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "download_pretrained_studiogan_cifar10.py"),
        "--output-dir",
        str(Path(args.checkpoints_root) / "StudioGAN" / "CIFAR10"),
    ]


def step_stage_ddpm(args: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "download_pretrained_ddpm_cifar10.py"),
        "--output-dir",
        str(Path(args.checkpoints_root) / "DDPM" / "CIFAR10"),
    ]


def step_stage_stylegan(args: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "download_pretrained_stylegan_celeba.py"),
        "--output-dir",
        str(Path(args.checkpoints_root) / "StyleGAN" / "CelebA"),
    ]


def step_train_dcgan_cifar10(args: argparse.Namespace):
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "train_dcgan.py"),
        "--dataset",
        "cifar10",
        "--data-root",
        str(Path(args.data_root) / "CIFAR10"),
        "--epochs",
        str(args.dcgan_epochs),
        "--batch-size",
        str(args.dcgan_batch_size),
        "--image-size",
        str(args.image_size),
        "--outf",
        str(Path(args.outputs_root) / "dcgan_cifar10"),
    ]
    append_subset_args(cmd, args)
    append_verbose_arg(cmd, args)
    if args.cuda:
        cmd.append("--cuda")
    return cmd


def step_train_dcgan_mnist(args: argparse.Namespace):
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "train_dcgan.py"),
        "--dataset",
        "mnist",
        "--data-root",
        str(Path(args.data_root) / "MNIST"),
        "--epochs",
        str(args.dcgan_epochs),
        "--batch-size",
        str(args.dcgan_batch_size),
        "--image-size",
        str(args.image_size),
        "--outf",
        str(Path(args.outputs_root) / "dcgan_mnist"),
    ]
    append_subset_args(cmd, args)
    append_verbose_arg(cmd, args)
    if args.cuda:
        cmd.append("--cuda")
    return cmd


def step_train_wgangp_cifar10(args: argparse.Namespace):
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "train_wgangp.py"),
        "--dataset",
        "cifar10",
        "--data-root",
        str(Path(args.data_root) / "CIFAR10"),
        "--epochs",
        str(args.wgangp_epochs),
        "--batch-size",
        str(args.wgangp_batch_size),
        "--image-size",
        str(args.image_size),
        "--out-dir",
        str(Path(args.outputs_root) / "wgangp_cifar10"),
    ]
    append_subset_args(cmd, args)
    append_verbose_arg(cmd, args)
    if args.cuda:
        cmd.append("--cuda")
    return cmd


def step_train_wgangp_chestxray14(args: argparse.Namespace):
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "train_wgangp.py"),
        "--dataset",
        "chestxray14",
        "--data-root",
        str(Path(args.data_root) / "ChestXray14"),
        "--epochs",
        str(args.wgangp_epochs),
        "--batch-size",
        str(args.wgangp_batch_size),
        "--image-size",
        str(args.image_size),
        "--out-dir",
        str(Path(args.outputs_root) / "wgangp_chestxray14"),
    ]
    append_subset_args(cmd, args)
    append_verbose_arg(cmd, args)
    if args.cuda:
        cmd.append("--cuda")
    return cmd


def step_smoke_models(_: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(TESTS_DIR / "call_models_files.py"),
    ]


def _dcgan_checkpoint_args(args: argparse.Namespace, *, dataset_name, channels):
    netg_path = args.dcgan_test_netg or str(Path(args.outputs_root) / f"dcgan_{dataset_name}" / "netG_latest.pth")
    return [
        "--netG",
        netg_path,
        "--num-samples",
        str(args.test_num_samples),
        "--batch-size",
        str(args.test_batch_size),
        "--image-size",
        str(args.image_size),
        "--channels",
        str(int(channels)),
    ]


def _wgangp_checkpoint_args(args: argparse.Namespace, *, dataset_name, channels):
    generator_path = args.wgangp_test_generator or str(
        Path(args.outputs_root) / f"wgangp_{dataset_name}" / "netG_latest.pth"
    )
    critic_path = args.wgangp_test_critic or str(
        Path(args.outputs_root) / f"wgangp_{dataset_name}" / "netD_latest.pth"
    )
    return [
        "--generator-checkpoint",
        generator_path,
        "--critic-checkpoint",
        critic_path,
        "--num-samples",
        str(args.test_num_samples),
        "--batch-size",
        str(args.test_batch_size),
        "--image-size",
        str(args.image_size),
        "--channels",
        str(int(channels)),
    ]


def _studiogan_checkpoint_args(args):
    checkpoint = args.studiogan_test_checkpoint or str(
        Path(args.checkpoints_root) / "StudioGAN" / "CIFAR10" / "studioGAN_generator.pkl"
    )
    return [
        "--checkpoint",
        checkpoint,
        "--num-samples",
        str(args.test_num_samples),
    ]


def _ddpm_checkpoint_args(args):
    checkpoint = args.ddpm_test_checkpoint or str(
        Path(args.checkpoints_root) / "DDPM" / "CIFAR10" / "ddpm_model.pth"
    )
    return [
        "--checkpoint",
        checkpoint,
        "--num-samples",
        str(args.test_num_samples),
    ]


def _stylegan2_checkpoint_args(args):
    checkpoint = args.stylegan2_test_checkpoint or str(
        Path(args.checkpoints_root) / "StyleGAN" / "CelebA" / "stylegan2_generator.pkl"
    )
    return [
        "--checkpoint",
        checkpoint,
        "--num-samples",
        str(args.test_num_samples),
    ]


TEST_STEP_SPECS: dict[str, dict[str, object]] = {
    "test_dcgan_cifar10": {
        "script_name": "test_dcgan.py",
        "out_dir_name": "dcgan_cifar10_test",
        "metrics_dataset": "cifar10",
        "metrics_data_root": "CIFAR10",
        "arg_builder": lambda args: _dcgan_checkpoint_args(args, dataset_name="cifar10", channels=3),
    },
    "test_dcgan_mnist": {
        "script_name": "test_dcgan.py",
        "out_dir_name": "dcgan_mnist_test",
        "metrics_dataset": "mnist",
        "metrics_data_root": "MNIST",
        "arg_builder": lambda args: _dcgan_checkpoint_args(args, dataset_name="mnist", channels=1),
    },
    "test_wgangp_cifar10": {
        "script_name": "test_wgangp.py",
        "out_dir_name": "wgangp_cifar10_test",
        "metrics_dataset": "cifar10",
        "metrics_data_root": "CIFAR10",
        "arg_builder": lambda args: _wgangp_checkpoint_args(args, dataset_name="cifar10", channels=3),
    },
    "test_wgangp_chestxray14": {
        "script_name": "test_wgangp.py",
        "out_dir_name": "wgangp_chestxray14_test",
        "metrics_dataset": "chestxray14",
        "metrics_data_root": "ChestXray14",
        "arg_builder": lambda args: _wgangp_checkpoint_args(args, dataset_name="chestxray14", channels=3),
    },
    "test_studiogan_cifar10": {
        "script_name": "test_studiogan.py",
        "out_dir_name": "studiogan_cifar10_test",
        "metrics_dataset": "cifar10",
        "metrics_data_root": "CIFAR10",
        "arg_builder": _studiogan_checkpoint_args,
    },
    "test_ddpm_cifar10": {
        "script_name": "test_ddpm.py",
        "out_dir_name": "ddpm_cifar10_test",
        "metrics_dataset": "cifar10",
        "metrics_data_root": "CIFAR10",
        "arg_builder": _ddpm_checkpoint_args,
    },
    "test_stylegan2_celeba": {
        "script_name": "test_stylegan2.py",
        "out_dir_name": "stylegan2_celeba_test",
        "metrics_dataset": "celeba",
        "metrics_data_root": "CelebA",
        "arg_builder": _stylegan2_checkpoint_args,
    },
}


def _append_common_test_metric_args(cmd, args, spec) :
    if not args.eval_metrics:
        return

    cmd.extend(
        [
            "--eval-metrics",
            "--metrics-dataset",
            str(spec["metrics_dataset"]),
            "--metrics-data-root",
            str(Path(args.data_root) / str(spec["metrics_data_root"])),
            "--metrics-samples",
            str(args.metrics_samples),
            "--metrics-feature-space",
            str(args.metrics_feature_space),
            "--metrics-feature-batch-size",
            str(args.metrics_feature_batch_size),
            "--metrics-feature-device",
            str(args.metrics_feature_device),
            "--metrics-bootstrap-samples",
            str(args.metrics_bootstrap_samples),
            "--metrics-bootstrap-seed",
            str(args.metrics_bootstrap_seed),
            "--metrics-bootstrap-alpha",
            str(args.metrics_bootstrap_alpha),
        ]
    )
    if args.metrics_download_if_missing:
        cmd.append("--metrics-download-if-missing")


def _build_test_step_command(args: argparse.Namespace, step_name: str) -> list[str]:
    spec = TEST_STEP_SPECS[step_name]
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / str(spec["script_name"])),
    ]
    cmd.extend(list(spec["arg_builder"](args)))
    cmd.extend(
        [
            "--out-dir",
            str(Path(args.outputs_root) / str(spec["out_dir_name"])),
        ]
    )
    append_generation_reference_seed_args(cmd, args)
    _append_common_test_metric_args(cmd, args, spec)
    if args.cuda:
        cmd.append("--cuda")
    append_verbose_arg(cmd, args)
    append_perturbation_args(cmd, args)
    if args.strict_tests:
        cmd.append("--strict")
    return cmd


def step_test_dcgan_cifar10(args: argparse.Namespace):
    return _build_test_step_command(args, "test_dcgan_cifar10")


def step_test_dcgan_mnist(args: argparse.Namespace):
    return _build_test_step_command(args, "test_dcgan_mnist")


def step_test_wgangp_cifar10(args: argparse.Namespace):
    return _build_test_step_command(args, "test_wgangp_cifar10")


def step_test_wgangp_chestxray14(args: argparse.Namespace):
    return _build_test_step_command(args, "test_wgangp_chestxray14")


def step_test_studiogan_cifar10(args: argparse.Namespace):
    return _build_test_step_command(args, "test_studiogan_cifar10")


def step_test_ddpm_cifar10(args: argparse.Namespace):
    return _build_test_step_command(args, "test_ddpm_cifar10")


def step_test_stylegan2_celeba(args: argparse.Namespace):
    return _build_test_step_command(args, "test_stylegan2_celeba")


def append_subset_args(cmd: list[str], args: argparse.Namespace):
    if args.subset_fraction is not None:
        cmd.extend(["--subset-fraction", str(args.subset_fraction)])
    if args.subset_max_samples is not None:
        cmd.extend(["--subset-max-samples", str(args.subset_max_samples)])
    cmd.extend(["--subset-seed", str(args.subset_seed)])
    cmd.extend(["--subset-strategy", args.subset_strategy])
    if args.subset_include_classes.strip():
        cmd.extend(["--subset-include-classes", args.subset_include_classes.strip()])
    if args.subset_drop_classes.strip():
        cmd.extend(["--subset-drop-classes", args.subset_drop_classes.strip()])


def append_generation_reference_seed_args(cmd: list[str], args: argparse.Namespace):
    cmd.extend(["--generation-seed", str(args.generation_seed)])
    cmd.extend(["--reference-seed", str(args.reference_seed)])


def append_verbose_arg(cmd: list[str], args: argparse.Namespace):
    if args.verbose:
        cmd.append("--verbose")


def _perturbations_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.use_perturbations
        or args.perturb_degrade
        or args.perturb_memoisation
        or args.perturb_class_removal
        or args.perturb_class_imbalance
        or args.perturb_sample_size
        or args.perturb_preprocessing
        or args.perturb_domain_shift
    )


def append_perturbation_args(cmd: list[str], args: argparse.Namespace):
    if not _perturbations_enabled(args):
        return

    cmd.append("--use-perturbations")
    cmd.extend(["--perturb-apply-to", args.perturb_apply_to])

    if args.perturb_degrade:
        cmd.append("--perturb-degrade")
        cmd.extend(["--perturb-degrade-severity", str(args.perturb_degrade_severity)])
        if args.perturb_degrade_gaussian_noise:
            cmd.append("--perturb-degrade-gaussian-noise")
        if args.perturb_degrade_gaussian_blur:
            cmd.append("--perturb-degrade-gaussian-blur")
        if args.perturb_degrade_jpeg_compression:
            cmd.append("--perturb-degrade-jpeg-compression")

    if args.perturb_memoisation:
        cmd.append("--perturb-memoisation")
        cmd.extend(["--perturb-memo-fraction", str(args.perturb_memo_fraction)])
        cmd.extend(["--perturb-memo-seed", str(args.perturb_memo_seed)])
    if args.perturb_class_removal:
        cmd.append("--perturb-class-removal")
        cmd.extend(["--perturb-class-removal-strategy", args.perturb_class_removal_strategy])
        cmd.extend(["--perturb-class-removal-targets", args.perturb_class_removal_targets])
        cmd.extend(["--perturb-class-removal-kmeans-k", str(args.perturb_class_removal_kmeans_k)])
        if args.perturb_class_removal_kmeans_cache_path.strip():
            cmd.extend(
                [
                    "--perturb-class-removal-kmeans-cache-path",
                    args.perturb_class_removal_kmeans_cache_path.strip(),
                ]
            )
        if args.perturb_class_removal_kmeans_recreate:
            cmd.append("--perturb-class-removal-kmeans-recreate")
        cmd.extend(["--perturb-class-removal-seed", str(args.perturb_class_removal_seed)])
        cmd.extend(
            [
                "--perturb-class-removal-label-threshold",
                str(args.perturb_class_removal_label_threshold),
            ]
        )
        cmd.extend(["--perturb-class-removal-min-kept", str(args.perturb_class_removal_min_kept)])
    if args.perturb_class_imbalance:
        cmd.append("--perturb-class-imbalance")
        cmd.extend(["--perturb-class-imbalance-strategy", args.perturb_class_imbalance_strategy])
        cmd.extend(["--perturb-class-imbalance-targets", args.perturb_class_imbalance_targets])
        cmd.extend(["--perturb-class-imbalance-balance", str(args.perturb_class_imbalance_balance)])
        cmd.extend(["--perturb-class-imbalance-kmeans-k", str(args.perturb_class_imbalance_kmeans_k)])
        if args.perturb_class_imbalance_kmeans_cache_path.strip():
            cmd.extend(
                [
                    "--perturb-class-imbalance-kmeans-cache-path",
                    args.perturb_class_imbalance_kmeans_cache_path.strip(),
                ]
            )
        if args.perturb_class_imbalance_kmeans_recreate:
            cmd.append("--perturb-class-imbalance-kmeans-recreate")
        cmd.extend(["--perturb-class-imbalance-seed", str(args.perturb_class_imbalance_seed)])
        cmd.extend(
            [
                "--perturb-class-imbalance-label-threshold",
                str(args.perturb_class_imbalance_label_threshold),
            ]
        )
        cmd.extend(["--perturb-class-imbalance-min-kept", str(args.perturb_class_imbalance_min_kept)])

    if args.perturb_class_removal or args.perturb_class_imbalance:
        if args.perturb_class_fixed_eval:
            cmd.append("--perturb-class-fixed-eval")
        else:
            cmd.append("--no-perturb-class-fixed-eval")
        cmd.extend(["--perturb-class-eval-count", str(args.perturb_class_eval_count)])
        cmd.extend(["--perturb-class-pool-size", str(args.perturb_class_pool_size)])
        cmd.extend(["--perturb-class-pool-multiplier", str(args.perturb_class_pool_multiplier)])

    if args.perturb_sample_size:
        cmd.append("--perturb-sample-size")
        cmd.extend(["--perturb-sample-size-n", str(args.perturb_sample_size_n)])
        cmd.extend(["--perturb-sample-size-seed", str(args.perturb_sample_size_seed)])
    if args.perturb_preprocessing:
        cmd.append("--perturb-preprocessing")
        cmd.extend(["--perturb-preprocessing-variant", str(args.perturb_preprocessing_variant)])
        cmd.extend(["--perturb-preprocessing-scale", str(args.perturb_preprocessing_scale)])
    if args.perturb_domain_shift:
        cmd.append("--perturb-domain-shift")
        cmd.extend(["--perturb-domain-shift-dataset", str(args.perturb_domain_shift_dataset)])
        cmd.extend(["--perturb-domain-shift-data-root", str(args.perturb_domain_shift_data_root)])
        cmd.extend(["--perturb-domain-shift-image-size", str(args.perturb_domain_shift_image_size)])

STEP_BUILDERS: dict[str, StepBuilder] = {
    "prep_mnist_cifar10": step_prep_mnist_cifar10,
    "prep_celeba": step_prep_celeba,
    "prep_chestxray14": step_prep_chestxray14,
    "stage_studiogan": step_stage_studiogan,
    "stage_ddpm": step_stage_ddpm,
    "stage_stylegan": step_stage_stylegan,
    "train_dcgan_cifar10": step_train_dcgan_cifar10,
    "train_dcgan_mnist": step_train_dcgan_mnist,
    "train_wgangp_cifar10": step_train_wgangp_cifar10,
    "train_wgangp_chestxray14": step_train_wgangp_chestxray14,
    "test_dcgan_cifar10": step_test_dcgan_cifar10,
    "test_dcgan_mnist": step_test_dcgan_mnist,
    "test_wgangp_cifar10": step_test_wgangp_cifar10,
    "test_wgangp_chestxray14": step_test_wgangp_chestxray14,
    "test_studiogan_cifar10": step_test_studiogan_cifar10,
    "test_ddpm_cifar10": step_test_ddpm_cifar10,
    "test_stylegan2_celeba": step_test_stylegan2_celeba,
    "smoke_models": step_smoke_models,
}


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
        "train_dcgan_mnist",
        "train_wgangp_cifar10",
        "train_wgangp_chestxray14",
    ],
    "test": [
        "test_dcgan_cifar10",
        "test_dcgan_mnist",
        "test_wgangp_cifar10",
        "test_wgangp_chestxray14",
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
        "train_dcgan_mnist",
        "train_wgangp_cifar10",
        "train_wgangp_chestxray14",
        "test_dcgan_cifar10",
        "test_dcgan_mnist",
        "test_wgangp_cifar10",
        "test_wgangp_chestxray14",
        "test_studiogan_cifar10",
        "test_ddpm_cifar10",
        "test_stylegan2_celeba",
        "smoke_models",
    ],
    "smoke": [
        "smoke_models",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repository operations dynamically via Scripts/*.py entrypoints."
    )
    parser.add_argument("--list", action="store_true", help="List profiles and steps, then exit.")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="setup")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=sorted(STEP_BUILDERS.keys()),
        help="Custom ordered steps. If provided, this overrides --profile.",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dcgan-epochs", type=int, default=1)
    parser.add_argument("--dcgan-batch-size", type=int, default=64)
    parser.add_argument("--wgangp-epochs", type=int, default=1)
    parser.add_argument("--wgangp-batch-size", type=int, default=64)
    parser.add_argument("--test-num-samples", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--generation-seed", type=int, default=10)
    parser.add_argument("--reference-seed", type=int, default=10)
    parser.add_argument("--dcgan-test-netg", type=str, default="")
    parser.add_argument("--wgangp-test-generator", type=str, default="")
    parser.add_argument("--wgangp-test-critic", type=str, default="")
    parser.add_argument("--studiogan-test-checkpoint", type=str, default="")
    parser.add_argument("--ddpm-test-checkpoint", type=str, default="")
    parser.add_argument("--stylegan2-test-checkpoint", type=str, default="")
    parser.add_argument(
        "--eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable metric evaluation in compatible test steps (currently DCGAN).",
    )
    parser.add_argument("--metrics-samples", type=int, default=64)
    parser.add_argument("--metrics-feature-space", type=str, default="inception_v3")
    parser.add_argument("--metrics-feature-batch-size", type=int, default=64)
    parser.add_argument(
        "--metrics-feature-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--metrics-bootstrap-samples", type=int, default=0)
    parser.add_argument("--metrics-bootstrap-seed", type=int, default=0)
    parser.add_argument("--metrics-bootstrap-alpha", type=float, default=0.05)
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow real dataset download fallback during metric evaluation.",
    )
    parser.add_argument(
        "--use-perturbations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable perturbation wrappers for generated/real samples in compatible test scripts.",
    )
    parser.add_argument(
        "--perturb-apply-to",
        choices=["fake", "real", "both"],
        default="fake",
        help="Which sample set perturbations should target (memoisation/class-removal/class-imbalance are fake-only).",
    )
    parser.add_argument(
        "--perturb-degrade",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable degradation perturbation.",
    )
    parser.add_argument("--perturb-degrade-severity", type=int, default=1)
    parser.add_argument(
        "--perturb-degrade-gaussian-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--perturb-degrade-gaussian-blur",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--perturb-degrade-jpeg-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--perturb-memoisation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable memoisation perturbation (inject real images into fake samples).",
    )
    parser.add_argument("--perturb-memo-fraction", type=float, default=0.1)
    parser.add_argument("--perturb-memo-seed", type=int, default=10)
    parser.add_argument(
        "--perturb-class-removal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable class-removal perturbation (drop selected fake classes/modes).",
    )
    parser.add_argument(
        "--perturb-class-removal-strategy",
        choices=["label", "kmeans"],
        default="label",
    )
    parser.add_argument(
        "--perturb-class-removal-targets",
        type=str,
        default="",
        help="Comma-separated labels/ids or kmeans label-cluster ids to drop.",
    )
    parser.add_argument("--perturb-class-removal-kmeans-k", type=int, default=8)
    parser.add_argument(
        "--perturb-class-removal-kmeans-cache-path",
        type=str,
        default="",
        help="Optional path to save/reuse kmeans label-cluster assignments.",
    )
    parser.add_argument(
        "--perturb-class-removal-kmeans-recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore existing kmeans cache and rebuild label clusters.",
    )
    parser.add_argument("--perturb-class-removal-seed", type=int, default=10)
    parser.add_argument("--perturb-class-removal-label-threshold", type=float, default=0.0)
    parser.add_argument("--perturb-class-removal-min-kept", type=int, default=4)
    parser.add_argument(
        "--perturb-class-fixed-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep class-removal and class-imbalance metrics on a fixed fake sample count.",
    )
    parser.add_argument(
        "--perturb-class-eval-count",
        type=int,
        default=0,
        help="Fixed fake sample count used after class perturbations. 0 falls back to metrics_samples.",
    )
    parser.add_argument(
        "--perturb-class-pool-size",
        type=int,
        default=0,
        help="Optional explicit fake pool size for class perturbation sweeps.",
    )
    parser.add_argument(
        "--perturb-class-pool-multiplier",
        type=float,
        default=3.0,
        help="Multiplier used to size the fake pool for class perturbation sweeps.",
    )
    parser.add_argument(
        "--perturb-class-imbalance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable class-imbalance perturbation (partially remove selected fake classes/modes).",
    )
    parser.add_argument(
        "--perturb-class-imbalance-strategy",
        choices=["label", "kmeans"],
        default="label",
    )
    parser.add_argument(
        "--perturb-class-imbalance-targets",
        type=str,
        default="",
        help="Comma-separated labels/ids or kmeans label-cluster ids to skew.",
    )
    parser.add_argument(
        "--perturb-class-imbalance-balance",
        type=str,
        default="0.5",
        help="Drop ratio for selected targets. Single value or comma-separated list.",
    )
    parser.add_argument("--perturb-class-imbalance-kmeans-k", type=int, default=8)
    parser.add_argument(
        "--perturb-class-imbalance-kmeans-cache-path",
        type=str,
        default="",
        help="Optional path to save/reuse kmeans label-cluster assignments.",
    )
    parser.add_argument(
        "--perturb-class-imbalance-kmeans-recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore existing kmeans cache and rebuild label clusters.",
    )
    parser.add_argument("--perturb-class-imbalance-seed", type=int, default=10)
    parser.add_argument("--perturb-class-imbalance-label-threshold", type=float, default=0.0)
    parser.add_argument("--perturb-class-imbalance-min-kept", type=int, default=4)
    parser.add_argument(
        "--perturb-sample-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sample-size variation perturbation.",
    )
    parser.add_argument("--perturb-sample-size-n", type=int, default=1000)
    parser.add_argument("--perturb-sample-size-seed", type=int, default=42)
    parser.add_argument(
        "--perturb-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable preprocessing variation perturbation.",
    )
    parser.add_argument(
        "--perturb-preprocessing-variant",
        type=str,
        default="downsample_bilinear",
        choices=[
            "downsample_nearest",
            "downsample_bilinear",
            "downsample_bicubic",
            "center_crop_pad",
            "grayscale_triplicate",
        ],
    )
    parser.add_argument("--perturb-preprocessing-scale", type=float, default=0.75)
    parser.add_argument(
        "--perturb-domain-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable domain-shift perturbation (alternate real reference dataset).",
    )
    parser.add_argument("--perturb-domain-shift-dataset", type=str, default="")
    parser.add_argument("--perturb-domain-shift-data-root", type=str, default="")
    parser.add_argument("--perturb-domain-shift-image-size", type=int, default=0)
    parser.add_argument(
        "--strict-tests",
        action="store_true",
        help="Fail test steps instead of skipping recoverable test-time errors.",
    )
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--subset-max-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=10)
    parser.add_argument(
        "--subset-strategy",
        choices=["random", "class_balanced"],
        default="random",
    )
    parser.add_argument(
        "--subset-include-classes",
        type=str,
        default="",
        help="Comma-separated class names or indices to keep.",
    )
    parser.add_argument(
        "--subset-drop-classes",
        type=str,
        default="",
        help="Comma-separated class names or indices to remove.",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag, only a dry-run plan is printed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep executing remaining steps even if one step fails.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging in pipeline child scripts and metric computation.",
    )
    return parser.parse_args()


def selected_steps(args: argparse.Namespace):
    if args.steps:
        return args.steps
    return PROFILES[args.profile]


def print_catalog():
    print("Available steps:")
    for step in sorted(STEP_BUILDERS.keys()):
        print(f"  - {step}")
    print("\nAvailable profiles:")
    for profile, steps in PROFILES.items():
        print(f"  - {profile}: {', '.join(steps)}")


def run_step(step_name: str, cmd: list[str], run: bool):
    print(f"\n[{step_name}]")
    print(" ".join(cmd))
    if not run:
        return 0
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return completed.returncode


def _extract_flag_value(cmd: list[str], flag: str) -> str:
    if flag not in cmd:
        return ""
    idx = cmd.index(flag)
    if idx + 1 >= len(cmd):
        return ""
    return cmd[idx + 1]


def _dataset_for_test_step(step_name: str) -> str:
    if step_name.endswith("_chestxray14"):
        return "chestxray14"
    if step_name.endswith("_mnist"):
        return "mnist"
    if step_name.endswith("_cifar10"):
        return "cifar10"
    if step_name.endswith("_celeba"):
        return "celeba"
    return ""


def _model_info_for_test_step(step_name: str) -> tuple[str, str]:
    if step_name.startswith("test_dcgan"):
        return "dcgan", "trained via Scripts/train_dcgan.py"
    if step_name.startswith("test_wgangp"):
        return "wgangp", "trained via Scripts/train_wgangp.py"
    if step_name.startswith("test_studiogan"):
        return "studiogan", "pretrained checkpoint"
    if step_name.startswith("test_ddpm"):
        return "ddpm", "pretrained checkpoint"
    if step_name.startswith("test_stylegan2"):
        return "stylegan2", "pretrained checkpoint"
    return step_name, ""


def _weights_for_test_step(step_name: str, cmd: list[str]) -> str:
    if step_name.startswith("test_dcgan"):
        return _extract_flag_value(cmd, "--netG")
    if step_name.startswith("test_wgangp"):
        gen_path = _extract_flag_value(cmd, "--generator-checkpoint")
        critic_path = _extract_flag_value(cmd, "--critic-checkpoint")
        if gen_path and critic_path:
            return f"generator={gen_path};critic={critic_path}"
        return gen_path or critic_path
    if step_name.startswith(("test_studiogan", "test_ddpm", "test_stylegan2")):
        return _extract_flag_value(cmd, "--checkpoint")
    return ""


def _subset_config_json(args: argparse.Namespace) -> str:
    subset = {
        "subset_fraction": args.subset_fraction,
        "subset_max_samples": args.subset_max_samples,
        "subset_seed": args.subset_seed,
        "subset_strategy": args.subset_strategy,
        "subset_include_classes": args.subset_include_classes.strip() or None,
        "subset_drop_classes": args.subset_drop_classes.strip() or None,
    }
    if not any(
        [
            subset["subset_fraction"] is not None,
            subset["subset_max_samples"] is not None,
            subset["subset_include_classes"] is not None,
            subset["subset_drop_classes"] is not None,
        ]
    ):
        return ""
    return json.dumps(subset, separators=(",", ":"))


def _perturbation_config_json_from_cmd(cmd: list[str]) -> str:
    if "--use-perturbations" not in cmd:
        return ""

    config = {
        "use_perturbations": True,
        "apply_to": _extract_flag_value(cmd, "--perturb-apply-to") or "fake",
        "degradation": {
            "enabled": "--perturb-degrade" in cmd,
            "severity": _extract_flag_value(cmd, "--perturb-degrade-severity") or "1",
            "gaussian_noise": "--perturb-degrade-gaussian-noise" in cmd,
            "gaussian_blur": "--perturb-degrade-gaussian-blur" in cmd,
            "jpeg_compression": "--perturb-degrade-jpeg-compression" in cmd,
        },
        "memoisation": {
            "enabled": "--perturb-memoisation" in cmd,
            "fraction": _extract_flag_value(cmd, "--perturb-memo-fraction") or "0.1",
            "seed": _extract_flag_value(cmd, "--perturb-memo-seed") or "10",
        },
        "class_removal": {
            "enabled": "--perturb-class-removal" in cmd,
            "strategy": _extract_flag_value(cmd, "--perturb-class-removal-strategy") or "label",
            "targets": _extract_flag_value(cmd, "--perturb-class-removal-targets"),
            "kmeans_k": _extract_flag_value(cmd, "--perturb-class-removal-kmeans-k") or "8",
            "kmeans_cache_path": _extract_flag_value(cmd, "--perturb-class-removal-kmeans-cache-path"),
            "kmeans_recreate": "--perturb-class-removal-kmeans-recreate" in cmd,
            "seed": _extract_flag_value(cmd, "--perturb-class-removal-seed") or "10",
            "label_threshold": _extract_flag_value(cmd, "--perturb-class-removal-label-threshold") or "0.0",
            "min_kept": _extract_flag_value(cmd, "--perturb-class-removal-min-kept") or "4",
        },
        "class_fixed_eval": {
            "enabled": "--perturb-class-fixed-eval" in cmd and "--no-perturb-class-fixed-eval" not in cmd,
            "evaluation_count": _extract_flag_value(cmd, "--perturb-class-eval-count") or "0",
            "pool_size": _extract_flag_value(cmd, "--perturb-class-pool-size") or "0",
            "pool_multiplier": _extract_flag_value(cmd, "--perturb-class-pool-multiplier") or "3.0",
        },
        "class_imbalance": {
            "enabled": "--perturb-class-imbalance" in cmd,
            "strategy": _extract_flag_value(cmd, "--perturb-class-imbalance-strategy") or "label",
            "targets": _extract_flag_value(cmd, "--perturb-class-imbalance-targets"),
            "balance": _extract_flag_value(cmd, "--perturb-class-imbalance-balance") or "0.5",
            "kmeans_k": _extract_flag_value(cmd, "--perturb-class-imbalance-kmeans-k") or "8",
            "kmeans_cache_path": _extract_flag_value(
                cmd, "--perturb-class-imbalance-kmeans-cache-path"
            ),
            "kmeans_recreate": "--perturb-class-imbalance-kmeans-recreate" in cmd,
            "seed": _extract_flag_value(cmd, "--perturb-class-imbalance-seed") or "10",
            "label_threshold": _extract_flag_value(
                cmd, "--perturb-class-imbalance-label-threshold"
            )
            or "0.0",
            "min_kept": _extract_flag_value(cmd, "--perturb-class-imbalance-min-kept") or "4",
        },
        "sample_size": {
            "enabled": "--perturb-sample-size" in cmd,
            "n": _extract_flag_value(cmd, "--perturb-sample-size-n") or "1000",
            "seed": _extract_flag_value(cmd, "--perturb-sample-size-seed") or "42",
        },
        "preprocessing": {
            "enabled": "--perturb-preprocessing" in cmd,
            "variant": _extract_flag_value(cmd, "--perturb-preprocessing-variant"),
            "scale": _extract_flag_value(cmd, "--perturb-preprocessing-scale") or "0.75",
        },
        "domain_shift": {
            "enabled": "--perturb-domain-shift" in cmd,
            "dataset": _extract_flag_value(cmd, "--perturb-domain-shift-dataset"),
            "data_root": _extract_flag_value(cmd, "--perturb-domain-shift-data-root"),
            "image_size": _extract_flag_value(cmd, "--perturb-domain-shift-image-size") or "0",
        },
    }
    return json.dumps(config, separators=(",", ":"))


def main():
    args = parse_args()

    if args.list:
        print_catalog()
        return

    steps = selected_steps(args)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Mode: {'EXECUTE' if args.run else 'DRY-RUN'}")
    print(f"Planned steps ({len(steps)}): {', '.join(steps)}")

    failures: list[tuple[str, int]] = []

    for step_name in steps:
        cmd = STEP_BUILDERS[step_name](args)
        code = run_step(step_name, cmd, run=args.run)
        if code != 0:
            failures.append((step_name, code))
            print(f"Step failed: {step_name} (exit code {code})")
            if not args.continue_on_error:
                break

    if failures:
        print("\nPipeline finished with failures:")
        for name, code in failures:
            print(f"  - {name}: exit code {code}")
        raise SystemExit(1)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()

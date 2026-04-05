"""
dynamic operations runner
 to execute common workflows by calling existing Scripts/*.py files.

examples (from repo root):

    to get information about available steps and profiles:
    py Tests/run_operations_pipeline.py --list

    running predefined profiles (setup, train, full, smoke):
    py Tests/run_operations_pipeline.py --profile setup --run
        setup runs all data preparation and staging steps for all datasets and pretrauned models, but no training.
        train runs just the training steps (DCGAN and WGAN-GP on CIFAR-10).
        full runs everything (setup + train + smoke test).
        smoke runs just the smoke test that calls all model files to check for import/runtime errors.

    running custom ordered step lists:
    py Tests/run_operations_pipeline.py --profile full --dcgan-epochs 1 --run
    py Tests/run_operations_pipeline.py --steps prep_mnist_cifar10 train_dcgan_cifar10 smoke_models --run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable


StepBuilder = Callable[[argparse.Namespace], list[str]]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Scripts"
TESTS_DIR = REPO_ROOT / "Tests"
PYTHON_EXE = sys.executable


def step_prep_mnist_cifar10(args: argparse.Namespace) :
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


def step_prep_chestxray14(args: argparse.Namespace) :
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
    if args.cuda:
        cmd.append("--cuda")
    return cmd


def step_smoke_models(_: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(TESTS_DIR / "call_models_files.py"),
    ]


STEP_BUILDERS: dict[str, StepBuilder] = {
    "prep_mnist_cifar10": step_prep_mnist_cifar10,
    "prep_celeba": step_prep_celeba,
    "prep_chestxray14": step_prep_chestxray14,
    "stage_studiogan": step_stage_studiogan,
    "stage_ddpm": step_stage_ddpm,
    "stage_stylegan": step_stage_stylegan,
    "train_dcgan_cifar10": step_train_dcgan_cifar10,
    "train_wgangp_cifar10": step_train_wgangp_cifar10,
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
        "train_wgangp_cifar10",
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

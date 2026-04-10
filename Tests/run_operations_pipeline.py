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
import csv
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
    if args.cuda:
        cmd.append("--cuda")
    return cmd


def step_smoke_models(_: argparse.Namespace):
    return [
        PYTHON_EXE,
        str(TESTS_DIR / "call_models_files.py"),
    ]


def step_test_dcgan_cifar10(args: argparse.Namespace):
    netg_path = args.dcgan_test_netg or str(Path(args.outputs_root) / "dcgan_cifar10" / "netG_latest.pth")
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "test_dcgan.py"),
        "--netG",
        netg_path,
        "--out-dir",
        str(Path(args.outputs_root) / "dcgan_cifar10_test"),
        "--num-samples",
        str(args.test_num_samples),
        "--batch-size",
        str(args.test_batch_size),
        "--image-size",
        str(args.image_size),
        "--channels",
        "3",
    ]
    if args.eval_metrics:
        cmd.extend(
            [
                "--eval-metrics",
                "--metrics-dataset",
                "cifar10",
                "--metrics-data-root",
                str(Path(args.data_root) / "CIFAR10"),
                "--metrics-samples",
                str(args.metrics_samples),
            ]
        )
        if args.metrics_download_if_missing:
            cmd.append("--metrics-download-if-missing")
    if args.cuda:
        cmd.append("--cuda")
    if args.strict_tests:
        cmd.append("--strict")
    return cmd


def step_test_wgangp_cifar10(args: argparse.Namespace):
    generator_path = args.wgangp_test_generator or str(
        Path(args.outputs_root) / "wgangp_cifar10" / "netG_latest.pth"
    )
    critic_path = args.wgangp_test_critic or str(
        Path(args.outputs_root) / "wgangp_cifar10" / "netD_latest.pth"
    )
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "test_wgangp.py"),
        "--generator-checkpoint",
        generator_path,
        "--critic-checkpoint",
        critic_path,
        "--out-dir",
        str(Path(args.outputs_root) / "wgangp_cifar10_test"),
        "--num-samples",
        str(args.test_num_samples),
        "--image-size",
        str(args.image_size),
        "--channels",
        "3",
    ]
    if args.cuda:
        cmd.append("--cuda")
    if args.strict_tests:
        cmd.append("--strict")
    return cmd


def step_test_studiogan_cifar10(args: argparse.Namespace):
    checkpoint = args.studiogan_test_checkpoint or str(
        Path(args.checkpoints_root) / "StudioGAN" / "CIFAR10" / "studiogan_generator.pth"
    )
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "test_studiogan.py"),
        "--checkpoint",
        checkpoint,
        "--out-dir",
        str(Path(args.outputs_root) / "studiogan_cifar10_test"),
        "--num-samples",
        str(args.test_num_samples),
    ]
    if args.cuda:
        cmd.append("--cuda")
    if args.strict_tests:
        cmd.append("--strict")
    return cmd


def step_test_ddpm_cifar10(args: argparse.Namespace):
    checkpoint = args.ddpm_test_checkpoint or str(
        Path(args.checkpoints_root) / "DDPM" / "CIFAR10" / "ddpm_model.pth"
    )
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "test_ddpm.py"),
        "--checkpoint",
        checkpoint,
        "--out-dir",
        str(Path(args.outputs_root) / "ddpm_cifar10_test"),
        "--num-samples",
        str(args.test_num_samples),
    ]
    if args.cuda:
        cmd.append("--cuda")
    if args.strict_tests:
        cmd.append("--strict")
    return cmd


def step_test_stylegan2_celeba(args: argparse.Namespace):
    checkpoint = args.stylegan2_test_checkpoint or str(
        Path(args.checkpoints_root) / "StyleGAN" / "CelebA" / "stylegan2_generator.pkl"
    )
    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "test_stylegan2.py"),
        "--checkpoint",
        checkpoint,
        "--out-dir",
        str(Path(args.outputs_root) / "stylegan2_celeba_test"),
        "--num-samples",
        str(args.test_num_samples),
    ]
    if args.eval_metrics:
        cmd.extend(
            [
                "--eval-metrics",
                "--metrics-dataset",
                "celeba",
                "--metrics-data-root",
                str(Path(args.data_root) / "CelebA"),
                "--metrics-samples",
                str(args.metrics_samples),
            ]
        )
        if args.metrics_download_if_missing:
            cmd.append("--metrics-download-if-missing")
    if args.cuda:
        cmd.append("--cuda")
    if args.strict_tests:
        cmd.append("--strict")
    return cmd


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


STEP_BUILDERS: dict[str, StepBuilder] = {
    "prep_mnist_cifar10": step_prep_mnist_cifar10,
    "prep_celeba": step_prep_celeba,
    "prep_chestxray14": step_prep_chestxray14,
    "stage_studiogan": step_stage_studiogan,
    "stage_ddpm": step_stage_ddpm,
    "stage_stylegan": step_stage_stylegan,
    "train_dcgan_cifar10": step_train_dcgan_cifar10,
    "train_wgangp_cifar10": step_train_wgangp_cifar10,
    "test_dcgan_cifar10": step_test_dcgan_cifar10,
    "test_wgangp_cifar10": step_test_wgangp_cifar10,
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
    parser.add_argument(
        "--metrics-download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow real dataset download fallback during metric evaluation.",
    )
    parser.add_argument(
        "--strict-tests",
        action="store_true",
        help="Fail test steps when placeholder/TODO test scripts are not fully implemented.",
    )
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--subset-max-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
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


def _metrics_json_for_out_dir(out_dir: Path, run_started_at: datetime) -> tuple[str, str]:
    metrics_path = out_dir / "metrics_report.json"
    if not metrics_path.exists():
        return str(metrics_path), ""
    try:
        modified_at = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc)
        if modified_at < run_started_at:
            return str(metrics_path), ""
        return str(metrics_path), metrics_path.read_text(encoding="utf-8")
    except OSError:
        return str(metrics_path), ""


def append_test_run_csv(
    args: argparse.Namespace,
    step_name: str,
    cmd: list[str],
    exit_code: int,
    run_started_at: datetime,
) -> None:
    csv_path = Path(args.outputs_root) / "test_runs_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_dir_value = _extract_flag_value(cmd, "--out-dir")
    out_dir = Path(out_dir_value) if out_dir_value else Path(args.outputs_root)
    model_name, trained_how = _model_info_for_test_step(step_name)
    metrics_path, metrics_json = _metrics_json_for_out_dir(out_dir, run_started_at=run_started_at)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step_name": step_name,
        "model_name": model_name,
        "trained_how": trained_how,
        "weights_path": _weights_for_test_step(step_name, cmd),
        "dataset": _dataset_for_test_step(step_name),
        "subset_config": _subset_config_json(args),
        "metrics_path": metrics_path,
        "metrics_report_json": metrics_json,
        "exit_code": str(exit_code),
        "output_dir": str(out_dir),
        "command": " ".join(cmd),
    }
    fieldnames = list(row.keys())

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
        started_at = datetime.now(timezone.utc)
        code = run_step(step_name, cmd, run=args.run)
        if args.run and step_name.startswith("test_"):
            append_test_run_csv(
                args=args,
                step_name=step_name,
                cmd=cmd,
                exit_code=code,
                run_started_at=started_at,
            )
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

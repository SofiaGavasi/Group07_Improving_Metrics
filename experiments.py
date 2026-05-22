from __future__ import annotations

from itertools import combinations
from typing import Any


def default_experiment_base_overrides():
    return {
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
        "PERTURB_SAMPLE_SIZE": False,
        "PERTURB_SAMPLE_SIZE_N": 128,
        "PERTURB_SAMPLE_SIZE_SEED": 10,
        "PERTURB_PREPROCESSING": False,
        "PERTURB_PREPROCESSING_VARIANT": "downsample_bilinear",
        "PERTURB_PREPROCESSING_SCALE": 0.75,
        "PERTURB_DOMAIN_SHIFT": False,
        "PERTURB_DOMAIN_SHIFT_DATASET": "",
        "PERTURB_DOMAIN_SHIFT_DATA_ROOT": "",
        "PERTURB_DOMAIN_SHIFT_IMAGE_SIZE": 0,
        "SUBSET_SEED": 10,
        "SUBSET_STRATEGY": "random",
    }


STYLEGAN2_STEP = "test_stylegan2_celeba"
STYLEGAN2_MODEL = "stylegan2"
STYLEGAN2_DATASET = "celeba"
EXHAUSTIVE_TARGET_COMBO_LIMIT = 12 # added this limit for celeba, otherwise it would have 1 trilion combinations....
SINGLE_LABEL_DATASET_CLASS_COUNT = 10
STYLEGAN2_CELEBA_LABEL_COUNT = 40
STYLEGAN2_CELEBA_KMEANS_K = 10


# i keep the target helpers here so every suite enumerates classes the same way.
def _targets_csv(indices):
    return ",".join(str(i) for i in indices)


def _target_name_tag(targets_csv):
    return str(targets_csv).replace(",", "_")


def _all_non_empty_target_csvs(*, target_count, sweep_name):
    total = int(target_count)
    if total <= 0:
        raise ValueError(f"{sweep_name} needs at least one target class.")

    # once the label space gets too large, a "complete" sweep stops being a real run plan
    if total > EXHAUSTIVE_TARGET_COMBO_LIMIT:
        combo_count = (2 ** total) - 1
        raise ValueError(
            f"{sweep_name} would create {combo_count} non-empty class combinations from {total} targets. "
            f"i only allow exhaustive sweeps up to {EXHAUSTIVE_TARGET_COMBO_LIMIT} targets because anything larger is not practical. "
            "for bigger spaces, reduce the label set first or use a clustered target space."
        )

    target_csvs: list[str] = []
    for combo_size in range(1, total + 1):
        for combo in combinations(range(total), combo_size):
            target_csvs.append(_targets_csv(combo))
    return target_csvs


def _format_fraction_pct(fraction: float):
    return f"{int(round(float(fraction) * 100)):02d}"


def _append_label_class_removal_sweep(
    *,
    experiments,
    make_experiment,
    target_csvs,):
    # this is the full label sweep: every single class and every cross-combination
    for targets_csv in target_csvs:
        experiments.append(
            make_experiment(
                name=f"class_removal_label_{_target_name_tag(targets_csv)}",
                override_updates={
                    "PERTURB_CLASS_REMOVAL": True,
                    "PERTURB_CLASS_REMOVAL_STRATEGY": "label",
                    "PERTURB_CLASS_REMOVAL_TARGETS": targets_csv,
                    "PERTURB_CLASS_REMOVAL_SEED": 10,
                    "PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD": 0.0,
                },
            )
        )


def _append_label_class_imbalance_sweep(
    *,
    experiments,
    make_experiment,
    target_csvs,
    balance_levels,
):
    # i use the same full target list here so imbalance and removal cover the same class space
    for targets_csv in target_csvs:
        target_tag = _target_name_tag(targets_csv)
        for balance in balance_levels:
            experiments.append(
                make_experiment(
                    name=f"class_imbalance_label_{target_tag}_{_format_fraction_pct(balance)}pct",
                    override_updates={
                        "PERTURB_CLASS_IMBALANCE": True,
                        "PERTURB_CLASS_IMBALANCE_STRATEGY": "label",
                        "PERTURB_CLASS_IMBALANCE_TARGETS": targets_csv,
                        "PERTURB_CLASS_IMBALANCE_BALANCE": str(balance),
                        "PERTURB_CLASS_IMBALANCE_SEED": 10,
                    },
                )
            )


def _stylegan2_experiment(
    name: str,
    override_updates: dict[str, Any],
    experiment_base_overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "steps": [STYLEGAN2_STEP],
        "model_name": STYLEGAN2_MODEL,
        "dataset_name": STYLEGAN2_DATASET,
        "overrides": {
            **experiment_base_overrides,
            **override_updates,
        },
    }


def _build_stylegan2_experiments(
    *,
    experiment_base_overrides: dict[str, Any],
):
    experiments: list[dict[str, Any]] = []
    label_target_csvs = _all_non_empty_target_csvs(
        target_count=STYLEGAN2_CELEBA_LABEL_COUNT,
        sweep_name="stylegan2 celeba label perturbation sweep",
    )
    kmeans_target_csvs = _all_non_empty_target_csvs(
        target_count=STYLEGAN2_CELEBA_KMEANS_K,
        sweep_name="stylegan2 celeba kmeans perturbation sweep",
    )

    experiments.append(
        _stylegan2_experiment(
            name="baseline_no_perturbation",
            override_updates={"USE_PERTURBATIONS": False},
            experiment_base_overrides=experiment_base_overrides,
        )
    )

    degrade_flags = [
        ("noise", "PERTURB_DEGRADE_GAUSSIAN_NOISE"),
        ("blur", "PERTURB_DEGRADE_GAUSSIAN_BLUR"),
        ("jpeg", "PERTURB_DEGRADE_JPEG_COMPRESSION"),
    ]
    for degrade_name, degrade_flag in degrade_flags:
        for severity in [1, 3, 5]:
            experiments.append(
                _stylegan2_experiment(
                    name=f"degrade_{degrade_name}_sev{severity}",
                    override_updates={
                        "PERTURB_DEGRADE": True,
                        "PERTURB_DEGRADE_SEVERITY": severity,
                        degrade_flag: True,
                    },
                    experiment_base_overrides=experiment_base_overrides,
                )
            )
    for severity in [1, 3, 5]:
        experiments.append(
            _stylegan2_experiment(
                name=f"degrade_all_sev{severity}",
                override_updates={
                    "PERTURB_DEGRADE": True,
                    "PERTURB_DEGRADE_SEVERITY": severity,
                    "PERTURB_DEGRADE_GAUSSIAN_NOISE": True,
                    "PERTURB_DEGRADE_GAUSSIAN_BLUR": True,
                    "PERTURB_DEGRADE_JPEG_COMPRESSION": True,
                },
                experiment_base_overrides=experiment_base_overrides,
            )
        )

    memo_fractions = [ 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    for fraction in memo_fractions:
        experiments.append(
            _stylegan2_experiment(
                name=f"memo_frac_{_format_fraction_pct(fraction)}pct",
                override_updates={
                    "PERTURB_MEMOISATION": True,
                    "PERTURB_MEMO_FRACTION": fraction,
                    "PERTURB_MEMO_SEED": 10,
                },
                experiment_base_overrides=experiment_base_overrides,
            )
        )

    _append_label_class_removal_sweep(
        experiments=experiments,
        make_experiment=lambda name, override_updates: _stylegan2_experiment(
            name=name,
            override_updates=override_updates,
            experiment_base_overrides=experiment_base_overrides,
        ),
        target_csvs=label_target_csvs,
    )

    # this stays exhaustive too, but k=10 is still a manageable space
    for targets_csv in kmeans_target_csvs:
        experiments.append(
            _stylegan2_experiment(
                name=f"class_removal_kmeans_k10_cluster_{_target_name_tag(targets_csv)}",
                override_updates={
                    "PERTURB_CLASS_REMOVAL": True,
                    "PERTURB_CLASS_REMOVAL_STRATEGY": "kmeans",
                    "PERTURB_CLASS_REMOVAL_KMEANS_K": STYLEGAN2_CELEBA_KMEANS_K,
                    "PERTURB_CLASS_REMOVAL_TARGETS": targets_csv,
                    "PERTURB_CLASS_REMOVAL_SEED": 10,
                },
                experiment_base_overrides=experiment_base_overrides,
            )
        )

    imbalance_levels = [0.90, 0.50, 0.2]
    _append_label_class_imbalance_sweep(
        experiments=experiments,
        make_experiment=lambda name, override_updates: _stylegan2_experiment(
            name=name,
            override_updates=override_updates,
            experiment_base_overrides=experiment_base_overrides,
        ),
        target_csvs=label_target_csvs,
        balance_levels=imbalance_levels,
    )

    for cluster_targets in kmeans_target_csvs:
        cluster_tag = _target_name_tag(cluster_targets)
        for balance in imbalance_levels:
            experiments.append(
                _stylegan2_experiment(
                    name=f"class_imbalance_kmeans_k10_cluster_{cluster_tag}_{_format_fraction_pct(balance)}pct",
                    override_updates={
                        "PERTURB_CLASS_IMBALANCE": True,
                        "PERTURB_CLASS_IMBALANCE_STRATEGY": "kmeans",
                        "PERTURB_CLASS_IMBALANCE_KMEANS_K": STYLEGAN2_CELEBA_KMEANS_K,
                        "PERTURB_CLASS_IMBALANCE_TARGETS": cluster_targets,
                        "PERTURB_CLASS_IMBALANCE_BALANCE": str(balance),
                        "PERTURB_CLASS_IMBALANCE_SEED": 10,
                    },
                    experiment_base_overrides=experiment_base_overrides,
                )
            )

    sample_size_candidates = [16, 32, 64, 128, 256, 512, 768, 1024, 1280]
    for sample_size_n in sample_size_candidates:
        experiments.append(
            _stylegan2_experiment(
                name=f"sample_size_{sample_size_n}",
                override_updates={
                    "PERTURB_SAMPLE_SIZE": True,
                    "PERTURB_SAMPLE_SIZE_N": sample_size_n,
                    "PERTURB_SAMPLE_SIZE_SEED": 10,
                    "PERTURB_APPLY_TO": "both",
                },
                experiment_base_overrides=experiment_base_overrides,
            )
        )

    preprocessing_variant_sweep = [
        ("downsample_nearest", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("downsample_bilinear", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("downsample_bicubic", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("center_crop_pad", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("grayscale_triplicate", [0.75]),
    ]
    for variant, scales in preprocessing_variant_sweep:
        for scale in scales:
            scale_tag = str(scale).replace(".", "p")
            experiments.append(
                _stylegan2_experiment(
                    name=f"preprocessing_{variant}_scale{scale_tag}",
                    override_updates={
                        "PERTURB_PREPROCESSING": True,
                        "PERTURB_PREPROCESSING_VARIANT": variant,
                        "PERTURB_PREPROCESSING_SCALE": scale,
                    },
                    experiment_base_overrides=experiment_base_overrides,
                )
            )

    domain_shift_sweep = [
        ("mnist", "data/MNIST", 32),
        ("cifar10", "data/CIFAR10", 32),
        ("chestxray14", "data/ChestXray14", 32),
    ]
    for dataset_name, data_root, image_size in domain_shift_sweep:
        experiments.append(
            _stylegan2_experiment(
                name=f"domain_shift_{dataset_name}",
                override_updates={
                    "PERTURB_DOMAIN_SHIFT": True,
                    "PERTURB_DOMAIN_SHIFT_DATASET": dataset_name,
                    "PERTURB_DOMAIN_SHIFT_DATA_ROOT": data_root,
                    "PERTURB_DOMAIN_SHIFT_IMAGE_SIZE": image_size,
                    "METRICS_DOWNLOAD_IF_MISSING": True,
                },
                experiment_base_overrides=experiment_base_overrides,
            )
        )

    return experiments


def _single_step_experiment(
    *,
    name: str,
    step: str,
    model_name: str,
    dataset_name: str,
    override_updates: dict[str, Any],
    experiment_base_overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "steps": [step],
        "model_name": model_name,
        "dataset_name": dataset_name,
        "overrides": {
            **experiment_base_overrides,
            **override_updates,
        },
    }


def _build_non_kmeans_perturbation_sweep(
    *,
    step: str,
    model_name: str,
    dataset_name: str,
    checkpoint_path: str,
    label_count: int,
    domain_shift_sweep: list[tuple[str, str, int]],
    experiment_base_overrides: dict[str, Any],
):
    experiments: list[dict[str, Any]] = []
    label_target_csvs = _all_non_empty_target_csvs(
        target_count=label_count,
        sweep_name=f"{model_name} {dataset_name} label perturbation sweep",
    )
    imbalance_levels = [0.90, 0.75, 0.50, 0.30, 0.15, 0.05]

    def _exp(name: str, override_updates: dict[str, Any]) -> dict[str, Any]:
        return _single_step_experiment(
            name=name,
            step=step,
            model_name=model_name,
            dataset_name=dataset_name,
            override_updates={
                "DCGAN_TEST_NETG": checkpoint_path,
                **override_updates,
            },
            experiment_base_overrides=experiment_base_overrides,
        )

    experiments.append(
        _exp(
            name="baseline_no_perturbation",
            override_updates={"USE_PERTURBATIONS": False},
        )
    )

    severity_levels = [1, 2, 3, 4, 5]
    degrade_flags = [
        ("noise", "PERTURB_DEGRADE_GAUSSIAN_NOISE"),
        ("blur", "PERTURB_DEGRADE_GAUSSIAN_BLUR"),
        ("jpeg", "PERTURB_DEGRADE_JPEG_COMPRESSION"),
    ]
    for degrade_name, degrade_flag in degrade_flags:
        for severity in severity_levels:
            experiments.append(
                _exp(
                    name=f"degrade_{degrade_name}_sev{severity}",
                    override_updates={
                        "PERTURB_DEGRADE": True,
                        "PERTURB_DEGRADE_SEVERITY": severity,
                        degrade_flag: True,
                    },
                )
            )
    for severity in severity_levels:
        experiments.append(
            _exp(
                name=f"degrade_all_sev{severity}",
                override_updates={
                    "PERTURB_DEGRADE": True,
                    "PERTURB_DEGRADE_SEVERITY": severity,
                    "PERTURB_DEGRADE_GAUSSIAN_NOISE": True,
                    "PERTURB_DEGRADE_GAUSSIAN_BLUR": True,
                    "PERTURB_DEGRADE_JPEG_COMPRESSION": True,
                },
            )
        )

    memo_fractions = [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    for fraction in memo_fractions:
        experiments.append(
            _exp(
                name=f"memo_frac_{_format_fraction_pct(fraction)}pct",
                override_updates={
                    "PERTURB_MEMOISATION": True,
                    "PERTURB_MEMO_FRACTION": fraction,
                    "PERTURB_MEMO_SEED": 10,
                },
            )
        )

    _append_label_class_removal_sweep(
        experiments=experiments,
        make_experiment=_exp,
        target_csvs=label_target_csvs,
    )

    _append_label_class_imbalance_sweep(
        experiments=experiments,
        make_experiment=_exp,
        target_csvs=label_target_csvs,
        balance_levels=imbalance_levels,
    )

    sample_size_candidates = [16, 32, 64, 128, 256, 512, 768, 1024, 1280]
    for sample_size_n in sample_size_candidates:
        experiments.append(
            _exp(
                name=f"sample_size_{sample_size_n}",
                override_updates={
                    "PERTURB_SAMPLE_SIZE": True,
                    "PERTURB_SAMPLE_SIZE_N": sample_size_n,
                    "PERTURB_SAMPLE_SIZE_SEED": 10,
                    "PERTURB_APPLY_TO": "both",
                },
            )
        )

    preprocessing_variant_sweep = [
        ("downsample_nearest", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("downsample_bilinear", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("downsample_bicubic", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("center_crop_pad", [0.90, 0.75, 0.60, 0.45, 0.30]),
        ("grayscale_triplicate", [0.75]),
    ]
    for variant, scales in preprocessing_variant_sweep:
        for scale in scales:
            scale_tag = str(scale).replace(".", "p")
            experiments.append(
                _exp(
                    name=f"preprocessing_{variant}_scale{scale_tag}",
                    override_updates={
                        "PERTURB_PREPROCESSING": True,
                        "PERTURB_PREPROCESSING_VARIANT": variant,
                        "PERTURB_PREPROCESSING_SCALE": scale,
                    },
                )
            )

    for shift_dataset, shift_root, shift_image_size in domain_shift_sweep:
        experiments.append(
            _exp(
                name=f"domain_shift_{shift_dataset}",
                override_updates={
                    "PERTURB_DOMAIN_SHIFT": True,
                    "PERTURB_DOMAIN_SHIFT_DATASET": shift_dataset,
                    "PERTURB_DOMAIN_SHIFT_DATA_ROOT": shift_root,
                    "PERTURB_DOMAIN_SHIFT_IMAGE_SIZE": shift_image_size,
                    "METRICS_DOWNLOAD_IF_MISSING": True,
                },
            )
        )

    return experiments


def _build_dcgan_cifar10_pretrained_experiments(
    *,
    dcgan_cifar10_pretrained_netg: str,
    experiment_base_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    return _build_non_kmeans_perturbation_sweep(
        step="test_dcgan_cifar10",
        model_name="dcgan",
        dataset_name="cifar10",
        checkpoint_path=dcgan_cifar10_pretrained_netg,
        label_count=SINGLE_LABEL_DATASET_CLASS_COUNT,
        domain_shift_sweep=[("mnist", "data/MNIST", 32)],
        experiment_base_overrides=experiment_base_overrides,
    )


def _build_dcgan_mnist_pretrained_experiments(
    *,
    dcgan_mnist_pretrained_netg: str,
    experiment_base_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    return _build_non_kmeans_perturbation_sweep(
        step="test_dcgan_mnist",
        model_name="dcgan",
        dataset_name="mnist",
        checkpoint_path=dcgan_mnist_pretrained_netg,
        label_count=SINGLE_LABEL_DATASET_CLASS_COUNT,
        domain_shift_sweep=[("cifar10", "data/CIFAR10", 32)],
        experiment_base_overrides=experiment_base_overrides,
    )


def build_experiments_for_suite(
    *,
    experiment_suite: str,
    dcgan_cifar10_pretrained_netg: str,
    dcgan_mnist_pretrained_netg: str,
    experiment_base_overrides: dict[str, Any] | None = None,
) :
    # this file only builds override grids now, so it does not need the sample count anymore
    base_overrides = experiment_base_overrides or default_experiment_base_overrides()
    suite = str(experiment_suite).strip().lower()
    if suite == "stylegan2_celeba":
        return _build_stylegan2_experiments(
            experiment_base_overrides=base_overrides,
        )
    if suite == "dcgan_cifar10_pretrained":
        return _build_dcgan_cifar10_pretrained_experiments(
            dcgan_cifar10_pretrained_netg=dcgan_cifar10_pretrained_netg,
            experiment_base_overrides=base_overrides,
        )
    if suite == "dcgan_mnist_pretrained":
        return _build_dcgan_mnist_pretrained_experiments(
            dcgan_mnist_pretrained_netg=dcgan_mnist_pretrained_netg,
            experiment_base_overrides=base_overrides,
        )
    if suite == "dcgan_pretrained_both":
        return [
            *_build_dcgan_cifar10_pretrained_experiments(
                dcgan_cifar10_pretrained_netg=dcgan_cifar10_pretrained_netg,
                experiment_base_overrides=base_overrides,
            ),
            *_build_dcgan_mnist_pretrained_experiments(
                dcgan_mnist_pretrained_netg=dcgan_mnist_pretrained_netg,
                experiment_base_overrides=base_overrides,
            ),
        ]
    raise ValueError(
        "Unknown EXPERIMENT_SUITE. Expected one of: "
        "stylegan2_celeba, dcgan_cifar10_pretrained, dcgan_mnist_pretrained, dcgan_pretrained_both."
    )

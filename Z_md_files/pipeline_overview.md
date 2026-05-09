# Pipeline Overview

## 1. Project Purpose

This repository is a full evaluation pipeline for generative models.  
It supports:

- dataset setup
- checkpoint staging for pretrained models
- training for self-trained models
- sample generation
- perturbation-based stress tests
- metric computation (FID, KID, IS, Precision, Recall, Density, Coverage)
- batch experiment execution and report aggregation

The main entry point is `main.py`.


## 2. End-to-end Execution Flow

The pipeline flow is:

1. Configure run settings in `main.py`.
2. Build one command for `Tests/run_operations_pipeline.py`.
3. `Tests/run_operations_pipeline.py` expands profile or explicit step list.
4. It calls scripts in `Scripts/` for each step.
5. Test scripts generate fake samples and optionally load real reference samples.
6. If perturbations are enabled, `Perturbation/pipeline_perturbations.py` applies them.
7. Metrics are computed through `Metrics/compute_all.py`.
8. Results are saved per test run in `outputs/.../metrics_report.json`.
9. Batch mode in `main.py` records each experiment in a campaign report JSON in `outputs/`.


## 3. Top-level Folders and Files

### `Datasets/`

This folder contains all dataset loading logic.

- `Datasets/unified_dataset_loader.py`
  - Defines `DatasetConfig` and `UnifiedDatasetLoader`.
  - Supports `mnist`, `cifar10`, `celeba`, `chestxray14`.
  - Applies shared resizing and normalization to `[-1, 1]`.
  - Applies subset filtering through `apply_dataset_subset`.

- `Datasets/dataset_subset.py`
  - Defines `DatasetSubsetConfig`.
  - Implements filtering by class include/drop.
  - Implements sample count control by `fraction` and `max_samples`.
  - Implements subset strategies:
    - `random`
    - `class_balanced` (single-label only)
  - Supports single-label and multi-label datasets.

- `Datasets/chestxray14_dataset.py`
  - Builds and caches local metadata index for ChestXray14.
  - Handles download/indexing via `kagglehub` when requested.
  - Creates train/val/test split metadata.
  - Exposes `ChestXray14Dataset` with multi-hot label vectors.

- `Datasets/__init__.py`
  - Re-exports dataset loader and subset interfaces.


### `Metrics/`

This folder contains all metric implementations.

- `Metrics/compute_all.py`
  - Main orchestrator.
  - Extracts Inception features once for real and fake samples.
  - Computes all metrics from those features/probabilities.
  - Adds bootstrap confidence intervals when configured.

- `Metrics/inception_features.py`
  - Loads pretrained torchvision Inception-v3.
  - Preprocesses images to Inception input format.
  - Extracts:
    - pool3 features (2048-d)
    - logits
    - class probabilities

- `Metrics/fid.py`
  - Computes FID from feature means and covariances.
  - Includes projection fallback to reduce covariance dimension when sample count is small.

- `Metrics/kid.py`
  - Computes KID using polynomial MMD estimator.
  - Uses repeated random subset sampling to return mean and std.

- `Metrics/is_score.py`
  - Computes Inception Score from probability vectors.
  - Returns split-based mean and std.

- `Metrics/precision_recall.py`
  - Computes PRDC precision and recall in feature space.

- `Metrics/density_coverage.py`
  - Computes PRDC density and coverage in feature space.

- `Metrics/prdc_utils.py`
  - Shared utilities:
    - pairwise distances
    - nearest-neighbor radii
    - ndarray conversion helpers

- `Metrics/statistics.py`
  - Bootstrap helpers for metric confidence intervals.
  - Percentile CI calculation.

- `Metrics/__init__.py`
  - Re-exports metric orchestrator and FID function.


### `Models/`

This folder contains model definitions and wrappers.

- `Models/dcgan.py`
  - `DCGANGenerator`
  - `DCGANDiscriminator`
  - dynamic architecture based on target image size
  - standard DCGAN weight initialization helper

- `Models/wgangp.py`
  - `WGANGPGenerator`
  - `WGANGPCritic`
  - gradient penalty function used during training

- `Models/pretrained_wrappers.py`
  - `StudioGANWrapper` for staged StudioGAN checkpoints
  - `DDPMWrapper` / `DDIMWrapper` via `diffusers`
  - `StyleGAN2Wrapper` for `.pkl` or torch checkpoints
  - unified `sample(...)` methods returning tensors in `[-1, 1]`

- `Models/__init__.py`
  - Exports DCGAN classes.


### `Perturbation/`

This folder contains perturbation definitions and the perturbation pipeline.

- `Perturbation/pipeline_perturbations.py`
  - Defines all perturbation CLI arguments.
  - Builds perturbation config dictionary.
  - Applies perturbations in a fixed order.
  - Returns updated fake/real tensors and serialized perturbation config.

- `Perturbation/degrade_dataset.py`
  - Applies resolution-aware noise/blur/JPEG degradation.
  - Uses severity levels 1 to 5.

- `Perturbation/memorization_dataset.py`
  - Replaces a fraction of fake samples with real samples.

- `Perturbation/class_removal.py`
  - Removes fake samples predicted as selected class targets.
  - Supports:
    - label strategy
    - kmeans-over-label-cooccurrence strategy

- `Perturbation/class_imbalance.py`
  - Downsamples selected fake class targets instead of removing all.
  - Supports:
    - label strategy
    - kmeans-over-label-cooccurrence strategy

- `Perturbation/__init__.py`
  - package marker.


### `Scripts/`

This folder contains executable scripts used by pipeline steps.

- Dataset setup scripts:
  - `Scripts/download_preprocess_mnist_cifar10.py`
  - `Scripts/download_preprocess_celeba.py`
  - `Scripts/download_preprocess_chestxray14.py`

- Pretrained checkpoint staging scripts:
  - `Scripts/download_pretrained_studiogan_cifar10.py`
  - `Scripts/download_pretrained_ddpm_cifar10.py`
  - `Scripts/download_pretrained_stylegan_celeba.py`

- Training scripts:
  - `Scripts/train_dcgan.py`
  - `Scripts/train_wgangp.py`

- Test scripts:
  - `Scripts/test_dcgan.py`
  - `Scripts/test_wgangp.py`
  - `Scripts/test_studiogan.py`
  - `Scripts/test_ddpm.py`
  - `Scripts/test_stylegan2.py`
  - All test scripts support perturbation args and optional metric evaluation.

- Analysis/helper scripts:
  - `Scripts/statistical_measures.py`
  - `Scripts/visualize_perturbation_testsV2X.py`
  - `Scripts/__init__.py`


### `Tests/`

This folder contains orchestration and smoke/example utilities.

- `Tests/run_operations_pipeline.py`
  - Defines all step builders.
  - Defines run profiles (`setup`, `train`, `test`, `full`, `smoke`).
  - Appends common args (metrics, subset, perturbations, verbosity).
  - Executes each step command.
  - Writes `test_runs_log.csv` with command and metric output snapshots.

- `Tests/call_models_files.py`
  - Simple runtime smoke checks for model module imports and tensor shape outputs.

- `Tests/examples_unified_dataset_loader.py`
  - Usage examples for dataset loader and subset filters.

- `Tests/__init__.py`
  - package marker.


### `Notebooks/`

This folder contains exploratory and validation notebooks.

- `Notebooks/metrics_validation_sanity.ipynb`
- `Notebooks/perturbation_validation_checks.ipynb`
- `Notebooks/batch_results_insights.ipynb`
- `Notebooks/class_removal_kmeans_experiments.ipynb`
- `Notebooks/class_imbalance_experiments.ipynb`
- `Notebooks/NotebookPrecision.ipynb`
- `Notebooks/NotebookTest_Density_and_Coverage.ipynb`
- `Notebooks/KID_notebook.ipynb`
- `Notebooks/Notebook_for_testing_loading_ddpm_and_ddim.ipynb`
- `Notebooks/dcgan_cifar10_100epochs_best_with_visualisations.ipynb`
- `Notebooks/datasets_eda.ipynb`
- `Notebooks/CleanNotebook.ipynb`

These notebooks are not called by the pipeline automatically, but are used for manual validation and analysis.


### `Z_md_files/`

Documentation and project notes.

- `Z_md_files/experiment_suites.txt`
- `Z_md_files/experiments.md`
- `Z_md_files/perturbations.md`
- `Z_md_files/METRIC_SOURCES.md`
- `Z_md_files/IMPLEMENTATION_CHANGES.md`
- `Z_md_files/running_instructions.md`


### Runtime artifact folders

- `data/`
  - Dataset files and downloaded metadata.

- `checkpoints/`
  - Pretrained and trained model weights.

- `outputs/`
  - Generated images, perturbation configs, metric JSON files, campaign reports, CSV logs.

- `.torch_cache/`
  - Torchvision cache used for Inception weights.

- `__pycache__/`, `.tmp_pycs/`
  - Python runtime caches and temporary bytecode folders.


### Root files

- `main.py`
  - Top-level controller.
  - Defines default settings.
  - Builds compact perturbation experiment list.
  - Runs either one pipeline command or batch mode.
  - Aggregates and updates report files incrementally.

- `README.md`
  - High-level project notes.

- `requirements.txt`
  - Python dependency list.

- `inception-2015-12-05.pt`
  - staged Inception model file (artifact).

- `netG_best.pth`, `netG_epoch_30.pth`
  - generator checkpoints (artifacts).

- `.gitignore`
  - git ignore rules.


## 4. How Files Connect

### Control path

- `main.py` creates command for `Tests/run_operations_pipeline.py`.
- `Tests/run_operations_pipeline.py` maps steps to scripts in `Scripts/`.

### Data path for test steps

1. Test script loads model through:
   - `Models/dcgan.py`, `Models/wgangp.py`, or `Models/pretrained_wrappers.py`.
2. Generates fake samples.
3. Loads real reference data through `Datasets/unified_dataset_loader.py`.
4. Applies perturbations through `Perturbation/pipeline_perturbations.py`.
5. Computes metrics through `Metrics/compute_all.py`.
6. Saves:
   - `generated_samples.png`
   - `perturbation_config.json`
   - `metrics_report.json`

### Batch reporting path

- `main.py` batch mode records per-experiment status and outputs.
- It writes one campaign JSON per model/dataset under `outputs/`.
- `Tests/run_operations_pipeline.py` also logs each test step in CSV.


## 5. Profiles and Step Composition

In `Tests/run_operations_pipeline.py`, profiles are fixed step lists:

- `setup`: data prep + checkpoint staging
- `train`: training scripts
- `test`: all model test scripts
- `full`: setup + train + test + smoke
- `smoke`: model import/shape checks only

`main.py` can override profile by passing explicit `CUSTOM_STEPS`.


## 6. Experiment Campaign Logic in `main.py`

`main.py` includes a compact experiment generator for StyleGAN2-CelebA:

- baseline
- degradation sweeps
- memoisation sweeps
- class removal sweeps (label and kmeans)
- class imbalance sweeps (label and kmeans)
- sample size sweeps
- preprocessing sweeps
- domain shift sweeps

Each experiment stores:

- deterministic experiment id
- exact command
- override dictionary
- exit status
- metric and perturbation outputs

On rerun, completed experiments can be skipped.


## 7. Output Contracts

Per test run output folder contains:

- `generated_samples.png`
- `metrics_report.json` (if metrics enabled and successful)
- `perturbation_config.json` (if perturbations enabled)

Batch report JSON contains:

- list of experiments
- command used
- status and exit code
- copied `metrics_report` and `perturbation_config` payloads
- timestamps

This contract is what `Notebooks/batch_results_insights.ipynb` reads for analysis.

# Running Instructions

This file is the practical guide for how this repository is meant to be used and extended.

The short version:
- `Models/` should stay model-focused (architecture and model-level helpers).
- `Scripts/` should handle training/testing/download workflows.
- `Datasets/` should handle loading and subset logic.
- `Metrics/` should expose metric functions.
- `Tests/run_operations_pipeline.py` is the top-level orchestrator for repeatable runs, which gets called by main.py

This is intentionally function-specific and modular, so you can swap pieces without rewriting the full stack.

## 1) Big Picture: How This Repo Is Organized

The project is structured as a pipeline with clean boundaries:

1. Data gets downloaded/prepared through `Scripts/download_*`.
2. Data gets loaded through `Datasets/unified_dataset_loader.py`.
3. Models are defined in `Models/` and should not own dataset-specific download logic.
4. Training/testing happens through `Scripts/train_*.py` and `Scripts/test_*.py`.
5. Metrics are computed through `Metrics/*` functions.
6. `main.py` is used to run multi-step workflows in order.

The design goal is that no model file should be the only way to run itself. A model should be callable as a module from scripts, tests, notebooks, or another pipeline

## 2) Repository Map

## Root files
- `README.md`
- `requirements.txt`: python dependencies
- `main.py`: editable top-level launcher for `Tests/run_operations_pipeline.py`.
- `running_instructions.md`

## `Datasets/`
- `Datasets/__init__.py`: package exports for dataset loaders/subsetting helpers
- `Datasets/unified_dataset_loader.py`: unified interface for MNIST/CIFAR10/CelebA/ChestXray14
- `Datasets/dataset_subset.py`: subset controls used across loaders/training
  - fraction/max sample filtering
  - random or class-balanced sampling
  - include/drop class filters
  - seeded reproducibility
- `Datasets/chestxray14_dataset.py`: ChestXray14 indexing + loader, including kagglehub flow

## `Models/`
- `Models/__init__.py`: model exports
- `Models/dcgan.py`: DCGAN architecture and weight init utilities only
  - `DCGANGenerator`
  - `DCGANDiscriminator`
  - `dcgan_weights_init`
- `Models/wgangp.py`: WGAN-GP skeleton/TODO.
- `Models/pretrained_wrappers.py`: pretrained wrapper skeletons/TODO.

## `Scripts/`
- `Scripts/__init__.py`: package marker.
- `Scripts/download_preprocess_mnist_cifar10.py`: MNIST/CIFAR10 setup.
- `Scripts/download_preprocess_celeba.py`: CelebA setup.
- `Scripts/download_preprocess_chestxray14.py`: ChestXray14 setup/indexing.
- `Scripts/download_pretrained_studiogan_cifar10.py`: pretrained staging (TODO details).
- `Scripts/download_pretrained_ddpm_cifar10.py`: pretrained staging (TODO details).
- `Scripts/download_pretrained_stylegan_celeba.py`: pretrained StyleGAN2 staging for CelebA-HQ (`.pkl`) with optional StyleGAN2-ADA source staging.
- `Scripts/train_dcgan.py`: DCGAN training entrypoint.
  - loads data through unified loader
  - supports subset controls
  - handles checkpoints/log images
- `Scripts/test_dcgan.py`: load a trained `netG` and generate sample images.
- `Scripts/test_wgangp.py`: WGAN-GP testing entrypoint with sample export and shared metric helper flow.
- `Scripts/test_studiogan.py`: StudioGAN wrapper testing entrypoint with shared metric helper flow.
- `Scripts/test_ddpm.py`: DDPM wrapper testing entrypoint with shared metric helper flow.
- `Scripts/test_stylegan2.py`: StyleGAN2 wrapper testing entrypoint (checkpoint loading, sampling, export, optional metrics).
- `Scripts/train_wgangp.py`: WGAN-GP training scaffold/TODO, now includes subset args.

## `Metrics/`
- `Metrics/__init__.py`: package marker/exports.
- `Metrics/compute_all.py`: wrapper to run all metrics.
- `Metrics/fid.py`: FID implementation helpers.
- `Metrics/is_score.py`: Inception Score skeleton/TODO.
- `Metrics/kid.py`: KID skeleton/TODO.
- `Metrics/precision_recall.py`: precision/recall metric implementation.
- `Metrics/density_coverage.py`: density/coverage skeleton/TODO.

## `Tests/`
- `Tests/__init__.py`: package marker.
- `Tests/run_operations_pipeline.py`: main orchestrator for setup/train/smoke workflows.
- `Tests/examples_unified_dataset_loader.py`: dataset-loader examples.
- `Tests/call_models_files.py`: model module smoke checks.

## `Notebooks/`
- `Notebooks/CleanNotebook.ipynb`: exploratory notebook.
- `Notebooks/datasets_eda.ipynb`: EDA notebook using unified loader.
- `Notebooks/NotebookPrecision.ipynb`: precision/recall workflow notebook updated to unified loader.

## Runtime/output folders
- `data/`: datasets live here (intentionally ignored in git).
- `outputs/`: training artifacts/checkpoints/images.

## 3) How To Run The Pipeline (Recommended)

If you want one place to set variables first and then launch the pipeline, use:

```powershell
py main.py
```

Edit the config block at the top of main.py:
- `PROFILE`, `CUSTOM_STEPS`
- `RUN`, `CUDA`, `CONTINUE_ON_ERROR`
- roots: `DATA_ROOT`, `CHECKPOINTS_ROOT`, `OUTPUTS_ROOT`
- training knobs: `IMAGE_SIZE`, `DCGAN_*`, `WGANGP_*`
- testing knobs: `TEST_*`, `*_TEST_CHECKPOINT`, `STRICT_TESTS`
- subset knobs: `SUBSET_*`

Example `main.py` setups:
- Dry-run everything: `PROFILE = "full"`, `RUN = False`
- Run setup only: `PROFILE = "setup"`, `RUN = True`
- Run testing only: `PROFILE = "test"`, `RUN = True`
- Train with subset: `PROFILE = "train"`, `RUN = True`, `SUBSET_FRACTION = 0.2`, `SUBSET_STRATEGY = "class_balanced"`, `SUBSET_DROP_CLASSES = "0"`

You can still run the pipeline script directly if you prefer full CLI control



## 5) Development Rules

If you continue this codebase, keep the boundaries clean:

## Models should stay standalone
- Put architecture and model-level utilities in `Models/`.
- Avoid dataset download/parsing and CLI-heavy logic in model files.
- A model module should be importable without side effects.

Good:
- `Models/dcgan.py` exports classes and utility init function.

Avoid:
- model file that can only be used by running it as a script with hardcoded paths.

## Scripts should own orchestration
- Training loop, optimizers, checkpoint writing, and command-line args belong in `Scripts/`.
- Scripts should consume models and datasets as dependencies, not redefine them.

## Datasets should own loading/subsetting
- Keep data format handling in `Datasets/`.
- Reuse `UnifiedDatasetLoader` for new workflows.
- Reuse `DatasetSubsetConfig` for consistent subset behavior.

## Metrics should stay function-style
- Metric modules should accept arrays/tensors and return values/dicts.
- Avoid hidden global state in metric code.

## Pipeline should remain thin
- `Tests/run_operations_pipeline.py` should only build/sequence commands.
- Keep model/data logic in modules/scripts, not in the pipeline file.


## 6) Make Paths Flexible (Repo Root Pattern)

For both scripts and notebooks, always resolve repo root dynamically.

Use this pattern:

```python
from pathlib import Path
import sys

CURRENT = Path.cwd().resolve()
if (CURRENT / "Datasets").exists():
    REPO_ROOT = CURRENT
elif (CURRENT.parent / "Datasets").exists():
    REPO_ROOT = CURRENT.parent
else:
    raise RuntimeError("Could not locate repository root.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

Then build paths from `REPO_ROOT`:

```python
data_root = REPO_ROOT / "data" / "CIFAR10"
```

This keeps notebooks runnable from either repo root or `Notebooks/` without edits.





# Group07 Improving Metrics

This repository is a configurable evaluation pipeline for generative models. It supports dataset preparation, checkpoint staging, local training, sample generation, perturbation-based stress tests, metric computation, batch experiment campaigns, and Unified Evaluation Frameword (RWFAS scoring).

The main controller is `main.py`. It is configuration-driven: you edit constants near the top of the file, then run:

```powershell
py main.py
```

## What The Pipeline Does


1. Prepare datasets in `data/`.
2. Stage pretrained checkpoints in `checkpoints/` when needed.
3. Train local models for workflows that use repo-trained weights.
4. Generate fake samples with a model-specific test script.
5. Optionally perturb fake samples, real samples, or both.
6. Compute FID, KID, Inception Score, Precision, Recall, Density, and Coverage.
7. Save run artifacts and JSON reports in `outputs/`.
8. Optionally aggregate perturbation campaigns into Task 3 / RWFAS analysis.

## Repository Map

| Path | Purpose |
| --- | --- |
| `main.py` | Top-level orchestrator for single runs and batch perturbation campaigns. |
| `experiments.py` | Builds predefined batch experiment suites and perturbation sweeps. |
| `Tests/run_operations_pipeline.py` | Step catalog and profile runner used by `main.py`. |
| `Scripts/` | Real entrypoints for setup, staging, training, and model testing. |
| `Datasets/` | Unified dataset loader plus subset and ChestXray14 helpers. |
| `Models/` | DCGAN, WGAN-GP, and pretrained wrapper implementations. |
| `Perturbation/` | Perturbation definitions and class-target selection logic. |
| `Metrics/` | Feature extraction and metric computation code. |
| `Task3/` | Downstream RWFAS scoring and cross-seed aggregation. |
| `Notebooks/` | Exploration and analysis notebooks. |
| `Z_md_files/` | Internal project notes and implementation writeups. |

## Supported Workflows

| Model family | Dataset | How weights are obtained | Main test step |
| --- | --- | --- | --- |
| DCGAN | MNIST | trained locally or explicit checkpoint override | `test_dcgan_mnist` |
| DCGAN | CIFAR-10 | trained locally or explicit checkpoint override | `test_dcgan_cifar10` |
| WGAN-GP | CIFAR-10 | trained locally or explicit checkpoint override | `test_wgangp_cifar10` |
| WGAN-GP | ChestXray14 | trained locally or explicit checkpoint override | `test_wgangp_chestxray14` |
| DDPM / DDIM | CIFAR-10 | staged pretrained `diffusers` checkpoint | `test_ddpm_cifar10` |
| StudioGAN | CIFAR-10 | staged pretrained checkpoint plus staged source tree | `test_studiogan_cifar10` |
| StyleGAN2 | CelebA | staged pretrained checkpoint plus staged StyleGAN2-ADA source | `test_stylegan2_celeba` |

## Installation And First-Time Setup

Install dependencies first:

```powershell
pip install -r requirements.txt
```

Notes:

- `torch`, `torchvision`, `scipy`, `scikit-learn`, and `diffusers` are required for the core pipeline.
- First-time setup needs network access for dataset downloads and pretrained checkpoints.
- ChestXray14 setup uses `kagglehub` through the dataset helper in `Datasets/chestxray14_dataset.py`.
- `data/`, `checkpoints/`, and `outputs/` are gitignored runtime folders.

## How `main.py` Decides What To Run

`main.py` has two execution modes.

### 1. Single-run mode

This mode runs one pipeline command assembled from `PROFILE` or `CUSTOM_STEPS`.


Important rule:

- Single-run mode only happens when `EXPERIMENTS` is empty.

In practice, if you want `PROFILE` or `CUSTOM_STEPS` to matter, set:

```python
EXPERIMENTS = []
```

If `EXPERIMENTS` is left as the result of `build_experiments_for_suite(...)`, `main.py` will ignore the single-run settings and launch a batch campaign instead.


In single-run mode, you are responsible for making sure the required setup, training, or staging steps are either already done or explicitly included in `CUSTOM_STEPS`.


### 2. Batch mode

This mode runs the perturbation suite defined by `EXPERIMENT_SUITE` and `EXPERIMENTS`.

Batch mode also does more automation than single-run mode:

- it preflights required datasets
- it preflights required checkpoints or staged model assets
- it can reuse in-process test sessions for large perturbation sweeps
- it writes resumable batch reports

## The Most Important `main.py` Settings

| Setting | Meaning |
| --- | --- |
| `PROFILE` | Profile name used when `CUSTOM_STEPS` is empty in single-run mode. |
| `CUSTOM_STEPS` | Exact ordered step list; overrides `PROFILE` in single-run mode. |
| `RUN` | `False` prints commands only, `True` executes them. |
| `CUDA` | Adds `--cuda` to compatible child scripts. |
| `EVAL_METRICS` | Enables metric computation during test steps. |
| `STRICT_TESTS` | Turns recoverable test-time skips into hard failures. |
| `CONTINUE_ON_ERROR` | Keeps going after a failed step or experiment. |
| `EXPERIMENT_SUITE` | Chooses the predefined batch perturbation suite. |
| `BATCH_NAME` | Label used in batch output paths and report filenames. |
| `GENERATION_SEED` | Seed for fake sample generation. |

### Baseline Safety Note

For clean single-run baseline testing, explicitly disable all perturbation flags unless you are intentionally running a stress test. `main.py` currently enables some perturbation-related defaults that can make a "normal" test run behave like a perturbation run if you do not turn them off.

A safe baseline block for single-run evaluation is:

```python
USE_PERTURBATIONS = False
PERTURB_DEGRADE = False
PERTURB_MEMOISATION = False
PERTURB_CLASS_REMOVAL = False
PERTURB_CLASS_IMBALANCE = False
PERTURB_SAMPLE_SIZE = False
PERTURB_PREPROCESSING = False
PERTURB_DOMAIN_SHIFT = False
```

## Profiles And Steps

`Tests/run_operations_pipeline.py` defines the profiles used by `main.py`.

| Profile | Steps |
| --- | --- |
| `setup` | `prep_mnist_cifar10`, `prep_celeba`, `prep_chestxray14`, `stage_studiogan`, `stage_ddpm`, `stage_stylegan` |
| `train` | `train_dcgan_cifar10`, `train_dcgan_mnist`, `train_wgangp_cifar10`, `train_wgangp_chestxray14` |
| `test` | `test_dcgan_cifar10`, `test_dcgan_mnist`, `test_wgangp_cifar10`, `test_wgangp_chestxray14`, `test_studiogan_cifar10`, `test_ddpm_cifar10`, `test_stylegan2_celeba` |
| `full` | setup + train + test + `smoke_models` |
| `smoke` | `smoke_models` only |

You can inspect the raw catalog directly with:

```powershell
py Tests/run_operations_pipeline.py --list
```

## End-To-End Flow During A Test Step

When a test step runs, the control flow is:

1. `main.py` builds a command for `Tests/run_operations_pipeline.py`.
2. The pipeline runner resolves the selected step into a script in `Scripts/`.
3. The test script loads a trained model or pretrained wrapper.
4. `Scripts/evaluation_runtime.py` generates or reuses cached fake samples.
5. If metrics or perturbations need real data, the runtime loads a real reference bundle through `Datasets/unified_dataset_loader.py`.
6. If perturbations are enabled, `Perturbation/pipeline_perturbations.py` applies them.
7. `Metrics/compute_all.py` extracts Inception-v3 features and computes the metric suite.
8. The runtime writes preview images, metric JSON, perturbation JSON, and cache JSON into the test output folder.

## How To Run Different Workflows From `main.py`

All single-run recipes below assume you first force single-run mode:

```python
EXPERIMENTS = []
RUN = True
```

Then run:

```powershell
py main.py
```

### 1. Setup-only workflow

Use this when you want to download datasets and stage pretrained assets without training or testing.

```python
EXPERIMENTS = []
PROFILE = "setup"
CUSTOM_STEPS = []
RUN = True
```

What it does:

- downloads MNIST and CIFAR-10
- downloads CelebA
- prepares ChestXray14 metadata/indexing
- stages StudioGAN, DDPM, and StyleGAN2 pretrained assets

### 2. One specific setup step

Use `CUSTOM_STEPS` when you only want a small part of setup.

Examples:

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["prep_mnist_cifar10"]
```

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["stage_ddpm"]
```

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["stage_stylegan"]
```

Remember:

- any non-empty `CUSTOM_STEPS` list overrides `PROFILE`

### 3. Training-only workflow

Use this when you want to produce local checkpoints without running test/evaluation yet.

Example: train DCGAN on CIFAR-10

```python
EXPERIMENTS = []
PROFILE = "train"
CUSTOM_STEPS = ["train_dcgan_cifar10"]
DCGAN_EPOCHS = 25
DCGAN_BATCH_SIZE = 64
CUDA = True
```

Example: train WGAN-GP on ChestXray14

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["prep_chestxray14", "train_wgangp_chestxray14"]
WGANGP_EPOCHS = 50
WGANGP_BATCH_SIZE = 64
CUDA = True
```

Training outputs are written under `outputs/`, for example:

- `outputs/dcgan_cifar10/`
- `outputs/dcgan_mnist/`
- `outputs/wgangp_cifar10/`
- `outputs/wgangp_chestxray14/`

The train scripts write rolling checkpoints such as `netG_latest.pth` and, for WGAN-GP, `netD_latest.pth`.

### 4. Test and metric-evaluation workflow

Use this when you already have weights and want generated samples plus metric reports.

Example: test a locally trained DCGAN on CIFAR-10

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["test_dcgan_cifar10"]
EVAL_METRICS = True
DCGAN_TEST_NETG = "outputs/dcgan_cifar10/netG_latest.pth"
TEST_NUM_SAMPLES = 1280
METRICS_SAMPLES = 1280
```

Example: test a staged DDPM checkpoint

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["test_ddpm_cifar10"]
EVAL_METRICS = True
DDPM_TEST_CHECKPOINT = "checkpoints/DDPM/CIFAR10"
TEST_NUM_SAMPLES = 1280
```

Example: test a staged StyleGAN2 checkpoint on CelebA

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["test_stylegan2_celeba"]
EVAL_METRICS = True
STYLEGAN2_TEST_CHECKPOINT = "checkpoints/StyleGAN/CelebA/stylegan2_generator.pkl"
```

Checkpoint override fields used by `main.py`:

- `DCGAN_TEST_NETG`
- `WGANGP_TEST_GENERATOR`
- `WGANGP_TEST_CRITIC`
- `DDPM_TEST_CHECKPOINT`
- `STUDIOGAN_TEST_CHECKPOINT`
- `STYLEGAN2_TEST_CHECKPOINT`

### 5. Full end-to-end workflow for one model

Use this when you want setup, training, and testing in one ordered run.

Example: full DCGAN CIFAR-10 path

```python
EXPERIMENTS = []
CUSTOM_STEPS = [
    "prep_mnist_cifar10",
    "train_dcgan_cifar10",
    "test_dcgan_cifar10",
]
EVAL_METRICS = True
CUDA = True
```

Example: full WGAN-GP ChestXray14 path

```python
EXPERIMENTS = []
CUSTOM_STEPS = [
    "prep_chestxray14",
    "train_wgangp_chestxray14",
    "test_wgangp_chestxray14",
]
EVAL_METRICS = True
CUDA = True
```

### 6. Perturbation workflow for one run

Use this when you want a one-off stress test rather than a whole batch suite.

Example: degradation stress test on DDPM CIFAR-10

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["test_ddpm_cifar10"]
EVAL_METRICS = True
USE_PERTURBATIONS = True
PERTURB_APPLY_TO = "fake"
PERTURB_DEGRADE = True
PERTURB_DEGRADE_SEVERITY = 3
PERTURB_DEGRADE_GAUSSIAN_NOISE = True

PERTURB_MEMOISATION = False
PERTURB_CLASS_REMOVAL = False
PERTURB_CLASS_IMBALANCE = False
PERTURB_SAMPLE_SIZE = False
PERTURB_PREPROCESSING = False
PERTURB_DOMAIN_SHIFT = False
```

Example: sample-size perturbation applied to both real and fake sets

```python
EXPERIMENTS = []
CUSTOM_STEPS = ["test_dcgan_cifar10"]
EVAL_METRICS = True
USE_PERTURBATIONS = True
PERTURB_APPLY_TO = "both"
PERTURB_SAMPLE_SIZE = True
PERTURB_SAMPLE_SIZE_N = 256
PERTURB_SAMPLE_SIZE_SEED = 10
```

Available perturbation families in `main.py`:

- degradation
- memoisation
- class removal
- class imbalance
- sample size
- preprocessing variation
- domain shift

### 7. Batch perturbation campaign workflow

Use this when you want the predefined experiment suite from `experiments.py`.

For batch mode, do not empty `EXPERIMENTS`. Keep the builder active and configure:

```python
PROFILE = "test"
CUSTOM_STEPS = []
BATCH_NAME = "seed_01"
GENERATION_SEED = 1
EXPERIMENT_SUITE = "ddpm_cifar10"
RUN = True
SKIP_COMPLETED_EXPERIMENTS = True
IN_PROCESS_BATCH_RUNNER = True
```

If you previously set `EXPERIMENTS = []` for single-run mode, restore the `build_experiments_for_suite(...)` assignment before running the batch campaign.

Then run:

```powershell
py main.py
```

Available predefined suites:

- `dcgan_pretrained_both`
- `dcgan_cifar10_pretrained`
- `dcgan_mnist_pretrained`
- `wgangp_cifar10`
- `wgangp_chestxray14`
- `ddpm_cifar10`
- `studiogan_cifar10`
- `stylegan2_celeba`

What batch mode does automatically:

- builds one experiment list from `experiments.py`
- assigns deterministic experiment IDs
- skips already completed experiments when `SKIP_COMPLETED_EXPERIMENTS = True`
- preloads missing datasets and staged model assets when possible
- writes per-model summary report JSON files in `outputs/`

## Output Structure

### Per-test output folder

A typical test output folder contains:

- `generated_samples.png`
- `metrics_report.json`
- `perturbation_config.json` when perturbations were active
- `cache_report.json`

### Shared cache

The evaluation runtime keeps reusable artifacts in:

- `outputs/shared_eval_cache/`

This cache is used for fake sample reuse, real reference reuse, feature reuse, and faster repeated batch experiments.

### Batch outputs

Per-experiment batch artifacts are written below:

```text
outputs/batch_runs/<batch_name>/<model_dataset>/<experiment_name>/<test_step_dir>/
```

Batch summary reports are written at the top of `outputs/`, for example:

```text
outputs/seed_01_ddpm_cifar10_perturbation_tests.json
```

Each report stores:

- experiment ID
- experiment name
- command used
- status and exit code
- output root
- copied metric payloads
- copied perturbation payloads
- copied cache payloads




## Task 3 / RWFAS Post-Processing

`Task3/` is downstream analysis on top of finished batch reports. It is not launched by `main.py` automatically.


The notebook `Task3/batch_results_insights.ipynb` allows to see generated plots of perturbations for each metric and trustworthiness analysis.

The notebook `Task3/task3_notebook.ipynb` is the interactive version of the unified evaluation protocol, with the process to obtain the RWFAS score.


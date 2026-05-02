# Experiments Plan for Metrics Performance Evaluation

## 1) Current Pipeline and Metrics: What Exists Today

### Pipeline execution flow
- `main.py` defines global settings and `EXPERIMENTS` overrides.
- Each experiment calls `Tests/run_operations_pipeline.py` with one or more `test_*` steps.
- Each test script writes outputs under its `--out-dir`:
  - `generated_samples.png`
  - `metrics_report.json` (when metric evaluation is enabled and supported)
  - `perturbation_config.json` (when perturbations are enabled)
- `main.py` batch mode creates deterministic experiment IDs, skips completed runs, and writes a per-model report JSON.

### Perturbation support status
- Implemented and wired in scripts with metric evaluation:
  - `test_dcgan.py` (CIFAR10, MNIST)
  - `test_ddpm.py` (CIFAR10)
  - `test_stylegan2.py` (CelebA)
- Not yet fully wired for this evaluation protocol:
  - `test_studiogan.py` is a standalone smoke script (no argparse pipeline, no metrics/perturbations/reporting).
  - `test_wgangp.py` is empty.
  - `train_wgangp.py` is a skeleton and currently only supports MNIST/CIFAR10, not ChestXray14.

### Metrics computation behavior
- Metrics are computed on flattened pixel tensors (images reshaped to 2D vectors), not on pretrained semantic features.
- Current metrics: `FID`, `IS`, `KID`, `precision/recall`, `density/coverage`.
- Consequence:
  - Low-level corruptions (noise/blur/jpeg) will strongly affect all distance-based metrics.
  - Domain-shift and semantic conclusions are weaker than with Inception/CLIP-style embeddings.
  - Results are still useful for relative stress testing inside this pipeline.

## 2) Common Experimental Protocol (applies to all perturbations)

- Keep one clean baseline per model/dataset.
- Use fixed seeds for generation and perturbations, then repeat with multiple seeds for CI/error bars.
- Recommended seeds: `10, 20, 30`.
- Use multiple sample sizes even for non-sample-size tests:
  - small: `64`
  - medium: `256`
  - large: `1024`
- Log all run metadata and perturbation configs.
- Compare perturbation run against the matching clean baseline from the same model, dataset, seed, and sample count.

## 3) Model/Dataset Matrix and Labeling Differences

- `dcgan` on `cifar10`: single-label 10 classes.
- `dcgan` on `mnist`: single-label 10 classes.
- `studiogan` on `cifar10`: single-label 10 classes (needs test-script integration).
- `ddpm` on `cifar10`: single-label 10 classes.
- `stylegan2` on `celeba`: multi-label 40 binary attributes.
- `wgangp/wcgan` on `chestxray14`: multi-label findings (medical; sparse positive labels).

Class-removal and class-imbalance differences:
- Single-label runs: target class IDs/names directly (`cat`, `airplane`, `7`, etc.).
- Multi-label runs: use attribute/finding names and tune label-threshold behavior.
- Multi-label runs should include both common and rare labels to test sensitivity.

## 4) Perturbation Experiment Definitions

## A) Degradation (noise / blur / jpeg / all)

### Variables to sweep
- corruption type: `noise`, `blur`, `jpeg`, `all`.
- severity: `1, 2, 3, 4, 5`.
- apply target: `fake` (primary), `real`, `both` (diagnostic controls).

### Runs to include
- For each model/dataset baseline:
  - 4 corruption types x 5 severities x 3 apply targets.
- Minimum exhaustive core if compute is limited:
  - apply target `fake` only, full type x severity grid.

### Expected metric behavior and why
- `FID`, `KID`: worsen as severity increases due to larger pixel-distribution mismatch.
- `IS`: usually decreases with heavy blur/noise because class evidence degrades.
- `Precision`: decreases first and strongly (fake manifold quality drops).
- `Recall` and `Coverage`: can decline slower at low severities, then drop at high severities.
- `Density`: decreases as fake points leave real-neighborhood radii.

## B) Memorisation

### Variables to sweep
- injection fraction: `0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.0`.
- seed: `10, 20, 30`.

### Runs to include
- All model/dataset pairs, fake-side replacement only.
- Keep sample count fixed first (e.g., `256`), then repeat for `64` and `1024`.

### Expected metric behavior and why
- `FID`, `KID`: improve (artificially) as real images are injected.
- `IS`: may improve slightly or remain stable, depending on replaced-image class confidence.
- `Precision`: increases.
- `Recall` and `Coverage`: may not improve proportionally; diversity can stagnate while closeness rises.
- Key interpretation: this perturbation checks whether a metric can be gamed by copying real data.

## C) Class Removal (mode dropping)

### Variables to sweep
- strategy: `label`, `kmeans`.
- target set size:
  - single-label datasets: remove `1`, `2`, `4`, `6` classes.
  - multi-label datasets: remove `1`, `2`, `4` attributes/findings.
- target prevalence:
  - high-frequency labels
  - mid-frequency labels
  - low-frequency labels
- kmeans parameters: `k=4,8,12`, drop `1` cluster then `2` clusters.
- `label_threshold` (multi-label): `0.0, 0.1, 0.2`.

### Runs to include by dataset type
- CIFAR10/MNIST:
  - explicit class-removal ladders (e.g., keep removing more classes).
- CelebA/ChestXray14:
  - attribute/finding removal ladders.
  - pairs of correlated labels (for co-occurrence stress).

### Expected metric behavior and why
- `Recall`, `Coverage`: strongest degradation (missing modes).
- `FID`, `KID`: worsen, but less specifically than recall/coverage for pure mode loss.
- `Precision`: can remain moderate/high if remaining modes are high quality.
- `Density`: can remain deceptively stable in surviving modes.

## D) Class Imbalance (representation bias)

### Variables to sweep
- strategy: `label`, `kmeans`.
- target labels: common vs rare.
- drop ratio (`balance`): `0.1, 0.2, 0.4, 0.6, 0.8, 0.9`.
- number of targeted labels: `1`, `2`, `4`.
- multi-label per-target ratios (list form): examples `0.2,0.6`, `0.4,0.8`.
- `label_threshold` (multi-label): `0.0, 0.1, 0.2`.

### Runs to include
- All model/dataset pairs.
- Separate runs for:
  - skewing a single dominant class.
  - skewing multiple classes.
  - skewing rare classes only.

### Expected metric behavior and why
- `Recall`, `Coverage`: decrease with stronger imbalance (partial mode loss).
- `FID`, `KID`: degrade gradually, often less sharply than full class removal.
- `Precision`: may stay stable in mild imbalance, then drop with severe skew.
- `IS`: may stay high if dominant classes are easy/high-confidence.

## E) Sample Size Variation (TODO to implement)

### Required pipeline changes
- Add dedicated experiment axis for metric sample count while generator output distribution is fixed.
- Keep generated pool fixed once, then evaluate subsets.

### Variables to sweep
- evaluation sample counts: `16, 32, 64, 128, 256, 512, 1024, 2048`.
- subset resampling repeats per count: `5` repeats.

### Expected metric behavior and why
- Mean metric values should stabilize as sample size grows.
- Variance should shrink with larger `N`.
- Unstable metrics will show large swings at small `N`.
- Use this to quantify estimator robustness and confidence intervals.

## F) Preprocessing Variation (TODO to implement)

### Required pipeline changes
- Add configurable real/fake preprocessing before feature flattening.
- Save preprocessing config in report.

### Variables to sweep
- resize kernel: `nearest`, `bilinear`, `bicubic`, `lanczos`.
- normalization mode: `[-1,1]`, `[0,1]`, per-channel z-score.
- optional center-crop vs no-crop.
- optional color conversions:
  - RGB to grayscale (replicated channels)
  - grayscale to RGB expansion.

### Expected metric behavior and why
- Robust implementations should show limited drift under equivalent preprocessing.
- Large drifts indicate implementation sensitivity, not real model-quality change.

## G) Domain Shift (TODO to implement)

### Required pipeline changes
- Allow `metrics_dataset` to differ from generator training dataset in a controlled matrix.
- Keep real reference sample count and preprocessing fixed.

### Suggested domain-shift pairs
- CIFAR10 generators (`dcgan`, `studiogan`, `ddpm`) vs MNIST, CelebA, ChestXray14.
- `stylegan2` (CelebA) vs CIFAR10 and ChestXray14.
- `wgangp/wcgan` (ChestXray14) vs CIFAR10 and CelebA.

### Expected metric behavior and why
- All comparison metrics should worsen strongly under true domain mismatch.
- If some metrics do not worsen, this indicates weak semantic validity in current feature space.

## 5) Model-Specific Notes and Priority Differences

- `dcgan` (`cifar10`, `mnist`): best first target for full exhaustive sweeps because scripts are already integrated.
- `ddpm` (`cifar10`): perturbation and metrics path is also integrated; include same sweeps as DCGAN.
- `stylegan2` (`celeba`): prioritize multi-label class-removal/imbalance with attribute prevalence buckets.
- `studiogan` (`cifar10`): first implement parity with `test_ddpm.py`/`test_stylegan2.py` (argparse, perturbations, metrics JSON).
- `wgangp/wcgan` (`chestxray14`):
  - implement train/test support for ChestXray14 and metric/perturbation integration.
  - prioritize medically meaningful multi-label imbalance/removal targets (common + rare findings).




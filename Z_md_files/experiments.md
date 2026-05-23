# Experiments Plan for Metrics Performance Evaluation

This document is the canonical overview of parameter sweeps currently configured in `main.py` + `experiments.py`, including expected metric behavior and rationale.

## 1) Global Evaluation Protocol

These settings are global defaults used by the current batch runs:

- `TEST_NUM_SAMPLES = 256`
- `TEST_BATCH_SIZE = 64`
- `METRICS_SAMPLES = 256`
- `METRICS_FEATURE_BATCH_SIZE = 64`
- `METRICS_BOOTSTRAP_SAMPLES = 50`
- `METRICS_BOOTSTRAP_SEED = 10`

Interpretation:
- Metrics are evaluated on paired real/fake subsets up to 256 samples.
- Confidence intervals are estimated via bootstrap with 50 resamples.
- Perturbation-specific seeds are fixed per experiment for repeatability.

## 2) Active Experiment Suites

Current batch suites:
- `dcgan_cifar10_pretrained`
- `dcgan_mnist_pretrained`
- `dcgan_pretrained_both`
- `stylegan2_celeba`

Each suite includes a baseline (`baseline_no_perturbation`) and the perturbation sweeps below.

## 3) Sweep Definitions by Perturbation Family

## A) Degradation

Applies to all active suites.

Sweep axes:
- corruption type: `noise`, `blur`, `jpeg`, `all`
- severity:
- DCGAN suites: `1, 2, 3, 4, 5`
- StyleGAN2 suite: `1, 3, 5`
- apply target: `fake`

Expected behavior and why:
- `FID`, `KID`: worsen with stronger corruption due to larger distribution mismatch.
- `IS`: usually degrades for severe blur/noise because classifier confidence drops.
- `Precision`, `Density`: typically drop first as fake sample quality degrades.
- `Recall`, `Coverage`: may degrade later, then collapse at high severity.

## B) Memoisation

Applies to all active suites.

Sweep axis:
- replacement fraction: `0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50`

Expected behavior and why:
- `FID`, `KID`: improve (artificially) as more real images are injected.
- `Precision`: often increases (fake samples are closer to real manifold).
- `Recall`, `Coverage`: may not improve proportionally if diversity is not truly improved.
- This sweep is a metric-gaming stress test.

## C) Class Imbalance

### DCGAN (MNIST, CIFAR10)

Targets:
- label targets: `0`, `1`, `2`, `0,1,2`

Balance levels:
- `0.90, 0.75, 0.50, 0.30, 0.15, 0.05`

### StyleGAN2 (CelebA)

Targets:
- kmeans targets only with `k=10`
- target sizes: `1`, `3`, `5`
- size `1`: all 10 single-cluster cases
- size `3`: up to 6 fixed random combinations
- size `5`: up to 6 fixed random combinations

Balance levels:
- `0.90, 0.50, 0.20`

Expected behavior and why:
- `Recall`, `Coverage`: decrease as under-represented modes disappear.
- `FID`, `KID`: worsen progressively with stronger imbalance.
- `Precision`: can remain relatively stable under mild imbalance and drop under severe skew.

## D) Class Removal (Mode Dropping)

### DCGAN (MNIST, CIFAR10)

Label-removal severity by number of removed classes:
- remove `1, 2, 4, 6, 8` classes

Implementation detail:
- targets are generated as first-N class indices (`0..N-1`).

### StyleGAN2 (CelebA)

KMeans cluster removal only:
- `k=10`
- target sizes: `1`, `3`, `5`
- size `1`: all 10 single-cluster cases
- size `3`: up to 6 fixed random combinations
- size `5`: up to 6 fixed random combinations

Expected behavior and why:
- `Recall`, `Coverage`: strongest degradation (missing modes/classes).
- `FID`, `KID`: worsen, but less specifically diagnostic than recall/coverage for pure mode dropping.
- `Precision`: can stay moderate/high if remaining modes are still sharp.

## E) Sample Size Variation

Applies to all active suites.

Sweep axis:
- `16, 32, 64, 128, 196, 256`

Expected behavior and why:
- Lower `N` increases estimator variance and instability.
- As `N` grows, metric estimates should stabilize and CIs should narrow.
- This quantifies robustness of the metric estimates themselves.

## F) Preprocessing Variation

Applies to all active suites.

Variants and scales:
- `downsample_nearest`: `0.90, 0.75, 0.60, 0.45, 0.30`
- `downsample_bilinear`: `0.90, 0.75, 0.60, 0.45, 0.30`
- `downsample_bicubic`: `0.90, 0.75, 0.60, 0.45, 0.30`
- `center_crop_pad`: `0.90, 0.75, 0.60, 0.45, 0.30`
- `grayscale_triplicate`: `0.75`

Expected behavior and why:
- Robust metrics should change smoothly with stronger preprocessing distortions.
- Large discontinuities suggest preprocessing sensitivity rather than true model-quality change.

## G) Domain Shift

### DCGAN suites
- `dcgan_cifar10_pretrained`: shift to `mnist`
- `dcgan_mnist_pretrained`: shift to `cifar10`

### StyleGAN2 suite
- shift to `mnist`, `cifar10`, `chestxray14`

Expected behavior and why:
- Strong domain mismatch should degrade all distribution-comparison metrics.
- If a metric barely changes, it may be weak for semantic distribution shift detection.

## 4) Dataset-Specific Coverage Notes

- MNIST/CIFAR10 (10 classes): class-removal counts map cleanly to class-count severity.
- CelebA: the active StyleGAN2 suite uses only the kmeans target space over label co-occurrence, not the direct 40-label sweep.
- ChestXray14-specific class-removal/class-imbalance sweeps are documented conceptually but are not yet in an active dedicated suite in the current `build_experiments_for_suite(...)` routing.


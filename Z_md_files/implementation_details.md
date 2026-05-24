# Implementation Details

## 1. Metrics implementation

This section explains both the code path and the statistical meaning of each metric.
Think of it as: what the metric measures, how this repository computes it, and how to interpret the output.

## 1.1 Shared metric pipeline

Main orchestrator:

- `Metrics/compute_all.py`

Entry function:

- `compute_all_metrics(real_samples, fake_samples, config)`

Execution flow in code:

1. Convert `real_samples` and `fake_samples` to CPU `float32` tensors.
2. Enforce paired sample count:
   - `paired_count = min(len(real), len(fake))`
   - both sets are truncated to `paired_count`
3. Validate minimum count:
   - if `paired_count < 4`, metric computation fails early.
4. Feature-space selection:
   - current implementation supports only `feature_space="inception_v3"`.
5. Extract representations:
   - real: Inception pool3 features
   - fake: Inception pool3 features + class probabilities
6. Compute base metrics:
   - FID, KID, Inception Score, Precision/Recall, Density/Coverage.
7. Optional bootstrap:
   - if `bootstrap_samples > 0`, attach percentile confidence intervals.

Error handling:

- Every metric call is wrapped in `_safe_metric_call`.
- If a metric fails, its output becomes `{"error": "...message..."}` and other metrics still run.

Returned payload keys:

- `metadata`
- `fid`
- `kid`
- `is`
- `precision_recall`
- `density_coverage`

Important implementation behavior:

- real/fake are truncated to equal length before feature extraction, so all metrics run on matched counts.
- bootstrap uses deterministic seeds with metric-specific offsets (`+11`, `+23`, etc.) to avoid identical resamples across metrics.


## 1.2 Inception feature extraction 

File:

- `Metrics/inception_features.py`

Main class:

- `InceptionFeatureExtractor`

Why this exists:

- Most image-generation metrics are not robust in raw pixel space.
- The extractor maps images to a semantic feature space learned by ImageNet-pretrained Inception-v3.
- These features (pool3, 2048-D) are a standard protocol for FID/KID/PRDC-style evaluation.

Model construction details:

- Uses `torchvision.models.inception_v3` with:
  - `weights=Inception_V3_Weights.IMAGENET1K_V1`
  - `aux_logits=True`
  - `transform_input=False`
- The model is set to `eval()` mode.
- A forward hook is registered on `model.avgpool` to capture pool3 activations.

Preprocessing pipeline per batch:

1. `[-1,1]` to `[0,1]` handling:
   - if min < 0 or max > 1, apply `(x + 1) / 2`.
   - clamp to `[0,1]`.
2. Channel handling:
   - 1-channel input is repeated to 3 channels.
   - 3-channel input is kept as-is.
   - other channel counts raise an error.
3. Resize:
   - bilinear resize to `299 x 299` (`align_corners=False`).
4. Normalize with ImageNet statistics:
   - mean `[0.485, 0.456, 0.406]`
   - std `[0.229, 0.224, 0.225]`

Forward outputs used:

- logits:
  - extracted from `raw_output.logits` when available.
- probabilities:
  - `softmax(logits, dim=1)` (used by Inception Score).
- features:
  - hook-captured `avgpool` output flattened to shape `[N, 2048]`.

Returned arrays:

- `features_np`: `float64`, shape `[N, 2048]`
- `logits_np`: `float64`, shape `[N, 1000]`
- `probs_np`: `float64`, shape `[N, 1000]`

Engineering details:

- `torch.hub.set_dir(Path.cwd() / ".torch_cache")` forces model-cache writes to a workspace path.
- `close()` removes the hook and should always be called (handled by `try/finally` in `compute_all_metrics`).

Interpretation note:

- Inception features are not "ground truth semantics"; they are a proxy learned from ImageNet categories.
- Metrics inherit this bias: they are strong for natural-image realism/diversity checks, weaker for domains far from ImageNet.


## 1.3 FID (Frechet Inception Distance)

File:

- `Metrics/fid.py`

Functions:

- `calculate_fid(mu1, sigma1, mu2, sigma2)`
- `compute_fid_from_features(real_features, fake_features, max_cov_dim=2048)`

What it measures:

- Distance between two multivariate Gaussians fitted to real and fake feature distributions.
- Captures both mean shift (quality mismatch) and covariance shift (diversity/structure mismatch).

Formula implemented:

- `FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 * sqrtm(Sigma_r * Sigma_f))`

How this code computes it:

1. Optionally project features with `_project_high_dim_features`.
2. Compute feature means and covariances with `np.mean` / `np.cov`.
3. Compute matrix square root with `scipy.linalg.sqrtm`.
4. Apply numerical safeguards:
   - if `sqrtm` output is non-finite, add diagonal jitter `eps=1e-6` and recompute.
   - if result is complex, keep real part.

Why projection exists:

- With small `N` and high feature dimension `D`, covariance can be rank-deficient and unstable.
- The code computes an SVD basis over combined centered features and projects both real/fake to a shared low-dimensional space.
- Effective cap:
  - `allowed_cov_dim = min(max_cov_dim, (N_real + N_fake - 1))`

How to interpret:

- Lower is better.
- FID is not an absolute "percent correct" score.
- It is sensitive to sample size and preprocessing protocol; compare only under identical settings.


## 1.4 KID (Kernel Inception Distance)

File:

- `Metrics/kid.py`

Functions:

- `polynomial_kernel(x, y)`
- `kid(real_features, fake_features)`
- `compute_kid(real_features, fake_features)`

What it measures:

- Maximum Mean Discrepancy (MMD) between real and fake feature distributions using a polynomial kernel.
- Like FID, lower is better, but KID is an unbiased estimator in finite samples.

Kernel used:

- `k(x, y) = ((x^T y / d) + 1)^3`, where `d` is feature dimension.

Estimator details:

- Uses unbiased MMD by removing diagonal self-similarity terms from `Kxx` and `Kyy`.
- Single-round estimate can be noisy, so `compute_kid` repeats random subset sampling.

Subset/round strategy in code:

- Convert features to `float32` to reduce memory.
- Compute a memory-based cap for subset size (rough 256MB budget).
- `subset_size = min(1000, n_real, n_fake, memory_cap)`
- rounds:
  - 100 if subset size >= 16
  - 20 otherwise
- return:
  - `(mean, std)` over rounds

Interpretation notes:

- Lower mean is better.
- KID can be slightly negative due to finite-sample variance; values near zero are typically "very close."
- `std` reflects instability across subset draws, not a formal confidence interval.


## 1.5 Inception Score (IS)

File:

- `Metrics/is_score.py`

Function:

- `compute_inception_score(probabilities, num_splits=10)`

What it measures:

- High-quality samples should produce confident per-image class predictions (`p(y|x)` low entropy).
- Diverse samples should cover many classes overall (marginal `p(y)` higher entropy).
- IS combines both via KL divergence between `p(y|x)` and `p(y)`.

Formula:

- For each split:
  - `IS_split = exp(E_x[ KL(p(y|x) || p(y)) ])`
- Final output:
  - mean and std across splits.

Input validation in code:

- probabilities must be 2D `[N, C]`.
- all rows must sum approximately to 1 (`atol=1e-3`).
- negative probabilities raise errors.

Split behavior:

- `num_splits` is clipped to `[1, N]`.
- each split uses contiguous chunks of samples.

Interpretation cautions:

- IS uses only fake samples; it does not compare against real data directly.
- It can reward classifiable but unrealistic images in some settings.
- Use together with FID/KID/PRDC, not alone.


## 1.6 Precision / Recall / Density / Coverage (PRDC family)

Files:

- `Metrics/precision_recall.py`
- `Metrics/density_coverage.py`
- `Metrics/prdc_utils.py`

Shared geometric idea:

- Build local neighborhoods in feature space using k-nearest-neighbor radii.
- Decide whether samples from one set fall inside neighborhoods of the other set.

Distance backend:

- `sklearn.metrics.pairwise_distances(..., metric="euclidean", n_jobs=8)`

Neighborhood radius computation:

- `nearest_neighbour_radii(features, nearest_k)` computes all pairwise distances within one set.
- Radius for each point is the `(k+1)`th smallest distance.
- `+1` is used because each point has zero distance to itself.

### Precision and Recall

Function:

- `compute_precision_recall(real_features, fake_features, k=5)`

Definitions:

- Precision:
  - fraction of fake samples that fall inside at least one real sample neighborhood.
  - "How much of generated content lies on the real data manifold?"
- Recall:
  - fraction of real samples that fall inside at least one fake sample neighborhood.
  - "How much of the real manifold is covered by generated samples?"

Ranges:

- both in `[0, 1]`
- higher is better

### Density and Coverage

Function:

- `compute_density_coverage(real_features, fake_features, k=5)`

Definitions:

- Density:
  - average number of real neighborhoods containing each fake sample, normalized by `k`.
  - reflects concentration around real modes.
- Coverage:
  - fraction of real samples whose nearest fake lies within that real sample's radius.
  - reflects support coverage of real set.

Ranges:

- Coverage is in `[0, 1]` (higher better).
- Density is non-negative and may exceed `1` (higher generally better, but too-high values can indicate over-concentration).


## 1.7 Bootstrapping and confidence intervals

Files:

- `Metrics/statistics.py`
- called from `Metrics/compute_all.py`

Method:

- percentile bootstrap with replacement.
- resample real and fake independently to their original sizes.

Utility flow:

1. `bootstrap_metric_distribution(...)` generates bootstrap metric values.
2. `bootstrap_percentile_interval(...)` computes quantiles:
   - low = `alpha/2`
   - high = `1 - alpha/2`
3. `with_bootstrap_summary(...)` formats:
   - point estimate
   - CI metadata (`method`, `alpha`, `low`, `high`, `bootstrap_samples`)

Metric-specific bootstrap choices:

- FID:
  - bootstrap scalar FID directly.
- KID:
  - bootstrap KID mean (not KID std).
- IS:
  - bootstrap by resampling fake probability rows.
- PR:
  - bootstrap precision and recall separately.
- DC:
  - bootstrap density and coverage separately.

Interpretation note:

- These are empirical uncertainty intervals for this exact evaluation protocol.
- They are not strict guarantees and depend on sample size, bootstrap count, and data assumptions.


## 1.8 Metric source references

References used in project docs (`Z_md_files/METRIC_SOURCES.md`):

- PRDC reference implementation:
  - https://github.com/clovaai/generative-evaluation-prdc
- FID/IS/KID protocol references:
  - https://github.com/toshas/torch-fidelity
  - https://github.com/GaParmar/clean-fid
  - https://github.com/mseitzer/pytorch-fid

Alignment notes for this codebase:

- PR/DC formulas and neighborhood logic follow PRDC reference conventions.
- FID uses the standard Frechet Gaussian formula with explicit numerical stabilizers.
- IS uses the standard split-KL implementation on Inception probabilities.
- KID uses the polynomial-kernel MMD estimator on extracted Inception features.


## 2. Model implementations

## 2.1 DCGAN (`Models/dcgan.py`)

Generator:

- latent input `[N, nz, 1, 1]`
- initial transposed convolution to `ngf*8` channels
- repeated upsampling blocks until requested image size
- final conv to channel count `nc`
- `tanh` output in `[-1,1]`

Discriminator:

- convolutional downsampling blocks
- first block without batchnorm (DCGAN convention)
- progressive downsampling to `4x4`
- final conv to scalar + sigmoid

Validation:

- `_validate_image_size` enforces power-of-2 size and minimum 32.

Initialization:

- `dcgan_weights_init`:
  - conv weights normal(0,0.02)
  - batchnorm gamma normal(1,0.02), beta 0


## 2.2 WGAN-GP (`Models/wgangp.py`)

Generator:

- DCGAN-like transposed conv architecture
- outputs in `[-1,1]`

Critic:

- convolutional network without sigmoid
- instance norm in intermediate blocks
- outputs one scalar critic value per sample

Gradient penalty:

- interpolates between real and fake
- computes gradient norm of critic output wrt interpolated input
- returns mean squared deviation from 1


## 2.3 Pretrained wrappers (`Models/pretrained_wrappers.py`)

### StudioGANWrapper

- stages source path into `sys.path`
- loads config and generator via StudioGAN source code
- loads checkpoint state dict (prefers EMA weights if available)
- sampling uses random latent and random class labels

### DDPMWrapper / DDIMWrapper

- loads `diffusers` pipeline from local directory checkpoint
- supports both DDPM and DDIM pipeline types
- sampling supports:
  - optional seed
  - optional number of inference steps
- output conversion path handles tensor, numpy, or PIL list
- always returns `[N,C,H,W]` tensor in `[-1,1]`

### StyleGAN2Wrapper

- loads `.pkl` or torch checkpoint
- supports StyleGAN2-ADA legacy unpickling path
- tries common generator extraction keys (`G_ema`, `G`)
- infers latent dimension from common attributes or first linear layer
- supports conditional and unconditional checkpoints
- supports multiple known call signatures for compatibility


## 3. Dataset implementations

## 3.1 Unified loader (`Datasets/unified_dataset_loader.py`)

Supported datasets:

- MNIST
- CIFAR10
- CelebA
- ChestXray14

Transform pipeline:

- resize to target square size
- `ToTensor()`
- optional normalize to `[-1,1]`
  - 1-channel stats for MNIST
  - 3-channel stats for other datasets

For each dataset, loader chooses corresponding torchvision/custom dataset class and returns train/test split as requested.


## 3.2 Subset system (`Datasets/dataset_subset.py`)

Configuration:

- `fraction`
- `max_samples`
- `seed`
- `strategy`
- `include_classes`
- `drop_classes`

Class handling:

- resolves class tokens by name or numeric index
- supports single-label and multi-label filters

Sampling:

- `random` strategy
- `class_balanced` strategy for single-label datasets only

The subset wrapper preserves original dataset attributes by forwarding unknown attributes to wrapped dataset.


## 3.3 ChestXray14 dataset (`Datasets/chestxray14_dataset.py`)

Preparation function:

- `prepare_chestxray14_dataset`

Behavior:

- resolves dataset root naming variants
- finds metadata CSV
- can download via `kagglehub`
- indexes image paths by filename
- creates split assignment from provided split files when available
- otherwise creates patient-level random split
- saves:
  - `chestxray14_index.csv`
  - `finding_classes.txt`

Dataset class:

- `ChestXray14Dataset`
- reads cached index
- filters by split
- loads RGB image
- produces multi-hot vector across findings


## 4. Training and test script implementation details

## 4.1 Training scripts

Files:

- `Scripts/train_dcgan.py`
- `Scripts/train_wgangp.py`

Common behavior:

- load dataset via `UnifiedDatasetLoader`
- apply subset config from CLI
- infer channel count from first dataset sample
- periodic image snapshots during training
- save epoch checkpoints and rolling latest checkpoints

DCGAN training:

- BCE loss
- alternating discriminator and generator updates

WGAN-GP training:

- critic updates per generator step (`critic-iters`)
- Wasserstein objective with gradient penalty


## 4.2 Test scripts

Files:

- `Scripts/test_dcgan.py`
- `Scripts/test_wgangp.py`
- `Scripts/test_studiogan.py`
- `Scripts/test_ddpm.py`
- `Scripts/test_stylegan2.py`

Common behavior:

- load model or wrapper
- generate fake samples in batches
- optional real reference loading via `UnifiedDatasetLoader`
- optional perturbation application
- save generated grid image
- optional metric evaluation using `compute_all_metrics`
- save `metrics_report.json`
- save `perturbation_config.json` when perturbations enabled

Strict mode:

- `--strict` converts failures from skip behavior into hard errors.


## 5. Pipeline orchestration implementation

## 5.1 Step-level orchestration (`Tests/run_operations_pipeline.py`)

Responsibilities:

- parse run arguments
- build step commands from `STEP_BUILDERS`
- choose profile or custom steps
- execute commands
- log failures and optionally continue
- for test steps, append rows to `test_runs_log.csv` with:
  - command
  - output dir
  - metrics path and payload snapshot
  - perturbation config snapshot


## 5.2 Batch campaign orchestration (`main.py`)

Responsibilities:

- holds top-level defaults for profile, metrics, perturbations, subsets
- builds compact experiment definitions for perturbation sweeps
- computes deterministic experiment ids from payload hashes
- writes and updates report JSON files incrementally
- supports resume by skipping completed experiments

Report entry includes:

- experiment metadata
- exact command
- override settings
- exit code and status
- collected test output payloads


## 6. Known implementation particularities

- Metric computation now uses Inception feature space, not flattened raw pixels.
- PRDC metrics can be slower because pairwise distances are computed with scikit-learn.
- FID path includes feature projection fallback for small sample and high feature dimension settings.
- Class removal and class imbalance assign pseudo-labels to fake samples using centroid distance in feature space, not model-provided labels.
- KMeans strategy for class perturbations clusters labels by co-occurrence, not images by appearance.
- Domain shift perturbation changes reference dataset loading path, not fake sample generation.


## 7. Dependency notes

Important runtime dependencies in `requirements.txt` include:

- `torch`, `torchvision`
- `numpy`, `scipy`, `scikit-learn`
- `diffusers`
- `kagglehub[pandas-datasets]`
- `torchmetrics`
- `clean-fid`

Some scripts require optional external assets:

- internet access for first-time checkpoint/data download
- staged StyleGAN2-ADA source for certain `.pkl` checkpoints
- staged StudioGAN source for StudioGAN checkpoint loading

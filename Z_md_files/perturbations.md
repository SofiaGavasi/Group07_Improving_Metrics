# Perturbations Implementation Details

## 1. Where perturbations are configured and applied

Perturbations are controlled from experiment settings in `main.py`, passed to `Tests/run_operations_pipeline.py`, then forwarded as CLI flags to model test scripts:

- `Scripts/test_dcgan.py`
- `Scripts/test_wgangp.py`
- `Scripts/test_studiogan.py`
- `Scripts/test_ddpm.py`
- `Scripts/test_stylegan2.py`

Inside test scripts, all perturbation logic is centralized in:

- `Perturbation/pipeline_perturbations.py`

The common call is:

- `apply_configured_perturbations(fake_samples, args, real_samples, reference_targets, reference_class_names, dataset_name)`

It returns:

- perturbed fake tensor
- perturbed real tensor (or unmodified / None)
- a full perturbation config dictionary that is saved as `perturbation_config.json`


## 2. Global perturbation controls

These controls are shared by all perturbations:

- `--use-perturbations`
- `--perturb-apply-to {fake,real,both}`

Behavior:

- `fake` means perturb only generated samples.
- `real` means perturb only reference samples used in metric computation.
- `both` means perturb both sides before metrics.

Important exception:

- `memoisation`, `class_removal`, and `class_imbalance` are fake-side operations.
- If `perturb_apply_to` excludes fake for these, they are skipped and reported in `config["skipped"]`.


## 3. Perturbation execution order

In `apply_configured_perturbations`, the order is fixed:

1. `degradation`
2. `memoisation`
3. `class_removal`
4. `class_imbalance`
5. `sample_size`
6. `preprocessing`
7. `domain_shift` (handled as dataset override in test script loading path)

Order matters because later steps consume outputs from earlier steps.


## 4. Degradation perturbation

### What it does

Applies image-quality corruption to sample tensors using `Perturbation/degrade_dataset.py`.

Enabled by:

- `--perturb-degrade`

Type switches:

- `--perturb-degrade-gaussian-noise`
- `--perturb-degrade-gaussian-blur`
- `--perturb-degrade-jpeg-compression`

Severity:

- `--perturb-degrade-severity` in `[1..5]`

### How it is implemented

`DegradedDataset` wraps a tensor dataset and applies selected transforms in `__getitem__`.

The base severity tables are:

- noise sigma (in 0..255 scale): `[6, 18, 32, 56, 96]`
- blur kernels: `[3, 7, 11, 17, 25]`
- blur sigmas: `[0.8, 1.8, 3.0, 5.0, 8.0]`
- JPEG qualities: `[75, 45, 25, 10, 3]`

The code applies resolution-aware scaling:

- corruption gets stronger on higher resolution images
- level 5 is intentionally very strong

JPEG path:

- tensor `[-1,1]` -> `[0,1]`
- convert to PIL
- save/load through JPEG in memory
- convert back to tensor and `[-1,1]`

### Effect on metric computation

Corrupted tensors are sent directly to Inception feature extraction.  
This changes feature geometry and usually worsens distance and neighborhood metrics as severity increases.


## 5. Memoisation perturbation

### What it does

Replaces a fraction of fake samples with real samples to simulate memorization.

Enabled by:

- `--perturb-memoisation`
- `--perturb-memo-fraction`
- `--perturb-memo-seed`

### How it is implemented

`Perturbation/memorization_dataset.py`:

- picks `n_inject = floor(len(fake) * fraction)` fake positions
- picks `n_inject` real indices
- maps fake index to real index
- returns real sample at injected positions, fake sample otherwise

In `pipeline_perturbations.py`, a safety check enforces:

- enough real samples must be available to support requested replacements

### Effect on metric computation

The fake distribution gets partially replaced by true data points.  
This can artificially improve FID/KID and precision-like metrics even when generator quality did not improve.


## 6. Class removal perturbation

### What it does

Simulates mode dropping by removing fake samples predicted to belong to selected targets.

Enabled by:

- `--perturb-class-removal`

Strategy:

- `--perturb-class-removal-strategy {label,kmeans}`

Targets:

- `--perturb-class-removal-targets` (comma separated names or indices)

Safeguards:

- `--perturb-class-removal-min-kept`

### 6.1 Label strategy

Single-label datasets:

- build reference class centroids from real features
- assign each fake to nearest centroid
- drop all fakes assigned to target class ids

Multi-label datasets:

- for each target label, compute positive and negative centroids
- predict fake as positive when `(dist_neg - dist_pos) > label_threshold`
- drop predicted positives

Control:

- `--perturb-class-removal-label-threshold`

### 6.2 KMeans strategy

This is not clustering images.  
It clusters labels using label co-occurrence features derived from real target matrix.

Process:

1. Build binary label matrix from real targets.
2. Compute label co-occurrence feature matrix.
3. Fit `KMeans(k)` on labels (not samples).
4. Map label indices to cluster ids.
5. Convert target cluster ids into label index set.
6. Reuse label strategy logic to drop samples for those labels.

Controls:

- `--perturb-class-removal-kmeans-k`
- `--perturb-class-removal-kmeans-cache-path`
- `--perturb-class-removal-kmeans-recreate`
- `--perturb-class-removal-seed`

Cache:

- saved in `outputs/perturbation_cache/...npz` by default
- reused if label count and k match

### Effect on metric computation

Class removal reduces support in fake distribution.  
Coverage and recall should usually drop, while precision can stay stable if remaining samples are still close to dense real regions.


## 7. Class imbalance perturbation

### What it does

Simulates representation bias by partially dropping selected fake class regions, not fully removing them.

Enabled by:

- `--perturb-class-imbalance`

Strategy:

- `--perturb-class-imbalance-strategy {label,kmeans}`

Targets:

- `--perturb-class-imbalance-targets`

Balance control:

- `--perturb-class-imbalance-balance`
- single value applies to all targets
- comma list can map per target in multi-label path

Safeguards:

- `--perturb-class-imbalance-min-kept`

### How it is implemented

Label and kmeans target resolution is parallel to class removal.

Single-label path:

- assign fake to nearest class centroid
- for each target class, randomly drop `int(class_count * balance)` samples

Multi-label path:

- compute predicted positives for each target label with positive/negative centroids
- drop each predicted-positive fake with probability = balance for that label

KMeans path:

- cluster labels by co-occurrence exactly as in class removal
- convert selected cluster ids to label indices
- apply label-based imbalance logic

### Effect on metric computation

Compared to full class removal, this causes gradual support loss.  
Recall and coverage are expected to degrade progressively with stronger imbalance settings.


## 8. Sample size variation perturbation

### What it does

Keeps only `n` samples by random subsampling.  
It is used to test metric estimator variance and stability.

Enabled by:

- `--perturb-sample-size`
- `--perturb-sample-size-n`
- `--perturb-sample-size-seed`

### How it is implemented

`_apply_sample_size_variation`:

- validates non-empty tensor
- validates `n > 0`
- validates `n <= available_count`
- selects random permutation with seeded `torch.Generator`
- slices first `n` indices

If `apply_to=both`, both fake and real are reduced to `n` (real uses `seed+1`).

### Effect on metric computation

This does not change image content distribution directly.  
It changes estimator reliability because fewer points are used for feature statistics and neighborhood decisions.


## 9. Preprocessing variation perturbation

### What it does

Applies deterministic preprocessing transforms before metric extraction to test implementation sensitivity.

Enabled by:

- `--perturb-preprocessing`
- `--perturb-preprocessing-variant`
- `--perturb-preprocessing-scale`

Variants:

- `downsample_nearest`
- `downsample_bilinear`
- `downsample_bicubic`
- `center_crop_pad`
- `grayscale_triplicate`

### How it is implemented

All variants are implemented in `_apply_preprocessing_variation` in `pipeline_perturbations.py`.

- Downsample variants: downsample to `scale * H/W`, then upsample back.
- Center crop pad: keep centered crop at scale, zero-pad back to original canvas.
- Grayscale triplicate: average channels, then repeat channel map.

### Effect on metric computation

These transformations modify the representation seen by Inception.  
A metric that is very sensitive to small preprocessing changes may vary strongly even when semantic content is mostly unchanged.


## 10. Domain shift perturbation

### What it does

Changes only the real reference dataset/domain used for metric computation.

Enabled by:

- `--perturb-domain-shift`
- `--perturb-domain-shift-dataset`
- `--perturb-domain-shift-data-root`
- `--perturb-domain-shift-image-size`

### How it is implemented

In `pipeline_perturbations.py`, `get_domain_shift_override` returns an override dictionary.  
The test script then uses this override when loading real samples for metric computation.

It does not alter fake samples.

### Effect on metric computation

All metrics compare fake against an out-of-domain real reference.  
This is expected to strongly degrade semantic validity of score interpretation and often leads to worse FID/KID and neighborhood metrics.


## 11. What is stored in `perturbation_config.json`

The saved config contains:

- top-level enable status
- target side (`fake`, `real`, `both`)
- active perturbation list
- detailed parameters for each perturbation type
- `applied` list with concrete operations executed
- `skipped` list with reasons
- perturbation-specific result metadata for class removal / imbalance

This JSON is the exact record needed to reproduce each experiment condition.


## 12. Interaction with metric pairing

Metric computation uses paired truncation:

- `paired_count = min(len(real), len(fake))`
- both tensors are truncated to the same count before feature extraction

This means perturbations that reduce sample count can indirectly change which samples are used in final metric computation, not only their content.

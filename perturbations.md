# Perturbations Overview + Experiment Plan

## Current Perturbations

All perturbations are configured from `main.py` and passed through the test pipeline.

### 1) Degradation perturbation

**Goal:** degrade visual quality while keeping the same semantic content.

Implemented in `Perturbation/degrade_dataset.py` and applied through `Perturbation/pipeline_perturbations.py`.

Controls:
- `PERTURB_DEGRADE` (on/off)
- `PERTURB_DEGRADE_SEVERITY` (1 to 5)
- `PERTURB_DEGRADE_GAUSSIAN_NOISE`
- `PERTURB_DEGRADE_GAUSSIAN_BLUR`
- `PERTURB_DEGRADE_JPEG_COMPRESSION`
- `PERTURB_APPLY_TO` (`"fake"`, `"real"`, `"both"`)

Expected effect: lower fidelity but similar semantic support.


### 2) Memoisation perturbation

**Goal:** simulate overfitting/memorization by injecting real samples into fake outputs.

Implemented in `Perturbation/memorization_dataset.py`.

Controls:
- `PERTURB_MEMOISATION` (on/off)
- `PERTURB_MEMO_FRACTION` (fraction of fake replaced by real)
- `PERTURB_MEMO_SEED`

Note:
- In practice, this is a fake-side perturbation. if `PERTURB_APPLY_TO` excludes fake, this perturbation is skipped.

Expected effect: fake set gets artificially closer to real set.


### 3) Class-removal perturbation (coverage perturbation)

**Goal:** simulate mode dropping by removing classes/modes from generated samples.

Implemented in `Perturbation/class_removal.py`.

Two strategies:

#### 3a) `label` strategy
- Drop classes directly by label name or index.
- Works for single-label and multi-label datasets.

Controls:
- `PERTURB_CLASS_REMOVAL` (on/off)
- `PERTURB_CLASS_REMOVAL_STRATEGY = "label"`
- `PERTURB_CLASS_REMOVAL_TARGETS` (comma-separated labels/ids)
- `PERTURB_CLASS_REMOVAL_LABEL_THRESHOLD` (used in multi-label assignment logic)
- `PERTURB_CLASS_REMOVAL_MIN_KEPT`


#### 3b) `kmeans` strategy (label co-occurrence kmeans)
- Clusters **labels** using label co-occurrence patterns from real reference labels.
- Then drops selected label-cluster ids.

Controls:
- `PERTURB_CLASS_REMOVAL` (on/off)
- `PERTURB_CLASS_REMOVAL_STRATEGY = "kmeans"`
- `PERTURB_CLASS_REMOVAL_KMEANS_K`
- `PERTURB_CLASS_REMOVAL_TARGETS` (cluster ids to drop)
- `PERTURB_CLASS_REMOVAL_KMEANS_CACHE_PATH`
- `PERTURB_CLASS_REMOVAL_KMEANS_RECREATE`
- `PERTURB_CLASS_REMOVAL_MIN_KEPT`

Notes:
- This strategy requires reference labels.
- Cache is supported to reuse label-cluster assignments.


---

##  Experiment Structure

Main goal: test whether metrics react correctly to known perturbations.

For each model/dataset pair, run the same seed/sample count and vary one perturbation axis at a time.

### A) Baseline
- No perturbation.
- Save all metric outputs and perturbation config logs.

### B) Degradation sweep
- Severity: `1, 2, 3, 4, 5`
- Repeat for:
  - noise only
  - blur only
  - jpeg only
  - combined (all enabled)

### C) Memoisation sweep
- Fraction: `0.0, 0.05, 0.10, 0.20, 0.30`
- Keep everything else fixed.

### D) Class-removal sweep
- `label` strategy:
  - single-label (MNIST/CIFAR10): drop 1 class, 2 classes, 4 classes
  - multi-label (CelebA/ChestXray14): drop 1 high-support label, 1 low-support label, and 2-label combination
- `kmeans` strategy:
  - vary `K` (e.g., 4, 6, 8)
  - drop 1 cluster, then 2 clusters
  - use same cached cluster assignment for fair comparisons


---

## What We Should Expect (Metric Behavior)

These are expected qualitative trends if metrics are behaving sensibly:

### Under degradation
- FID/KID: should worsen as severity increases.
- Precision: should usually drop.
- Recall/Coverage: may drop less than precision (content still similar but quality lower).

### Under memoisation
- Distribution-distance metrics (FID/KID): may improve artificially (because fake includes real samples).
- Precision: may increase.
- Recall/Coverage: may not improve proportionally; can expose mismatch between "closeness" and true generative diversity.

### Under class removal
- Recall/Coverage: should drop strongly (missing modes).
- Precision: may stay stable or even look decent (remaining modes can still be high quality).
- FID/KID: should degrade, but often less specifically than coverage-style metrics for mode loss.


---



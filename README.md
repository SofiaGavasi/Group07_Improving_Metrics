# Group07 Improving Metrics

## Repository Layout

- `Models/`: generative model implementations and sample interface
- `Datasets/`: dataset helpers and unified loader
- `Metrics/`: FID/IS/KID/Precision-Recall/Density-Coverage API
- `Scripts/`: dataset prep, checkpoint setup, and training entrypoints
- `Notebooks/`: exploratory analysis/evaluation notebooks
- `Tests/`: interface tests




## Files Overview

- `Datasets/unified_dataset_loader.py`: unified loader for MNIST, CIFAR-10, CelebA, and ChestX-ray14
- `Datasets/chestxray14_dataset.py`: ChestX-ray14 dataset indexing/loading via kagglehub + local split metadata
- `Models/generation.py`: single `generate_samples(model, n, ...)` interface across models
- `Models/wgangp.py`: WGAN-GP architecture/training utility skeleton (TODO placeholders)
- `Models/pretrained_wrappers.py`: wrapper skeletons for StudioGAN, DDPM/DDIM, and StyleGAN2 checkpoints
- `Metrics/compute_all.py`: wrapper that runs all metric functions and returns a unified results dictionary
- `Scripts/download_preprocess_mnist_cifar10.py`: MNIST and CIFAR-10 download/setup entrypoint
- `Scripts/download_preprocess_celeba.py`: CelebA download/setup entrypoint
- `Scripts/download_preprocess_chestxray14.py`: ChestX-ray14 download/index setup entrypoint
- `Scripts/download_pretrained_studiogan_cifar10.py`: checkpoint staging entrypoint for StudioGAN (CIFAR-10)
- `Scripts/download_pretrained_ddpm_cifar10.py`: checkpoint staging entrypoint for DDPM (CIFAR-10)
- `Scripts/download_pretrained_stylegan_celeba.py`: checkpoint staging entrypoint for StyleGAN/StyleGAN2 (CelebA)
- `Scripts/train_dcgan.py`: training entrypoint that dispatches to `Models/dcgan.py` (currently CIFAR-10 path supported)
- `Scripts/train_wgangp.py`: WGAN-GP training entrypoint scaffold

## Tests Files

- `Tests/examples_unified_dataset_loader.py`: runnable examples for calling `UnifiedDatasetLoader` on each dataset type
- `Tests/call_models_files.py`: module-level smoke runner for all files in `Models/` (implemented parts run, TODO parts report error`)
- `Tests/run_operations_pipeline.py`: dynamic pipeline/orchestration runner that chains `Scripts/*.py` operations via profiles or custom steps
I put examples on how to run each in each of these test files in the comments in each


## Open TODOs

- `Models/wgangp.py`: replace placeholder WGAN-GP generator architecture with final architecture. implement `WGANGPGenerator.forward`. replace placeholder WGAN-GP critic architecture with final architecture. implement `WGANGPCritic.forward`.  implement WGAN-GP `gradient_penalty`

- `Models/pretrained_wrappers.py`: load StudioGAN generator from checkpoint,  implement `StudioGANWrapper.sample`
- `Models/pretrained_wrappers.py`: load DDPM model from checkpoint, add DDIM sampling path, implement `DDPMWrapper.sample`
- `Models/pretrained_wrappers.py`: load StyleGAN/StyleGAN2 generator from checkpoint, implement `StyleGAN2Wrapper.sample`

- `Metrics/is_score.py`: implement Inception Score with split-based mean/std output
- `Metrics/kid.py`: implement Kernel Inception Distance with subset mean/std output
- `Metrics/precision_recall.py`: implement Precision and Recall metric 
- `Metrics/density_coverage.py`: implement Density and Coverage metric
- `Metrics/fid.py`: add optional clean-fid mode/settings arguments after eval protocol is finalized

- `Scripts/download_preprocess_mnist_cifar10.py`: add project-specific preprocessing/export steps (cached tensors, stats, splits).
- `Scripts/download_preprocess_celeba.py`: decide aligned vs unaligned variant and exact split policy, verify CelebA download endpoint/access requirements in target runtime
- `Scripts/download_pretrained_studiogan_cifar10.py`: add concrete StudioGAN checkpoint URLs and download logic,  add verification and extraction logic
- `Scripts/download_pretrained_ddpm_cifar10.py`: add DDPM checkpoint download logic, add DDIM sampler config and checkpoint compatibility checks
- `Scripts/download_pretrained_stylegan_celeba.py`: add StyleGAN/StyleGAN2 checkpoint download logic,  add checkpoint conversion when source format differs from wrapper runtime
- `Scripts/train_dcgan.py`: update `Models/dcgan.py` dataset pipeline for MNIST training support, add logging hooks and standardized checkpoint names.
- `Scripts/train_wgangp.py`: instantiate `WGANGPGenerator` and `WGANGPCritic` from `Models/wgangp.py`.  build dataloaders via `Datasets/unified_dataset_loader.py`. implement full WGAN-GP training loop (critic iterations, gradient penalty, checkpointing). add evaluation hooks calling `Metrics/compute_all.py` on generated samples.

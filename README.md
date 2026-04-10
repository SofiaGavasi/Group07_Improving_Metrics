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
- `Datasets/dataset_subset.py`: reusable subset controls (fraction/max size, class-balanced sampling, include/drop classes)
- `Models/wgangp.py`: WGAN-GP architecture/training utility skeleton (TODO placeholders)
- `Models/pretrained_wrappers.py`: pretrained wrapper module (StyleGAN2 loading/sampling implemented; StudioGAN and DDPM remain TODO)
- `Metrics/compute_all.py`: wrapper that runs all metric functions and returns a unified results dictionary
- `Scripts/download_preprocess_mnist_cifar10.py`: MNIST and CIFAR-10 download/setup entrypoint
- `Scripts/download_preprocess_celeba.py`: CelebA download/setup entrypoint
- `Scripts/download_preprocess_chestxray14.py`: ChestX-ray14 download/index setup entrypoint
- `Scripts/download_pretrained_studiogan_cifar10.py`: checkpoint staging entrypoint for StudioGAN (CIFAR-10)
- `Scripts/download_pretrained_ddpm_cifar10.py`: checkpoint staging entrypoint for DDPM (CIFAR-10)
- `Scripts/download_pretrained_stylegan_celeba.py`: stages a pretrained StyleGAN2 CelebA-HQ checkpoint (`.pkl`) and optional StyleGAN2-ADA source
- `Scripts/train_dcgan.py`: DCGAN training entrypoint using `UnifiedDatasetLoader` (+ subset controls)
- `Scripts/test_dcgan.py`: DCGAN generator sampling entrypoint from a trained `netG` checkpoint
- `Scripts/train_wgangp.py`: WGAN-GP training entrypoint scaffold
- `Scripts/test_wgangp.py`: WGAN-GP testing entrypoint with shared metrics hook (`_evaluate_and_save_metrics`)
- `Scripts/test_studiogan.py`: StudioGAN pretrained wrapper testing entrypoint with shared metrics hook
- `Scripts/test_ddpm.py`: DDPM pretrained wrapper testing entrypoint with shared metrics hook
- `Scripts/test_stylegan2.py`: StyleGAN2 pretrained wrapper testing entrypoint (generation + optional metrics)



## Open TODOs

- `Models/wgangp.py`: replace placeholder WGAN-GP generator architecture with final architecture. implement `WGANGPGenerator.forward`. replace placeholder WGAN-GP critic architecture with final architecture. implement `WGANGPCritic.forward`.  implement WGAN-GP `gradient_penalty`

- `Models/pretrained_wrappers.py`: load StudioGAN generator from checkpoint,  implement `StudioGANWrapper.sample`
- `Models/pretrained_wrappers.py`: load DDPM model from checkpoint, add DDIM sampling path, implement `DDPMWrapper.sample`

- `Metrics/is_score.py`: implement Inception Score with split-based mean/std output

- `Scripts/download_preprocess_mnist_cifar10.py`: add project-specific preprocessing/export steps (cached tensors, stats, splits).
- `Scripts/download_preprocess_celeba.py`: decide aligned vs unaligned variant and exact split policy, verify CelebA download endpoint/access requirements in target runtime
- `Scripts/download_pretrained_studiogan_cifar10.py`: add concrete StudioGAN checkpoint URLs and download logic,  add verification and extraction logic
- `Scripts/download_pretrained_ddpm_cifar10.py`: add DDPM checkpoint download logic, add DDIM sampler config and checkpoint compatibility checks
- `Scripts/train_wgangp.py`: instantiate `WGANGPGenerator` and `WGANGPCritic` from `Models/wgangp.py`.  build dataloaders via `Datasets/unified_dataset_loader.py`. implement full WGAN-GP training loop (critic iterations, gradient penalty, checkpointing). add evaluation hooks calling `Metrics/compute_all.py` on generated samples.
- `Scripts/test_wgangp.py`: finalize sampling behavior once `Models/wgangp.py` architecture/forward paths are implemented.
- `Scripts/test_studiogan.py`: complete `StudioGANWrapper.sample` so sample export + metric evaluation become fully operational.
- `Scripts/test_ddpm.py`: complete `DDPMWrapper.sample` (DDPM/DDIM) so sample export + metric evaluation become fully operational.

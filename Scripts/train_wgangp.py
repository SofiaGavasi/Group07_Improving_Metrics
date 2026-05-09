# Example:
# py Scripts/train_wgangp.py --dataset cifar10 --data-root data/CIFAR10 --epochs 1 --batch-size 64 --image-size 32 --out-dir outputs/wgangp_cifar10
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datasets.dataset_subset import DatasetSubsetConfig, parse_class_identifiers
from Datasets.unified_dataset_loader import make_default_loader
from Models.wgangp import WGANGPCritic, WGANGPGenerator, gradient_penalty


# parse args
def parse_args():
    parser = argparse.ArgumentParser(description="Train WGAN-GP with UnifiedDatasetLoader.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "chestxray14"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--g-base", type=int, default=64)
    parser.add_argument("--d-base", type=int, default=64)
    parser.add_argument("--critic-iters", type=int, default=5)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--out-dir", type=str, default="outputs/wgangp")
    parser.add_argument("--manual-seed", type=int, default=None)
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--subset-max-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--subset-strategy", type=str, default="random", choices=["random", "class_balanced"])
    parser.add_argument("--subset-include-classes", type=str, default="")
    parser.add_argument("--subset-drop-classes", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try download/setup if local dataset files are missing.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging for dataset/model/training setup details.",
    )
    return parser.parse_args()


# load train dataset
def _load_train_dataset(args: argparse.Namespace):
    subset_config = DatasetSubsetConfig(
        # Reuse shared subset semantics so train and test pipelines match.
        fraction=args.subset_fraction,
        max_samples=args.subset_max_samples,
        seed=args.subset_seed,
        strategy=args.subset_strategy,
        include_classes=parse_class_identifiers(args.subset_include_classes),
        drop_classes=parse_class_identifiers(args.subset_drop_classes),
    )
    loader = make_default_loader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_size=args.image_size,
        subset_config=subset_config,
    )
    try:
        return loader.get_dataset(train=True, download=False)
    except (FileNotFoundError, RuntimeError) as exc:
        if not args.download_if_missing:
            raise
        print(f"{args.dataset}: local load failed ({exc})")
        print(f"{args.dataset}: attempting download/setup...")
        return loader.get_dataset(train=True, download=True)


# entry point when running this script
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.manual_seed if args.manual_seed is not None else random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    cudnn.benchmark = args.cuda and torch.cuda.is_available()
    if args.verbose:
        print(
            f"[train_wgangp] device={device} dataset={args.dataset} image_size={args.image_size} "
            f"batch_size={args.batch_size} critic_iters={args.critic_iters}",
            flush=True,
        )

    dataset = _load_train_dataset(args)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty after filtering.")
    if args.verbose:
        print(f"[train_wgangp] dataset_size_after_filtering={len(dataset)}", flush=True)

    sample_x, _ = dataset[0]
    if not torch.is_tensor(sample_x) or sample_x.ndim != 3:
        raise ValueError("Expected dataset to return image tensor in CHW format.")
    channels = int(sample_x.shape[0])
    if args.verbose:
        print(f"[train_wgangp] inferred channels={channels} sample_shape={tuple(sample_x.shape)}", flush=True)

    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        drop_last=True,
    )
    if args.verbose:
        print(f"[train_wgangp] dataloader_batches={len(dataloader)}", flush=True)

    generator = WGANGPGenerator(
        latent_dim=int(args.latent_dim),
        out_channels=channels,
        base_channels=int(args.g_base),
    ).to(device)
    critic = WGANGPCritic(
        in_channels=channels,
        base_channels=int(args.d_base),
    ).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    fixed_noise = torch.randn(max(1, int(args.batch_size)), int(args.latent_dim), 1, 1, device=device)

    global_step = 0
    for epoch in range(int(args.epochs)):
        for step, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = int(real.shape[0])

            # Critic update(s): maximize E[D(real)] - E[D(fake)] + GP.
            for _ in range(max(1, int(args.critic_iters))):
                noise = torch.randn(batch_size, int(args.latent_dim), 1, 1, device=device)
                fake = generator(noise).detach()

                critic_real = critic(real)
                critic_fake = critic(fake)
                gp = gradient_penalty(critic=critic, real=real, fake=fake, device=device)
                loss_d = -(critic_real.mean() - critic_fake.mean()) + float(args.gp_lambda) * gp

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            # Generator update: maximize E[D(fake)] (equivalently minimize -E[D(fake)]).
            noise = torch.randn(batch_size, int(args.latent_dim), 1, 1, device=device)
            fake = generator(noise)
            loss_g = -critic(fake).mean()

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            if global_step % max(1, int(args.log_interval)) == 0:
                print(
                    f"[{epoch + 1}/{args.epochs}][{step}/{len(dataloader)}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} GP: {gp.item():.4f}"
                )
                save_image(real[:64], out_dir / "real_samples.png", normalize=True)
                with torch.no_grad():
                    preview = generator(fixed_noise)
                save_image(preview[:64], out_dir / f"fake_samples_epoch_{epoch + 1:03d}.png", normalize=True)

            global_step += 1

        # Save epoch and rolling checkpoints for pipeline test scripts.
        torch.save(generator.state_dict(), out_dir / f"netG_epoch_{epoch + 1}.pth")
        torch.save(critic.state_dict(), out_dir / f"netD_epoch_{epoch + 1}.pth")
        torch.save(generator.state_dict(), out_dir / "netG_latest.pth")
        torch.save(critic.state_dict(), out_dir / "netD_latest.pth")


if __name__ == "__main__":
    main()

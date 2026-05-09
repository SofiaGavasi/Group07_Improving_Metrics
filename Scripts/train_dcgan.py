# example: py Scripts/train_dcgan.py --dataset cifar10 --data-root data/CIFAR10 --epochs 1 --batch-size 64 --image-size 32 --outf outputs/dcgan_cifar10 --subset-fraction 0.2 --subset-strategy class_balanced --subset-drop-classes 0
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
# allow direct script execution without requiring package install.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datasets.dataset_subset import DatasetSubsetConfig, parse_class_identifiers
from Datasets.unified_dataset_loader import make_default_loader
from Models.dcgan import DCGANDiscriminator, DCGANGenerator, dcgan_weights_init


# parse args
def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN using UnifiedDatasetLoader.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--outf", type=str, default="outputs/dcgan")
    parser.add_argument("--netG", type=str, default="", help="Optional checkpoint path for generator.")
    parser.add_argument("--netD", type=str, default="", help="Optional checkpoint path for discriminator.")
    parser.add_argument("--manual-seed", type=int, default=None)
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--subset-max-samples", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--subset-strategy",
        type=str,
        default="random",
        choices=["random", "class_balanced"],
    )
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
def load_train_dataset(args, subset_config: DatasetSubsetConfig):
    loader = make_default_loader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_size=args.image_size,
        subset_config=subset_config,
    )
    try:
        # first try local data to avoid unnecessary downloads
        return loader.get_dataset(train=True, download=False)
    except (FileNotFoundError, RuntimeError) as exc:
        if not args.download_if_missing:
            raise
        # fallback is explicit so pipeline/test logs clearly show what happened
        print(f"{args.dataset}: local load failed ({exc})")
        print(f"{args.dataset}: attempting download/setup...")
        return loader.get_dataset(train=True, download=True)


# entry point when running this script
def main():
    args = parse_args()

    out_dir = Path(args.outf)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.manual_seed if args.manual_seed is not None else random.randint(1, 10000)
    print("Random Seed:", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    cudnn.benchmark = args.cuda and torch.cuda.is_available()
    if args.verbose:
        print(
            f"[train_dcgan] device={device} dataset={args.dataset} image_size={args.image_size} "
            f"batch_size={args.batch_size}",
            flush=True,
        )

    subset_config = DatasetSubsetConfig(
        # centralized subset config keeps behavior consistent across scripts/pipeline
        fraction=args.subset_fraction,
        max_samples=args.subset_max_samples,
        seed=args.subset_seed,
        strategy=args.subset_strategy,
        include_classes=parse_class_identifiers(args.subset_include_classes),
        drop_classes=parse_class_identifiers(args.subset_drop_classes),
    )

    dataset = load_train_dataset(args, subset_config=subset_config)
    print(f"Dataset size after filtering: {len(dataset)}")
    if args.verbose:
        print(
            "[train_dcgan] subset_config="
            f"{subset_config}",
            flush=True,
        )

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty after filtering.")

    # infer channels from actual dataset output so this works for mnist and cifar10
    sample_x, _ = dataset[0]
    if not torch.is_tensor(sample_x) or sample_x.ndim != 3:
        raise ValueError("Expected dataset to return image tensor in CHW format.")
    channels = int(sample_x.shape[0])
    if args.verbose:
        print(f"[train_dcgan] inferred channels={channels} sample_shape={tuple(sample_x.shape)}", flush=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    if args.verbose:
        print(f"[train_dcgan] dataloader_batches={len(dataloader)}", flush=True)

    netG = DCGANGenerator(
        ngpu=0,
        nc=channels,
        nz=args.nz,
        ngf=args.ngf,
        image_size=args.image_size,
    ).to(device)
    netD = DCGANDiscriminator(
        ngpu=0,
        nc=channels,
        ndf=args.ndf,
        image_size=args.image_size,
    ).to(device)

    if args.netG:
        # resume path if checkpoint is provided
        netG.load_state_dict(torch.load(args.netG, map_location=device))
    else:
        netG.apply(dcgan_weights_init)

    if args.netD:
        netD.load_state_dict(torch.load(args.netD, map_location=device))
    else:
        netD.apply(dcgan_weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0
    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)

    for epoch in range(args.epochs):
        for step, data in enumerate(dataloader):
            real = data[0].to(device)
            batch_size = real.size(0)

            # 1) update D: maximize log(D(x)) + log(1 - D(G(z))).
            netD.zero_grad(set_to_none=True)
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            output_real = netD(real)
            errD_real = criterion(output_real, labels)
            errD_real.backward()
            d_x = output_real.mean().item()

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            labels.fill_(fake_label)

            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, labels)
            errD_fake.backward()
            d_g_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # 2) update G: maximize log(D(G(z))) (equivalently minimize BCE vs real labels)
            netG.zero_grad(set_to_none=True)
            labels.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, labels)
            errG.backward()
            d_g_z2 = output.mean().item()
            optimizerG.step()

            if step % args.log_interval == 0:
                print(
                    f"[{epoch + 1}/{args.epochs}][{step}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}"
                )
                # keep one real/fake preview for quick training sanity checks
                save_image(real[:64], out_dir / "real_samples.png", normalize=True)
                with torch.no_grad():
                    preview = netG(fixed_noise)
                save_image(
                    preview.detach(),
                    out_dir / f"fake_samples_epoch_{epoch + 1:03d}.png",
                    normalize=True,
                )

        # keep both per-epoch and rolling latest checkpointss
        torch.save(netG.state_dict(), out_dir / f"netG_epoch_{epoch + 1}.pth")
        torch.save(netD.state_dict(), out_dir / f"netD_epoch_{epoch + 1}.pth")
        torch.save(netG.state_dict(), out_dir / "netG_latest.pth")
        torch.save(netD.state_dict(), out_dir / "netD_latest.pth")


if __name__ == "__main__":
    main()

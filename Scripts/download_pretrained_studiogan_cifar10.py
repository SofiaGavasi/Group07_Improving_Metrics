# example: py Scripts/download_pretrained_studiogan_cifar10.py --output-dir checkpoints/StudioGAN/CIFAR10

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request
import zipfile
import shutil
import tempfile

DEFAULT_STUDIOGAN_REPO = ('https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.git')

DEFAULT_CHECKPOINT_URL = 'https://huggingface.co/Mingguksky/PyTorch-StudioGAN/resolve/main/studiogan_official_ckpt/CIFAR10_tailored/CIFAR10-SNGAN-train-2022_03_06_02_24_46/model%3DG-best-weights-step%3D88000.pth'


# download file
def _download_file(url, destination):
    if url is None:
        print("No checkpoint URL")
        return    
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(destination)
    print(f"Saved to: {destination}")

# helper for stage studiogan source
def _stage_studiogan_source(source_url: str, target_dir: Path, force: bool) -> None:
    required_path = target_dir / "src" / "models" / "model.py"

    if not force and required_path.exists():
        print(f"StudioGAN source already staged: {target_dir}")
        return

    if target_dir.exists() and not force:
        raise FileExistsError(
            f"{target_dir} exists but is incomplete. Re-run with --force to replace it."
        )

    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        archive_path = tmp_dir / "studiogan.zip"
        _download_file(source_url, archive_path)

        with zipfile.ZipFile(archive_path, mode="r") as archive:
            archive.extractall(tmp_dir)

        extracted_roots = [p for p in tmp_dir.iterdir() if p.is_dir()]
        source_root = extracted_roots[0]

        shutil.move(str(source_root), str(target_dir))
        print(f"Staged studioGAN source at: {target_dir}")


# entry point when running this script
def main():
    parser = argparse.ArgumentParser(description="Stage StudioGAN checkpoints for /CIFAR10.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/StudioGAN/CIFAR10")
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        default=DEFAULT_CHECKPOINT_URL
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="studioGAN_generator.pkl"
    )
    parser.add_argument(
        "--stage-studiogan-source",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        "--studioGAN-source-url",
        type=str,
        default=DEFAULT_STUDIOGAN_REPO
    )
    parser.add_argument(
        "--studioGAN-source-dirname",
        type=str,
        default="studioGAN_src"
    )
    parser.add_argument(
        "--force",
        action="store_true"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.checkpoint_name

    if checkpoint_path.exists() and not args.force:
        print(f"Checkpoint already exists, skipping download: {checkpoint_path}")
    else:
        if args.checkpoint_url is not None:
            _download_file(args.checkpoint_url, checkpoint_path)
            print(f"Staged checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint downloaded. Please add model.pth manually.")

    if args.stage_studiogan_source:
        source_dir = output_dir / args.studioGAN_source_dirname
        _stage_studiogan_source(
            source_url=args.studioGAN_source_url,
            target_dir=source_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()

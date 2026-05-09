# example: py Scripts/download_pretrained_stylegan_celeba.py --output-dir checkpoints/StyleGAN/CelebA
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_CHECKPOINT_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/"
    "files/stylegan2-celebahq-256x256.pkl"
)
DEFAULT_STYLEGAN2_ADA_SOURCE_URL = (
    "https://github.com/NVlabs/stylegan2-ada-pytorch/archive/refs/heads/main.zip"
)


# download file
def _download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(destination)


# helper for stage stylegan2 ada source
def _stage_stylegan2_ada_source(source_url: str, target_dir: Path, force: bool) -> None:
    required_paths = (target_dir / "legacy.py", target_dir / "training", target_dir / "torch_utils")
    if not force and all(path.exists() for path in required_paths):
        print(f"StyleGAN2-ADA source already staged: {target_dir}")
        return

    if target_dir.exists() and not force:
        raise FileExistsError(
            f"{target_dir} exists but is incomplete. Re-run with --force to replace it."
        )

    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        archive_path = tmp_dir / "stylegan2_ada_source.zip"
        _download_file(source_url, archive_path)

        with zipfile.ZipFile(archive_path, mode="r") as archive:
            archive.extractall(tmp_dir)

        extracted_roots = [path for path in tmp_dir.iterdir() if path.is_dir()]
        source_root = next((path for path in extracted_roots if (path / "legacy.py").exists()), None)
        if source_root is None:
            raise RuntimeError("Could not locate stylegan2-ada source root after extraction.")

        shutil.move(str(source_root), str(target_dir))
        print(f"Staged StyleGAN2-ADA source at: {target_dir}")


# entry point when running this script
def main():
    parser = argparse.ArgumentParser(description="Stage StyleGAN/StyleGAN2 checkpoints for CelebA.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/StyleGAN/CelebA")
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        default=DEFAULT_CHECKPOINT_URL
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="stylegan2_generator.pkl"
    )
    parser.add_argument(
        "--stage-stylegan2-ada-source",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        "--stylegan2-ada-source-url",
        type=str,
        default=DEFAULT_STYLEGAN2_ADA_SOURCE_URL
    )
    parser.add_argument(
        "--stylegan2-ada-source-dirname",
        type=str,
        default="stylegan2_ada_src"
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
        _download_file(args.checkpoint_url, checkpoint_path)
        print(f"Staged checkpoint: {checkpoint_path}")

    if args.stage_stylegan2_ada_source:
        source_dir = output_dir / args.stylegan2_ada_source_dirname
        _stage_stylegan2_ada_source(
            source_url=args.stylegan2_ada_source_url,
            target_dir=source_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()

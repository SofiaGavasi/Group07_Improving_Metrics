from __future__ import annotations

from pathlib import Path


# load repo and outputs root
def get_repo_and_outputs_root() -> tuple[Path, Path]:
    repo_root = Path.cwd().resolve().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd().resolve()
    outputs_root = repo_root / "outputs"
    return repo_root, outputs_root


# helper for find report files
def find_report_files(outputs_root: Path, batch_name: str, report_suffix: str) -> list[Path]:
    return sorted(outputs_root.glob(f"{batch_name}_*_{report_suffix}.json"))

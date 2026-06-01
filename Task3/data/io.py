"""
this file holds small path helpers for task3 batch analysis

"""

from pathlib import Path


def get_repo_and_outputs_root():
    cwd = Path.cwd().resolve()
    if cwd.name.lower() in {"notebooks", "task3"}:
        repo_root = cwd.parent
    else:
        repo_root = cwd

    outputs_root = repo_root / "outputs"
    return repo_root, outputs_root


def find_report_files(outputs_root, batch_name, report_suffix):
    outputs_root = Path(outputs_root)
    pattern = f"{batch_name}_*_{report_suffix}.json"
    return sorted(outputs_root.glob(pattern))

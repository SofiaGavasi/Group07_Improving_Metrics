from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DATASET_HANDLE = "nih-chest-xrays/data"
INDEX_FILE_NAME = "chestxray14_index.csv"
FINDING_CLASSES_FILE_NAME = "finding_classes.txt"

DEFAULT_FINDING_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

METADATA_FILE_CANDIDATES = [
    "Data_Entry_2017_v2020.csv",
    "Data_Entry_2017.csv",
    "metadata/Data_Entry_2017_v2020.csv",
    "metadata/Data_Entry_2017.csv",
]
TRAIN_SPLIT_FILE_CANDIDATES = [
    "train_val_list.txt",
    "metadata/train_val_list.txt",
]
TEST_SPLIT_FILE_CANDIDATES = [
    "test_list.txt",
    "metadata/test_list.txt",
]

#normalizing root path so it looks for data/ChestXray14 or data/chestxray14 for example
def _resolve_dataset_root(data_root):
    root = Path(data_root)
    if root.name.lower() == "chestxray14":
        return root

    camel_case = root / "ChestXray14"
    lower_case = root / "chestxray14"
    if camel_case.exists() or not lower_case.exists():
        return camel_case
    return lower_case

#first existing path found
def _find_first_existing_file(root, candidates):
    for relative in candidates:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None

#  find image files and map filename -> absolute path
def _collect_image_paths(root):
    if not root.exists():
        return {}

    image_paths: dict[str, str] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for image_path in root.rglob(pattern):
            image_paths.setdefault(image_path.name, str(image_path.resolve()))
    return image_paths

# read split list from file, return set of image indices
def _read_split_list(path):
    if path is None or not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def _load_metadata_with_kagglehub():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for ChestX-ray14 download. "
            "Install with: pip install kagglehub[pandas-datasets]"
        ) from exc

    for file_path in METADATA_FILE_CANDIDATES:
        try:
            frame = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                DATASET_HANDLE,
                file_path,
            )
            if isinstance(frame, pd.DataFrame):
                return frame, file_path
        except Exception:
            continue

# case-insensitive column matcher for metadata schema variations
def _resolve_column_name(frame, expected_columns):
    normalized = {column.strip().lower(): column for column in frame.columns}
    for expected in expected_columns:
        candidate = normalized.get(expected.lower())
        if candidate is not None:
            return candidate
    return None

# fallback split strategy (train/val/test) by patient ID to reduce leakage
def _assign_patient_split(
    frame: pd.DataFrame,
    seed: int,
    val_ratio: float,
    test_ratio: float,
):
    result = frame.copy()
    unique_patients = np.array(sorted(result["patient_id"].unique()))
    if unique_patients.size == 0:
        result["split"] = "train"
        return result

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(unique_patients)

    test_count = int(round(unique_patients.size * test_ratio))
    val_count = int(round(unique_patients.size * val_ratio))

    if test_ratio > 0 and test_count == 0 and unique_patients.size > 2:
        test_count = 1
    if val_ratio > 0 and val_count == 0 and unique_patients.size > 2:
        val_count = 1

    test_patients = set(unique_patients[:test_count].tolist())
    remaining = unique_patients[test_count:]
    val_patients = set(remaining[:val_count].tolist())

    result["split"] = "train"
    result.loc[result["patient_id"].isin(test_patients), "split"] = "test"
    result.loc[result["patient_id"].isin(val_patients), "split"] = "val"
    return result

# carves out val from train with seed
def _inject_validation_split(frame, seed, val_ratio):
    result = frame.copy()
    train_only = result[result["split"] == "train"]
    train_patients = np.array(sorted(train_only["patient_id"].unique()))
    if train_patients.size == 0:
        return result

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(train_patients)

    val_count = int(round(train_patients.size * val_ratio))
    if val_ratio > 0 and val_count == 0 and train_patients.size > 2:
        val_count = 1

    val_patients = set(train_patients[:val_count].tolist())
    result.loc[
        (result["split"] == "train") & (result["patient_id"].isin(val_patients)),
        "split",
    ] = "val"
    return result

#parse pipe-separated finding labels and builds sorted class list
def _extract_finding_classes(labels):
    findings: set[str] = set()
    for raw in labels.fillna("No Finding").astype(str):
        for token in raw.split("|"):
            label = token.strip()
            if not label or label == "No Finding":
                continue
            findings.add(label)
    if findings:
        return sorted(findings)
    return list(DEFAULT_FINDING_CLASSES)




# MAIN SETUP/INDEX BUILDER
# if dataset csv already there, we reuse it (unless download= true)
def prepare_chestxray14_dataset(
    data_root,
    download: bool = False,
    seed: int = 10,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    dataset_root = _resolve_dataset_root(data_root)
    metadata_root = dataset_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    index_path = metadata_root / INDEX_FILE_NAME

    if index_path.exists():
        return index_path

    cache_path_file = metadata_root / "kaggle_cache_path.txt"
    kaggle_cache_root: Optional[Path] = None

    if download:
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for ChestX-ray14 download. "
                "Install with: pip install kagglehub[pandas-datasets]"
            ) from exc
        kaggle_cache_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
        cache_path_file.write_text(str(kaggle_cache_root), encoding="utf-8")
    elif cache_path_file.exists():
        cached = Path(cache_path_file.read_text(encoding="utf-8").strip())
        if cached.exists():
            kaggle_cache_root = cached

    metadata_frame: Optional[pd.DataFrame] = None
    metadata_file = _find_first_existing_file(dataset_root, METADATA_FILE_CANDIDATES)

    if metadata_file is None and kaggle_cache_root is not None:
        metadata_file = _find_first_existing_file(kaggle_cache_root, METADATA_FILE_CANDIDATES)

    if metadata_file is not None:
        metadata_frame = pd.read_csv(metadata_file)
    elif download:
        metadata_frame, source_name = _load_metadata_with_kagglehub()
        metadata_file = metadata_root / Path(source_name).name
        metadata_frame.to_csv(metadata_file, index=False)
    else:
        raise FileNotFoundError(
            f"ChestX-ray14 metadata not found under '{dataset_root}'. "
            "Run Scripts/download_preprocess_chestxray14.py first or call with download=True."
        )

    image_paths = _collect_image_paths(dataset_root)
    if kaggle_cache_root is not None:
        image_paths.update(_collect_image_paths(kaggle_cache_root))

    image_col = _resolve_column_name(metadata_frame, ["Image Index", "image_index"])
    finding_col = _resolve_column_name(metadata_frame, ["Finding Labels", "finding_labels"])
    patient_col = _resolve_column_name(metadata_frame, ["Patient ID", "patient_id"])

    if image_col is None or finding_col is None:
        raise ValueError(
            "ChestX-ray14 metadata must contain image and finding-label columns."
        )

    index_frame = pd.DataFrame(
        {
            "image_index": metadata_frame[image_col].astype(str).str.strip(),
            "finding_labels": metadata_frame[finding_col].fillna("No Finding").astype(str),
        }
    )

    if patient_col is None:
        index_frame["patient_id"] = np.arange(index_frame.shape[0], dtype=np.int64)
    else:
        numeric_patient_ids = pd.to_numeric(
            metadata_frame[patient_col],
            errors="coerce",
        ).fillna(-1)
        index_frame["patient_id"] = numeric_patient_ids.astype(np.int64)

    index_frame["image_path"] = index_frame["image_index"].map(image_paths)
    index_frame = index_frame.dropna(subset=["image_path"]).reset_index(drop=True)
    if index_frame.empty:
        raise FileNotFoundError(
            "No ChestX-ray14 images were found. "
            "Check dataset download or ensure image files exist under the configured root."
        )

    split_search_roots = [dataset_root]
    if kaggle_cache_root is not None:
        split_search_roots.append(kaggle_cache_root)

    train_split_file = None
    test_split_file = None
    for candidate_root in split_search_roots:
        train_split_file = _find_first_existing_file(candidate_root, TRAIN_SPLIT_FILE_CANDIDATES)
        test_split_file = _find_first_existing_file(candidate_root, TEST_SPLIT_FILE_CANDIDATES)
        if train_split_file is not None and test_split_file is not None:
            break

    train_set = _read_split_list(train_split_file)
    test_set = _read_split_list(test_split_file)
    if train_set and test_set:
        index_frame["split"] = "train"
        index_frame.loc[index_frame["image_index"].isin(test_set), "split"] = "test"
        in_known_splits = index_frame["image_index"].isin(train_set.union(test_set))
        index_frame.loc[~in_known_splits, "split"] = "train"
        index_frame = _inject_validation_split(index_frame, seed=seed, val_ratio=val_ratio)
    else:
        index_frame = _assign_patient_split(
            index_frame,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    finding_classes = _extract_finding_classes(index_frame["finding_labels"])
    classes_path = metadata_root / FINDING_CLASSES_FILE_NAME
    classes_path.write_text("\n".join(finding_classes), encoding="utf-8")

    index_frame = index_frame[
        ["image_index", "image_path", "finding_labels", "patient_id", "split"]
    ]
    index_frame.to_csv(index_path, index=False)

    return index_path

# reads saved class names, falls back to defaults (i took them from original paper linked in kaggle)
def _load_finding_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        return list(DEFAULT_FINDING_CLASSES)
    lines = classes_file.read_text(encoding="utf-8").splitlines()
    classes = [line.strip() for line in lines if line.strip()]
    if classes:
        return classes
    return list(DEFAULT_FINDING_CLASSES)



# DATASET CLASS
class ChestXray14Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        download: bool = False,
    ):
        self.root = _resolve_dataset_root(root)
        self.transform = transform

        normalized_split = split.lower()
        split_aliases = {"valid": "val", "validation": "val"}
        normalized_split = split_aliases.get(normalized_split, normalized_split)
        if normalized_split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of: train, val, test, all")
        self.split = normalized_split

        index_path = prepare_chestxray14_dataset(self.root, download=download)
        index_frame = pd.read_csv(index_path)
        if self.split != "all":
            index_frame = index_frame[index_frame["split"] == self.split].reset_index(drop=True)

        if index_frame.empty:
            raise ValueError(
                f"No ChestX-ray14 samples found for split '{self.split}'. "
                "Run setup again or choose another split."
            )

        classes_file = index_path.parent / FINDING_CLASSES_FILE_NAME
        self.finding_classes = _load_finding_classes(classes_file)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.finding_classes)}

        self.image_paths = index_frame["image_path"].tolist()
        self.finding_labels = index_frame["finding_labels"].astype(str).tolist()

    def __len__(self): #number of samples
        return len(self.image_paths)

    def __getitem__(self, idx: int): 
        # opens image as RGV, 
        # applies transform
        # builds multi-hot target tensor of size num_findings
        # for No Finding, target stays all zeros

        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.zeros(len(self.finding_classes), dtype=torch.float32)
        raw_labels = self.finding_labels[idx]
        if raw_labels and raw_labels != "No Finding":
            for label in raw_labels.split("|"):
                normalized = label.strip()
                label_index = self.class_to_idx.get(normalized)
                if label_index is not None:
                    target[label_index] = 1.0

        return image, target

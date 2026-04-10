from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset





@dataclass
class DatasetSubsetConfig:
    # fraction/max_samples define how much data to keep after class filtering
    fraction: Optional[float] = None
    max_samples: Optional[int] = None
    # seed used by random and class-balanced sampling
    seed: int = 10
    strategy: str = "random"  # random or class_balanced
    #class filters accept names or numeric indices
    include_classes: list[str] = field(default_factory=list)
    drop_classes: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return any(
            [
                self.fraction is not None,
                self.max_samples is not None,
                bool(self.include_classes),
                bool(self.drop_classes),
                self.strategy != "random",
            ]
        )







class DatasetSubset(Dataset):
    # wrapper that exposes only the selected sample indices
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __getattr__(self, item):
        # forward unknown attributes so original dataset metadata is still accessible
        return getattr(self.dataset, item)







def parse_class_identifiers(raw):
    # normalized tokens for classes
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _target_size(total, config):
    #     fraction and max_samples caps in sequence
    target = total
    if config.fraction is not None:
        if config.fraction <= 0 or config.fraction > 1:
            raise ValueError("--subset-fraction must be in (0, 1].")
        target = min(target, max(1, int(round(total * config.fraction))))
    if config.max_samples is not None:
        if config.max_samples <= 0:
            raise ValueError("--subset-max-samples must be > 0.")
        target = min(target, config.max_samples)
    return target


def _dataset_class_names(dataset: Dataset):
    # trying common class-name fields across torchvision/custom datasets
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return [str(x) for x in list(dataset.classes)]
    if hasattr(dataset, "attr_names"):
        names = list(dataset.attr_names)
        if names:
            return [str(x) for x in names]
    if hasattr(dataset, "finding_classes"):
        names = list(dataset.finding_classes)
        if names:
            return [str(x) for x in names]
    return []


def _extract_single_labels(dataset: Dataset):
    #  single-label integer targets when available (so for minst and cifar)
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
        if labels.ndim == 1:
            return labels.astype(np.int64)
        if labels.ndim == 2 and labels.shape[1] == 1:
            return labels.reshape(-1).astype(np.int64)

    if hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels)
        if labels.ndim == 1:
            return labels.astype(np.int64)
    return None


def _extract_multilabel_matrix(dataset: Dataset):
    #  multi-label datasets ( CelebA attrs, ChestXray14 findings)
    if hasattr(dataset, "attr"):
        attr = dataset.attr
        if isinstance(attr, torch.Tensor):
            matrix = attr.detach().cpu().numpy()
        else:
            matrix = np.asarray(attr)
        if matrix.ndim == 2:
            return (matrix > 0).astype(np.uint8)

    if hasattr(dataset, "finding_labels") and hasattr(dataset, "class_to_idx"):
        rows = len(dataset.finding_labels)
        cols = len(dataset.class_to_idx)
        matrix = np.zeros((rows, cols), dtype=np.uint8)
        for row_idx, raw in enumerate(dataset.finding_labels):
            text = str(raw)
            if not text or text == "No Finding":
                continue
            for token in text.split("|"):
                label = token.strip()
                class_idx = dataset.class_to_idx.get(label)
                if class_idx is not None:
                    matrix[row_idx, class_idx] = 1
        return matrix
    return None


def _resolve_class_indices(
    class_identifiers: list[str],
    class_names: list[str],
    upper_bound: int,
):
    # resolving class tokens to numeric class indices
    if not class_identifiers:
        return set()

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    resolved: set[int] = set()

    for token in class_identifiers:
        if token in class_to_idx:
            resolved.add(class_to_idx[token])
            continue
        try:
            numeric = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Unknown class identifier '{token}'. Use class name or numeric index."
            ) from exc
        if numeric < 0 or numeric >= upper_bound:
            raise ValueError(f"Class index out of range: {numeric}")
        resolved.add(numeric)

    return resolved


###############################
#keep or drop samples by label for single-label datasets and multi-label ones

def _filter_indices_single(
    indices: np.ndarray,
    labels: np.ndarray,
    include_classes: set[int],
    drop_classes: set[int],):
    
    mask = np.ones(indices.shape[0], dtype=bool)
    if include_classes:
        include_mask = np.isin(labels[indices], list(include_classes))
        mask &= include_mask
    if drop_classes:
        drop_mask = np.isin(labels[indices], list(drop_classes))
        mask &= ~drop_mask
    return indices[mask]
def _filter_indices_multi(
    indices: np.ndarray,
    matrix: np.ndarray,
    include_classes: set[int],
    drop_classes: set[int]) :
    mask = np.ones(indices.shape[0], dtype=bool)
    if include_classes:
        include_cols = sorted(include_classes)
        include_hits = matrix[indices][:, include_cols].sum(axis=1) > 0
        mask &= include_hits
    if drop_classes:
        drop_cols = sorted(drop_classes)
        drop_hits = matrix[indices][:, drop_cols].sum(axis=1) > 0
        mask &= ~drop_hits
    return indices[mask]


#########################
#choosing random or balanced selection of samples

def _choose_random(indices: np.ndarray, size: int, seed: int):
    if size >= indices.shape[0]:
        return indices.tolist()
    rng = np.random.default_rng(seed=seed)
    chosen = rng.choice(indices, size=size, replace=False)
    chosen.sort()
    return chosen.tolist()

def _choose_class_balanced(
    indices: np.ndarray,
    labels: np.ndarray,
    size: int,
    seed: int,
):
    if size >= indices.shape[0]:
        return indices.tolist()

    rng = np.random.default_rng(seed=seed)
    classes = sorted(set(labels[indices].tolist()))
    buckets: dict[int, list[int]] = {}
    for class_id in classes:
        class_indices = indices[labels[indices] == class_id].tolist()
        rng.shuffle(class_indices)
        buckets[class_id] = class_indices

    if not classes:
        return []

    base = size // len(classes)
    remainder = size % len(classes)
    selected: list[int] = []
    remaining: dict[int, list[int]] = {}

    for class_id in classes:
        take = min(base, len(buckets[class_id]))
        selected.extend(buckets[class_id][:take])
        remaining[class_id] = buckets[class_id][take:]

    class_order = classes.copy()
    rng.shuffle(class_order)
    for class_id in class_order:
        if remainder == 0:
            break
        if remaining[class_id]:
            selected.append(remaining[class_id].pop())
            remainder -= 1

    if len(selected) < size:
        pool: list[int] = []
        for class_id in class_order:
            pool.extend(remaining[class_id])
        rng.shuffle(pool)
        selected.extend(pool[: size - len(selected)])

    selected = selected[:size]
    selected.sort()
    return selected





# Main entrypoint: class filtering first, then size/strategy sampling
def apply_dataset_subset(
    dataset: Dataset,
    config: Optional[DatasetSubsetConfig],
    dataset_name: str = "",
):
    
    if config is None or not config.is_active():
        return dataset

    indices = np.arange(len(dataset), dtype=np.int64)
    labels = _extract_single_labels(dataset)
    multi_matrix = None if labels is not None else _extract_multilabel_matrix(dataset)

    class_names = _dataset_class_names(dataset)
    num_classes = len(class_names)
    if num_classes == 0:
        if labels is not None:
            num_classes = int(labels.max()) + 1 if labels.size else 0
        elif multi_matrix is not None:
            num_classes = int(multi_matrix.shape[1])

    include_classes = _resolve_class_indices(config.include_classes, class_names, num_classes)
    drop_classes = _resolve_class_indices(config.drop_classes, class_names, num_classes)

    if labels is not None:
        filtered = _filter_indices_single(indices, labels, include_classes, drop_classes)
    elif multi_matrix is not None:
        filtered = _filter_indices_multi(indices, multi_matrix, include_classes, drop_classes)
    else:
        if include_classes or drop_classes:
            raise ValueError(
                f"Class-based filtering is not available for dataset '{dataset_name}'."
            )
        filtered = indices

    if filtered.size == 0:
        raise ValueError(f"Subset filtering produced zero samples for dataset '{dataset_name}'.")

    target = _target_size(int(filtered.size), config)
    if config.strategy == "random":
        chosen = _choose_random(filtered, target, config.seed)
    elif config.strategy == "class_balanced":
        if labels is None:
            raise ValueError(
                "class_balanced strategy is only supported for single-label datasets."
            )
        chosen = _choose_class_balanced(filtered, labels, target, config.seed)
    else:
        raise ValueError("Unknown subset strategy. Use 'random' or 'class_balanced'.")

    if len(chosen) == len(dataset):
        return dataset
    return DatasetSubset(dataset, chosen)

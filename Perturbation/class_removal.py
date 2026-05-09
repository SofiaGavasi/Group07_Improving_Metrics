from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parse_token_list(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _feature_matrix(samples: torch.Tensor) -> np.ndarray:
    return (
        samples.detach()
        .cpu()
        .float()
        .reshape(samples.shape[0], -1)
        .numpy()
        .astype(np.float64, copy=False)
    )


def _resolve_named_targets(
    raw_targets: list[str],
    class_names: list[str],
    upper_bound: int,
) -> list[int]:
    if not raw_targets:
        raise ValueError("Class-removal perturbation needs at least one target class/cluster id.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_to_idx_lower = {name.lower(): idx for idx, name in enumerate(class_names)}
    resolved: list[int] = []
    for token in raw_targets:
        if token in class_to_idx:
            resolved.append(class_to_idx[token])
            continue
        lower = token.lower()
        if lower in class_to_idx_lower:
            resolved.append(class_to_idx_lower[lower])
            continue
        try:
            numeric = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Unknown class-removal target '{token}'. use a class name or numeric index."
            ) from exc
        if numeric < 0 or numeric >= upper_bound:
            raise ValueError(
                f"Class-removal target index out of range: {numeric} (valid 0..{upper_bound - 1})."
            )
        resolved.append(numeric)
    return sorted(set(resolved))


def _prepare_reference_targets(reference_targets: torch.Tensor) -> np.ndarray:
    targets_np = reference_targets.detach().cpu().numpy()
    if targets_np.ndim == 0:
        targets_np = targets_np.reshape(1)
    if targets_np.ndim == 2 and targets_np.shape[1] == 1:
        targets_np = targets_np.reshape(-1)
    return targets_np


def _class_names_for_label_count(
    class_names: list[str] | None,
    label_count: int,
) -> list[str]:
    names = [str(name) for name in list(class_names or [])]
    if len(names) > int(label_count):
        return names[: int(label_count)]
    if len(names) < int(label_count):
        start = len(names)
        names.extend([f"class_{idx}" for idx in range(start, int(label_count))])
    return names


def _reference_targets_to_binary_matrix(
    reference_targets: torch.Tensor,
    class_names: list[str] | None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    prepared = _prepare_reference_targets(reference_targets)
    if prepared.ndim == 1:
        labels = prepared.astype(np.int64, copy=False)
        inferred = int(labels.max()) + 1 if labels.size else 1
        label_count = max(inferred, len(class_names or []), 1)
        matrix = np.zeros((labels.shape[0], label_count), dtype=np.uint8)
        if labels.size:
            matrix[np.arange(labels.shape[0]), labels] = 1
    elif prepared.ndim == 2:
        matrix = (prepared > 0).astype(np.uint8)
        label_count = int(matrix.shape[1])
    else:
        raise ValueError("Unsupported reference target shape for class-removal.")

    names = _class_names_for_label_count(class_names=class_names, label_count=label_count)
    return matrix, names, prepared


def _label_cooccurrence_features(label_matrix: np.ndarray) -> np.ndarray:
    # labels are the points we cluster; each point is its co-occurrence profile across all labels.
    co_counts = label_matrix.T @ label_matrix
    supports = label_matrix.sum(axis=0).astype(np.float64)
    supports[supports <= 0] = 1.0
    features = co_counts.astype(np.float64) / supports[:, None]
    np.fill_diagonal(features, 1.0)
    return features


def _apply_class_removal_kmeans(
    fake_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    reference_targets: torch.Tensor,
    reference_class_names: list[str] | None,
    targets_raw: str,
    kmeans_k: int,
    kmeans_cache_path: str,
    kmeans_recreate: bool,
    out_dir: str,
    dataset_name: str,
    seed: int,
    label_threshold: float,
    min_kept: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if kmeans_k < 2:
        raise ValueError("--perturb-class-removal-kmeans-k must be >= 2.")

    label_matrix, class_names, prepared_targets = _reference_targets_to_binary_matrix(
        reference_targets=reference_targets,
        class_names=reference_class_names,
    )
    label_count = int(label_matrix.shape[1])
    if int(kmeans_k) > label_count:
        raise ValueError(
            f"--perturb-class-removal-kmeans-k ({int(kmeans_k)}) cannot exceed label count ({label_count})."
        )
    label_features = _label_cooccurrence_features(label_matrix)

    def _default_cache_path() -> Path:
        # Use a stable cache root by default. Per-run output folders may be transient
        # and are more likely to race/fail on synced filesystems.
        base = Path("outputs")
        safe_dataset = dataset_name.strip() or "dataset"
        return (
            base
            / "perturbation_cache"
            / f"class_removal_label_kmeans_{safe_dataset}_labels{label_count}_k{int(kmeans_k)}_seed{int(seed)}.npz"
        )

    def _resolve_cache_path() -> Path:
        raw = str(kmeans_cache_path).strip()
        return Path(raw) if raw else _default_cache_path()

    cache_path = _resolve_cache_path()
    cache_hit = False
    cache_status = "not_checked"
    label_cluster_ids: np.ndarray | None = None

    if not bool(kmeans_recreate):
        if cache_path.exists():
            try:
                saved = np.load(cache_path)
                cached_cluster_ids = np.asarray(saved["label_cluster_ids"], dtype=np.int64)
                cached_k = int(saved["kmeans_k"])
                cached_label_count = int(saved["label_count"])
                if (
                    cached_cluster_ids.ndim == 1
                    and cached_k == int(kmeans_k)
                    and cached_label_count == label_count
                    and cached_cluster_ids.shape[0] == label_count
                ):
                    label_cluster_ids = cached_cluster_ids
                    cache_hit = True
                    cache_status = "loaded_existing"
                else:
                    cache_status = "mismatch_refit"
            except Exception:
                cache_status = "load_failed_refit"
        else:
            cache_status = "missing_refit"
    else:
        cache_status = "forced_refit"

    if label_cluster_ids is None:
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required to fit label-kmeans class-removal clusters. "
                "Install scikit-learn or reuse an existing cache file."
            ) from exc

        # clustering labels using co-occurrence features.
        kmeans = KMeans(n_clusters=int(kmeans_k), random_state=int(seed), n_init=10)
        label_cluster_ids = kmeans.fit_predict(label_features).astype(np.int64, copy=False)
        cache_payload = {
            "label_cluster_ids": label_cluster_ids,
            "kmeans_k": np.int64(kmeans_k),
            "label_count": np.int64(label_count),
            "dataset_name": np.asarray(dataset_name),
            "seed": np.int64(seed),
            "class_names": np.asarray(class_names),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez(cache_path, **cache_payload)
        except OSError:
            # Fallback to global cache root if custom/per-run path is not writable.
            fallback_path = (
                Path("outputs")
                / "perturbation_cache"
                / f"class_removal_label_kmeans_{dataset_name.strip() or 'dataset'}_labels{label_count}_k{int(kmeans_k)}_seed{int(seed)}.npz"
            )
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fallback_path, **cache_payload)
            cache_path = fallback_path
        if cache_status in {"missing_refit", "forced_refit", "mismatch_refit", "load_failed_refit"}:
            cache_status = f"{cache_status}_saved"
        else:
            cache_status = "refit_saved"

    target_tokens = _parse_token_list(targets_raw)
    target_cluster_ids = _resolve_named_targets(
        raw_targets=target_tokens,
        class_names=[str(i) for i in range(int(kmeans_k))],
        upper_bound=int(kmeans_k),
    )
    drop_label_indices = np.nonzero(np.isin(label_cluster_ids, target_cluster_ids))[0].tolist()
    if not drop_label_indices:
        raise ValueError("Selected kmeans label-clusters contain no labels to drop.")

    drop_labels_raw = ",".join(str(int(idx)) for idx in drop_label_indices)
    if prepared_targets.ndim == 1:
        filtered, drop_details = _apply_class_removal_single_label(
            fake_samples=fake_samples,
            reference_samples=reference_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=drop_labels_raw,
            min_kept=int(min_kept),
        )
    else:
        filtered, drop_details = _apply_class_removal_multi_label(
            fake_samples=fake_samples,
            reference_samples=reference_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=drop_labels_raw,
            label_threshold=float(label_threshold),
            min_kept=int(min_kept),
        )

    label_cluster_map = {
        class_names[idx]: int(cluster_id)
        for idx, cluster_id in enumerate(label_cluster_ids.tolist())
    }
    details = {
        "strategy": "kmeans",
        "kmeans_basis": "label_cooccurrence",
        "kmeans_k": int(kmeans_k),
        "kmeans_cache_path": str(cache_path),
        "kmeans_cache_hit": bool(cache_hit),
        "kmeans_cache_status": cache_status,
        "drop_clusters": target_cluster_ids,
        "drop_label_indices": [int(idx) for idx in drop_label_indices],
        "drop_label_names": [class_names[idx] for idx in drop_label_indices],
        "label_cluster_assignments": label_cluster_map,
        "drop_result": drop_details,
        "removed_count": int(drop_details["removed_count"]),
        "kept_count": int(drop_details["kept_count"]),
    }
    return filtered, details


def _apply_class_removal_single_label(
    fake_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    reference_targets: torch.Tensor,
    class_names: list[str],
    targets_raw: str,
    min_kept: int,
):
    ref_features = _feature_matrix(reference_samples)
    fake_features = _feature_matrix(fake_samples)
    labels = _prepare_reference_targets(reference_targets).astype(np.int64, copy=False)
    unique_labels = sorted(set(labels.tolist()))
    if not unique_labels:
        raise ValueError("Label-based class-removal could not find any reference classes.")

    centroids: list[np.ndarray] = []
    centroid_ids: list[int] = []
    for class_id in unique_labels:
        class_mask = labels == class_id
        if not np.any(class_mask):
            continue
        centroids.append(ref_features[class_mask].mean(axis=0))
        centroid_ids.append(int(class_id))

    if not centroids:
        raise ValueError("Label-based class-removal failed to build class centroids.")

    centroid_matrix = np.stack(centroids, axis=0)
    distances = ((fake_features[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    predicted_labels = np.asarray([centroid_ids[idx] for idx in nearest], dtype=np.int64)

    upper_bound = max(max(centroid_ids) + 1, len(class_names))
    resolved_drop = _resolve_named_targets(
        raw_targets=_parse_token_list(targets_raw),
        class_names=class_names,
        upper_bound=upper_bound,
    )
    keep_mask = ~np.isin(predicted_labels, resolved_drop)
    keep_indices = np.nonzero(keep_mask)[0]
    if int(keep_indices.size) < int(min_kept):
        raise ValueError(
            "class-removal perturbation kept too few fake samples after dropping label classes. "
            f"kept={int(keep_indices.size)} min_kept={int(min_kept)}"
        )

    filtered = fake_samples[torch.as_tensor(keep_indices, dtype=torch.long)]
    unique_ids, counts = np.unique(predicted_labels, return_counts=True)
    pred_hist = {str(int(idx)): int(count) for idx, count in zip(unique_ids, counts)}
    details = {
        "strategy": "label",
        "label_mode": "single_label",
        "drop_classes": resolved_drop,
        "drop_class_names": [class_names[idx] if idx < len(class_names) else str(idx) for idx in resolved_drop],
        "predicted_label_histogram_fake": pred_hist,
        "removed_count": int(fake_samples.shape[0]) - int(filtered.shape[0]),
        "kept_count": int(filtered.shape[0]),
    }
    return filtered, details


def _apply_class_removal_multi_label(
    fake_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    reference_targets: torch.Tensor,
    class_names: list[str],
    targets_raw: str,
    label_threshold: float,
    min_kept: int,
) :
    ref_features = _feature_matrix(reference_samples)
    fake_features = _feature_matrix(fake_samples)
    target_matrix = (_prepare_reference_targets(reference_targets) > 0).astype(np.uint8)
    if target_matrix.ndim != 2:
        raise ValueError("Expected multi-label matrix for class-removal on this dataset.")

    label_count = int(target_matrix.shape[1])
    resolved_drop = _resolve_named_targets(
        raw_targets=_parse_token_list(targets_raw),
        class_names=class_names,
        upper_bound=label_count,
    )

    drop_mask = np.zeros(fake_features.shape[0], dtype=bool)
    per_label_predicted_positives: dict[str, int] = {}
    for label_idx in resolved_drop:
        positive = target_matrix[:, label_idx] == 1
        negative = target_matrix[:, label_idx] == 0
        if positive.sum() == 0 or negative.sum() == 0:
            per_label_predicted_positives[str(label_idx)] = 0
            continue

        positive_centroid = ref_features[positive].mean(axis=0)
        negative_centroid = ref_features[negative].mean(axis=0)
        dist_pos = ((fake_features - positive_centroid[None, :]) ** 2).sum(axis=1)
        dist_neg = ((fake_features - negative_centroid[None, :]) ** 2).sum(axis=1)
        # if closer to positive centroid by threshold margin, we call it a positive prediction.
        predicted_positive = (dist_neg - dist_pos) > float(label_threshold)
        per_label_predicted_positives[str(label_idx)] = int(predicted_positive.sum())
        drop_mask |= predicted_positive

    keep_indices = np.nonzero(~drop_mask)[0]
    if int(keep_indices.size) < int(min_kept):
        raise ValueError(
            "class-removal perturbation kept too few fake samples after dropping multi-label classes. "
            f"kept={int(keep_indices.size)} min_kept={int(min_kept)}"
        )

    filtered = fake_samples[torch.as_tensor(keep_indices, dtype=torch.long)]
    details = {
        "strategy": "label",
        "label_mode": "multi_label",
        "drop_classes": resolved_drop,
        "drop_class_names": [class_names[idx] if idx < len(class_names) else str(idx) for idx in resolved_drop],
        "label_threshold": float(label_threshold),
        "predicted_positive_counts": per_label_predicted_positives,
        "removed_count": int(fake_samples.shape[0]) - int(filtered.shape[0]),
        "kept_count": int(filtered.shape[0]),
    }
    return filtered, details


def apply_class_removal(
    fake_samples: torch.Tensor,
    config: dict[str, Any],
    real_samples: torch.Tensor | None,
    reference_targets: torch.Tensor | None,
    reference_class_names: list[str] | None,
    dataset_name: str,
) :
    if config["strategy"] == "kmeans":
        if real_samples is None or reference_targets is None:
            raise ValueError("kmeans class-removal needs real reference samples and labels.")
        return _apply_class_removal_kmeans(
            fake_samples=fake_samples,
            reference_samples=real_samples,
            reference_targets=reference_targets,
            reference_class_names=reference_class_names,
            targets_raw=str(config["targets_raw"]),
            kmeans_k=int(config["kmeans_k"]),
            kmeans_cache_path=str(config.get("kmeans_cache_path", "")),
            kmeans_recreate=bool(config.get("kmeans_recreate", False)),
            out_dir=str(config.get("out_dir", "")),
            dataset_name=dataset_name,
            seed=int(config["seed"]),
            label_threshold=float(config["label_threshold"]),
            min_kept=int(config["min_kept"]),
        )

    if real_samples is None or reference_targets is None:
        raise ValueError("Label-based class-removal needs real reference samples and labels.")

    class_names = list(reference_class_names or [])
    if not class_names:
        # fallback if dataset class names are not available.
        prepared = _prepare_reference_targets(reference_targets)
        if prepared.ndim == 1:
            upper = int(prepared.max()) + 1 if prepared.size else 1
        else:
            upper = int(prepared.shape[1])
        class_names = [str(i) for i in range(max(upper, 1))]

    prepared_targets = _prepare_reference_targets(reference_targets)
    if prepared_targets.ndim == 1:
        return _apply_class_removal_single_label(
            fake_samples=fake_samples,
            reference_samples=real_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=str(config["targets_raw"]),
            min_kept=int(config["min_kept"]),
        )
    if prepared_targets.ndim == 2:
        return _apply_class_removal_multi_label(
            fake_samples=fake_samples,
            reference_samples=real_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=str(config["targets_raw"]),
            label_threshold=float(config["label_threshold"]),
            min_kept=int(config["min_kept"]),
        )
    raise ValueError(
        f"Unsupported reference target shape for class-removal on dataset '{dataset_name}'."
    )

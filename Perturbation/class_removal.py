from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .class_assignment_cache import (
    LabelAssignmentContext,
    build_label_assignment_context,
    class_names_for_label_count,
    default_class_names_from_targets,
    prepare_reference_targets,
)
from .class_fixed_eval import (
    select_single_label_weighted_subset,
    select_uniform_subset,
)


def _parse_token_list(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


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


def _reference_targets_to_binary_matrix(
    reference_targets: torch.Tensor,
    class_names: list[str] | None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    prepared = prepare_reference_targets(reference_targets)
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

    names = class_names_for_label_count(class_names=class_names, label_count=label_count)
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
    evaluation_count: int | None,
    assignment_context = None,
) :
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
            seed=int(seed),
            min_kept=int(min_kept),
            evaluation_count=evaluation_count,
            assignment_context=assignment_context,
        )
    else:
        filtered, drop_details = _apply_class_removal_multi_label(
            fake_samples=fake_samples,
            reference_samples=reference_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=drop_labels_raw,
            label_threshold=float(label_threshold),
            seed=int(seed),
            min_kept=int(min_kept),
            evaluation_count=evaluation_count,
            assignment_context=assignment_context,
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
    seed: int,
    min_kept: int,
    evaluation_count: int | None,
    assignment_context= None,
):
    if assignment_context is not None and assignment_context.label_mode == "single_label":
        predicted_labels = np.asarray(assignment_context.predicted_labels, dtype=np.int64)
        pred_hist = dict(assignment_context.predicted_label_histogram or {})
        upper_bound = max(int(predicted_labels.max()) + 1 if predicted_labels.size else 1, len(class_names))
    else:
        assignment_context = build_label_assignment_context(
            fake_samples=fake_samples,
            reference_samples=reference_samples,
            reference_targets=reference_targets,
            class_names=class_names,
        )
        predicted_labels = np.asarray(assignment_context.predicted_labels, dtype=np.int64)
        pred_hist = dict(assignment_context.predicted_label_histogram or {})
        upper_bound = max(int(predicted_labels.max()) + 1 if predicted_labels.size else 1, len(class_names))

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

    # for fixed-count eval, i keep the full survivor list and then sample from it.
    evaluation_indices = keep_indices.astype(np.int64, copy=False)
    evaluation_sampling = {
        "mode": "all_survivors",
        "seed": int(seed),
        "with_replacement": False,
    }
    if evaluation_count is not None and int(evaluation_count) > 0:
        class_weights = {
            int(class_id): 0.0 if int(class_id) in resolved_drop else 1.0
            for class_id in sorted(set(predicted_labels.tolist()))
        }
        evaluation_indices = select_single_label_weighted_subset(
            predicted_labels=predicted_labels,
            class_weights=class_weights,
            target_count=int(evaluation_count),
            seed=int(seed),
            context="class-removal fixed evaluation",
        )
        evaluation_sampling = {
            "mode": "single_label_exact_quota",
            "seed": int(seed),
            "with_replacement": False,
            "target_count": int(evaluation_count),
        }

    filtered = fake_samples[torch.as_tensor(evaluation_indices, dtype=torch.long)]
    details = {
        "strategy": "label",
        "label_mode": "single_label",
        "drop_classes": resolved_drop,
        "drop_class_names": [class_names[idx] if idx < len(class_names) else str(idx) for idx in resolved_drop],
        "predicted_label_histogram_fake": pred_hist,
        "kept_indices": [int(idx) for idx in keep_indices.tolist()],
        "evaluation_indices": [int(idx) for idx in evaluation_indices.tolist()],
        "evaluation_sampling": evaluation_sampling,
        "removed_count": int(fake_samples.shape[0]) - int(keep_indices.size),
        "kept_count": int(keep_indices.size),
        "survivor_count": int(keep_indices.size),
        "evaluation_count": int(evaluation_indices.size),
        "returned_count": int(filtered.shape[0]),
        "pool_count": int(fake_samples.shape[0]),
    }
    return filtered, details


def _apply_class_removal_multi_label(
    fake_samples: torch.Tensor,
    reference_samples: torch.Tensor,
    reference_targets: torch.Tensor,
    class_names: list[str],
    targets_raw: str,
    label_threshold: float,
    seed: int,
    min_kept: int,
    evaluation_count: int | None,
    assignment_context = None,
) :
    if assignment_context is not None and assignment_context.label_mode == "multi_label":
        margin_scores = np.asarray(assignment_context.margin_scores, dtype=np.float64)
        target_matrix = (np.asarray(assignment_context.prepared_targets) > 0).astype(np.uint8)
    else:
        assignment_context = build_label_assignment_context(
            fake_samples=fake_samples,
            reference_samples=reference_samples,
            reference_targets=reference_targets,
            class_names=class_names,
        )
        if assignment_context.label_mode != "multi_label":
            raise ValueError("Expected multi-label assignment context for class-removal.")
        margin_scores = np.asarray(assignment_context.margin_scores, dtype=np.float64)
        target_matrix = (np.asarray(assignment_context.prepared_targets) > 0).astype(np.uint8)

    if target_matrix.ndim != 2:
        raise ValueError("Expected multi-label matrix for class-removal on this dataset.")

    label_count = int(target_matrix.shape[1])
    resolved_drop = _resolve_named_targets(
        raw_targets=_parse_token_list(targets_raw),
        class_names=class_names,
        upper_bound=label_count,
    )

    drop_mask = np.zeros(margin_scores.shape[0], dtype=bool)
    per_label_predicted_positives: dict[str, int] = {}
    for label_idx in resolved_drop:
        margin_column = margin_scores[:, label_idx]
        if not np.isfinite(margin_column).any():
            per_label_predicted_positives[str(label_idx)] = 0
            continue
        # i reuse the stored margin scores here, so threshold changes are still cheap.
        predicted_positive = margin_column > float(label_threshold)
        per_label_predicted_positives[str(label_idx)] = int(predicted_positive.sum())
        drop_mask |= predicted_positive

    keep_indices = np.nonzero(~drop_mask)[0]
    if int(keep_indices.size) < int(min_kept):
        raise ValueError(
            "class-removal perturbation kept too few fake samples after dropping multi-label classes. "
            f"kept={int(keep_indices.size)} min_kept={int(min_kept)}"
        )

    # multi-label is messier, so i sample from the survivors after the drop mask is built.
    evaluation_indices = keep_indices.astype(np.int64, copy=False)
    evaluation_sampling = {
        "mode": "all_survivors",
        "seed": int(seed),
        "with_replacement": False,
    }
    if evaluation_count is not None and int(evaluation_count) > 0:
        evaluation_indices = select_uniform_subset(
            keep_indices,
            target_count=int(evaluation_count),
            seed=int(seed),
            context="class-removal fixed evaluation",
            pool_count=int(fake_samples.shape[0]),
        )
        evaluation_sampling = {
            "mode": "uniform_survivor_subset",
            "seed": int(seed),
            "with_replacement": False,
            "target_count": int(evaluation_count),
        }

    filtered = fake_samples[torch.as_tensor(evaluation_indices, dtype=torch.long)]
    details = {
        "strategy": "label",
        "label_mode": "multi_label",
        "drop_classes": resolved_drop,
        "drop_class_names": [class_names[idx] if idx < len(class_names) else str(idx) for idx in resolved_drop],
        "label_threshold": float(label_threshold),
        "predicted_positive_counts": per_label_predicted_positives,
        "kept_indices": [int(idx) for idx in keep_indices.tolist()],
        "evaluation_indices": [int(idx) for idx in evaluation_indices.tolist()],
        "evaluation_sampling": evaluation_sampling,
        "removed_count": int(fake_samples.shape[0]) - int(keep_indices.size),
        "kept_count": int(keep_indices.size),
        "survivor_count": int(keep_indices.size),
        "evaluation_count": int(evaluation_indices.size),
        "returned_count": int(filtered.shape[0]),
        "pool_count": int(fake_samples.shape[0]),
    }
    return filtered, details


def apply_class_removal(
    fake_samples: torch.Tensor,
    config: dict[str, Any],
    real_samples: torch.Tensor | None,
    reference_targets: torch.Tensor | None,
    reference_class_names: list[str] | None,
    dataset_name: str,
    runtime_context= None,
) :
    assignment_context = None
    evaluation_count = None
    if isinstance(runtime_context, dict):
        assignment_context = runtime_context.get("label_assignment_context")
        if runtime_context.get("class_fixed_eval_enabled", False):
            raw_count = runtime_context.get("class_evaluation_count")
            if raw_count is not None:
                evaluation_count = int(raw_count)

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
            evaluation_count=evaluation_count,
            assignment_context=assignment_context,
        )

    if real_samples is None or reference_targets is None:
        raise ValueError("Label-based class-removal needs real reference samples and labels.")

    class_names = list(reference_class_names or [])
    if not class_names:
        # fallback if dataset class names are not available.
        class_names = default_class_names_from_targets(
            reference_targets=reference_targets,
            class_names=reference_class_names,
        )

    prepared_targets = prepare_reference_targets(reference_targets)
    if prepared_targets.ndim == 1:
        return _apply_class_removal_single_label(
            fake_samples=fake_samples,
            reference_samples=real_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=str(config["targets_raw"]),
            seed=int(config["seed"]),
            min_kept=int(config["min_kept"]),
            evaluation_count=evaluation_count,
            assignment_context=assignment_context,
        )
    if prepared_targets.ndim == 2:
        return _apply_class_removal_multi_label(
            fake_samples=fake_samples,
            reference_samples=real_samples,
            reference_targets=reference_targets,
            class_names=class_names,
            targets_raw=str(config["targets_raw"]),
            label_threshold=float(config["label_threshold"]),
            seed=int(config["seed"]),
            min_kept=int(config["min_kept"]),
            evaluation_count=evaluation_count,
            assignment_context=assignment_context,
        )
    raise ValueError(
        f"Unsupported reference target shape for class-removal on dataset '{dataset_name}'."
    )

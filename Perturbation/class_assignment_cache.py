from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# this object is the reusable output of the expensive assignment step
# i keep both the raw target view and the prediction helpers here so later code does not recompute them
@dataclass
class LabelAssignmentContext:
    label_mode: str
    class_names: list[str]
    prepared_targets: np.ndarray
    predicted_labels: np.ndarray | None = None
    predicted_label_histogram: dict[str, int] | None = None
    margin_scores: np.ndarray | None = None


# this turns images into flat vectors for the cheap assignment rules we use in the perturbation sweep
def feature_matrix(samples):
    return (
        samples.detach()
        .cpu()
        .float()
        .reshape(samples.shape[0], -1)
        .numpy()
        .astype(np.float64, copy=False)
    )


# this normalizes target tensors so the rest of the code can treat single label and multi label data in one place
def prepare_reference_targets(reference_targets):
    targets_np = reference_targets.detach().cpu().numpy()
    if targets_np.ndim == 0:
        targets_np = targets_np.reshape(1)
    if targets_np.ndim == 2 and targets_np.shape[1] == 1:
        targets_np = targets_np.reshape(-1)
    return targets_np


# this makes sure i always have a usable class name list even when the dataset metadata is incomplete
def class_names_for_label_count(
    class_names,
    label_count,
):
    names = [str(name) for name in list(class_names or [])]
    if len(names) > int(label_count):
        return names[: int(label_count)]
    if len(names) < int(label_count):
        start = len(names)
        names.extend([f"class_{idx}" for idx in range(start, int(label_count))])
    return names


# this is the fallback path when the dataset did not hand me names directly
def default_class_names_from_targets(
    reference_targets,
    class_names,
):
    prepared = prepare_reference_targets(reference_targets)
    if prepared.ndim == 1:
        upper = int(prepared.max()) + 1 if prepared.size else 1
    elif prepared.ndim == 2:
        upper = int(prepared.shape[1])
    else:
        raise ValueError("Unsupported reference target shape.")
    return class_names_for_label_count(class_names=class_names, label_count=max(upper, 1))


# this is the main cached assignment builder for class removal and class imbalance
# i run the heavy fake to reference comparison once here and then the whole sweep can reuse it
# for single label data i assign each fake sample to the nearest class centroid
# for multi label data i keep margin scores for every label so threshold changes stay cheap
def build_label_assignment_context(
    *,
    fake_samples,
    reference_samples,
    reference_targets,
    class_names,
) :
    # i do the expensive fake/reference comparisons here once, then the sweep reuses them
    prepared = prepare_reference_targets(reference_targets)

    if prepared.ndim == 1:
        ref_features = feature_matrix(reference_samples)
        fake_features = feature_matrix(fake_samples)
        labels = prepared.astype(np.int64, copy=False)
        unique_labels = sorted(set(labels.tolist()))
        if not unique_labels:
            raise ValueError("Could not build single-label assignment context without reference classes.")

        centroids: list[np.ndarray] = []
        centroid_ids: list[int] = []
        for class_id in unique_labels:
            class_mask = labels == class_id
            if not np.any(class_mask):
                continue
            centroids.append(ref_features[class_mask].mean(axis=0))
            centroid_ids.append(int(class_id))

        if not centroids:
            raise ValueError("Could not build single-label assignment centroids.")

        centroid_matrix = np.stack(centroids, axis=0)
        distances = ((fake_features[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
        nearest = distances.argmin(axis=1)
        predicted_labels = np.asarray([centroid_ids[idx] for idx in nearest], dtype=np.int64)
        unique_ids, counts = np.unique(predicted_labels, return_counts=True)
        pred_hist = {str(int(idx)): int(count) for idx, count in zip(unique_ids, counts)}
        resolved_names = class_names_for_label_count(
            class_names=class_names,
            label_count=max(max(centroid_ids) + 1, len(class_names or []), 1),
        )
        return LabelAssignmentContext(
            label_mode="single_label",
            class_names=resolved_names,
            prepared_targets=prepared,
            predicted_labels=predicted_labels,
            predicted_label_histogram=pred_hist,
        )

    if prepared.ndim == 2:
        ref_features = feature_matrix(reference_samples)
        fake_features = feature_matrix(fake_samples)
        target_matrix = (prepared > 0).astype(np.uint8)
        label_count = int(target_matrix.shape[1])
        resolved_names = class_names_for_label_count(class_names=class_names, label_count=label_count)

        # i store the raw margin scores so different target sets can reuse the same work
        margin_scores = np.full((fake_features.shape[0], label_count), fill_value=-np.inf, dtype=np.float64)
        for label_idx in range(label_count):
            positive = target_matrix[:, label_idx] == 1
            negative = target_matrix[:, label_idx] == 0
            if positive.sum() == 0 or negative.sum() == 0:
                continue

            positive_centroid = ref_features[positive].mean(axis=0)
            negative_centroid = ref_features[negative].mean(axis=0)
            dist_pos = ((fake_features - positive_centroid[None, :]) ** 2).sum(axis=1)
            dist_neg = ((fake_features - negative_centroid[None, :]) ** 2).sum(axis=1)
            margin_scores[:, label_idx] = dist_neg - dist_pos

        return LabelAssignmentContext(
            label_mode="multi_label",
            class_names=resolved_names,
            prepared_targets=prepared,
            margin_scores=margin_scores,
        )

    raise ValueError("Unsupported reference target shape for assignment context.")

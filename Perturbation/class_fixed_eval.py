from __future__ import annotations

import math

import numpy as np


# keep the fixed-count helpers here so removal and imbalance share the same rules
class InsufficientClassEvaluationPoolError(RuntimeError):
    def __init__(
        self,
        *,
        context: str,
        available_count: int,
        required_count: int,
        pool_count: int,
    ) :
        safe_available = max(1, int(available_count))
        growth = float(required_count) / float(safe_available)
        recommended_pool_size = max(
            int(required_count),
            int(math.ceil(float(pool_count) * growth * 1.1)),
        )
        super().__init__(
            f"{context} only kept {int(available_count)} usable fake samples from a pool of {int(pool_count)}. "
            f"Need at least {int(required_count)} for fixed-count evaluation."
        )
        self.context = str(context)
        self.available_count = int(available_count)
        self.required_count = int(required_count)
        self.pool_count = int(pool_count)
        self.recommended_pool_size = int(recommended_pool_size)


def select_uniform_subset(
    indices,
    target_count: int,
    seed: int,
    *,
    context = "class fixed evaluation",
    pool_count= None,
) :
    chosen = np.asarray(indices, dtype=np.int64)
    need = int(target_count)
    if chosen.size < need:
        raise InsufficientClassEvaluationPoolError(
            context=context,
            available_count=int(chosen.size),
            required_count=need,
            pool_count=int(chosen.size) if pool_count is None else int(pool_count),
        )

    rng = np.random.default_rng(int(seed))
    picked = rng.choice(chosen, size=need, replace=False)
    return np.asarray(picked, dtype=np.int64)


# this gives us integer class counts without changing the total
def _allocate_weighted_class_quotas(
    *,
    available_counts,
    class_weights,
    target_count,
) :
    quotas = {int(class_id): 0 for class_id in available_counts.keys()}
    remaining_capacity = {int(class_id): int(count) for class_id, count in available_counts.items()}
    remaining_target = int(target_count)

    while remaining_target > 0:
        active = [
            int(class_id)
            for class_id, count in remaining_capacity.items()
            if int(count) > 0 and float(class_weights.get(int(class_id), 0.0)) > 0.0
        ]
        if not active:
            break

        masses = np.asarray(
            [
                float(remaining_capacity[class_id]) * float(class_weights.get(class_id, 0.0))
                for class_id in active
            ],
            dtype=np.float64,
        )
        total_mass = float(masses.sum())
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            break

        raw = (float(remaining_target) * masses) / total_mass
        base = np.floor(raw).astype(np.int64)

        if int(base.sum()) == 0:
            order = np.argsort(-raw)
            for idx in order.tolist():
                if remaining_target <= 0:
                    break
                class_id = int(active[idx])
                if int(remaining_capacity[class_id]) <= 0:
                    continue
                quotas[class_id] += 1
                remaining_capacity[class_id] -= 1
                remaining_target -= 1
            continue

        for idx, class_id in enumerate(active):
            give = min(int(base[idx]), int(remaining_capacity[class_id]))
            if give <= 0:
                continue
            quotas[class_id] += give
            remaining_capacity[class_id] -= give
            remaining_target -= give

        if remaining_target <= 0:
            break

        frac_order = np.argsort(-(raw - np.floor(raw)))
        for idx in frac_order.tolist():
            if remaining_target <= 0:
                break
            class_id = int(active[idx])
            if int(remaining_capacity[class_id]) <= 0:
                continue
            quotas[class_id] += 1
            remaining_capacity[class_id] -= 1
            remaining_target -= 1

    return quotas


def select_single_label_weighted_subset(
    *,
    predicted_labels,
    class_weights,
    target_count: int,
    seed: int,
    context: str,
) :
    labels = np.asarray(predicted_labels, dtype=np.int64)
    indices_by_class: dict[int, np.ndarray] = {}
    available_counts: dict[int, int] = {}

    for class_id in sorted(set(labels.tolist())):
        class_indices = np.nonzero(labels == int(class_id))[0].astype(np.int64, copy=False)
        indices_by_class[int(class_id)] = class_indices
        available_counts[int(class_id)] = int(class_indices.size)

    eligible_capacity = int(
        sum(
            available_counts[class_id]
            for class_id, weight in class_weights.items()
            if float(weight) > 0.0 and int(class_id) in available_counts
        )
    )
    if eligible_capacity < int(target_count):
        raise InsufficientClassEvaluationPoolError(
            context=context,
            available_count=eligible_capacity,
            required_count=int(target_count),
            pool_count=int(labels.shape[0]),
        )

    quotas = _allocate_weighted_class_quotas(
        available_counts=available_counts,
        class_weights=class_weights,
        target_count=int(target_count),
    )
    assigned = int(sum(quotas.values()))
    if assigned < int(target_count):
        raise InsufficientClassEvaluationPoolError(
            context=context,
            available_count=assigned,
            required_count=int(target_count),
            pool_count=int(labels.shape[0]),
        )

    rng = np.random.default_rng(int(seed))
    # once the quotas are fixed, i only need a uniform pick inside each class
    selected_parts: list[np.ndarray] = []
    for class_id, quota in quotas.items():
        need = int(quota)
        if need <= 0:
            continue
        class_indices = indices_by_class[int(class_id)]
        if need >= int(class_indices.size):
            selected_parts.append(class_indices.copy())
            continue
        picked = rng.choice(class_indices, size=need, replace=False)
        selected_parts.append(np.asarray(picked, dtype=np.int64))

    if not selected_parts:
        raise InsufficientClassEvaluationPoolError(
            context=context,
            available_count=0,
            required_count=int(target_count),
            pool_count=int(labels.shape[0]),
        )

    selected = np.concatenate(selected_parts, axis=0)
    if int(selected.shape[0]) > int(target_count):
        selected = rng.choice(selected, size=int(target_count), replace=False)
    if int(selected.shape[0]) < int(target_count):
        raise InsufficientClassEvaluationPoolError(
            context=context,
            available_count=int(selected.shape[0]),
            required_count=int(target_count),
            pool_count=int(labels.shape[0]),
        )

    shuffled = rng.permutation(selected)
    return np.asarray(shuffled, dtype=np.int64)

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class PreparedTestRun:
    model_name: str
    generation_payload: dict[str, Any]
    generate_samples: Callable[[int | None], torch.Tensor]
    resolve_reference_request: Callable[[Any, int], tuple[str, str, int]]
    cleanup: Callable[[], None] | None = None


def close_prepared_test_run(prepared):
    if prepared is None or prepared.cleanup is None:
        return
    prepared.cleanup()


# helper for set deterministic seed
def set_deterministic_seed(seed: int, verbose: bool = False, context: str = "test") -> None:
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if verbose:
        print(f"[{context}] deterministic seed set to {seed_value}", flush=True)


# build torch generator
def make_torch_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


# helper for annotate memoisation effective count
def annotate_memoisation_effective_count(
    perturbation_info: dict[str, Any] | None,
    evaluation_subset_size: int,
    verbose: bool = False,
    context: str = "test",
) -> dict[str, Any] | None:
    if not isinstance(perturbation_info, dict):
        return perturbation_info

    memo_cfg = perturbation_info.get("memoisation")
    if not isinstance(memo_cfg, dict):
        return perturbation_info

    memo_result = memo_cfg.get("result")
    if not isinstance(memo_result, dict):
        return perturbation_info

    injected_positions = memo_result.get("injected_positions")
    if not isinstance(injected_positions, list):
        return perturbation_info

    subset = max(0, int(evaluation_subset_size))
    effective_count = int(sum(1 for pos in injected_positions if int(pos) < subset))
    effective_frac = float(effective_count / subset) if subset > 0 else 0.0
    total_injected = int(memo_result.get("injected_count", 0))
    total_samples = int(memo_result.get("total_fake_samples", 0))

    memo_result["effective_in_evaluation_subset"] = {
        "evaluation_subset_size": subset,
        "effective_replaced_count": effective_count,
        "effective_replaced_fraction": effective_frac,
        "total_injected_count": total_injected,
        "total_fake_samples": total_samples,
    }
    if verbose:
        print(
            f"[{context}] memoisation effective replacements in evaluated subset: "
            f"{effective_count}/{subset} ({effective_frac:.4f})",
            flush=True,
        )

    return perturbation_info

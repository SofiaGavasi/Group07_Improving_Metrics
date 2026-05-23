from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Metrics.compute_all import MetricComputationConfig, compute_all_metrics_from_extracted
from Metrics.inception_features import InceptionFeatureConfig, InceptionFeatureExtractor
from Perturbation.class_assignment_cache import build_label_assignment_context
from Perturbation.class_fixed_eval import InsufficientClassEvaluationPoolError
from Perturbation.pipeline_perturbations import (
    apply_configured_perturbations,
    perturbation_needs_real_reference,
    perturbation_needs_reference_targets,
    perturbations_enabled,
)
from Scripts.test_runtime_utils import annotate_memoisation_effective_count, make_torch_generator


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "outputs" / "shared_eval_cache"
CACHE_SCHEMA_VERSION = 1
CLASS_POOL_GROWTH_FACTOR = 2.0
CLASS_POOL_MAX_ATTEMPTS = 4



#__________________________________________________________


# this is the common cache record shape to write into reports
# i use the same object for fake samples real references and feature artifacts
@dataclass
class CacheArtifact:
    kind: str
    key: str
    path: Path
    cache_hit: bool
    metadata: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "path": str(self.path),
            "cache_hit": bool(self.cache_hit),
            "metadata": self.metadata,
        }


# this keeps a generated tensor together with the cache metadata that produced it
@dataclass
class TensorArtifact:
    tensor: torch.Tensor
    cache: CacheArtifact


# this bundles the real reference tensors and labels so later steps do not pass them around separately
@dataclass
class ReferenceArtifact:
    samples: torch.Tensor
    targets: torch.Tensor
    class_names: list[str]
    cache: CacheArtifact


# this stores extracted features and optional inception probabilities for one source key
@dataclass
class FeatureArtifact:
    features: np.ndarray
    probs: np.ndarray | None
    cache: CacheArtifact


# this tells the cache layer how to rebuild a derived feature set from an existing one
#  for subset and replace cases so we can avoid a fresh inception pass
@dataclass
class FeatureReusePlan:
    mode: str
    base_source_key: str
    indices: list[int] | None = None
    replacement_source_key: str | None = None
    replacement_indices: list[int] | None = None
    positions: list[int] | None = None


# this is the per-run output bundle that goes back to the test script and then into the batch report
@dataclass
class EvaluationArtifacts:
    output_dir: Path
    metrics_path: Path
    metrics_report: dict[str, Any] | None
    perturbation_config_path: Path
    perturbation_config: dict[str, Any] | None
    cache_report_path: Path
    cache_report: dict[str, Any]

    def to_test_output(self, step_name):
        return {
            "step_name": step_name,
            "output_dir": str(self.output_dir),
            "metrics_path": str(self.metrics_path),
            "metrics_report": self.metrics_report,
            "perturbation_config_path": str(self.perturbation_config_path),
            "perturbation_config": self.perturbation_config,
            "cache_report_path": str(self.cache_report_path),
            "cache_report": self.cache_report,
        }


# this is the long-lived state for one in-process sweep
# we keep the shared fake pool real bundle assignment context and extractor here so later experiments stay cheap
@dataclass
class EvaluationReuseSession:
    fake_artifacts: dict[str, TensorArtifact]
    reference_artifacts: dict[str, ReferenceArtifact]
    label_assignment_contexts: dict[str, Any]
    memory_feature_artifacts: dict[str, FeatureArtifact]
    extractor_holders: dict[str, dict[str, InceptionFeatureExtractor]]
    preview_written_keys: set[str]

    def close(self):
        for holder in self.extractor_holders.values():
            extractor = holder.get("extractor")
            if extractor is not None:
                extractor.close()
        self.extractor_holders.clear()


# this builds an empty reusable session for one batch group
def create_evaluation_reuse_session():
    return EvaluationReuseSession(
        fake_artifacts={},
        reference_artifacts={},
        label_assignment_contexts={},
        memory_feature_artifacts={},
        extractor_holders={},
        preview_written_keys=set(),
    )



#__________________________________________________________
# json and hashing stay here so every cache key is built the same way


# this makes every cache key payload deterministic before hashing
def _stable_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# this is the one place where we build cache keys for all artifact types
def _hash_payload(kind, payload):
    wrapped = {
        "kind": kind,
        "schema_version": CACHE_SCHEMA_VERSION,
        "payload": payload,
    }
    return hashlib.sha1(_stable_json(wrapped).encode("utf-8")).hexdigest()


# this writes small metadata json files in a consistent way
def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# this maps one cache kind and one key to its on-disk folder
def _cache_dir(kind, key):
    return CACHE_ROOT / kind / key


# this resolves paths before they go into cache metadata so keys stay stable
def _normalize_path(path_like):
    return str(Path(path_like).resolve())


# this records enough file metadata to make cache keys react to checkpoint changes
def file_signature(path_like) :
    path = Path(path_like).resolve()
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


# this pulls class names from whichever dataset field is available
def dataset_class_names(dataset):
    # i keep this in one place because every test script was doing the same checks
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return [str(name) for name in list(dataset.classes)]
    if hasattr(dataset, "attr_names") and dataset.attr_names is not None:
        return [str(name) for name in list(dataset.attr_names)]
    if hasattr(dataset, "finding_classes") and dataset.finding_classes is not None:
        return [str(name) for name in list(dataset.finding_classes)]
    return []


def _generation_seed(args):
    value = getattr(args, "generation_seed", None)
    if value is None:
        value = getattr(args, "seed", 0)
    return int(value)


def _reference_seed(args):
    value = getattr(args, "reference_seed", None)
    if value is None:
        value = getattr(args, "seed", 0)
    return int(value)




#__________________________________________________________
# shared real reference loading lives here now, so the cache can sit above every model script


# this loads or builds the real reference bundle for metrics and label-based perturbations
# the goal is that every compatible experiment shares the same real tensor and target cache
def load_or_create_real_reference(
    *,
    dataset_name,
    data_root,
    image_size,
    sample_count,
    download_if_missing,
    seed,
    verbose: bool = False,
):
    payload = {
        "dataset_name": str(dataset_name),
        "data_root": _normalize_path(data_root),
        "image_size": int(image_size),
        "sample_count": int(sample_count),
        "seed": int(seed),
    }
    key = _hash_payload("real_reference", payload)
    cache_dir = _cache_dir("real_reference", key)
    bundle_path = cache_dir / "bundle.pt"
    metadata_path = cache_dir / "metadata.json"

    if bundle_path.exists():
        cached = torch.load(bundle_path, map_location="cpu")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else payload
        if verbose:
            print(f"[eval_cache] reusing real reference bundle {key}", flush=True)
        return ReferenceArtifact(
            samples=torch.as_tensor(cached["samples"]).detach().cpu(),
            targets=torch.as_tensor(cached["targets"]).detach().cpu(),
            class_names=[str(name) for name in cached.get("class_names", [])],
            cache=CacheArtifact(
                kind="real_reference",
                key=key,
                path=bundle_path,
                cache_hit=True,
                metadata=metadata,
            ),
        )

    from Datasets.unified_dataset_loader import make_default_loader

    loader = make_default_loader(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
    )

    try:
        dataset = loader.get_dataset(train=False, download=False)
    except (FileNotFoundError, RuntimeError):
        if not download_if_missing:
            raise
        dataset = loader.get_dataset(train=False, download=True)

    dataloader = DataLoader(
        dataset,
        batch_size=min(128, int(sample_count)),
        shuffle=True,
        num_workers=0,
        generator=make_torch_generator(seed),
    )

    image_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    total = 0
    for images, targets in dataloader:
        image_batches.append(images)
        target_batches.append(torch.as_tensor(targets))
        total += int(images.shape[0])
        if total >= int(sample_count):
            break

    if not image_batches:
        raise ValueError("Could not load real samples for evaluation.")

    samples = torch.cat(image_batches, dim=0)[: int(sample_count)].detach().cpu()
    targets = torch.cat(target_batches, dim=0)[: int(sample_count)].detach().cpu()
    class_names = dataset_class_names(dataset)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples,
            "targets": targets,
            "class_names": class_names,
        },
        bundle_path,
    )
    metadata = {
        **payload,
        "class_name_count": int(len(class_names)),
        "sample_shape": list(samples.shape),
    }
    _write_json(metadata_path, metadata)
    if verbose:
        print(f"[eval_cache] wrote real reference bundle {key}", flush=True)

    return ReferenceArtifact(
        samples=samples,
        targets=targets,
        class_names=class_names,
        cache=CacheArtifact(
            kind="real_reference",
            key=key,
            path=bundle_path,
            cache_hit=False,
            metadata=metadata,
        ),
    )


#__________________________________________________________

# fake generation is still model specific, but caching it is the same everywhere

# this is the fake-sample cache entrypoint
# the model script still owns the actual generation function, but this wrapper makes the artifact reusable
def load_or_create_fake_samples(
    *,
    generation_payload,
    generate_samples,
    verbose: bool = False,
):
    key = _hash_payload("fake_samples", generation_payload)
    cache_dir = _cache_dir("fake_samples", key)
    samples_path = cache_dir / "samples.pt"
    metadata_path = cache_dir / "metadata.json"

    if samples_path.exists():
        cached = torch.load(samples_path, map_location="cpu")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else generation_payload
        if verbose:
            print(f"[eval_cache] reusing generated samples {key}", flush=True)
        return TensorArtifact(
            tensor=torch.as_tensor(cached["samples"]).detach().cpu(),
            cache=CacheArtifact(
                kind="fake_samples",
                key=key,
                path=samples_path,
                cache_hit=True,
                metadata=metadata,
            ),
        )

    samples = generate_samples().detach().cpu()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples}, samples_path)
    metadata = {
        **generation_payload,
        "sample_shape": list(samples.shape),
    }
    _write_json(metadata_path, metadata)
    if verbose:
        print(f"[eval_cache] wrote generated samples {key}", flush=True)

    return TensorArtifact(
        tensor=samples,
        cache=CacheArtifact(
            kind="fake_samples",
            key=key,
            path=samples_path,
            cache_hit=False,
            metadata=metadata,
        ),
    )


# this first checks the in-memory session cache and then falls back to the on-disk fake cache
def _session_fake_artifact(
    *,
    session,
    generation_payload,
    generate_samples,
    verbose,
):
    key = _hash_payload("fake_samples", generation_payload)
    if session is not None:
        cached = session.fake_artifacts.get(key)
        if cached is not None:
            return cached

    artifact = load_or_create_fake_samples(
        generation_payload=generation_payload,
        generate_samples=generate_samples,
        verbose=verbose,
    )
    if session is not None:
        session.fake_artifacts[key] = artifact
    return artifact


# this does the same session-aware reuse for the real reference bundle
def _session_reference_artifact(
    *,
    session,
    dataset_name,
    data_root,
    image_size,
    sample_count,
    download_if_missing,
    seed,
    verbose,
) :
    payload = {
        "dataset_name": str(dataset_name),
        "data_root": _normalize_path(data_root),
        "image_size": int(image_size),
        "sample_count": int(sample_count),
        "seed": int(seed),
    }
    key = _hash_payload("real_reference", payload)
    if session is not None:
        cached = session.reference_artifacts.get(key)
        if cached is not None:
            return cached

    artifact = load_or_create_real_reference(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        sample_count=sample_count,
        download_if_missing=download_if_missing,
        seed=seed,
        verbose=verbose,
    )
    if session is not None:
        session.reference_artifacts[key] = artifact
    return artifact


# this builds the shared class assignment context once per fake pool and reference bundle
# class removal and class imbalance reuse this across many experiments
def _session_label_assignment_context(
    *,
    session,
    fake_artifact,
    reference_artifact,
) :
    if reference_artifact is None:
        return None

    context_key = _hash_payload(
        "label_assignment_context",
        {
            "fake_key": str(fake_artifact.cache.key),
            "reference_key": str(reference_artifact.cache.key),
        },
    )
    if session is not None:
        cached = session.label_assignment_contexts.get(context_key)
        if cached is not None:
            return cached

    context = build_label_assignment_context(
        fake_samples=fake_artifact.tensor,
        reference_samples=reference_artifact.samples,
        reference_targets=reference_artifact.targets,
        class_names=reference_artifact.class_names,
    )
    if session is not None:
        session.label_assignment_contexts[context_key] = context
    return context


# this key lets one session reuse the same inception extractor across compatible metric runs
def _metric_holder_key(metric_config) :
    payload = {
        "feature_space": str(metric_config.feature_space),
        "feature_batch_size": int(metric_config.feature_batch_size),
        "feature_device": str(metric_config.feature_device),
    }
    return _hash_payload("metric_extractor_holder", payload)


# this decides whether the current run should use the fixed-count class evaluation path
def _class_fixed_eval_enabled(args) -> bool:
    return bool(
        getattr(args, "eval_metrics", False)
        and getattr(args, "perturb_class_fixed_eval", True)
        and (
            getattr(args, "perturb_class_removal", False)
            or getattr(args, "perturb_class_imbalance", False)
        )
    )


# this resolves the target fake count used for class-removal and class-imbalance metric evaluation
def _class_evaluation_count(args) -> int:
    requested = int(getattr(args, "perturb_class_eval_count", 0))
    if requested > 0:
        return requested
    fallback = int(getattr(args, "metrics_samples", 0))
    if fallback > 0:
        return fallback
    return max(1, int(getattr(args, "num_samples", 1)))


# this picks the first fake pool size for class sweeps
# i either use the explicit pool size or a multiplier on the fixed evaluation count
def _initial_class_pool_count(args, evaluation_count: int) -> int:
    base_count = max(1, int(getattr(args, "num_samples", evaluation_count)))
    explicit = int(getattr(args, "perturb_class_pool_size", 0))
    if explicit > 0:
        return max(base_count, int(evaluation_count), explicit)
    multiplier = max(1.0, float(getattr(args, "perturb_class_pool_multiplier", 3.0)))
    return max(base_count, int(evaluation_count), int(np.ceil(float(evaluation_count) * multiplier)))


# this grows the fake pool after a class experiment says the pool was too small
def _next_class_pool_count(
    *,
    current_count: int,
    evaluation_count: int,
    error: InsufficientClassEvaluationPoolError | None,
) -> int:
    target = max(int(current_count) + 1, int(evaluation_count))
    if error is not None:
        target = max(target, int(error.recommended_pool_size))
    return max(target, int(np.ceil(float(current_count) * CLASS_POOL_GROWTH_FACTOR)))


# this decides how many real samples we really need to load for a run
# most class sweeps do not need a giant real bundle even if the fake pool is oversized
def _reference_bundle_count(args, generated_pool_count: int) -> int:
    base_count = max(1, int(getattr(args, "num_samples", generated_pool_count)))
    metrics_count = max(1, int(getattr(args, "metrics_samples", base_count)))
    if bool(getattr(args, "perturb_memoisation", False)) or str(getattr(args, "perturb_apply_to", "fake")) in {"real", "both"}:
        return max(metrics_count, int(generated_pool_count))
    return max(metrics_count, base_count)



#__________________________________________________________

# this is the stable cache key for one extracted feature artifact
def _feature_cache_key(source_key: str, feature_space: str, needs_probs: bool) -> str:
    payload = {
        "source_key": str(source_key),
        "feature_space": str(feature_space),
        "needs_probs": bool(needs_probs),
    }
    return _hash_payload("metric_features", payload)


# this wraps an in-memory-only feature artifact when i do not want to write a derived subset to disk
def _transient_feature_artifact(
    *,
    source_key,
    feature_space,
    needs_probs,
    features,
    probs,
    metadata,
) :
    key = _feature_cache_key(source_key, feature_space, needs_probs)
    return FeatureArtifact(
        features=np.asarray(features, dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64) if probs is not None else None,
        cache=CacheArtifact(
            kind="metric_features",
            key=key,
            path=Path(),
            cache_hit=False,
            metadata={
                **metadata,
                "transient_in_memory": True,
            },
        ),
    )


# this loads cached extracted features if they already exist on disk
def _load_feature_artifact(source_key: str, feature_space: str, needs_probs: bool) :
    key = _feature_cache_key(source_key, feature_space, needs_probs)
    cache_dir = _cache_dir("metric_features", key)
    arrays_path = cache_dir / "arrays.npz"
    metadata_path = cache_dir / "metadata.json"
    if not arrays_path.exists():
        return None

    loaded = np.load(arrays_path)
    probs = loaded["probs"] if ("probs" in loaded and loaded["probs"].size > 0) else None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return FeatureArtifact(
        features=np.asarray(loaded["features"], dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64) if probs is not None else None,
        cache=CacheArtifact(
            kind="metric_features",
            key=key,
            path=arrays_path,
            cache_hit=True,
            metadata=metadata,
        ),
    )


# this writes extracted features to disk in the common cache layout
def _save_feature_artifact(
    *,
    source_key,
    feature_space,
    needs_probs,
    features,
    probs,
    metadata ,
    cache_hit,
):
    key = _feature_cache_key(source_key, feature_space, needs_probs)
    cache_dir = _cache_dir("metric_features", key)
    arrays_path = cache_dir / "arrays.npz"
    metadata_path = cache_dir / "metadata.json"

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        features=np.asarray(features, dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64) if probs is not None else np.asarray([], dtype=np.float64),
    )
    _write_json(metadata_path, metadata)
    return FeatureArtifact(
        features=np.asarray(features, dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64) if probs is not None else None,
        cache=CacheArtifact(
            kind="metric_features",
            key=key,
            path=arrays_path,
            cache_hit=cache_hit,
            metadata=metadata,
        ),
    )


# this is the actual inception extraction step for one tensor source
def _extract_feature_artifact(
    *,
    source_key,
    samples,
    feature_space,
    needs_probs,
    metric_config,
    extractor,
    verbose: bool = False,
):
    if verbose:
        print(f"[eval_cache] extracting {source_key} into {feature_space} features", flush=True)

    features, _, probs = extractor.extract(samples.detach().cpu().float())
    metadata = {
        "source_key": str(source_key),
        "feature_space": str(feature_space),
        "needs_probs": bool(needs_probs),
        "sample_count": int(samples.shape[0]),
        "sample_shape": list(samples.shape),
        "feature_batch_size": int(metric_config.feature_batch_size),
    }
    return _save_feature_artifact(
        source_key=source_key,
        feature_space=feature_space,
        needs_probs=needs_probs,
        features=features,
        probs=probs if needs_probs else None,
        metadata=metadata,
        cache_hit=False,
    )


# this is the main feature reuse resolver
# i first try memory then disk then subset or replace plans and only extract as a last resort
def _resolve_feature_artifact(
    *,
    source_key: str,
    feature_space: str,
    needs_probs,
    metric_config,
    sample_sources,
    feature_plans,
    verbose,
    memory_feature_artifacts,
    persist_derived_feature_artifacts,
    extractor_holder,
) :
    feature_key = _feature_cache_key(source_key, feature_space, needs_probs)
    memory_cached = memory_feature_artifacts.get(feature_key)
    if memory_cached is not None:
        if verbose:
            print(f"[eval_cache] reusing in-memory metric features {source_key}", flush=True)
        return memory_cached

    cached = _load_feature_artifact(source_key, feature_space, needs_probs)
    if cached is not None:
        if verbose:
            print(f"[eval_cache] reusing metric features {source_key}", flush=True)
        memory_feature_artifacts[feature_key] = cached
        return cached

    plan = feature_plans.get(source_key)
    if plan is not None and plan.mode == "subset":
        base_artifact = _resolve_feature_artifact(
            source_key=plan.base_source_key,
            feature_space=feature_space,
            needs_probs=needs_probs,
            metric_config=metric_config,
            sample_sources=sample_sources,
            feature_plans=feature_plans,
            verbose=verbose,
            memory_feature_artifacts=memory_feature_artifacts,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            extractor_holder=extractor_holder,
        )
        if plan.indices is None:
            raise ValueError("subset feature reuse plan is missing indices.")
        subset = np.asarray(plan.indices, dtype=np.int64)
        derived_probs = base_artifact.probs[subset] if (needs_probs and base_artifact.probs is not None) else None
        metadata = {
            "source_key": str(source_key),
            "feature_space": str(feature_space),
            "needs_probs": bool(needs_probs),
            "derived_from": str(plan.base_source_key),
            "reuse_mode": "subset",
            "selected_count": int(subset.shape[0]),
        }
        if verbose:
            print(f"[eval_cache] deriving subset features for {source_key}", flush=True)
        if persist_derived_feature_artifacts:
            artifact = _save_feature_artifact(
                source_key=source_key,
                feature_space=feature_space,
                needs_probs=needs_probs,
                features=base_artifact.features[subset],
                probs=derived_probs,
                metadata=metadata,
                cache_hit=False,
            )
        else:
            artifact = _transient_feature_artifact(
                source_key=source_key,
                feature_space=feature_space,
                needs_probs=needs_probs,
                features=base_artifact.features[subset],
                probs=derived_probs,
                metadata=metadata,
            )
        memory_feature_artifacts[feature_key] = artifact
        return artifact

    if plan is not None and plan.mode == "replace":
        base_artifact = _resolve_feature_artifact(
            source_key=plan.base_source_key,
            feature_space=feature_space,
            needs_probs=True,
            metric_config=metric_config,
            sample_sources=sample_sources,
            feature_plans=feature_plans,
            verbose=verbose,
            memory_feature_artifacts=memory_feature_artifacts,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            extractor_holder=extractor_holder,
        )
        replacement_artifact = _resolve_feature_artifact(
            source_key=str(plan.replacement_source_key),
            feature_space=feature_space,
            needs_probs=True,
            metric_config=metric_config,
            sample_sources=sample_sources,
            feature_plans=feature_plans,
            verbose=verbose,
            memory_feature_artifacts=memory_feature_artifacts,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            extractor_holder=extractor_holder,
        )
        if plan.positions is None or plan.replacement_indices is None:
            raise ValueError("replace feature reuse plan is missing positions or source indices.")

        derived_features = np.array(base_artifact.features, copy=True)
        derived_probs = np.array(base_artifact.probs, copy=True) if base_artifact.probs is not None else None
        positions = np.asarray(plan.positions, dtype=np.int64)
        replacement_indices = np.asarray(plan.replacement_indices, dtype=np.int64)
        derived_features[positions] = replacement_artifact.features[replacement_indices]
        if derived_probs is not None and replacement_artifact.probs is not None:
            derived_probs[positions] = replacement_artifact.probs[replacement_indices]
        metadata = {
            "source_key": str(source_key),
            "feature_space": str(feature_space),
            "needs_probs": bool(needs_probs),
            "derived_from": str(plan.base_source_key),
            "replacement_source": str(plan.replacement_source_key),
            "reuse_mode": "replace",
            "replacement_count": int(positions.shape[0]),
        }
        if verbose:
            print(f"[eval_cache] deriving replaced features for {source_key}", flush=True)
        if persist_derived_feature_artifacts:
            artifact = _save_feature_artifact(
                source_key=source_key,
                feature_space=feature_space,
                needs_probs=needs_probs,
                features=derived_features,
                probs=derived_probs if needs_probs else None,
                metadata=metadata,
                cache_hit=False,
            )
        else:
            artifact = _transient_feature_artifact(
                source_key=source_key,
                feature_space=feature_space,
                needs_probs=needs_probs,
                features=derived_features,
                probs=derived_probs if needs_probs else None,
                metadata=metadata,
            )
        memory_feature_artifacts[feature_key] = artifact
        return artifact

    samples = sample_sources.get(source_key)
    if samples is None:
        raise KeyError(f"No samples available for feature source '{source_key}'.")

    extractor = extractor_holder.get("extractor")
    if extractor is None:
        extractor = InceptionFeatureExtractor(
            config=InceptionFeatureConfig(
                batch_size=int(metric_config.feature_batch_size),
                device=str(metric_config.feature_device),
                num_workers=0,
            )
        )
        extractor_holder["extractor"] = extractor

    artifact = _extract_feature_artifact(
        source_key=source_key,
        samples=samples,
        feature_space=feature_space,
        needs_probs=needs_probs,
        metric_config=metric_config,
        extractor=extractor,
        verbose=verbose,
    )
    memory_feature_artifacts[feature_key] = artifact
    return artifact




#__________________________________________________________
# i build source keys from stable config only

# this records which fake-side perturbations change the feature source identity
def _fake_side_signature(perturbation_info):
    if not isinstance(perturbation_info, dict) or not bool(perturbation_info.get("enabled", False)):
        return {"enabled": False}

    signature: dict[str, Any] = {
        "enabled": True,
        "apply_to": str(perturbation_info.get("apply_to", "fake")),
    }
    degradation = perturbation_info.get("degradation", {})
    if bool(degradation.get("enabled", False)):
        signature["degradation"] = {
            "severity": int(degradation.get("severity", 1)),
            "gaussian_noise": bool(degradation.get("gaussian_noise", False)),
            "gaussian_blur": bool(degradation.get("gaussian_blur", False)),
            "jpeg_compression": bool(degradation.get("jpeg_compression", False)),
        }
    memo = perturbation_info.get("memoisation", {})
    if bool(memo.get("enabled", False)):
        signature["memoisation"] = {
            "fraction": float(memo.get("fraction", 0.0)),
            "seed": int(memo.get("seed", 0)),
        }
    class_removal = perturbation_info.get("class_removal", {})
    if bool(class_removal.get("enabled", False)):
        signature["class_removal"] = {
            "strategy": str(class_removal.get("strategy", "")),
            "targets_raw": str(class_removal.get("targets_raw", "")),
            "kmeans_k": int(class_removal.get("kmeans_k", 0)),
            "seed": int(class_removal.get("seed", 0)),
            "label_threshold": float(class_removal.get("label_threshold", 0.0)),
            "min_kept": int(class_removal.get("min_kept", 0)),
        }
    class_imbalance = perturbation_info.get("class_imbalance", {})
    if bool(class_imbalance.get("enabled", False)):
        signature["class_imbalance"] = {
            "strategy": str(class_imbalance.get("strategy", "")),
            "targets_raw": str(class_imbalance.get("targets_raw", "")),
            "balance": class_imbalance.get("balance"),
            "kmeans_k": int(class_imbalance.get("kmeans_k", 0)),
            "seed": int(class_imbalance.get("seed", 0)),
            "label_threshold": float(class_imbalance.get("label_threshold", 0.0)),
            "min_kept": int(class_imbalance.get("min_kept", 0)),
        }
    class_fixed_eval = perturbation_info.get("class_fixed_eval", {})
    if bool(class_fixed_eval.get("enabled", False)) and (
        bool(class_removal.get("enabled", False)) or bool(class_imbalance.get("enabled", False))
    ):
        signature["class_fixed_eval"] = {
            "evaluation_count": int(class_fixed_eval.get("evaluation_count", 0)),
            "pool_size": int(class_fixed_eval.get("pool_size", 0)),
            "pool_multiplier": float(class_fixed_eval.get("pool_multiplier", 0.0)),
        }
    sample_size = perturbation_info.get("sample_size", {})
    if bool(sample_size.get("enabled", False)):
        signature["sample_size"] = {
            "n": int(sample_size.get("n", 0)),
            "seed": int(sample_size.get("seed", 0)),
        }
    preprocessing = perturbation_info.get("preprocessing", {})
    if bool(preprocessing.get("enabled", False)):
        signature["preprocessing"] = {
            "variant": str(preprocessing.get("variant", "")),
            "scale": float(preprocessing.get("scale", 1.0)),
        }
    return signature


# this does the same for real-side perturbations that can affect metric features
def _real_side_signature(perturbation_info):
    if not isinstance(perturbation_info, dict) or not bool(perturbation_info.get("enabled", False)):
        return {"enabled": False}

    signature: dict[str, Any] = {
        "enabled": True,
        "apply_to": str(perturbation_info.get("apply_to", "fake")),
    }
    degradation = perturbation_info.get("degradation", {})
    if bool(degradation.get("enabled", False)) and str(perturbation_info.get("apply_to", "fake")) in {"real", "both"}:
        signature["degradation"] = {
            "severity": int(degradation.get("severity", 1)),
            "gaussian_noise": bool(degradation.get("gaussian_noise", False)),
            "gaussian_blur": bool(degradation.get("gaussian_blur", False)),
            "jpeg_compression": bool(degradation.get("jpeg_compression", False)),
        }
    sample_size = perturbation_info.get("sample_size", {})
    if bool(sample_size.get("enabled", False)) and str(perturbation_info.get("apply_to", "fake")) in {"real", "both"}:
        signature["sample_size"] = {
            "n": int(sample_size.get("n", 0)),
            "seed": int(sample_size.get("seed", 0)),
        }
    preprocessing = perturbation_info.get("preprocessing", {})
    if bool(preprocessing.get("enabled", False)) and str(perturbation_info.get("apply_to", "fake")) in {"real", "both"}:
        signature["preprocessing"] = {
            "variant": str(preprocessing.get("variant", "")),
            "scale": float(preprocessing.get("scale", 1.0)),
        }
    return signature


# this turns a baseline cache key plus a perturbation signature into a derived feature source key
def _feature_source_key(base_key, side, signature):
    if not signature.get("enabled", False):
        return base_key
    payload = {
        "base_key": str(base_key),
        "side": str(side),
        "signature": signature,
    }
    return _hash_payload("feature_source", payload)


# this tells the cache layer when a fake-side perturbation can be rebuilt by subsetting or replacing rows
def _build_fake_feature_plan(
    *,
    baseline_fake_key,
    baseline_real_key,
    perturbation_info,
):
    if not isinstance(perturbation_info, dict):
        return None

    applied = [str(item) for item in perturbation_info.get("applied", []) if str(item).endswith(":fake")]
    if not applied:
        return None

    if applied == ["sample_size:fake"]:
        result = perturbation_info.get("sample_size", {}).get("result", {})
        indices = result.get("selected_indices_fake")
        if isinstance(indices, list):
            return FeatureReusePlan(
                mode="subset",
                base_source_key=baseline_fake_key,
                indices=[int(idx) for idx in indices],
            )

    if applied == ["class_removal:fake"]:
        result = perturbation_info.get("class_removal", {}).get("result", {})
        indices = result.get("evaluation_indices")
        if not isinstance(indices, list):
            indices = result.get("kept_indices")
        if isinstance(indices, list):
            return FeatureReusePlan(
                mode="subset",
                base_source_key=baseline_fake_key,
                indices=[int(idx) for idx in indices],
            )

    if applied == ["class_imbalance:fake"]:
        result = perturbation_info.get("class_imbalance", {}).get("result", {})
        indices = result.get("evaluation_indices")
        if not isinstance(indices, list):
            indices = result.get("kept_indices")
        if isinstance(indices, list):
            return FeatureReusePlan(
                mode="subset",
                base_source_key=baseline_fake_key,
                indices=[int(idx) for idx in indices],
            )

    if applied == ["memoisation:fake"]:
        result = perturbation_info.get("memoisation", {}).get("result", {})
        positions = result.get("injected_positions")
        replacement_indices = result.get("injected_real_indices")
        if isinstance(positions, list) and isinstance(replacement_indices, list):
            return FeatureReusePlan(
                mode="replace",
                base_source_key=baseline_fake_key,
                replacement_source_key=baseline_real_key,
                positions=[int(idx) for idx in positions],
                replacement_indices=[int(idx) for idx in replacement_indices],
            )

    return None


# this is the real-side version of the subset reuse plan builder
def _build_real_feature_plan(
    *,
    baseline_real_key,
    perturbation_info,
):
    if not isinstance(perturbation_info, dict):
        return None

    applied = [str(item) for item in perturbation_info.get("applied", []) if str(item).endswith(":real")]
    if applied == ["sample_size:real"]:
        result = perturbation_info.get("sample_size", {}).get("result", {})
        indices = result.get("selected_indices_real")
        if isinstance(indices, list):
            return FeatureReusePlan(
                mode="subset",
                base_source_key=baseline_real_key,
                indices=[int(idx) for idx in indices],
            )
    return None


# this is a small helper for deciding whether the real tensor used for metrics was actually changed
def _has_real_side_perturbation(perturbation_info):
    if not isinstance(perturbation_info, dict):
        return False
    return any(str(item).endswith(":real") for item in perturbation_info.get("applied", []))


# this picks the real tensor that should go into metric evaluation
# it keeps the perturbed real set when that side was intentionally changed
def _metric_real_samples(
    *,
    base_real_samples,
    perturbed_real_samples,
    metrics_samples: int,
    perturbation_info,
) :
    if perturbed_real_samples is not None and _has_real_side_perturbation(perturbation_info):
        # this fixes the old bug where a real-side sample-size perturbation was silently ignored
        limit = min(int(perturbed_real_samples.shape[0]), int(metrics_samples))
        return perturbed_real_samples[:limit].detach().cpu()
    if perturbed_real_samples is not None and int(perturbed_real_samples.shape[0]) >= int(metrics_samples):
        return perturbed_real_samples[: int(metrics_samples)].detach().cpu()
    if base_real_samples is None:
        raise ValueError("Real samples are required for metric evaluation.")
    return base_real_samples[: int(metrics_samples)].detach().cpu()


# this is the current bootstrap policy gate
# right now i only keep bootstrap for the baseline and sample-size runs
def _should_bootstrap_metrics(perturbation_info):
    if not isinstance(perturbation_info, dict):
        return True
    if not bool(perturbation_info.get("enabled", False)):
        return True
    sample_size = perturbation_info.get("sample_size", {})
    return bool(sample_size.get("enabled", False))


# this is the cached metric entrypoint once fake and real tensors are already decided
# i resolve feature artifacts here and then hand the arrays to Metrics/compute_all.py
def evaluate_metrics_with_cache(
    *,
    real_samples,
    fake_samples,
    real_source_key,
    fake_source_key,
    sample_sources,
    feature_plans,
    out_dir,
    metric_config,
    memory_feature_artifacts,
    persist_derived_feature_artifacts = True,
    extractor_holder = None,
    verbose = False,
) :
    paired_count = min(int(real_samples.shape[0]), int(fake_samples.shape[0]))
    if paired_count < 4:
        raise ValueError("Need at least 4 paired real/fake samples to compute metrics robustly.")

    trimmed_real = real_samples[:paired_count].detach().cpu()
    trimmed_fake = fake_samples[:paired_count].detach().cpu()
    sample_sources = {
        **sample_sources,
        real_source_key: trimmed_real,
        fake_source_key: trimmed_fake,
    }

    holder = extractor_holder if extractor_holder is not None else {}
    try:
        real_features = _resolve_feature_artifact(
            source_key=real_source_key,
            feature_space=metric_config.feature_space,
            needs_probs=False,
            metric_config=metric_config,
            sample_sources=sample_sources,
            feature_plans=feature_plans,
            verbose=verbose,
            memory_feature_artifacts=memory_feature_artifacts,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            extractor_holder=holder,
        )
        fake_features = _resolve_feature_artifact(
            source_key=fake_source_key,
            feature_space=metric_config.feature_space,
            needs_probs=True,
            metric_config=metric_config,
            sample_sources=sample_sources,
            feature_plans=feature_plans,
            verbose=verbose,
            memory_feature_artifacts=memory_feature_artifacts,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            extractor_holder=holder,
        )
    finally:
        if extractor_holder is None:
            extractor = holder.get("extractor")
            if extractor is not None:
                extractor.close()

    results = compute_all_metrics_from_extracted(
        real_features=real_features.features,
        fake_features=fake_features.features,
        fake_probs=fake_features.probs,
        config=metric_config,
    )
    metrics_path = out_dir / "metrics_report.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if verbose:
        print(f"[eval_cache] wrote metrics to {metrics_path}", flush=True)

    cache_report = {
        "real_features": real_features.cache.to_report(),
        "fake_features": fake_features.cache.to_report(),
        "real_source_key": str(real_source_key),
        "fake_source_key": str(fake_source_key),
    }
    return results, cache_report


# this is the main shared flow the model scripts call now
# it handles fake pool sizing real reference reuse perturbations feature reuse metrics and per-run reports
def run_cached_evaluation(
    *,
    args,
    model_name,
    generation_payload,
    generate_samples,
    resolve_reference_request,
    session = None,
    write_preview = True,
    persist_derived_feature_artifacts = True,
    bootstrap_samples_override = None,
    bootstrap_policy= "full",
):
    verbose = bool(getattr(args, "verbose", False))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    metrics_path = out_dir / "metrics_report.json"
    perturbation_path = out_dir / "perturbation_config.json"
    cache_report_path = out_dir / "cache_report.json"

    class_fixed_eval_enabled = _class_fixed_eval_enabled(args)
    class_evaluation_count = _class_evaluation_count(args) if class_fixed_eval_enabled else None
    if class_fixed_eval_enabled:
        current_generation_count = _initial_class_pool_count(
            args,
            int(class_evaluation_count or getattr(args, "num_samples", 1)),
        )
    else:
        current_generation_count = max(1, int(getattr(args, "num_samples", 1)))
    class_pool_attempt = 0

    fake_artifact: TensorArtifact | None = None
    reference_artifact: ReferenceArtifact | None = None
    baseline_fake_samples: torch.Tensor | None = None
    samples: torch.Tensor | None = None
    perturbation_info: dict[str, Any] | None = None
    perturbed_real_samples: torch.Tensor | None = None

    # class sweeps may need a bigger pool than the default test sample count.
    while True:
        effective_generation_payload = dict(generation_payload)
        effective_generation_payload["num_samples"] = int(current_generation_count)

        fake_artifact = _session_fake_artifact(
            session=session,
            generation_payload=effective_generation_payload,
            generate_samples=lambda: generate_samples(int(current_generation_count)),
            verbose=verbose,
        )
        baseline_fake_samples = fake_artifact.tensor.detach().cpu()
        samples = baseline_fake_samples.clone()

        metric_image_size = int(getattr(args, "metrics_image_size", 0))
        if metric_image_size <= 0 and samples.ndim == 4 and int(samples.shape[-1]) == int(samples.shape[-2]):
            metric_image_size = int(samples.shape[-1])
        if metric_image_size <= 0:
            metric_image_size = int(getattr(args, "image_size", 0)) or int(samples.shape[-1])

        ref_dataset, ref_root, ref_image_size = resolve_reference_request(args, metric_image_size)

        needs_reference_bundle = bool(getattr(args, "eval_metrics", False)) or perturbation_needs_real_reference(args)
        reference_count = _reference_bundle_count(args, generated_pool_count=int(current_generation_count))

        reference_artifact = None
        if needs_reference_bundle:
            reference_artifact = _session_reference_artifact(
                session=session,
                dataset_name=str(ref_dataset),
                data_root=str(ref_root),
                image_size=int(ref_image_size),
                sample_count=reference_count,
                download_if_missing=bool(getattr(args, "metrics_download_if_missing", False)),
                seed=_reference_seed(args),
                verbose=verbose,
            )

        perturbed_real_samples = None
        perturbation_info = None

        try:
            if perturbations_enabled(args):
                needs_reference_targets = perturbation_needs_reference_targets(args)
                if reference_artifact is not None and (
                    perturbation_needs_real_reference(args) or needs_reference_targets
                ):
                    perturbed_real_samples = reference_artifact.samples.clone()

                perturbation_runtime_context = None
                if reference_artifact is not None and (
                    bool(getattr(args, "perturb_class_removal", False))
                    or bool(getattr(args, "perturb_class_imbalance", False))
                ):
                    perturbation_runtime_context = {
                        "label_assignment_context": _session_label_assignment_context(
                            session=session,
                            fake_artifact=fake_artifact,
                            reference_artifact=reference_artifact,
                        ),
                        "class_fixed_eval_enabled": bool(class_fixed_eval_enabled),
                        "class_evaluation_count": int(class_evaluation_count or 0),
                    }

                samples, perturbed_real_samples, perturbation_info = apply_configured_perturbations(
                    fake_samples=samples,
                    args=args,
                    real_samples=perturbed_real_samples,
                    reference_targets=reference_artifact.targets if (needs_reference_targets and reference_artifact is not None) else None,
                    reference_class_names=reference_artifact.class_names if reference_artifact is not None else None,
                    dataset_name=str(ref_dataset),
                    runtime_context=perturbation_runtime_context,
                )
            break
        except InsufficientClassEvaluationPoolError as exc:
            if (not class_fixed_eval_enabled) or int(class_pool_attempt) + 1 >= int(CLASS_POOL_MAX_ATTEMPTS):
                raise
            next_count = _next_class_pool_count(
                current_count=int(current_generation_count),
                evaluation_count=int(class_evaluation_count or getattr(args, "num_samples", 1)),
                error=exc,
            )
            if verbose:
                print(
                    f"[eval_cache] class pool too small ({current_generation_count} -> {next_count}) "
                    f"after {exc.available_count} usable samples for {exc.required_count} targets",
                    flush=True,
                )
            current_generation_count = next_count
            class_pool_attempt += 1
            continue

    preview_key = str(fake_artifact.cache.key)
    preview_should_write = bool(write_preview)
    if session is not None and preview_key in session.preview_written_keys:
        preview_should_write = False
    if preview_should_write:
        save_image(samples, out_dir / "generated_samples.png", nrow=8, normalize=True)
        print(f"Saved {samples.shape[0]} samples to {out_dir / 'generated_samples.png'}")
        if session is not None:
            session.preview_written_keys.add(preview_key)

    cache_report: dict[str, Any] = {
        "fake_samples": fake_artifact.cache.to_report(),
        "real_reference": reference_artifact.cache.to_report() if reference_artifact is not None else None,
        "class_fixed_eval": {
            "enabled": bool(class_fixed_eval_enabled),
            "evaluation_count": int(class_evaluation_count or 0),
            "pool_count": int(current_generation_count),
            "attempts": int(class_pool_attempt) + 1,
        },
        "metrics": None,
    }
    metrics_report: dict[str, Any] | None = None

    if bool(getattr(args, "eval_metrics", False)):
        try:
            base_real_samples = reference_artifact.samples if reference_artifact is not None else None
            real_samples = _metric_real_samples(
                base_real_samples=base_real_samples,
                perturbed_real_samples=perturbed_real_samples,
                metrics_samples=int(getattr(args, "metrics_samples", 0)),
                perturbation_info=perturbation_info,
            )

            if perturbation_info is not None:
                evaluation_subset_size = min(int(real_samples.shape[0]), int(samples.shape[0]))
                perturbation_info = annotate_memoisation_effective_count(
                    perturbation_info=perturbation_info,
                    evaluation_subset_size=evaluation_subset_size,
                    verbose=verbose,
                    context=f"test_{model_name}",
                )

            metric_device = str(getattr(args, "metrics_feature_device", "cpu"))
            if metric_device == "cuda" and not torch.cuda.is_available():
                metric_device = "cpu"

            requested_bootstrap_samples = int(getattr(args, "metrics_bootstrap_samples", 0))
            if bootstrap_samples_override is None:
                if _should_bootstrap_metrics(perturbation_info):
                    effective_bootstrap_samples = requested_bootstrap_samples
                    effective_bootstrap_policy = "baseline_and_sample_size_only"
                else:
                    effective_bootstrap_samples = 0
                    effective_bootstrap_policy = "baseline_and_sample_size_only"
            else:
                effective_bootstrap_samples = int(bootstrap_samples_override)
                effective_bootstrap_policy = str(bootstrap_policy)
            metric_config = MetricComputationConfig(
                feature_space=str(getattr(args, "metrics_feature_space", "inception_v3")),
                feature_batch_size=int(getattr(args, "metrics_feature_batch_size", 64)),
                feature_device=metric_device,
                bootstrap_samples=effective_bootstrap_samples,
                bootstrap_seed=int(getattr(args, "metrics_bootstrap_seed", 0)),
                bootstrap_alpha=float(getattr(args, "metrics_bootstrap_alpha", 0.05)),
                requested_bootstrap_samples=requested_bootstrap_samples,
                bootstrap_policy=effective_bootstrap_policy,
                verbose=verbose,
            )

            baseline_real_key = reference_artifact.cache.key if reference_artifact is not None else ""
            fake_signature = _fake_side_signature(perturbation_info)
            real_signature = _real_side_signature(perturbation_info)
            fake_source_key = _feature_source_key(fake_artifact.cache.key, "fake", fake_signature)
            real_source_key = _feature_source_key(baseline_real_key, "real", real_signature) if baseline_real_key else ""

            feature_plans: dict[str, FeatureReusePlan] = {}
            fake_plan = _build_fake_feature_plan(
                baseline_fake_key=fake_artifact.cache.key,
                baseline_real_key=baseline_real_key,
                perturbation_info=perturbation_info,
            )
            if fake_plan is not None and fake_source_key != fake_artifact.cache.key:
                feature_plans[fake_source_key] = fake_plan

            real_plan = _build_real_feature_plan(
                baseline_real_key=baseline_real_key,
                perturbation_info=perturbation_info,
            )
            if real_plan is not None and real_source_key != baseline_real_key:
                feature_plans[real_source_key] = real_plan

            sample_sources = {
                fake_artifact.cache.key: baseline_fake_samples,
            }
            if reference_artifact is not None:
                sample_sources[baseline_real_key] = reference_artifact.samples

            extractor_holder: dict[str, InceptionFeatureExtractor] | None = None
            memory_feature_artifacts: dict[str, FeatureArtifact]
            if session is not None:
                holder_key = _metric_holder_key(metric_config)
                extractor_holder = session.extractor_holders.setdefault(holder_key, {})
                memory_feature_artifacts = session.memory_feature_artifacts
            else:
                memory_feature_artifacts = {}

            metrics, metrics_cache_report = evaluate_metrics_with_cache(
                real_samples=real_samples,
                fake_samples=samples,
                real_source_key=real_source_key or baseline_real_key,
                fake_source_key=fake_source_key,
                sample_sources=sample_sources,
                feature_plans=feature_plans,
                out_dir=out_dir,
                metric_config=metric_config,
                memory_feature_artifacts=memory_feature_artifacts,
                persist_derived_feature_artifacts=persist_derived_feature_artifacts,
                extractor_holder=extractor_holder,
                verbose=verbose,
            )
            print(f"Metric summary keys: {list(metrics.keys())}")
            cache_report["metrics"] = metrics_cache_report
            metrics_report = metrics
        except Exception as exc:
            metrics_report = {"error": str(exc)}
            metrics_path.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
            if bool(getattr(args, "strict", False)):
                raise
            print(f"Metric evaluation skipped/failed: {exc}")

    if perturbation_info is not None:
        perturbation_path.write_text(json.dumps(perturbation_info, indent=2), encoding="utf-8")
        if verbose:
            print(f"[eval_cache] wrote perturbation config to {perturbation_path}", flush=True)

    cache_report_path.write_text(json.dumps(cache_report, indent=2), encoding="utf-8")
    if verbose:
        print(f"[eval_cache] wrote cache report to {cache_report_path}", flush=True)
    return EvaluationArtifacts(
        output_dir=out_dir,
        metrics_path=metrics_path,
        metrics_report=metrics_report,
        perturbation_config_path=perturbation_path,
        perturbation_config=perturbation_info,
        cache_report_path=cache_report_path,
        cache_report=cache_report,
    )

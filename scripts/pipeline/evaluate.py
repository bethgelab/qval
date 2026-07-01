"""Step 3: Evaluate — compute correlations between predictions.

Loads Dataset pickles and prediction JSON files, computes correlations between
GT and eval predictions, and writes summary JSONs. The script has two modes:

* **Discovery (default).** With no ``--config``/``--dataset``/``--predictions-dir``,
  it walks the standardized prediction layout under ``data/predictions/<env>/``
  (``GT_*`` and ``EVAL_*`` directories), pairs every existing GT/eval
  combination using the registry catalog, and evaluates each one in-process.
  Outputs land in the canonical evaluation tree under ``--output-root``.

* **Single config.** Given ``--config some.yaml`` (or the ``--dataset`` /
  ``--predictions-dir`` / ``--output-dir`` triple), it runs exactly one explicit
  evaluation. Used by the SLURM wrapper and for one-off comparisons.

Usage:
    # Evaluate everything discoverable under data/predictions/
    python scripts/pipeline/evaluate.py

    # Evaluate a single explicit comparison
    python scripts/pipeline/evaluate.py --config configs/evaluation_frozen_lake.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

from qval import PipelineEvaluationConfig
from qval.correlation import compute_correlation, compute_ranking_correlation
from qval.data_cache import load_dataset, resolve_dataset_path
from qval.experiment_logging import sanitize_for_json
from qval.prediction_store import (
    load_predictions_dir,
    load_predictions_from_sources,
)
from qval.ranking_identity import (
    build_ranking_coverage_metadata,
    ranking_point_hash,
)
from qval.registry import Signal, active_registry, add_catalog_arg, set_active
from qval.registry import applicability, naming
from qval.transforms import sum_per_trajectory, transform_predictions
from qval.types import CorrelationMethod, SignalType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

format_evaluation_output_dir_name = naming.format_evaluation_output_dir_name


# --- Vocabularies, datasets, and the method matrix are all derived from the
#     active registry catalog (the single source of truth, src/qval/registry).
#     Edit the catalog there, not these module constants. The catalog-independent
#     constants are bound here directly; everything keyed by the catalog is
#     (re)built by ``_rebuild_registry_tables()`` — once at import for the bundled
#     default, and again in ``main()`` after a ``--catalog`` selection. The
#     discovery parsers and ``build_evaluation_jobs`` read these module globals at
#     call time, so re-pointing the catalog reassigns them (not just the defaults).
MODALITIES = naming.MODALITIES
NULL_PREDICTION_REPORT_THRESHOLD = 0.10

SIGNAL_DISPLAY_NAMES = {
    "qv": "q-value",
    "sv": "state-value",
}
SIGNAL_ASSUMPTIONS = {s.value: s.config_signal_type for s in Signal}
POINTWISE_SIGNALS = tuple(SIGNAL_DISPLAY_NAMES)
RANKING_QV_KEY = applicability.RANKING_QV_KEY
CODEGEN_REPORT_METHOD = "codegen"


def _rebuild_registry_tables() -> None:
    """(Re)derive every catalog-backed module constant from the active catalog.

    Called once at import (binding the bundled default) and again in ``main()``
    after ``--catalog`` selection. The discovery functions and the
    ``build_evaluation_jobs`` table arguments read these module globals at call
    time, so re-pointing the catalog reaches all of them.
    """
    global ENVIRONMENTS, ENV_DISPLAY_NAMES, OPEN_APPS_DATASET_ID
    global _LLM_MODELS, _ALL_LLM_ACTORS, LLM_ACTORS, GT_ACTORS
    global VISION_BACKBONE_ACTORS, PRETRAINED_ACTORS, ALL_EVAL_ACTORS, ALL_ACTORS
    global _codegen_spec, CODEGEN_SAMPLE_METHODS, _CODEGEN_COMPONENT_METHODS
    global EXPECTED_ENV_DATASETS, EXPECTED_ACTORS_BY_ENV_MODALITY
    global EXPECTED_GT_ACTORS_BY_ENV_DATASET, RANKING_GT_ACTORS_BY_ENV_DATASET
    global DATASET_PATHS, OUTPUT_DATASET_IDS

    registry = active_registry()
    ENVIRONMENTS = tuple(e.env for e in registry.environments)
    ENV_DISPLAY_NAMES = {e.env: e.display for e in registry.environments}
    _open_apps = registry.env_by_name("open_apps")
    OPEN_APPS_DATASET_ID = _open_apps.dataset_id if _open_apps is not None else ""

    # LLM evaluation actors (not ground-truth-only, not an embedding/pretrained
    # actor class). ``include_in_results`` excludes a model from results discovery.
    _LLM_MODELS = tuple(
        m for m in registry.models if not m.is_gt_only and not m.is_non_llm
    )
    _ALL_LLM_ACTORS = tuple(m.actor for m in _LLM_MODELS)
    LLM_ACTORS = tuple(m.actor for m in _LLM_MODELS if m.include_in_results)
    GT_ACTORS = registry.gt_actors
    VISION_BACKBONE_ACTORS = tuple(
        m.actor for m in registry.models if m.is_embedding_backbone
    )
    PRETRAINED_ACTORS = tuple(m.actor for m in registry.models if m.is_pretrained)
    ALL_EVAL_ACTORS = LLM_ACTORS + VISION_BACKBONE_ACTORS + PRETRAINED_ACTORS
    ALL_ACTORS = ALL_EVAL_ACTORS + GT_ACTORS

    # A codegen run emits ``codegen-mean`` plus ``codegen-s0..codegen-s{N-1}``; N
    # is the registry's sample count. ``default_expected_methods`` expands the
    # single ``codegen`` method base into these on-disk discovery components.
    _codegen_spec = registry.method_by_base("codegen")
    CODEGEN_SAMPLE_METHODS = tuple(
        f"codegen-s{i}"
        for i in range((_codegen_spec.num_samples or 0) if _codegen_spec else 0)
    )
    _CODEGEN_COMPONENT_METHODS = ("codegen-mean", *CODEGEN_SAMPLE_METHODS)

    EXPECTED_ENV_DATASETS = {e.env: (e.dataset_id,) for e in registry.environments}
    EXPECTED_ACTORS_BY_ENV_MODALITY = {
        e.env: {"text": e.text_actors, "vision": e.vision_actors}
        for e in registry.environments
    }
    EXPECTED_GT_ACTORS_BY_ENV_DATASET = {
        (e.env, e.dataset_id): e.gt_actors for e in registry.environments
    }
    RANKING_GT_ACTORS_BY_ENV_DATASET = {
        (e.env, e.dataset_id): e.ranking_gt_actors
        for e in registry.environments
        if e.ranking_gt_actors
    }
    DATASET_PATHS = {
        (e.env, e.dataset_id): Path(e.dataset_path) for e in registry.environments
    }
    OUTPUT_DATASET_IDS = {
        (e.env, e.dataset_id): e.dataset_id for e in registry.environments
    }


_rebuild_registry_tables()


# ---------------------------------------------------------------------------
# Correlation engine: load one dataset + a set of predictions, compute
# correlations, write one summary JSON. Used by both run modes.
# ---------------------------------------------------------------------------


def _ranking_candidate_counts(ranking_points: list[Any]) -> list[int]:
    """Return the candidate count for each ranking point."""
    return [len(point.candidates) for point in ranking_points]


class _RankingDatasetIndex(NamedTuple):
    ranking_points: list[Any]
    candidate_counts: list[int]
    point_hashes: list[str]
    hash_to_index: dict[str, int]


class _ResolvedRankingPrediction(NamedTuple):
    candidate_counts: list[int]
    dataset_indices: list[int]
    ranking_point_hashes: list[str] | None
    legacy_mode: bool


def _build_ranking_dataset_index(ranking_points: list[Any]) -> _RankingDatasetIndex:
    """Index the current dataset's ranking points by stable identity hash."""
    candidate_counts = _ranking_candidate_counts(ranking_points)
    point_hashes: list[str] = []
    hash_to_index: dict[str, int] = {}
    for idx, point in enumerate(ranking_points):
        point_id = ranking_point_hash(point)
        if point_id in hash_to_index:
            raise ValueError(
                "Current dataset contains duplicate ranking point hash "
                f"{point_id!r} at indices {hash_to_index[point_id]} and {idx}."
            )
        point_hashes.append(point_id)
        hash_to_index[point_id] = idx
    return _RankingDatasetIndex(
        ranking_points=ranking_points,
        candidate_counts=candidate_counts,
        point_hashes=point_hashes,
        hash_to_index=hash_to_index,
    )


def _validate_ranking_prediction(
    pred: Any,
    ranking_points_or_index: list[Any] | _RankingDatasetIndex,
    *,
    label: str,
) -> _ResolvedRankingPrediction:
    """Resolve a ranking prediction's covered dataset points and chunk layout."""
    dataset_index = (
        ranking_points_or_index
        if isinstance(ranking_points_or_index, _RankingDatasetIndex)
        else _build_ranking_dataset_index(ranking_points_or_index)
    )
    candidate_counts = dataset_index.candidate_counts
    config = pred.config or {}
    metadata = pred.metadata or {}
    actual = len(pred.values)
    if actual == 0:
        raise ValueError(
            f"Invalid {label} ranking prediction {pred.method_name!r}: "
            "must cover at least one ranking point."
        )

    counts_raw = config.get("candidate_counts")
    total_candidates = config.get("total_candidates")
    if counts_raw is not None:
        pred_counts = [int(count) for count in counts_raw]
        if not pred_counts:
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                "must cover at least one ranking point."
            )
        if any(count <= 0 for count in pred_counts):
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                "candidate_counts must all be positive."
            )
        if sum(pred_counts) != actual:
            raise ValueError(
                f"Ranking prediction {pred.method_name!r} is internally inconsistent: "
                f"config.candidate_counts sums to {sum(pred_counts)} "
                f"but values has length {actual}."
            )
    else:
        pred_counts = None

    if total_candidates is not None and int(total_candidates) != actual:
        raise ValueError(
            f"Ranking prediction {pred.method_name!r} is internally inconsistent: "
            f"config.total_candidates={total_candidates} but values has length {actual}."
        )

    point_hashes_raw = metadata.get("ranking_point_hashes")
    if point_hashes_raw is not None:
        if pred_counts is None:
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                "ranking_point_hashes require config.candidate_counts."
            )
        point_hashes = [str(value) for value in point_hashes_raw]
        if len(point_hashes) != len(pred_counts):
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                f"ranking_point_hashes has length {len(point_hashes)} but "
                f"candidate_counts has length {len(pred_counts)}."
            )
        if len(set(point_hashes)) != len(point_hashes):
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                "ranking_point_hashes contains duplicate hash entries."
            )
        dataset_indices: list[int] = []
        matched_counts: list[int] = []
        for point_id in point_hashes:
            if point_id not in dataset_index.hash_to_index:
                raise ValueError(
                    f"Invalid {label} ranking prediction {pred.method_name!r}: "
                    f"ranking point hash {point_id!r} is not present in the "
                    "current dataset."
                )
            idx = dataset_index.hash_to_index[point_id]
            dataset_indices.append(idx)
            matched_counts.append(candidate_counts[idx])
        if matched_counts != pred_counts:
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                f"config.candidate_counts={pred_counts} do not match the "
                f"current dataset's matched counts {matched_counts}."
            )
        current_metadata = build_ranking_coverage_metadata(
            metadata.get("dataset_fingerprint"),
            [dataset_index.ranking_points[idx] for idx in dataset_indices],
        )
        if (
            metadata.get("ranking_points_hash") is not None
            and metadata["ranking_points_hash"] != current_metadata["ranking_points_hash"]
        ):
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                "ranking_points_hash does not match the current dataset coverage."
            )
        return _ResolvedRankingPrediction(
            candidate_counts=pred_counts,
            dataset_indices=dataset_indices,
            ranking_point_hashes=point_hashes,
            legacy_mode=False,
        )

    logger.warning(
        "%s ranking prediction %r lacks metadata.ranking_point_hashes; "
        "subset/order identity cannot be verified. Falling back to legacy "
        "dataset-prefix validation. Regenerate the artifact for subset-safe "
        "evaluation.",
        label,
        pred.method_name,
    )

    if pred_counts is not None:
        if len(pred_counts) > len(candidate_counts):
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: covers "
                f"{len(pred_counts)} ranking points but dataset only has "
                f"{len(candidate_counts)}."
            )
        dataset_prefix = candidate_counts[: len(pred_counts)]
        if pred_counts != dataset_prefix:
            raise ValueError(
                f"Invalid {label} ranking prediction {pred.method_name!r}: "
                f"legacy prefix candidate_counts={pred_counts} do not match "
                f"the dataset prefix {dataset_prefix}. Without ranking point "
                "hashes, non-prefix subsets cannot be verified."
            )
        return _ResolvedRankingPrediction(
            candidate_counts=pred_counts,
            dataset_indices=list(range(len(pred_counts))),
            ranking_point_hashes=None,
            legacy_mode=True,
        )

    cum = 0
    for idx, count in enumerate(candidate_counts):
        cum += count
        if cum == actual:
            return _ResolvedRankingPrediction(
                candidate_counts=list(candidate_counts[: idx + 1]),
                dataset_indices=list(range(idx + 1)),
                ranking_point_hashes=None,
                legacy_mode=True,
            )
    raise ValueError(
        f"Invalid {label} ranking prediction {pred.method_name!r}: "
        f"got {actual} flattened values, which matches no prefix of the dataset's "
        f"candidate counts (total points: {len(candidate_counts)}, "
        f"full sum: {sum(candidate_counts)}, full list: {candidate_counts})"
    )


def _ranking_scores_to_ranks(
    values: list[float],
    candidate_counts: list[int],
) -> list[float]:
    """Convert flattened per-candidate scores to flattened per-candidate ranks."""
    import math

    import numpy as np
    from scipy import stats

    expected = sum(candidate_counts)
    if len(values) != expected:
        raise ValueError(
            f"ranking scores length mismatch: expected {expected}, got {len(values)}"
        )

    flat_ranks: list[float] = []
    offset = 0
    for width in candidate_counts:
        chunk = values[offset : offset + width]
        offset += width
        if any(math.isnan(value) for value in chunk):
            flat_ranks.extend([float("nan")] * width)
            continue
        ranks = stats.rankdata(-np.array(chunk, dtype=float), method="average")
        flat_ranks.extend(float(rank) for rank in ranks)
    return flat_ranks

_DEFAULT_CORRELATION_METHODS = (
    CorrelationMethod.PEARSON,
    CorrelationMethod.SPEARMAN,
    CorrelationMethod.KENDALL_TAU,
    CorrelationMethod.SIGN_AGREEMENT,
)
_MIN_USABLE_CORRELATION_POINTS = 50


def _catalog_root_for_config(config_path: Path) -> Path:
    """Resolve the catalog root that owns an evaluation config.

    Hand-maintained evaluation configs live at
    ``<catalog_root>/configs/evaluation/<env>/<file>.yaml``, alongside the
    catalog's ``catalog.py``. Their ``output_dir`` and ``prediction_sources``
    paths (e.g. ``data/{evaluations,predictions}/<env>/<dir>``) are
    catalog-relative, so they join this root. Walk up to the directory
    containing ``catalog.py``; if none is found (an ad-hoc config outside any
    catalog) fall back to the current working directory, preserving the
    historical CWD-relative behaviour.
    """
    for parent in config_path.resolve().parents:
        if (parent / "catalog.py").is_file():
            return parent
    return Path.cwd()


def evaluate_from_config(
    eval_config: PipelineEvaluationConfig | None,
    *,
    dataset_arg: str | None = None,
    predictions_dir_arg: str | None = None,
    output_dir_arg: str | None = None,
    config_path: Path | None = None,
) -> None:
    """Run one evaluation: load a dataset + predictions, correlate, write summary.

    Inputs come from ``eval_config`` (a :class:`PipelineEvaluationConfig`) and/or
    the explicit CLI overrides ``dataset_arg`` / ``predictions_dir_arg`` /
    ``output_dir_arg``. At least a dataset path and an output directory must be
    resolvable from one of these sources.

    Path resolution follows the three-bucket model: ``dataset_path`` is a shared
    input (its ``shared/`` prefix resolves repo-relative); ``output_dir`` and
    ``prediction_sources`` paths are catalog outputs that join the catalog root
    derived from ``config_path`` (the dir containing ``catalog.py``). CLI
    overrides stay CWD-relative. Absolute paths are preserved throughout, so the
    discovery path (which passes absolute paths and no ``config_path``) is
    unaffected.
    """
    catalog_root = (
        _catalog_root_for_config(config_path)
        if config_path is not None
        else Path.cwd()
    )

    dataset_path = dataset_arg or (eval_config.dataset_path if eval_config else None)
    if not dataset_path:
        raise ValueError("dataset_path is required (set in config or CLI)")

    resolved_dataset_path = resolve_dataset_path(dataset_path)
    logger.info("Loading dataset from %s", resolved_dataset_path)
    dataset = load_dataset(resolved_dataset_path)
    logger.info(
        "Loaded: %d eval points, %d trajectory returns",
        len(dataset.evaluation_points),
        len(dataset.trajectory_returns),
    )

    # 2. Load predictions
    if predictions_dir_arg:
        logger.info("Loading predictions from %s", predictions_dir_arg)
        all_predictions = load_predictions_dir(predictions_dir_arg)
        if not all_predictions:
            logger.error("No prediction files found in %s", predictions_dir_arg)
            return
    elif eval_config:
        all_predictions = load_predictions_from_sources(
            eval_config.prediction_sources, base=catalog_root
        )
    else:
        raise ValueError("predictions_dir or evaluation config is required")

    all_gt_preds = {
        k: v for k, v in all_predictions.items() if v.method_type == "gt"
    }
    all_eval_preds = {
        k: v for k, v in all_predictions.items() if v.method_type == "eval"
    }
    logger.info("GT predictions: %s", list(all_gt_preds.keys()))
    logger.info("Eval predictions: %s", list(all_eval_preds.keys()))

    if not all_gt_preds:
        logger.error("No GT predictions found")
        return
    has_gt_vs_gt = bool(
        eval_config is not None and eval_config.gt_vs_gt_comparisons
    )
    if not all_eval_preds and not has_gt_vs_gt:
        logger.error("No eval predictions found and no GT×GT comparisons configured")
        return

    # Split ranking predictions from pointwise ones
    ranking_gt_preds = {
        k: v
        for k, v in all_gt_preds.items()
        if v.config.get("prediction_format") == "ranking_q_values"
    }
    ranking_eval_preds = {
        k: v
        for k, v in all_eval_preds.items()
        if v.config.get("prediction_format") in {"ranking_predicted", "ranking_scores"}
    }

    gt_preds = {k: v for k, v in all_gt_preds.items() if k not in ranking_gt_preds}
    eval_preds = {
        k: v for k, v in all_eval_preds.items() if k not in ranking_eval_preds
    }

    # 3. Parse correlation methods
    if eval_config and eval_config.correlation_methods:
        correlation_methods = eval_config.correlation_methods
    else:
        # Try from dataset config, fall back to defaults
        config_methods = dataset.config.get("correlation_methods")
        if config_methods:
            correlation_methods = tuple(
                CorrelationMethod[m.upper()] for m in config_methods
            )
        else:
            correlation_methods = _DEFAULT_CORRELATION_METHODS

    has_explicit_comparisons = (
        eval_config is not None and eval_config.comparisons is not None
    )
    exclude_zero_points = bool(
        eval_config is not None and eval_config.exclude_zero_points
    )

    if eval_preds:
        if has_explicit_comparisons:
            # Per-pair signal type: resolve from each eval prediction's config.
            # Use a default for experiment_info only.
            signal_type = _resolve_signal_type_lenient(eval_preds)
        else:
            signal_type = _resolve_signal_type(
                config_signal_type=(
                    eval_config.signal_type.name.lower()
                    if eval_config and eval_config.signal_type is not None
                    else None
                ),
                eval_predictions=eval_preds,
            )
    elif eval_config and eval_config.signal_type is not None:
        signal_type = eval_config.signal_type
    elif ranking_eval_preds:
        signal_type = SignalType.Q_VALUE
    elif has_gt_vs_gt:
        signal_type = _resolve_signal_type_from_gt(
            gt_preds, eval_config.gt_vs_gt_comparisons
        )
    else:
        raise ValueError(
            "No eval predictions available for pointwise correlations, "
            "no ranking predictions, and no GT×GT comparisons configured"
        )

    points = dataset.evaluation_points
    num_trajectories = len(dataset.trajectory_returns)

    # 4. Compute correlations
    results_summary: dict[str, dict] = {}
    comparison_pairs: list[tuple[str, Any, str, Any]] = []
    if gt_preds and eval_preds:
        comparison_pairs = list(
            _iter_comparison_pairs(
                gt_preds,
                eval_preds,
                comparisons=(
                    tuple((c.gt, c.eval) for c in eval_config.comparisons)
                    if has_explicit_comparisons
                    else None
                ),
            )
        )

        grouped_pairs: dict[str, list[tuple[str, Any]]] = {}
        for gt_name, gt_pred, eval_name, eval_pred in comparison_pairs:
            grouped_pairs.setdefault(gt_name, []).append((eval_name, eval_pred))

        for gt_name, gt_pred in gt_preds.items():
            selected_eval_pairs = grouped_pairs.get(gt_name, [])
            if not selected_eval_pairs:
                continue
            gt_assumption_str = gt_pred.config.get("assumption", "state_value")
            gt_assumption = SignalType[gt_assumption_str.upper()]

            logger.info("GT: %s (assumption=%s)", gt_name, gt_assumption.name)

            if gt_assumption == SignalType.SHAPED_REWARD:
                # Trajectory-level comparison: sum eval predictions per trajectory
                gt_values = dataset.trajectory_returns
                methods_data: dict[str, dict] = {}

                for eval_name, eval_pred in selected_eval_pairs:
                    if len(eval_pred.values) != len(points):
                        raise ValueError(
                            f"SHAPED_REWARD comparison ({gt_name!r}, {eval_name!r}): "
                            f"eval prediction has {len(eval_pred.values)} values but "
                            f"dataset has {len(points)} evaluation points. Prefix "
                            f"predictions (first N points) are not supported for "
                            f"SHAPED_REWARD yet — trajectory boundaries must be "
                            f"resolved first. See TODO.md."
                        )
                    traj_sums = sum_per_trajectory(
                        eval_pred.values,
                        points,
                        num_trajectories,
                    )
                    corr_data, num_dropped = _correlations_with_optional_filter(
                        gt_values,
                        traj_sums,
                        correlation_methods,
                        gt_name,
                        eval_name,
                        exclude_zero_points=exclude_zero_points,
                    )
                    methods_data[eval_name] = {
                        "correlations": corr_data,
                        "predicted_trajectory_sums": traj_sums,
                        "num_dropped_zero": num_dropped,
                    }

                methods_data = _aggregate_sample_groups(methods_data, eval_preds)
                results_summary[gt_name] = {
                    "assumption": gt_assumption.name.lower(),
                    "granularity": "trajectory",
                    "ground_truth_values": gt_values,
                    "methods": methods_data,
                }
            else:
                # Point-level comparison. GT and eval may both be prefixes of
                # the dataset and may differ in length; both are assumed to
                # cover the first N evaluation points (sampling_strategy="first"
                # at prediction time). Each pair is compared on
                # min(len(gt), len(eval)) values; the longer side is truncated
                # to that prefix.
                gt_values = gt_pred.values
                methods_data = {}

                for eval_name, eval_pred in selected_eval_pairs:
                    n = len(eval_pred.values)
                    comparison_n = min(len(gt_values), n)
                    if comparison_n > len(points):
                        logger.warning(
                            "Comparison (%r, %r) has %d shared prediction values "
                            "but dataset has only %d evaluation points. Truncating "
                            "to the dataset prefix.",
                            gt_name,
                            eval_name,
                            comparison_n,
                            len(points),
                        )
                        comparison_n = len(points)
                    if comparison_n == 0:
                        raise ValueError(
                            f"Empty comparison for ({gt_name!r}, {eval_name!r}): "
                            f"GT covers {len(gt_values)} points, eval covers {n}."
                        )
                    if len(gt_values) != n:
                        logger.info(
                            "Length mismatch for (%r, %r): GT=%d, eval=%d. "
                            "Truncating both to first %d values.",
                            gt_name, eval_name, len(gt_values), n, comparison_n,
                        )
                    eval_values = (
                        eval_pred.values
                        if n == comparison_n
                        else eval_pred.values[:comparison_n]
                    )
                    gt_values_slice = (
                        gt_values
                        if len(gt_values) == comparison_n
                        else gt_values[:comparison_n]
                    )
                    points_slice = (
                        points if comparison_n == len(points)
                        else points[:comparison_n]
                    )

                    # Per-pair signal type: resolve from eval prediction config
                    if has_explicit_comparisons:
                        eval_st_str = str(
                            eval_pred.config.get("signal_type", "state_value")
                        )
                        pair_signal_type = SignalType[eval_st_str.upper()]
                    else:
                        pair_signal_type = signal_type
                    # Transform eval predictions based on signal_type -> assumption
                    transformed = transform_predictions(
                        eval_values,
                        points_slice,
                        pair_signal_type,
                        gt_assumption,
                    )
                    corr_data, num_dropped = _correlations_with_optional_filter(
                        gt_values_slice,
                        transformed,
                        correlation_methods,
                        gt_name,
                        eval_name,
                        exclude_zero_points=exclude_zero_points,
                    )
                    methods_data[eval_name] = {
                        "correlations": corr_data,
                        "predicted_values": eval_pred.values,
                        "num_dropped_zero": num_dropped,
                    }

                methods_data = _aggregate_sample_groups(methods_data, eval_preds)
                results_summary[gt_name] = {
                    "assumption": gt_assumption.name.lower(),
                    "granularity": "point",
                    "ground_truth_values": gt_values,
                    "methods": methods_data,
                }
    elif ranking_eval_preds and ranking_gt_preds:
        logger.info("Skipping pointwise correlations; only ranking predictions found")

    # 4a-bis. GT×GT comparisons (explicit opt-in via config.gt_vs_gt_comparisons)
    gt_vs_gt_summary: dict[str, dict] = {}
    gt_vs_gt_pairs: list[tuple[str, Any, str, Any]] = []
    if has_gt_vs_gt:
        gt_vs_gt_pairs = list(
            _iter_gt_vs_gt_pairs(
                gt_preds,
                comparisons=tuple(
                    (c.lhs, c.rhs) for c in eval_config.gt_vs_gt_comparisons
                ),
            )
        )
        grouped_gt_pairs: dict[str, list[tuple[str, Any]]] = {}
        for lhs_name, _, rhs_name, rhs_pred in gt_vs_gt_pairs:
            grouped_gt_pairs.setdefault(lhs_name, []).append((rhs_name, rhs_pred))

        for lhs_name, rhs_list in grouped_gt_pairs.items():
            lhs_pred = gt_preds[lhs_name]
            lhs_assumption_str = lhs_pred.config.get("assumption", "state_value")
            lhs_assumption = SignalType[lhs_assumption_str.upper()]
            if lhs_assumption == SignalType.SHAPED_REWARD:
                raise ValueError(
                    f"GT×GT comparison for {lhs_name!r}: SHAPED_REWARD assumption "
                    "is not supported for GT×GT correlations."
                )
            logger.info("GT×GT lhs: %s (assumption=%s)", lhs_name, lhs_assumption.name)
            methods_data: dict[str, dict] = {}
            for rhs_name, rhs_pred in rhs_list:
                rhs_assumption_str = rhs_pred.config.get("assumption", "state_value")
                rhs_assumption = SignalType[rhs_assumption_str.upper()]
                if rhs_assumption != lhs_assumption:
                    raise ValueError(
                        f"GT×GT comparison ({lhs_name!r}, {rhs_name!r}): "
                        f"assumption mismatch (lhs={lhs_assumption.name.lower()}, "
                        f"rhs={rhs_assumption.name.lower()}). Both GTs must share "
                        "the same assumption."
                    )
                if len(rhs_pred.values) != len(lhs_pred.values):
                    raise ValueError(
                        f"GT×GT comparison ({lhs_name!r}, {rhs_name!r}): "
                        f"length mismatch (lhs={len(lhs_pred.values)}, "
                        f"rhs={len(rhs_pred.values)}). Both GTs must cover "
                        "the same evaluation points."
                    )
                corr_data, num_dropped = _correlations_with_optional_filter(
                    lhs_pred.values,
                    rhs_pred.values,
                    correlation_methods,
                    lhs_name,
                    rhs_name,
                    exclude_zero_points=exclude_zero_points,
                )
                methods_data[rhs_name] = {
                    "correlations": corr_data,
                    "predicted_values": rhs_pred.values,
                    "num_dropped_zero": num_dropped,
                }
            gt_vs_gt_summary[lhs_name] = {
                "assumption": lhs_assumption.name.lower(),
                "granularity": "point",
                "ground_truth_values": lhs_pred.values,
                "methods": methods_data,
            }

    # 4b. Ranking evaluations — detect via prediction_format
    if ranking_gt_preds and ranking_eval_preds:
        ranking_dataset_index = _build_ranking_dataset_index(dataset.ranking_points)
        resolved_ranking_gt = {
            gt_name: _validate_ranking_prediction(
                gt_pred,
                ranking_dataset_index,
                label="gt",
            )
            for gt_name, gt_pred in ranking_gt_preds.items()
        }
        resolved_ranking_eval = {
            eval_name: _validate_ranking_prediction(
                eval_pred,
                ranking_dataset_index,
                label="eval",
            )
            for eval_name, eval_pred in ranking_eval_preds.items()
        }
        for gt_name, gt_pred in ranking_gt_preds.items():
            gt_coverage = resolved_ranking_gt[gt_name]

            methods_data = {}
            for eval_name, eval_pred in ranking_eval_preds.items():
                eval_coverage = resolved_ranking_eval[eval_name]

                # Allow prefix-truncation when one side has fewer points than the
                # other. Required because predictions can be generated against
                # different prefix lengths of the same dataset (the typical knob
                # is "first N points"). The shared prefix must agree exactly on
                # dataset_indices and candidate_counts; non-prefix mismatches
                # still raise. When both sides have ranking_point_hashes, the
                # prefix is hash-verified — otherwise it's a load-bearing
                # assumption ("eval was generated on a prefix of GT's dataset")
                # logged at WARNING.
                common_n = min(
                    len(eval_coverage.dataset_indices),
                    len(gt_coverage.dataset_indices),
                )
                if (
                    eval_coverage.dataset_indices[:common_n]
                        != gt_coverage.dataset_indices[:common_n]
                    or eval_coverage.candidate_counts[:common_n]
                        != gt_coverage.candidate_counts[:common_n]
                ):
                    raise ValueError(
                        f"Ranking coverage mismatch for ({gt_name!r}, {eval_name!r}): "
                        f"GT resolves to dataset indices {gt_coverage.dataset_indices} "
                        f"with counts={gt_coverage.candidate_counts}; eval resolves to "
                        f"{eval_coverage.dataset_indices} with counts="
                        f"{eval_coverage.candidate_counts}. The first "
                        f"min(len(eval), len(gt))={common_n} ranking points must "
                        "agree exactly (eval is expected to be a prefix of GT)."
                    )

                common_counts = eval_coverage.candidate_counts[:common_n]
                common_flat = sum(common_counts)
                gt_values_used = gt_pred.values[:common_flat]
                eval_values_used = eval_pred.values[:common_flat]

                if (
                    len(gt_coverage.dataset_indices) != common_n
                    or len(eval_coverage.dataset_indices) != common_n
                ):
                    longer_side = (
                        "gt" if len(gt_coverage.dataset_indices) > common_n else "eval"
                    )
                    longer_n = max(
                        len(eval_coverage.dataset_indices),
                        len(gt_coverage.dataset_indices),
                    )
                    hashes_verified = (
                        gt_coverage.ranking_point_hashes is not None
                        and eval_coverage.ranking_point_hashes is not None
                    )
                    logger.warning(
                        "Truncating %s ranking prediction (%d -> %d points) for "
                        "(%r, %r): %s",
                        longer_side, longer_n, common_n,
                        gt_name, eval_name,
                        "shared prefix is hash-verified."
                        if hashes_verified
                        else "shared prefix assumed (no ranking_point_hashes); "
                             "regenerate artifacts with hashes for verified subset.",
                    )

                if eval_pred.config.get("prediction_format") == "ranking_scores":
                    eval_values_used = _ranking_scores_to_ranks(
                        eval_values_used,
                        common_counts,
                    )

                ranking_corr = compute_ranking_correlation(
                    gt_values_used,
                    eval_values_used,
                    common_counts,
                )
                logger.info(
                    "  %s vs %s RANKING_SPEARMAN: mean=%.4f std=%.4f "
                    "(valid=%d, skipped=%d, coverage=%d of %d points)",
                    eval_name, gt_name,
                    ranking_corr["mean_spearman"],
                    ranking_corr["std_spearman"],
                    ranking_corr["num_valid"],
                    ranking_corr["num_skipped"],
                    common_n,
                    len(ranking_dataset_index.candidate_counts),
                )
                methods_data[eval_name] = {
                    "correlations": {
                        "ranking_spearman": {
                            "mean": ranking_corr["mean_spearman"],
                            "std": ranking_corr["std_spearman"],
                            "num_valid": ranking_corr["num_valid"],
                            "num_skipped": ranking_corr["num_skipped"],
                            "num_points": ranking_corr["num_points"],
                        },
                    },
                    "per_point_correlations": ranking_corr["per_point"],
                }

            if methods_data:
                results_summary[gt_name] = {
                    "assumption": gt_pred.config.get("assumption", "q_value"),
                    "granularity": "ranking",
                    "candidate_counts": gt_coverage.candidate_counts,
                    "methods": methods_data,
                }

    # 5. Save summary
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    elif eval_config and eval_config.output_dir:
        output_dir = catalog_root / eval_config.output_dir
    else:
        raise ValueError("output_dir is required (set in config or CLI)")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_info": {
            "dataset_source": str(resolved_dataset_path),
            "predictions_dir": str(predictions_dir_arg)
            if predictions_dir_arg
            else None,
            "env_name": dataset.metadata.get("env_name"),
            "signal_type": signal_type.name.lower(),
            "num_evaluation_points": len(points),
            "num_trajectories": num_trajectories,
            "gt_methods": list(all_gt_preds.keys()),
            "eval_methods": list(all_eval_preds.keys()),
            "comparisons": [
                {"gt": gt_name, "eval": eval_name}
                for gt_name, _, eval_name, _ in comparison_pairs
            ],
            "gt_vs_gt_comparisons": [
                {"lhs": lhs_name, "rhs": rhs_name}
                for lhs_name, _, rhs_name, _ in gt_vs_gt_pairs
            ],
            "correlation_methods": [m.name.lower() for m in correlation_methods],
            "exclude_zero_points": exclude_zero_points,
            "min_usable_correlation_points": _MIN_USABLE_CORRELATION_POINTS,
            "timestamp": datetime.now().astimezone().isoformat(),
        },
        "dataset_config": dataset.config,
        "dataset_metadata": dataset.metadata,
        "estimations": results_summary,
        "gt_vs_gt_estimations": gt_vs_gt_summary,
    }

    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_{ts}.json"
    with summary_path.open("w") as f:
        json.dump(sanitize_for_json(summary), f, indent=4)
    logger.info("Summary saved to %s", summary_path)


def _filter_zero_pairs(
    lhs: list[float],
    rhs: list[float],
) -> tuple[list[float], list[float], int]:
    """Drop paired entries where either side equals 0.0 exactly.

    Returns ``(filtered_lhs, filtered_rhs, num_dropped)``. Uses ``==`` so
    ``0.0``, ``-0.0``, and integer ``0`` are all treated as zero; any other
    value (including ``1e-12``) is kept.
    """
    filtered_lhs: list[float] = []
    filtered_rhs: list[float] = []
    dropped = 0
    for l_val, r_val in zip(lhs, rhs):
        if l_val == 0 or r_val == 0:
            dropped += 1
            continue
        filtered_lhs.append(l_val)
        filtered_rhs.append(r_val)
    return filtered_lhs, filtered_rhs, dropped


def _correlations_with_optional_filter(
    lhs_values: list[float],
    rhs_values: list[float],
    correlation_methods: tuple[CorrelationMethod, ...],
    lhs_name: str,
    rhs_name: str,
    *,
    exclude_zero_points: bool,
) -> tuple[dict[str, dict], int]:
    """Run ``_compute_correlations`` after optionally dropping zero-pairs."""
    if exclude_zero_points:
        lhs_v, rhs_v, dropped = _filter_zero_pairs(lhs_values, rhs_values)
    else:
        lhs_v, rhs_v, dropped = lhs_values, rhs_values, 0
    corr = _compute_correlations(
        lhs_v, rhs_v, correlation_methods, lhs_name, rhs_name
    )
    return corr, dropped


def _compute_correlations(
    gt_values: list[float],
    predicted: list[float],
    correlation_methods: tuple[CorrelationMethod, ...],
    gt_name: str,
    eval_name: str,
) -> dict[str, dict]:
    """Compute all correlations between GT and predicted values."""
    corr_data: dict[str, dict] = {}
    for corr_method in correlation_methods:
        try:
            usable_points = _count_usable_correlation_pairs(gt_values, predicted)
            if usable_points < _MIN_USABLE_CORRELATION_POINTS:
                corr_data[corr_method.name.lower()] = {
                    "correlation": float("nan"),
                    "p_value": float("nan"),
                    "num_points": usable_points,
                }
                logger.warning(
                    "  %s vs %s %s: insufficient usable points for correlation "
                    "(n=%d < %d); recording NaN",
                    eval_name,
                    gt_name,
                    corr_method.name,
                    usable_points,
                    _MIN_USABLE_CORRELATION_POINTS,
                )
                continue
            corr = compute_correlation(gt_values, predicted, corr_method)
            corr_data[corr_method.name.lower()] = {
                "correlation": corr.correlation,
                "p_value": corr.p_value,
                "num_points": corr.num_points,
            }
            logger.info(
                "  %s vs %s %s: r=%.4f p=%.6f (n=%d)",
                eval_name,
                gt_name,
                corr_method.name,
                corr.correlation,
                corr.p_value,
                corr.num_points,
            )
        except ValueError as e:
            logger.warning(
                "  Failed %s vs %s %s: %s",
                eval_name,
                gt_name,
                corr_method.name,
                e,
            )
    return corr_data


def _count_usable_correlation_pairs(
    ground_truth: list[float],
    predicted: list[float],
) -> int:
    """Count pairs that survive correlation-time NaN filtering."""
    if len(ground_truth) != len(predicted):
        msg = (
            f"ground_truth and predicted must have the same length, "
            f"got {len(ground_truth)} and {len(predicted)}"
        )
        raise ValueError(msg)
    return sum(
        1
        for g, p in zip(ground_truth, predicted)
        if not (math.isnan(g) or math.isnan(p))
    )


def _resolve_signal_type_lenient(eval_predictions: dict[str, Any]) -> SignalType:
    """Resolve a representative signal type from eval predictions.

    Used when explicit comparisons are specified and per-pair resolution
    is done instead. Returns the first signal type found, or STATE_VALUE
    as a fallback. Does NOT reject mixed signal types.
    """
    for pred in eval_predictions.values():
        st = pred.config.get("signal_type")
        if st is not None:
            return SignalType[str(st).upper()]
    return SignalType.STATE_VALUE


def _resolve_signal_type(
    *,
    config_signal_type: str | None,
    eval_predictions: dict[str, Any],
) -> SignalType:
    """Resolve the eval signal type from config or eval prediction artifacts."""
    if config_signal_type is not None:
        return SignalType[config_signal_type.upper()]

    signal_types = {
        str(pred.config.get("signal_type")).lower()
        for pred in eval_predictions.values()
        if pred.config.get("signal_type") is not None
    }

    if not signal_types:
        raise ValueError(
            "Could not determine eval signal_type. Set evaluation config "
            "signal_type or ensure eval prediction artifacts persist it."
        )
    if len(signal_types) > 1:
        raise ValueError(
            "Multiple eval signal types found in prediction artifacts: "
            f"{sorted(signal_types)}. Set evaluation config signal_type "
            "explicitly or evaluate one signal type at a time."
        )

    signal_type_str = next(iter(signal_types))
    return SignalType[signal_type_str.upper()]


def _resolve_signal_type_from_gt(
    gt_preds: dict[str, Any],
    gt_vs_gt_comparisons: tuple[Any, ...],
) -> SignalType:
    """Pick an experiment-info signal_type from the first GT×GT lhs assumption.

    Used only for the summary's ``experiment_info.signal_type`` field when no
    eval predictions exist and per-pair GT×GT correlations drive the run.
    Falls back to STATE_VALUE if the referenced lhs is missing an assumption.
    """
    for comparison in gt_vs_gt_comparisons:
        lhs = comparison.lhs
        if lhs in gt_preds:
            assumption = gt_preds[lhs].config.get("assumption", "state_value")
            return SignalType[str(assumption).upper()]
    return SignalType.STATE_VALUE


def _iter_gt_vs_gt_pairs(
    gt_preds: dict[str, Any],
    comparisons: tuple[tuple[str, str | tuple[str, ...]], ...] | None,
) -> list[tuple[str, Any, str, Any]]:
    """Return selected (lhs, rhs) GT/GT prediction pairs.

    Unlike ``_iter_comparison_pairs``, there is no Cartesian default: an
    empty or ``None`` ``comparisons`` returns ``[]``. Each entry's ``rhs`` can
    be a single method name or a tuple of names (1:N expansion).
    """
    if not comparisons:
        return []

    pairs: list[tuple[str, Any, str, Any]] = []
    for lhs_name, rhs_names in comparisons:
        if lhs_name not in gt_preds:
            raise ValueError(
                f"Unknown lhs GT method in gt_vs_gt_comparisons: {lhs_name!r}. "
                f"Available GT methods: {sorted(gt_preds)}"
            )
        if isinstance(rhs_names, str):
            rhs_iter: tuple[str, ...] = (rhs_names,)
        else:
            rhs_iter = tuple(rhs_names)
        for rhs_name in rhs_iter:
            if rhs_name not in gt_preds:
                raise ValueError(
                    f"Unknown rhs GT method in gt_vs_gt_comparisons: {rhs_name!r}. "
                    f"Available GT methods: {sorted(gt_preds)}"
                )
            pairs.append(
                (lhs_name, gt_preds[lhs_name], rhs_name, gt_preds[rhs_name])
            )
    return pairs


def _iter_comparison_pairs(
    gt_preds: dict[str, Any],
    eval_preds: dict[str, Any],
    comparisons: tuple[tuple[str, str | tuple[str, ...]], ...] | None,
) -> list[tuple[str, Any, str, Any]]:
    """Return selected GT/eval prediction pairs.

    When ``comparisons`` is ``None``, returns the full Cartesian product.
    Otherwise validates and returns only the requested pairs in order.
    Each comparison entry can specify a single eval name or a tuple of
    eval names (1:N expansion).
    """
    if comparisons is None:
        return [
            (gt_name, gt_pred, eval_name, eval_pred)
            for gt_name, gt_pred in gt_preds.items()
            for eval_name, eval_pred in eval_preds.items()
        ]

    pairs: list[tuple[str, Any, str, Any]] = []
    for gt_name, eval_names in comparisons:
        if gt_name not in gt_preds:
            raise ValueError(
                f"Unknown GT method in comparisons: {gt_name!r}. "
                f"Available GT methods: {sorted(gt_preds)}"
            )
        # Normalize: single string → iterable
        if isinstance(eval_names, str):
            eval_names_iter: tuple[str, ...] = (eval_names,)
        else:
            eval_names_iter = tuple(eval_names)
        for eval_name in eval_names_iter:
            if eval_name not in eval_preds:
                raise ValueError(
                    f"Unknown eval method in comparisons: {eval_name!r}. "
                    f"Available eval methods: {sorted(eval_preds)}"
                )
            pairs.append((gt_name, gt_preds[gt_name], eval_name, eval_preds[eval_name]))
    return pairs


def _aggregate_sample_correlations(
    per_sample_data: list[dict],
) -> dict[str, dict]:
    """Compute aggregate statistics across per-sample correlation results.

    Returns a dict keyed by correlation method name, each containing
    mean/std/min/max/median/num_samples/num_nan. NaN and None entries are
    filtered before computing statistics; ``num_nan`` counts how many per-
    sample entries were dropped. If all samples are NaN/None, every stat is
    NaN and ``num_samples`` is 0.
    """
    import math as _math

    by_method_valid: dict[str, list[float]] = {}
    by_method_dropped: dict[str, int] = {}
    for sample_entry in per_sample_data:
        corr_dict = sample_entry.get("correlations", {})
        for method_name, method_data in corr_dict.items():
            corr_val = method_data.get("correlation")
            if corr_val is None or (
                isinstance(corr_val, float) and _math.isnan(corr_val)
            ):
                by_method_dropped[method_name] = (
                    by_method_dropped.get(method_name, 0) + 1
                )
                by_method_valid.setdefault(method_name, [])
                continue
            by_method_valid.setdefault(method_name, []).append(corr_val)

    result: dict[str, dict] = {}
    for method_name, values in by_method_valid.items():
        n = len(values)
        num_nan = by_method_dropped.get(method_name, 0)
        if n == 0:
            result[method_name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "num_samples": 0,
                "num_nan": num_nan,
            }
            continue
        result[method_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "num_samples": n,
            "num_nan": num_nan,
        }
    return result


def _aggregate_sample_groups(
    methods_data: dict[str, dict],
    eval_preds: dict[str, Any],
) -> dict[str, dict]:
    """Group multi-sample predictions and replace with aggregate entries.

    Scans ``eval_preds`` for ``config.sample_group`` to identify grouped
    predictions. Groups member entries from ``methods_data``, replaces
    them with a single entry containing ``aggregate_correlations`` plus a
    ``per_sample`` list. Ungrouped methods pass through unchanged.
    """
    # Identify groups: sample_group -> list of member names
    groups: dict[str, list[str]] = {}
    group_num_samples: dict[str, int] = {}
    for name, pred in eval_preds.items():
        sample_group = pred.config.get("sample_group")
        if sample_group is not None and name in methods_data:
            groups.setdefault(sample_group, []).append(name)
            ns = pred.config.get("num_samples")
            if ns is not None:
                group_num_samples[sample_group] = ns

    if not groups:
        return methods_data

    grouped_members = set()
    for members in groups.values():
        grouped_members.update(members)

    result: dict[str, dict] = {}
    # Pass through ungrouped methods
    for name, data in methods_data.items():
        if name not in grouped_members:
            result[name] = data

    # Build aggregate entries
    for group_name, members in groups.items():
        # Sort by sample_index for deterministic ordering
        members.sort(
            key=lambda n: eval_preds[n].config.get("sample_index", 0)
        )
        per_sample = [methods_data[m] for m in members if m in methods_data]
        aggregate_corr = _aggregate_sample_correlations(per_sample)
        result[group_name] = {
            "num_samples": group_num_samples.get(group_name, len(per_sample)),
            "aggregate_correlations": aggregate_corr,
            "per_sample": per_sample,
        }

    return result


# ---------------------------------------------------------------------------
# Discovery: walk the standardized data/predictions/<env>/ tree, pair every
# existing GT/eval combination using the registry catalog, and evaluate each
# in-process via ``evaluate_from_config``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionDirInfo:
    env: str
    path: Path
    dataset_key: str
    dataset_id: str
    actor: str
    modality: str | None = None


@dataclass(frozen=True)
class MissingCombination:
    kind: str
    message: str
    env: str
    dataset_id: str | None = None
    gt_actor: str | None = None
    actor: str | None = None
    modality: str | None = None
    method: str | None = None
    signal: str | None = None


@dataclass(frozen=True)
class EvaluationJob:
    env: str
    dataset_id: str
    gt_actor: str
    actor: str
    modality: str
    dataset_path: Path
    gt_dir: Path
    eval_dir: Path
    gt_paths: tuple[Path, ...]
    eval_paths: tuple[Path, ...]
    output_dir: Path
    comparisons: Mapping[str, tuple[str, ...]]

    def config_dict(self, *, exclude_zero_points: bool = False) -> dict:
        gt_methods = sorted(self.comparisons)
        eval_methods = sorted(
            {eval_method for methods in self.comparisons.values() for eval_method in methods}
        )
        pointwise_comparisons = {
            gt_method: eval_methods_for_gt
            for gt_method, eval_methods_for_gt in self.comparisons.items()
            if not _is_ranking_gt_method(gt_method)
        }
        config = {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "prediction_sources": [
                {
                    "paths": [str(path) for path in self.gt_paths],
                    "type": "gt",
                    "include_methods": gt_methods,
                },
                {
                    "paths": [str(path) for path in self.eval_paths],
                    "type": "eval",
                    "include_methods": eval_methods,
                },
            ],
            "comparisons": [
                {"gt": gt_method, "eval": list(eval_methods_for_gt)}
                for gt_method, eval_methods_for_gt in pointwise_comparisons.items()
            ],
        }
        if exclude_zero_points:
            config["exclude_zero_points"] = True
        return config


ExpectedMethodsFn = Callable[[str, str], Mapping[str, tuple[str, ...]]]


@dataclass(frozen=True)
class PredictionSummary:
    method_name: str
    method_type: str
    config: Mapping[str, Any]
    path: Path
    signal: str | None = None
    prediction_format: str | None = None
    values_count: int | None = None
    null_count: int | None = None
    null_fraction: float | None = None
    null_percentage: float | None = None


@dataclass(frozen=True)
class ComparisonSelection:
    comparisons: Mapping[str, tuple[str, ...]]
    gt_paths: tuple[Path, ...]
    eval_paths: tuple[Path, ...]


@dataclass
class DetectedPredictionFile:
    path: str
    role: str
    status: str
    reason: str
    env: str
    dataset_id: str
    actor: str
    modality: str | None
    filename_method_name: str | None = None
    method_name: str | None = None
    method_type: str | None = None
    signal: str | None = None
    prediction_format: str | None = None
    timestamp: str | None = None
    selected_path: str | None = None
    values_count: int | None = None
    null_count: int | None = None
    null_fraction: float | None = None
    null_percentage: float | None = None
    evaluable: bool = False


def parse_eval_dir_name(path: Path) -> PredictionDirInfo | None:
    """Parse ``EVAL_<dataset>_<actor>_<modality>`` directory names."""
    name = path.name
    if not name.startswith("EVAL_"):
        return None
    rest = name[len("EVAL_") :]
    modality = None
    for candidate in MODALITIES:
        suffix = f"_{candidate}"
        if rest.endswith(suffix):
            modality = candidate
            rest = rest[: -len(suffix)]
            break
    if modality is None:
        return None
    parsed = _split_dataset_actor(rest, ALL_EVAL_ACTORS)
    if parsed is None:
        return None
    dataset_key, actor = parsed
    env = path.parent.name
    return PredictionDirInfo(
        env=env,
        path=path,
        dataset_key=dataset_key,
        dataset_id=_normalize_dataset_id(dataset_key),
        actor=actor,
        modality=modality,
    )


def parse_gt_dir_name(path: Path) -> PredictionDirInfo | None:
    """Parse ``GT_<dataset>_<actor>`` directory names."""
    name = path.name
    if not name.startswith("GT_"):
        return None
    rest = name[len("GT_") :]
    parsed = _split_dataset_actor(rest, ALL_ACTORS)
    if parsed is None:
        return None
    dataset_key, actor = parsed
    env = path.parent.name
    return PredictionDirInfo(
        env=env,
        path=path,
        dataset_key=dataset_key,
        dataset_id=_normalize_dataset_id(dataset_key),
        actor=actor,
    )


def default_expected_methods(actor: str, modality: str) -> Mapping[str, tuple[str, ...]]:
    """Return expected method bases by signal for an actor/modality pair.

    Thin adapter over the registry's :func:`applicability.expected_methods_for_actor`:
    it expands the single ``codegen`` method base into its per-sample on-disk
    components.
    """
    matrix = applicability.expected_methods_for_actor(actor, modality)
    return {
        key: _expand_codegen_components(bases) if key in POINTWISE_SIGNALS else bases
        for key, bases in matrix.items()
    }


def _expand_codegen_components(bases: tuple[str, ...]) -> tuple[str, ...]:
    """Replace the ``codegen`` method base with its on-disk discovery
    components (``codegen-mean`` + ``codegen-s0..codegen-s{N-1}``)."""
    expanded: list[str] = []
    for base in bases:
        if base == CODEGEN_REPORT_METHOD:
            expanded.extend(_CODEGEN_COMPONENT_METHODS)
        else:
            expanded.append(base)
    return tuple(expanded)


def build_evaluation_jobs(
    *,
    predictions_root: Path,
    output_root: Path,
    dataset_paths: Mapping[tuple[str, str], Path] | None = None,
    expected_env_datasets: Mapping[str, tuple[str, ...]] | None = None,
    expected_actors_by_env_modality: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ]
    | None = None,
    expected_gt_actors_by_env_dataset: Mapping[
        tuple[str, str], tuple[str, ...]
    ]
    | None = None,
    output_dataset_ids: Mapping[tuple[str, str], str] | None = None,
    expected_methods_fn: ExpectedMethodsFn | None = None,
    detected_files: list[DetectedPredictionFile] | None = None,
) -> tuple[list[EvaluationJob], list[MissingCombination]]:
    """Discover evaluable experiments and missing standardized combinations.

    The table arguments default to the module globals (which follow the active
    catalog) resolved at *call* time, so a ``--catalog`` selection that rebuilt
    them in ``main()`` is honoured without re-binding default arguments.
    """
    if dataset_paths is None:
        dataset_paths = DATASET_PATHS
    if expected_env_datasets is None:
        expected_env_datasets = EXPECTED_ENV_DATASETS
    if expected_actors_by_env_modality is None:
        expected_actors_by_env_modality = EXPECTED_ACTORS_BY_ENV_MODALITY
    if expected_gt_actors_by_env_dataset is None:
        expected_gt_actors_by_env_dataset = EXPECTED_GT_ACTORS_BY_ENV_DATASET
    if output_dataset_ids is None:
        output_dataset_ids = OUTPUT_DATASET_IDS
    if expected_methods_fn is None:
        expected_methods_fn = default_expected_methods
    predictions_root = Path(predictions_root)
    output_root = Path(output_root)

    eval_dirs = _discover_eval_dirs(predictions_root)
    gt_dirs = _discover_gt_dirs(predictions_root)
    missing: list[MissingCombination] = []
    jobs: list[EvaluationJob] = []

    expected_keys: set[tuple[str, str, str, str]] = set()
    expected_dataset_pairs: set[tuple[str, str]] = set()
    for env, dataset_ids in expected_env_datasets.items():
        for dataset_id in dataset_ids:
            expected_dataset_pairs.add((env, dataset_id))
            modality_map = expected_actors_by_env_modality.get(env, {})
            for modality, actors in modality_map.items():
                for actor in actors:
                    expected_keys.add((env, dataset_id, actor, modality))

    discovered_expected_eval_keys: set[tuple[str, str, str, str]] = set()
    for key in sorted(eval_dirs):
        env, dataset_id, actor, modality = key
        if (env, dataset_id) in expected_dataset_pairs:
            discovered_expected_eval_keys.add(key)
        else:
            missing.append(
                _unexpected_prediction_dir(
                    kind="unexpected_eval_dir",
                    env=env,
                    dataset_id=dataset_id,
                    actor=actor,
                    modality=modality,
                )
            )

    for env, dataset_id, actor in sorted(gt_dirs):
        if (env, dataset_id) not in expected_dataset_pairs:
            missing.append(
                _unexpected_prediction_dir(
                    kind="unexpected_gt_dir",
                    env=env,
                    dataset_id=dataset_id,
                    actor=actor,
                    modality=None,
                )
            )

    candidate_keys = expected_keys | discovered_expected_eval_keys
    for env, dataset_id, actor, modality in sorted(candidate_keys):
        eval_info = eval_dirs.get((env, dataset_id, actor, modality))
        if eval_info is None:
            missing.append(
                _missing_eval_dir(env, dataset_id, actor, modality)
            )
            continue

        dataset_path = dataset_paths.get((env, dataset_id))
        if dataset_path is None:
            missing.append(
                MissingCombination(
                    kind="dataset",
                    message=(
                        f"Dataset path missing for environment {_display_env(env)} "
                        f"and dataset {dataset_id!r}; skipping {actor} {modality}."
                    ),
                    env=env,
                    dataset_id=dataset_id,
                    actor=actor,
                    modality=modality,
                )
            )
            continue

        gt_infos, missing_gt_actors = _select_gt_dirs(
            env,
            dataset_id,
            gt_dirs,
            expected_gt_actors_by_env_dataset,
        )
        for gt_actor in missing_gt_actors:
            missing.append(
                _missing_gt_dir(
                    env,
                    dataset_id,
                    actor,
                    modality,
                    gt_actor=gt_actor,
                )
            )
        if not gt_infos:
            if not missing_gt_actors:
                missing.append(
                    _missing_gt_dir(
                        env,
                        dataset_id,
                        actor,
                        modality,
                        gt_actor=None,
                    )
                )
            continue

        for gt_info in gt_infos:
            selection = _build_comparisons_for_dirs(
                gt_info,
                eval_info,
                _expected_methods_for_gt_actor(
                    expected_methods_fn(actor, modality),
                    env=env,
                    dataset_id=dataset_id,
                    gt_actor=gt_info.actor,
                ),
                missing,
                detected_files,
            )
            if not selection.comparisons:
                logger.warning(
                    "No existing qv/sv or ranking comparisons for %s %s GT %s "
                    "vs %s %s; skipping.",
                    env,
                    dataset_id,
                    gt_info.actor,
                    actor,
                    modality,
                )
                continue

            output_dataset_id = output_dataset_ids.get((env, dataset_id), dataset_id)
            jobs.append(
                EvaluationJob(
                    env=env,
                    dataset_id=dataset_id,
                    gt_actor=gt_info.actor,
                    actor=actor,
                    modality=modality,
                    dataset_path=Path(dataset_path),
                    gt_dir=gt_info.path,
                    eval_dir=eval_info.path,
                    gt_paths=selection.gt_paths,
                    eval_paths=selection.eval_paths,
                    output_dir=(
                        output_root
                        / env
                        / format_evaluation_output_dir_name(
                            output_dataset_id,
                            gt_actor=gt_info.actor,
                            eval_actor=actor,
                            modality=modality,
                        )
                    ),
                    comparisons=selection.comparisons,
                )
            )

    return jobs, _collapse_reported_missing_methods(list(dict.fromkeys(missing)))


def run_evaluation_jobs(
    jobs: list[EvaluationJob],
    *,
    dry_run: bool,
    exclude_zero_points: bool,
    verbose_configs: bool = False,
) -> int:
    """Run one in-process evaluation per discovered job.

    Each job's config dict is turned into a :class:`PipelineEvaluationConfig`
    and passed to :func:`evaluate_from_config` directly — no temporary files
    and no subprocess. A job that raises is logged and counted as a failure so
    one bad experiment does not abort the rest of the batch.
    """
    failures = 0
    for job in jobs:
        config = job.config_dict(exclude_zero_points=exclude_zero_points)
        logger.info(
            "%s %s %s GT %s vs %s %s -> %s",
            "Would evaluate" if dry_run else "Evaluating",
            job.env,
            job.dataset_id,
            job.gt_actor,
            job.actor,
            job.modality,
            job.output_dir,
        )
        if dry_run:
            if verbose_configs:
                logger.info("Dry run config: %s", json.dumps(config, sort_keys=True))
            continue
        try:
            evaluate_from_config(PipelineEvaluationConfig.from_dict(config))
        except Exception:
            failures += 1
            logger.exception(
                "Evaluation failed for %s %s %s %s",
                job.env,
                job.dataset_id,
                job.actor,
                job.modality,
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: Compute correlations between GT and eval predictions. "
            "With no single-evaluation arguments, discovers and evaluates every "
            "standardized experiment under --predictions-root in-process."
        )
    )
    # Single-evaluation mode (any of these selects it).
    parser.add_argument("--dataset", default=None, help="Path to dataset pickle")
    parser.add_argument(
        "--predictions-dir", default=None, help="Directory with prediction JSONs"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory for a single summary"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional evaluation config YAML for a single explicit evaluation",
    )
    # Discovery mode (the default).
    parser.add_argument(
        "--predictions-root",
        default=None,
        help=(
            "Root containing standardized prediction subfolders (discovery mode). "
            "Defaults to <catalog>/data/predictions."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root where per-experiment summary folders are written (discovery "
            "mode). Defaults to <catalog>/data/evaluations."
        ),
    )
    parser.add_argument(
        "--missing-report",
        default=None,
        help="Optional JSON path for missing-directory/missing-method report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered jobs and missing combinations without evaluating.",
    )
    parser.add_argument(
        "--verbose-configs",
        action="store_true",
        help="With --dry-run, print full generated per-job config dictionaries.",
    )
    parser.add_argument(
        "--exclude-zero-points",
        action="store_true",
        help="Pass exclude_zero_points through to each evaluation.",
    )
    add_catalog_arg(parser)
    args = parser.parse_args()

    # Select the catalog and rebuild every catalog-derived discovery table before
    # any discovery runs (the parsers and build_evaluation_jobs read them).
    ctx = set_active(args.catalog)
    _rebuild_registry_tables()

    single_mode = bool(args.config or args.dataset or args.predictions_dir)
    if single_mode:
        eval_config = (
            PipelineEvaluationConfig.from_yaml(args.config) if args.config else None
        )
        evaluate_from_config(
            eval_config,
            dataset_arg=args.dataset,
            predictions_dir_arg=args.predictions_dir,
            output_dir_arg=args.output_dir,
            config_path=Path(args.config) if args.config else None,
        )
        return

    # Discovery roots default to the active catalog's output tree; flags override
    # (CWD-relative).
    predictions_root = (
        Path(args.predictions_root)
        if args.predictions_root is not None
        else ctx.root / "data" / "predictions"
    )
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else ctx.root / "data" / "evaluations"
    )
    detected_files: list[DetectedPredictionFile] = []
    jobs, missing = build_evaluation_jobs(
        predictions_root=predictions_root,
        output_root=output_root,
        detected_files=detected_files,
    )

    for item in missing:
        logger.warning(item.message)

    if args.missing_report:
        _write_missing_report(Path(args.missing_report), missing, detected_files)

    logger.info("Discovered %d evaluable experiments.", len(jobs))

    failures = run_evaluation_jobs(
        jobs,
        dry_run=args.dry_run,
        exclude_zero_points=args.exclude_zero_points,
        verbose_configs=args.verbose_configs,
    )
    if failures:
        raise SystemExit(failures)


def _discover_eval_dirs(
    predictions_root: Path,
) -> dict[tuple[str, str, str, str], PredictionDirInfo]:
    dirs: dict[tuple[str, str, str, str], PredictionDirInfo] = {}
    for env in ENVIRONMENTS:
        env_root = predictions_root / env
        if not env_root.is_dir():
            continue
        for child in sorted(env_root.iterdir()):
            if not child.is_dir():
                continue
            info = parse_eval_dir_name(child)
            if info is None:
                continue
            dirs[(info.env, info.dataset_id, info.actor, info.modality or "")] = info
    return dirs


def _discover_gt_dirs(
    predictions_root: Path,
) -> dict[tuple[str, str, str], PredictionDirInfo]:
    dirs: dict[tuple[str, str, str], PredictionDirInfo] = {}
    for env in ENVIRONMENTS:
        env_root = predictions_root / env
        if not env_root.is_dir():
            continue
        for child in sorted(env_root.iterdir()):
            if not child.is_dir():
                continue
            info = parse_gt_dir_name(child)
            if info is None:
                continue
            dirs[(info.env, info.dataset_id, info.actor)] = info
    return dirs


def _build_comparisons_for_dirs(
    gt_info: PredictionDirInfo,
    eval_info: PredictionDirInfo,
    expected_methods: Mapping[str, tuple[str, ...]],
    missing: list[MissingCombination],
    detected_files: list[DetectedPredictionFile] | None,
) -> ComparisonSelection:
    gt_preds = _load_prediction_summaries_dir(
        gt_info.path / "gt",
        role="gt",
        dir_info=gt_info,
        detected_files=detected_files,
    )
    eval_preds = _load_prediction_summaries_dir(
        eval_info.path / "eval",
        role="eval",
        dir_info=eval_info,
        detected_files=detected_files,
    )
    expected_pointwise = _pointwise_expected_methods(expected_methods)
    expected_ranking_bases = expected_methods.get(RANKING_QV_KEY, ())

    gt_methods_by_signal = _gt_methods_by_signal(gt_preds)
    ranking_gt_methods = _ranking_gt_methods(gt_preds)
    pointwise_eval_by_signal = _pointwise_eval_methods_by_signal(eval_preds)
    ranking_eval_methods = _ranking_eval_methods(eval_preds)
    selected_eval_by_signal: dict[str, list[str]] = {"qv": [], "sv": []}
    selected_ranking_eval_methods: list[str] = []

    for signal, expected_bases in expected_pointwise.items():
        if not gt_methods_by_signal.get(signal):
            if expected_bases:
                missing.append(
                    MissingCombination(
                        kind="gt_method",
                        message=(
                            f"{SIGNAL_DISPLAY_NAMES[signal]} GT predictions missing "
                            f"on environment {_display_env(eval_info.env)} for "
                            f"dataset {eval_info.dataset_id!r}, GT actor "
                            f"{gt_info.actor}; skipping "
                            f"{eval_info.actor} {eval_info.modality} "
                            f"{SIGNAL_DISPLAY_NAMES[signal]} methods."
                        ),
                        env=eval_info.env,
                        dataset_id=eval_info.dataset_id,
                        gt_actor=gt_info.actor,
                        actor=eval_info.actor,
                        modality=eval_info.modality,
                        signal=signal,
                    )
                )
            continue
        for method_base in expected_bases:
            method_name = f"{method_base}_{signal}"
            if method_name in pointwise_eval_by_signal.get(signal, ()):
                selected_eval_by_signal[signal].append(method_name)
            else:
                missing.append(
                    _missing_eval_method(
                        eval_info,
                        method=method_base,
                        signal=signal,
                    )
                )

    if expected_ranking_bases and not ranking_gt_methods:
        missing.append(
            MissingCombination(
                kind="ranking_gt_method",
                message=(
                    "ranking q-value GT predictions missing on environment "
                    f"{_display_env(eval_info.env)} for dataset "
                    f"{eval_info.dataset_id!r}, GT actor {gt_info.actor}; "
                    "ranking q-value methods for "
                    f"{eval_info.actor} {eval_info.modality} cannot be evaluated."
                ),
                env=eval_info.env,
                dataset_id=eval_info.dataset_id,
                gt_actor=gt_info.actor,
                actor=eval_info.actor,
                modality=eval_info.modality,
                signal="qv",
            )
        )

    for method_base in expected_ranking_bases:
        method_name = f"{method_base}_qv"
        if method_name in ranking_eval_methods:
            if ranking_gt_methods:
                selected_ranking_eval_methods.append(method_name)
        else:
            missing.append(
                _missing_ranking_eval_method(
                    eval_info,
                    method=method_base,
                )
            )

    comparisons: dict[str, tuple[str, ...]] = {}
    for signal, gt_methods in gt_methods_by_signal.items():
        eval_methods = tuple(dict.fromkeys(selected_eval_by_signal.get(signal, [])))
        if not eval_methods:
            continue
        for gt_method in gt_methods:
            comparisons[gt_method] = eval_methods

    ranking_eval_tuple = tuple(dict.fromkeys(selected_ranking_eval_methods))
    if ranking_eval_tuple:
        for gt_method in ranking_gt_methods:
            comparisons[gt_method] = ranking_eval_tuple

    _mark_selected_file_usage(
        detected_files,
        gt_info=gt_info,
        eval_info=eval_info,
        gt_preds=gt_preds,
        eval_preds=eval_preds,
        comparisons=comparisons,
    )
    gt_paths = tuple(gt_preds[gt_method].path for gt_method in comparisons)
    eval_paths = tuple(
        dict.fromkeys(
            eval_preds[eval_method].path
            for eval_methods in comparisons.values()
            for eval_method in eval_methods
        )
    )
    return ComparisonSelection(
        comparisons=comparisons,
        gt_paths=gt_paths,
        eval_paths=eval_paths,
    )


def _gt_methods_by_signal(gt_preds: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    methods: dict[str, list[str]] = {"qv": [], "sv": []}
    for name, pred in gt_preds.items():
        config = getattr(pred, "config", {}) or {}
        if _is_non_pointwise_gt(name, config.get("prediction_format")):
            continue
        assumption = str(config.get("assumption", "")).lower()
        if assumption == SIGNAL_ASSUMPTIONS["qv"] or name.startswith("qv_"):
            methods["qv"].append(name)
        elif assumption == SIGNAL_ASSUMPTIONS["sv"] or name.startswith("sv_"):
            methods["sv"].append(name)
    return {signal: tuple(sorted(names)) for signal, names in methods.items() if names}


def _ranking_gt_methods(gt_preds: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, pred in gt_preds.items()
            if _is_non_pointwise_gt(
                name,
                (getattr(pred, "config", {}) or {}).get("prediction_format"),
            )
            and _filename_signal("gt", name) == "qv"
        )
    )


def _pointwise_eval_methods_by_signal(
    eval_preds: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    methods: dict[str, list[str]] = {"qv": [], "sv": []}
    for name, pred in eval_preds.items():
        config = getattr(pred, "config", {}) or {}
        if _is_non_pointwise_eval(config.get("prediction_format")):
            continue
        signal = _method_signal(name)
        if signal is not None:
            methods[signal].append(name)
    return {signal: tuple(sorted(names)) for signal, names in methods.items() if names}


def _ranking_eval_methods(eval_preds: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, pred in eval_preds.items()
            if _is_non_pointwise_eval(
                (getattr(pred, "config", {}) or {}).get("prediction_format")
            )
            and _method_signal(name) == "qv"
        )
    )


def _pointwise_expected_methods(
    expected_methods: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    return {
        signal: tuple(expected_methods.get(signal, ()))
        for signal in POINTWISE_SIGNALS
    }


_TIMESTAMPED_STEM_RE = re.compile(r"^(?P<base>.+)_(?P<ts>\d{8}_\d{6})$")
_CODEGEN_SAMPLE_BASE_RE = re.compile(r"^codegen-s\d+$")
_NON_POINTWISE_EVAL_FORMATS = {"ranking_predicted", "ranking_scores"}
_NON_POINTWISE_GT_FORMATS = {"ranking_q_values"}


def _load_prediction_summaries_dir(
    dir_path: Path,
    *,
    role: str,
    dir_info: PredictionDirInfo,
    detected_files: list[DetectedPredictionFile] | None,
) -> dict[str, PredictionSummary]:
    """Load just enough prediction JSON metadata for discovery.

    The real evaluation loads predictions through
    ``qval.prediction_store`` in ``evaluate_from_config``. This
    lightweight reader avoids importing environment and plotting dependencies
    during dry-run discovery.
    """
    valid_candidates: dict[str, list[tuple[Path, PredictionSummary, str]]] = {}
    for path in sorted(dir_path.glob("*.json")):
        match = _TIMESTAMPED_STEM_RE.match(path.stem)
        if match is None:
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_invalid_filename",
                reason=(
                    "Prediction JSON filename must be "
                    "<method_name>_<YYYYMMDD_HHMMSS>.json"
                ),
            )
            continue

        filename_method_name = match.group("base")
        timestamp = match.group("ts")
        try:
            with path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_json_decode_error",
                reason=str(exc),
                filename_method_name=filename_method_name,
                timestamp=timestamp,
            )
            continue
        if not isinstance(data, dict):
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_invalid_json_schema",
                reason="Prediction JSON root must be an object",
                filename_method_name=filename_method_name,
                timestamp=timestamp,
            )
            continue

        value_record_kwargs = _value_record_kwargs(data)
        method_name = data.get("method_name")
        method_type = data.get("method_type")
        config = data.get("config")
        if not isinstance(config, dict):
            config = {}
        prediction_format = config.get("prediction_format")
        if prediction_format is not None:
            prediction_format = str(prediction_format)
        if not isinstance(method_name, str) or not method_name:
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_missing_method_name",
                reason="Prediction JSON must contain a non-empty method_name string",
                filename_method_name=filename_method_name,
                method_type=str(method_type) if method_type is not None else None,
                prediction_format=prediction_format,
                timestamp=timestamp,
                **value_record_kwargs,
            )
            continue
        if method_name != filename_method_name:
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_method_name_mismatch",
                reason=(
                    "JSON method_name must equal filename basename without "
                    "timestamp suffix"
                ),
                filename_method_name=filename_method_name,
                method_name=method_name,
                method_type=str(method_type) if method_type is not None else None,
                prediction_format=prediction_format,
                timestamp=timestamp,
                **value_record_kwargs,
            )
            continue
        if method_type != role:
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_method_type_mismatch",
                reason=f"JSON method_type must be {role!r} for this directory",
                filename_method_name=filename_method_name,
                method_name=method_name,
                method_type=str(method_type) if method_type is not None else None,
                prediction_format=prediction_format,
                timestamp=timestamp,
                **value_record_kwargs,
            )
            continue

        signal = _filename_signal(role, method_name)
        if signal is None:
            if role == "eval":
                status = "skipped_missing_signal_suffix"
                reason = (
                    "Eval prediction JSON filename must end with _qv or _sv "
                    "before the timestamp suffix"
                )
            else:
                status = "skipped_missing_signal_prefix"
                reason = (
                    "GT prediction JSON filename must start with qv_ or sv_, "
                    "or use ranking_gt_qv_ / ranking_gt_sv_ for ranking GT "
                    "artifacts"
                )
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status=status,
                reason=reason,
                filename_method_name=filename_method_name,
                method_name=method_name,
                method_type=str(method_type) if method_type is not None else None,
                signal=signal,
                prediction_format=prediction_format,
                timestamp=timestamp,
                **value_record_kwargs,
            )
            continue
        config_signal = _config_signal(role, config)
        if (
            config_signal is not None
            and config_signal != signal
            and not (
                role == "gt"
                and _is_non_pointwise_gt(method_name, prediction_format)
            )
        ):
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="skipped_signal_config_mismatch",
                reason=(
                    "Filename qv/sv signal token must match JSON "
                    "signal_type/assumption metadata"
                ),
                filename_method_name=filename_method_name,
                method_name=method_name,
                method_type=str(method_type) if method_type is not None else None,
                signal=signal,
                prediction_format=prediction_format,
                timestamp=timestamp,
                **value_record_kwargs,
            )
            continue

        summary = PredictionSummary(
            method_name=method_name,
            method_type=method_type,
            config=config,
            path=path,
            signal=signal,
            prediction_format=prediction_format,
            values_count=value_record_kwargs["values_count"],
            null_count=value_record_kwargs["null_count"],
            null_fraction=value_record_kwargs["null_fraction"],
            null_percentage=value_record_kwargs["null_percentage"],
        )
        valid_candidates.setdefault(method_name, []).append((path, summary, timestamp))

    predictions: dict[str, PredictionSummary] = {}
    for method_name, candidates in valid_candidates.items():
        selected_path, selected_summary, selected_timestamp = max(
            candidates,
            key=lambda item: (item[2], item[0].name),
        )
        predictions[method_name] = selected_summary
        for path, summary, timestamp in candidates:
            is_selected = path == selected_path
            _record_detected_file(
                detected_files,
                path=path,
                role=role,
                dir_info=dir_info,
                status="selected" if is_selected else "superseded",
                reason=(
                    "Selected latest valid timestamp for method"
                    if is_selected
                    else "Superseded by newer timestamp for same method"
                ),
                filename_method_name=method_name,
                method_name=summary.method_name,
                method_type=summary.method_type,
                signal=summary.signal,
                prediction_format=summary.prediction_format,
                timestamp=timestamp,
                selected_path=str(selected_path) if not is_selected else None,
                values_count=summary.values_count,
                null_count=summary.null_count,
                null_fraction=summary.null_fraction,
                null_percentage=summary.null_percentage,
            )
    return predictions


def _mark_selected_file_usage(
    detected_files: list[DetectedPredictionFile] | None,
    *,
    gt_info: PredictionDirInfo,
    eval_info: PredictionDirInfo,
    gt_preds: Mapping[str, PredictionSummary],
    eval_preds: Mapping[str, PredictionSummary],
    comparisons: Mapping[str, tuple[str, ...]],
) -> None:
    if detected_files is None:
        return

    used_gt_methods = set(comparisons)
    used_eval_methods = {
        method_name
        for method_names in comparisons.values()
        for method_name in method_names
    }
    for item in detected_files:
        if item.status not in {
            "selected",
            "selected_not_evaluated",
            "selected_ignored_non_pointwise",
        }:
            continue
        if item.role == "gt" and _detected_file_matches_info(item, gt_info):
            summary = gt_preds.get(item.method_name or "")
            if item.method_name in used_gt_methods:
                item.status = "selected_for_evaluation"
                item.reason = "Selected latest valid GT artifact for evaluation"
                item.evaluable = True
            elif summary is not None and _is_non_pointwise_gt(
                summary.method_name, summary.prediction_format
            ):
                item.status = "selected_ignored_non_pointwise"
                item.reason = (
                    "Selected latest valid ranking GT artifact but no compatible "
                    "ranking eval method was selected"
                )
                item.evaluable = False
            elif item.status == "selected":
                item.status = "selected_not_evaluated"
                item.reason = (
                    "Selected latest valid GT artifact but no compatible eval "
                    "method was selected"
                )
                item.evaluable = False
        elif item.role == "eval" and _detected_file_matches_info(item, eval_info):
            summary = eval_preds.get(item.method_name or "")
            if item.method_name in used_eval_methods:
                item.status = "selected_for_evaluation"
                item.reason = "Selected latest valid eval artifact for evaluation"
                item.evaluable = True
            elif summary is not None and _is_non_pointwise_eval(
                summary.prediction_format
            ):
                item.status = "selected_ignored_non_pointwise"
                item.reason = (
                    "Selected latest valid ranking eval artifact but no compatible "
                    "ranking GT method was selected"
                )
                item.evaluable = False
            elif item.status == "selected":
                item.status = "selected_not_evaluated"
                item.reason = (
                    "Selected latest valid eval artifact but no compatible GT "
                    "method was selected"
                )
                item.evaluable = False


def _detected_file_matches_info(
    item: DetectedPredictionFile,
    info: PredictionDirInfo,
) -> bool:
    return (
        item.env == info.env
        and item.dataset_id == info.dataset_id
        and item.actor == info.actor
        and item.modality == info.modality
    )


def _record_detected_file(
    detected_files: list[DetectedPredictionFile] | None,
    *,
    path: Path,
    role: str,
    dir_info: PredictionDirInfo,
    status: str,
    reason: str,
    filename_method_name: str | None = None,
    method_name: str | None = None,
    method_type: str | None = None,
    signal: str | None = None,
    prediction_format: str | None = None,
    timestamp: str | None = None,
    selected_path: str | None = None,
    values_count: int | None = None,
    null_count: int | None = None,
    null_fraction: float | None = None,
    null_percentage: float | None = None,
) -> None:
    if detected_files is None:
        return
    path_str = str(path)
    if any(item.path == path_str for item in detected_files):
        return
    detected_files.append(
        DetectedPredictionFile(
            path=path_str,
            role=role,
            status=status,
            reason=reason,
            env=dir_info.env,
            dataset_id=dir_info.dataset_id,
            actor=dir_info.actor,
            modality=dir_info.modality,
            filename_method_name=filename_method_name,
            method_name=method_name,
            method_type=method_type,
            signal=signal,
            prediction_format=prediction_format,
            timestamp=timestamp,
            selected_path=selected_path,
            values_count=values_count,
            null_count=null_count,
            null_fraction=null_fraction,
            null_percentage=null_percentage,
        )
    )


def _value_record_kwargs(data: Mapping[str, Any]) -> dict[str, Any]:
    values = data.get("values")
    if not isinstance(values, list):
        return {
            "values_count": None,
            "null_count": None,
            "null_fraction": None,
            "null_percentage": None,
        }
    values_count = len(values)
    null_count = sum(1 for value in values if value is None)
    null_fraction = null_count / values_count if values_count else 0.0
    return {
        "values_count": values_count,
        "null_count": null_count,
        "null_fraction": null_fraction,
        "null_percentage": null_fraction * 100.0,
    }


def _filename_signal(role: str, method_name: str) -> str | None:
    if role == "eval":
        return _method_signal(method_name)
    if method_name.startswith("qv_"):
        return "qv"
    if method_name.startswith("sv_"):
        return "sv"
    if method_name.startswith("ranking_gt_qv_"):
        return "qv"
    if method_name.startswith("ranking_gt_sv_"):
        return "sv"
    return None


def _method_signal(method_name: str) -> str | None:
    if method_name.endswith("_qv"):
        return "qv"
    if method_name.endswith("_sv"):
        return "sv"
    return None


def _config_signal(role: str, config: Mapping[str, Any]) -> str | None:
    key = "signal_type" if role == "eval" else "assumption"
    value = str(config.get(key, "")).lower()
    for signal, assumption in SIGNAL_ASSUMPTIONS.items():
        if value == assumption:
            return signal
    return None


def _is_non_pointwise_eval(prediction_format: str | None) -> bool:
    return prediction_format in _NON_POINTWISE_EVAL_FORMATS


def _is_non_pointwise_gt(method_name: str, prediction_format: str | None) -> bool:
    return (
        prediction_format in _NON_POINTWISE_GT_FORMATS
        or method_name.startswith("ranking_gt_")
    )


def _is_ranking_gt_method(method_name: str) -> bool:
    return method_name.startswith("ranking_gt_")


def _expected_methods_for_gt_actor(
    expected_methods: Mapping[str, tuple[str, ...]],
    *,
    env: str,
    dataset_id: str,
    gt_actor: str,
) -> Mapping[str, tuple[str, ...]]:
    """Adjust the expected method catalog for GT-set-specific coverage."""
    methods = {signal: tuple(bases) for signal, bases in expected_methods.items()}
    env_spec = active_registry().env_by_name(env)
    if env_spec is not None and not env_spec.supports_sv:
        methods["sv"] = ()
    ranking_gt_actors = RANKING_GT_ACTORS_BY_ENV_DATASET.get((env, dataset_id))
    if ranking_gt_actors is not None and gt_actor not in ranking_gt_actors:
        methods[RANKING_QV_KEY] = ()
    return methods


def _select_gt_dirs(
    env: str,
    dataset_id: str,
    gt_dirs: Mapping[tuple[str, str, str], PredictionDirInfo],
    expected_gt_actors_by_env_dataset: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[tuple[PredictionDirInfo, ...], tuple[str, ...]]:
    expected_actors = expected_gt_actors_by_env_dataset.get((env, dataset_id), ())
    if expected_actors:
        selected: list[PredictionDirInfo] = []
        missing: list[str] = []
        for actor in expected_actors:
            info = gt_dirs.get((env, dataset_id, actor))
            if info is None:
                missing.append(actor)
            else:
                selected.append(info)
        return tuple(selected), tuple(missing)

    matches = [
        info
        for (gt_env, gt_dataset_id, _), info in gt_dirs.items()
        if gt_env == env and gt_dataset_id == dataset_id
    ]
    return tuple(sorted(matches, key=lambda info: info.path.name)), ()


def _split_dataset_actor(
    value: str,
    actors: tuple[str, ...],
) -> tuple[str, str] | None:
    for actor in sorted(actors, key=len, reverse=True):
        suffix = f"_{actor}"
        if value.endswith(suffix):
            dataset_key = value[: -len(suffix)]
            if dataset_key:
                return dataset_key, actor
    return None


def _normalize_dataset_id(dataset_key: str) -> str:
    for separator in ("_", "-"):
        marker = f"pt{separator}"
        if marker in dataset_key:
            prefix, rest = dataset_key.split(separator, 1)
            if prefix.endswith("pt") and prefix[:-2].isdigit():
                return rest
    if dataset_key.endswith("pt") and dataset_key[:-2].isdigit():
        return ""
    return dataset_key


def _missing_eval_dir(
    env: str,
    dataset_id: str,
    actor: str,
    modality: str,
) -> MissingCombination:
    return MissingCombination(
        kind="eval_dir",
        message=(
            f"evaluation directory missing on environment {_display_env(env)}, "
            f"dataset {dataset_id!r}, actor {actor}, {modality} modality."
        ),
        env=env,
        dataset_id=dataset_id,
        actor=actor,
        modality=modality,
    )


def _unexpected_prediction_dir(
    *,
    kind: str,
    env: str,
    dataset_id: str,
    actor: str,
    modality: str | None,
) -> MissingCombination:
    modality_description = (
        f", {modality} modality" if modality is not None else ""
    )
    return MissingCombination(
        kind=kind,
        message=(
            f"ignoring standardized prediction directory on environment "
            f"{_display_env(env)} for unexpected dataset {dataset_id!r}, "
            f"actor {actor}{modality_description}."
        ),
        env=env,
        dataset_id=dataset_id,
        actor=actor,
        modality=modality,
    )


def _missing_gt_dir(
    env: str,
    dataset_id: str,
    actor: str,
    modality: str,
    *,
    gt_actor: str | None,
) -> MissingCombination:
    gt_description = f" for GT actor {gt_actor}" if gt_actor is not None else ""
    return MissingCombination(
        kind="gt_dir",
        message=(
            f"GT directory missing on environment {_display_env(env)} "
            f"for dataset {dataset_id!r}{gt_description}; skipping "
            f"{actor} {modality}."
        ),
        env=env,
        dataset_id=dataset_id,
        gt_actor=gt_actor,
        actor=actor,
        modality=modality,
    )


def _missing_eval_method(
    eval_info: PredictionDirInfo,
    *,
    method: str,
    signal: str,
) -> MissingCombination:
    return MissingCombination(
        kind="eval_method",
        message=(
            f"{method} method {SIGNAL_DISPLAY_NAMES[signal]} predictions missing "
            f"on environment {_display_env(eval_info.env)}, "
            f"dataset {eval_info.dataset_id!r}, actor {eval_info.actor}, "
            f"{eval_info.modality} modality."
        ),
        env=eval_info.env,
        dataset_id=eval_info.dataset_id,
        actor=eval_info.actor,
        modality=eval_info.modality,
        method=method,
        signal=signal,
    )


def _collapse_reported_missing_methods(
    missing: list[MissingCombination],
) -> list[MissingCombination]:
    """Collapse codegen component misses into one report entry.

    A single codegen prediction run emits `codegen-mean` plus sixteen
    `codegen-s<N>` samples. When the run is absent, reporting all 17 as
    separate missing methods inflates counts without adding actionable detail.
    """
    collapsed: list[MissingCombination] = []
    seen: set[MissingCombination] = set()
    for item in missing:
        replacement = _collapse_missing_codegen_component(item)
        if replacement in seen:
            continue
        seen.add(replacement)
        collapsed.append(replacement)
    return collapsed


def _collapse_missing_codegen_component(
    item: MissingCombination,
) -> MissingCombination:
    if item.kind != "eval_method" or item.signal is None:
        return item
    if not _is_codegen_component_base(item.method):
        return item
    signal_name = SIGNAL_DISPLAY_NAMES[item.signal]
    return replace(
        item,
        method=CODEGEN_REPORT_METHOD,
        message=(
            f"{CODEGEN_REPORT_METHOD} method {signal_name} predictions missing "
            f"on environment {_display_env(item.env)}, "
            f"dataset {item.dataset_id!r}, actor {item.actor}, "
            f"{item.modality} modality."
        ),
    )


def _is_codegen_component_base(method: str | None) -> bool:
    return (
        method == "codegen-mean"
        or (method is not None and _CODEGEN_SAMPLE_BASE_RE.match(method) is not None)
    )


def _missing_ranking_eval_method(
    eval_info: PredictionDirInfo,
    *,
    method: str,
) -> MissingCombination:
    return MissingCombination(
        kind="ranking_eval_method",
        message=(
            f"{method} method ranking q-value predictions missing "
            f"on environment {_display_env(eval_info.env)}, "
            f"dataset {eval_info.dataset_id!r}, actor {eval_info.actor}, "
            f"{eval_info.modality} modality."
        ),
        env=eval_info.env,
        dataset_id=eval_info.dataset_id,
        actor=eval_info.actor,
        modality=eval_info.modality,
        method=method,
        signal="qv",
    )


def _display_env(env: str) -> str:
    return ENV_DISPLAY_NAMES.get(env, env)


def _write_missing_report(
    path: Path,
    missing: list[MissingCombination],
    detected_files: list[DetectedPredictionFile],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detected_by_status = _count_by_status(detected_files)
    high_null_prediction_files = _high_null_prediction_files(detected_files)
    report = {
        "summary": {
            "missing_total": len(missing),
            "detected_prediction_files_total": len(detected_files),
            "detected_by_status": detected_by_status,
            "evaluable_prediction_files_total": sum(
                1 for item in detected_files if item.evaluable
            ),
            "null_prediction_threshold_fraction": (
                NULL_PREDICTION_REPORT_THRESHOLD
            ),
            "high_null_prediction_files_total": len(high_null_prediction_files),
        },
        "missing": [asdict(item) for item in missing],
        "high_null_prediction_files": high_null_prediction_files,
        "detected_prediction_files": [asdict(item) for item in detected_files],
    }
    path.write_text(json.dumps(report, indent=2))
    logger.info("Missing-combination report written to %s", path)


_HIGH_NULL_REPORT_STATUSES = {
    "selected_for_evaluation",
    "selected_not_evaluated",
    "selected_ignored_non_pointwise",
}


def _high_null_prediction_files(
    detected_files: list[DetectedPredictionFile],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in detected_files:
        if item.status not in _HIGH_NULL_REPORT_STATUSES:
            continue
        if item.null_fraction is None:
            continue
        if item.null_fraction <= NULL_PREDICTION_REPORT_THRESHOLD:
            continue
        rows.append(
            {
                "path": item.path,
                "role": item.role,
                "status": item.status,
                "env": item.env,
                "dataset_id": item.dataset_id,
                "actor": item.actor,
                "modality": item.modality,
                "filename_method_name": item.filename_method_name,
                "method_name": item.method_name,
                "method_type": item.method_type,
                "signal": item.signal,
                "prediction_format": item.prediction_format,
                "prediction_count": item.values_count,
                "null_count": item.null_count,
                "null_fraction": item.null_fraction,
                "null_percentage": item.null_percentage,
                "evaluable": item.evaluable,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            str(item["env"]),
            str(item["dataset_id"]),
            str(item["actor"]),
            str(item["modality"]),
            str(item["method_name"]),
            str(item["path"]),
        ),
    )


def _count_by_status(detected_files: list[DetectedPredictionFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detected_files:
        counts[item.status] = counts.get(item.status, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()

"""Data caching for separating collection from evaluation.

Provides ``Dataset`` for storing evaluation points, per-trajectory returns,
and metadata. Serialized via pickle (State objects contain env-specific
hidden state that is not JSON-serializable).
"""

from __future__ import annotations

import logging
import os
import pickle
import random
import tempfile
from collections.abc import Mapping
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qval.path_resolution import resolve_latest_artifact_path
from qval.script_utils import resolve_effective_max_steps
from qval.types import EvaluationPoint, RankingPoint

if TYPE_CHECKING:
    from llenvs.evaluation.runner import TrajectoryResult

    from qval.config import EvaluationPointsConfig

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Cached data from the collection phase of a benchmark.

    Stores evaluation points and trajectory returns needed by the predict
    and evaluate scripts.

    Attributes:
        evaluation_points: Full EvaluationPoint objects with State references.
        trajectory_returns: Per-trajectory total returns (one per trajectory).
        config: Serialized config as a plain dict.
        metadata: Additional metadata (env name, model, adapter, timestamps).
        trajectory_results: Raw TrajectoryResult objects from collection.
            Enables resampling evaluation points without re-collecting
            trajectories. ``None`` when not stored.
    """

    evaluation_points: list[EvaluationPoint]
    trajectory_returns: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ranking_points: list[RankingPoint] = field(default_factory=list)
    trajectory_results: list[TrajectoryResult] | None = None

    @property
    def has_images(self) -> bool:
        """True if any evaluation point's state carries images.

        Used by methods/evaluation to decide whether image-requiring methods
        can run on this dataset. The check is env-agnostic — it only reads
        ``extract_images()`` on each point's state. Cheap; scans until first
        non-empty match.
        """
        from qval.methods.serialization import extract_images

        for point in self.evaluation_points:
            if extract_images(point.state).all:
                return True
        return False


def _hidden_metadata_value(hidden: Any, name: str) -> Any:
    """Read a metadata-like field from hidden state objects or dicts."""
    if hidden is None:
        return None
    if isinstance(hidden, dict):
        return hidden.get(name)
    return getattr(hidden, name, None)


def _task_name_for_result(result: Any) -> str | None:
    """Best-effort task name extraction for a stored trajectory result."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        task_name = metadata.get("task_name")
        if task_name not in (None, ""):
            return str(task_name)
        reset_info = metadata.get("reset_info")
        if isinstance(reset_info, Mapping):
            reset_task_name = reset_info.get("task_name")
            if reset_task_name not in (None, ""):
                return str(reset_task_name)

    initial_state = getattr(getattr(result, "trajectory", None), "initial_state", None)
    hidden = getattr(initial_state, "hidden", None)
    hidden_task_name = _hidden_metadata_value(hidden, "task_name")
    if hidden_task_name not in (None, ""):
        return str(hidden_task_name)

    return None


def _task_index_for_result(result: Any) -> int | None:
    """Best-effort task index extraction for a stored trajectory result."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        task_index = metadata.get("task_index")
        if isinstance(task_index, int):
            return task_index

    initial_state = getattr(getattr(result, "trajectory", None), "initial_state", None)
    hidden = getattr(initial_state, "hidden", None)
    hidden_task_index = _hidden_metadata_value(hidden, "task_index")
    if isinstance(hidden_task_index, int):
        return hidden_task_index

    return None


def _task_key_for_result(result: Any) -> Any | None:
    """Resolve the preferred task key for unique-task trajectory sampling."""
    task_name = _task_name_for_result(result)
    if task_name is not None:
        return task_name
    return _task_index_for_result(result)


def _task_name_for_point(point: EvaluationPoint | RankingPoint) -> str | None:
    """Best-effort task name extraction for a stored point."""
    replay_spec = getattr(point, "replay_spec", None)
    if replay_spec is not None:
        task_name = getattr(replay_spec, "task_name", None)
        if task_name not in (None, ""):
            return str(task_name)

    for state_name in ("state", "next_state"):
        state = getattr(point, state_name, None)
        hidden = getattr(state, "hidden", None)
        task_name = _hidden_metadata_value(hidden, "task_name")
        if task_name not in (None, ""):
            return str(task_name)

    return None


def _infer_source_trajectory_indices(dataset: Dataset) -> list[int]:
    """Infer the source trajectory indices represented in a dataset."""
    source_count = _source_num_trajectories(dataset)
    indices = set(range(source_count))
    indices.update(point.trajectory_index for point in dataset.evaluation_points)
    indices.update(point.trajectory_index for point in dataset.ranking_points)
    return sorted(indices)


def _trajectory_task_names(dataset: Dataset) -> dict[int, str]:
    """Recover per-trajectory task names from results or stored points."""
    names_by_trajectory: dict[int, str] = {}

    def record(trajectory_index: int, task_name: str | None) -> None:
        if task_name is None:
            return
        existing = names_by_trajectory.get(trajectory_index)
        if existing is not None and existing != task_name:
            raise ValueError(
                "Conflicting task names for trajectory_index="
                f"{trajectory_index}: {existing!r} vs {task_name!r}"
            )
        names_by_trajectory[trajectory_index] = task_name

    if dataset.trajectory_results is not None:
        for trajectory_index, result in enumerate(dataset.trajectory_results):
            record(trajectory_index, _task_name_for_result(result))

    for point in dataset.evaluation_points:
        record(point.trajectory_index, _task_name_for_point(point))
    for point in dataset.ranking_points:
        record(point.trajectory_index, _task_name_for_point(point))

    return names_by_trajectory


def save_dataset(data: Dataset, path: str | Path) -> None:
    """Save Dataset to a pickle file.

    Creates parent directories if they don't exist.

    Args:
        data: The dataset to save.
        path: File path for the pickle output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def resolve_dataset_path(path: str | Path) -> Path:
    """Resolve a dataset path to an existing pickle.

    Supports exact files, directories with timestamped pickles, and missing base
    paths with timestamped siblings such as ``dataset_YYYYmmdd_HHMMSS.pkl``.
    """
    return resolve_latest_artifact_path(
        path,
        suffix=".pkl",
        artifact_label="dataset",
    )


def load_dataset(path: str | Path) -> Dataset:
    """Load Dataset from a pickle file.

    Args:
        path: File path to the pickle file.

    Returns:
        The loaded Dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)  # noqa: S301
    if not isinstance(data, Dataset):
        raise TypeError(
            f"Expected Dataset, got {type(data).__name__}"
        )
    return data


def _point_key(point: EvaluationPoint | RankingPoint) -> tuple[int, int]:
    """Return the stable dataset identity key for a point."""
    return (point.trajectory_index, point.step_index)


def _validate_unique_point_keys(
    points: list[EvaluationPoint] | list[RankingPoint],
    *,
    label: str,
) -> dict[tuple[int, int], Any]:
    """Index dataset points by key and reject duplicates."""
    indexed: dict[tuple[int, int], Any] = {}
    for point in points:
        key = _point_key(point)
        if key in indexed:
            raise ValueError(
                f"Duplicate {label} key encountered: "
                f"trajectory_index={key[0]}, step_index={key[1]}"
            )
        indexed[key] = point
    return indexed


def _source_num_trajectories(dataset: Dataset) -> int:
    """Infer the source trajectory count for artifact-local metadata."""
    if dataset.trajectory_results is not None:
        return len(dataset.trajectory_results)
    if dataset.trajectory_returns:
        return len(dataset.trajectory_returns)
    num_trajectories = dataset.metadata.get("num_trajectories")
    if isinstance(num_trajectories, int):
        return num_trajectories
    return 0


def _filter_sequence_by_indices(
    values: list[Any],
    kept_indices: list[int],
    *,
    label: str,
) -> list[Any]:
    """Filter a list by original trajectory indices with bounds checking."""
    if not kept_indices:
        return []
    max_index = max(kept_indices)
    if max_index >= len(values):
        raise ValueError(
            f"Dataset contains point with trajectory_index={max_index} outside "
            f"{label} bounds [0, {len(values)})"
        )
    return [values[index] for index in kept_indices]


def _filter_task_indices_from_metadata(
    metadata: dict[str, Any],
    kept_indices: list[int],
) -> list[Any]:
    """Recover per-trajectory task indices when raw trajectory results are absent."""
    for key in ("collected_task_indices", "task_indices"):
        value = metadata.get(key)
        if not isinstance(value, list):
            continue
        return _filter_sequence_by_indices(
            value,
            kept_indices,
            label=f"metadata[{key!r}]",
        )
    return []


def prune_ranking_dataset(
    dataset: Dataset,
    *,
    min_candidates: int,
    source_path: str | Path | None = None,
) -> Dataset:
    """Prune a dataset to points with sufficiently wide ranking candidate sets.

    Keeps only ranking points whose ``len(candidates) >= min_candidates`` and
    the matching evaluation points with the same ``(trajectory_index, step_index)``
    key. Evaluation points without a matching ranking point are dropped. The
    surviving trajectories are compacted and reindexed so the output dataset
    remains semantically consistent for trajectory-level evaluation.
    """
    if min_candidates <= 0:
        raise ValueError("min_candidates must be positive")

    eval_by_key = _validate_unique_point_keys(
        dataset.evaluation_points,
        label="evaluation point",
    )
    ranking_by_key = _validate_unique_point_keys(
        dataset.ranking_points,
        label="ranking point",
    )
    ranking_keys_meeting_threshold = {
        key
        for key, point in ranking_by_key.items()
        if len(point.candidates) >= min_candidates
    }
    kept_keys = set(eval_by_key) & ranking_keys_meeting_threshold

    kept_evaluation_points = [
        point
        for point in dataset.evaluation_points
        if _point_key(point) in kept_keys
    ]
    kept_ranking_points = [
        point
        for point in dataset.ranking_points
        if _point_key(point) in kept_keys
    ]

    kept_trajectory_indices = sorted({
        point.trajectory_index for point in kept_evaluation_points
    })
    index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(kept_trajectory_indices)
    }
    pruned_evaluation_points = [
        replace(point, trajectory_index=index_map[point.trajectory_index])
        for point in kept_evaluation_points
    ]
    pruned_ranking_points = [
        replace(point, trajectory_index=index_map[point.trajectory_index])
        for point in kept_ranking_points
    ]

    trajectory_returns = _filter_sequence_by_indices(
        dataset.trajectory_returns,
        kept_trajectory_indices,
        label="trajectory_returns",
    )
    trajectory_results = (
        None
        if dataset.trajectory_results is None
        else _filter_sequence_by_indices(
            dataset.trajectory_results,
            kept_trajectory_indices,
            label="trajectory_results",
        )
    )

    new_metadata = dict(dataset.metadata)
    if source_path is not None:
        new_metadata["pruned_from"] = str(source_path)
    new_metadata["prune_timestamp"] = datetime.now().astimezone().isoformat()

    final_trajectory_count = len(kept_trajectory_indices)
    if trajectory_results is not None:
        final_task_indices = [
            getattr(result, "metadata", {}).get("task_index")
            for result in trajectory_results
        ]
    else:
        final_task_indices = _filter_task_indices_from_metadata(
            new_metadata,
            kept_trajectory_indices,
        )
    final_task_indices_are_ints = (
        bool(final_task_indices)
        and all(isinstance(index, int) for index in final_task_indices)
    )

    new_metadata["num_trajectories"] = final_trajectory_count
    new_metadata["collected_task_indices"] = final_task_indices
    if final_task_indices_are_ints:
        new_metadata["task_indices"] = list(final_task_indices)
    else:
        new_metadata.pop("task_indices", None)
    if "requested_trajectories" in new_metadata:
        new_metadata["requested_trajectories"] = final_trajectory_count
    if "retained_trajectories" in new_metadata:
        new_metadata["retained_trajectories"] = final_trajectory_count
    if "dropped_trajectories" in new_metadata:
        new_metadata["dropped_trajectories"] = 0
    if "num_attempted_trajectories" in new_metadata:
        new_metadata["num_attempted_trajectories"] = final_trajectory_count
    if "num_skipped_trajectories" in new_metadata:
        new_metadata["num_skipped_trajectories"] = 0
    if "drop_rate" in new_metadata:
        new_metadata["drop_rate"] = 0.0
    if "total_trajectories" in new_metadata:
        new_metadata["total_trajectories"] = final_trajectory_count
    if "num_tasks" in new_metadata:
        if final_task_indices_are_ints:
            new_metadata["num_tasks"] = len(set(final_task_indices))
        else:
            new_metadata.pop("num_tasks", None)
    if "samples_per_task" in new_metadata:
        if final_task_indices_are_ints:
            task_counts = Counter(final_task_indices)
            distinct_counts = set(task_counts.values())
            if len(distinct_counts) == 1:
                new_metadata["samples_per_task"] = next(iter(distinct_counts))
            else:
                new_metadata.pop("samples_per_task", None)
        else:
            new_metadata.pop("samples_per_task", None)
    if "max_steps" in new_metadata or "make_kwargs" in new_metadata:
        new_metadata["max_steps"] = resolve_effective_max_steps(
            new_metadata.get("max_steps"),
            new_metadata.get("make_kwargs"),
        )

    source_trajectory_count = _source_num_trajectories(dataset)
    new_metadata["ranking_candidate_pruning"] = {
        "min_candidates": min_candidates,
        "source_num_evaluation_points": len(dataset.evaluation_points),
        "kept_num_evaluation_points": len(pruned_evaluation_points),
        "dropped_num_evaluation_points": (
            len(dataset.evaluation_points) - len(pruned_evaluation_points)
        ),
        "source_num_ranking_points": len(dataset.ranking_points),
        "kept_num_ranking_points": len(pruned_ranking_points),
        "dropped_num_ranking_points": (
            len(dataset.ranking_points) - len(pruned_ranking_points)
        ),
        "source_num_trajectories": source_trajectory_count,
        "kept_num_trajectories": final_trajectory_count,
        "dropped_num_trajectories": (
            source_trajectory_count - final_trajectory_count
        ),
    }

    return Dataset(
        evaluation_points=pruned_evaluation_points,
        trajectory_returns=trajectory_returns,
        config=dict(dataset.config),
        metadata=new_metadata,
        ranking_points=pruned_ranking_points,
        trajectory_results=trajectory_results,
    )


def prune_dataset_tasks(
    dataset: Dataset,
    *,
    task_names: set[str] | list[str] | tuple[str, ...],
    source_path: str | Path | None = None,
) -> Dataset:
    """Prune a dataset by removing every trajectory belonging to named tasks.

    The kept trajectories are compacted and reindexed so the output dataset
    remains self-consistent across evaluation points, ranking points, cached
    returns, and raw trajectory results.
    """
    normalized_task_names: set[str] = set()
    for task_name in task_names:
        if task_name is None:
            continue
        task_name_str = str(task_name).strip()
        if task_name_str:
            normalized_task_names.add(task_name_str)
    if not normalized_task_names:
        raise ValueError("task_names must contain at least one non-empty task name")

    source_trajectory_indices = _infer_source_trajectory_indices(dataset)
    names_by_trajectory = _trajectory_task_names(dataset)
    kept_trajectory_indices = [
        trajectory_index
        for trajectory_index in source_trajectory_indices
        if names_by_trajectory.get(trajectory_index) not in normalized_task_names
    ]
    kept_trajectory_index_set = set(kept_trajectory_indices)
    index_map = {
        old_index: new_index
        for new_index, old_index in enumerate(kept_trajectory_indices)
    }

    kept_evaluation_points = [
        replace(point, trajectory_index=index_map[point.trajectory_index])
        for point in dataset.evaluation_points
        if point.trajectory_index in kept_trajectory_index_set
    ]
    kept_ranking_points = [
        replace(point, trajectory_index=index_map[point.trajectory_index])
        for point in dataset.ranking_points
        if point.trajectory_index in kept_trajectory_index_set
    ]

    trajectory_returns = _filter_sequence_by_indices(
        dataset.trajectory_returns,
        kept_trajectory_indices,
        label="trajectory_returns",
    )
    trajectory_results = (
        None
        if dataset.trajectory_results is None
        else _filter_sequence_by_indices(
            dataset.trajectory_results,
            kept_trajectory_indices,
            label="trajectory_results",
        )
    )

    new_metadata = dict(dataset.metadata)
    if source_path is not None:
        new_metadata["pruned_from"] = str(source_path)
    new_metadata["prune_timestamp"] = datetime.now().astimezone().isoformat()

    final_trajectory_count = len(kept_trajectory_indices)
    if trajectory_results is not None:
        final_task_indices = [
            _task_index_for_result(result)
            for result in trajectory_results
        ]
        final_task_keys = [
            _task_key_for_result(result)
            for result in trajectory_results
        ]
    else:
        final_task_indices = _filter_task_indices_from_metadata(
            new_metadata,
            kept_trajectory_indices,
        )
        final_task_keys = [
            names_by_trajectory.get(index)
            for index in kept_trajectory_indices
        ]
        if all(key is None for key in final_task_keys):
            final_task_keys = list(final_task_indices)

    final_task_indices_are_ints = (
        bool(final_task_indices)
        and all(isinstance(index, int) for index in final_task_indices)
    )
    final_task_keys = [key for key in final_task_keys if key is not None]

    new_metadata["num_trajectories"] = final_trajectory_count
    new_metadata["collected_task_indices"] = final_task_indices
    if final_task_indices_are_ints:
        new_metadata["task_indices"] = list(final_task_indices)
    else:
        new_metadata.pop("task_indices", None)
    if "requested_trajectories" in new_metadata:
        new_metadata["requested_trajectories"] = final_trajectory_count
    if "retained_trajectories" in new_metadata:
        new_metadata["retained_trajectories"] = final_trajectory_count
    if "dropped_trajectories" in new_metadata:
        new_metadata["dropped_trajectories"] = 0
    if "num_attempted_trajectories" in new_metadata:
        new_metadata["num_attempted_trajectories"] = final_trajectory_count
    if "num_skipped_trajectories" in new_metadata:
        new_metadata["num_skipped_trajectories"] = 0
    if "drop_rate" in new_metadata:
        new_metadata["drop_rate"] = 0.0
    if "total_trajectories" in new_metadata:
        new_metadata["total_trajectories"] = final_trajectory_count
    if "num_tasks" in new_metadata:
        if final_task_keys:
            new_metadata["num_tasks"] = len(set(final_task_keys))
        else:
            new_metadata.pop("num_tasks", None)
    if "samples_per_task" in new_metadata:
        if final_task_keys:
            task_counts = Counter(final_task_keys)
            distinct_counts = set(task_counts.values())
            if len(distinct_counts) == 1:
                new_metadata["samples_per_task"] = next(iter(distinct_counts))
            else:
                new_metadata.pop("samples_per_task", None)
        else:
            new_metadata.pop("samples_per_task", None)
    if "max_steps" in new_metadata or "make_kwargs" in new_metadata:
        new_metadata["max_steps"] = resolve_effective_max_steps(
            new_metadata.get("max_steps"),
            new_metadata.get("make_kwargs"),
        )

    source_trajectory_count = len(source_trajectory_indices)
    new_metadata["task_pruning"] = {
        "removed_task_names": sorted(normalized_task_names),
        "source_num_evaluation_points": len(dataset.evaluation_points),
        "kept_num_evaluation_points": len(kept_evaluation_points),
        "dropped_num_evaluation_points": (
            len(dataset.evaluation_points) - len(kept_evaluation_points)
        ),
        "source_num_ranking_points": len(dataset.ranking_points),
        "kept_num_ranking_points": len(kept_ranking_points),
        "dropped_num_ranking_points": (
            len(dataset.ranking_points) - len(kept_ranking_points)
        ),
        "source_num_trajectories": source_trajectory_count,
        "kept_num_trajectories": final_trajectory_count,
        "dropped_num_trajectories": (
            source_trajectory_count - final_trajectory_count
        ),
    }

    return Dataset(
        evaluation_points=kept_evaluation_points,
        trajectory_returns=trajectory_returns,
        config=dict(dataset.config),
        metadata=new_metadata,
        ranking_points=kept_ranking_points,
        trajectory_results=trajectory_results,
    )


def resample_dataset(
    dataset: Dataset,
    points_config: EvaluationPointsConfig,
    *,
    source_path: str | Path | None = None,
    strip_trajectories: bool = False,
    trajectory_subset_count: int | None = None,
    trajectory_subset_seed: int | None = None,
    unique_tasks: bool = False,
) -> Dataset:
    """Re-extract evaluation points from stored trajectory results.

    Runs :func:`~qval.evaluation_points.collect_evaluation_points`
    with new ``points_config`` parameters, recomputes trajectory returns,
    and returns a new :class:`Dataset`. Ranking points are dropped because
    they depend on specific evaluation points and live environment interaction.

    Args:
        dataset: A Dataset with ``trajectory_results`` populated.
        points_config: New evaluation point sampling configuration.
        source_path: Path to the source dataset (recorded in metadata).
        strip_trajectories: If ``True``, the returned dataset has
            ``trajectory_results=None`` (smaller file, no further resampling).
        trajectory_subset_count: Optional number of cleaned stored
            trajectories to sample uniformly without replacement before
            evaluation-point extraction.
        trajectory_subset_seed: Optional RNG seed for trajectory sampling.
            ``None`` falls back to ``dataset.metadata["seed"]`` and then ``42``.
        unique_tasks: When ``True``, sample a balanced quota from every unique
            task instead of sampling raw trajectories. Requires
            ``trajectory_subset_count``.

    Returns:
        A new Dataset with resampled evaluation points.

    Raises:
        ValueError: If ``dataset.trajectory_results`` is ``None``.
    """
    from qval.evaluation_points import collect_evaluation_points
    from qval.rollout import trajectory_return

    if dataset.trajectory_results is None:
        raise ValueError(
            "Dataset does not contain trajectory_results. "
            "Only datasets collected with trajectory storage can be resampled."
        )
    if unique_tasks and trajectory_subset_count is None:
        raise ValueError("unique_tasks requires trajectory_subset_count")

    raw_trajectory_results = dataset.trajectory_results
    dropped_failed_trajectory_count = 0
    dropped_failed_trajectory_indices: list[int] = []
    cleaned_source_indices: list[int] = []
    trajectory_results = []
    for original_idx, result in enumerate(raw_trajectory_results):
        error = getattr(result, "metadata", {}).get("error")
        if error:
            dropped_failed_trajectory_count += 1
            dropped_failed_trajectory_indices.append(original_idx)
            logger.warning(
                "Dropping failed trajectory %d during resample: %s",
                original_idx,
                error,
            )
            continue
        cleaned_source_indices.append(original_idx)
        trajectory_results.append(result)

    trajectory_selection_metadata: dict[str, Any] | None = None
    if trajectory_subset_count is not None:
        if trajectory_subset_count <= 0:
            raise ValueError("trajectory_subset_count must be positive")
        source_num_trajectories = len(raw_trajectory_results)
        clean_num_trajectories = len(trajectory_results)
        effective_seed = trajectory_subset_seed
        if effective_seed is None:
            metadata_seed = dataset.metadata.get("seed")
            effective_seed = 42 if metadata_seed is None else int(metadata_seed)
        rng = random.Random(effective_seed)
        if unique_tasks:
            grouped_cleaned_indices: dict[Any, list[int]] = {}
            for cleaned_index, result in enumerate(trajectory_results):
                task_key = _task_key_for_result(result)
                if task_key is None:
                    raise ValueError(
                        "unique_tasks requires task_name or task_index metadata "
                        f"for every trajectory (missing on source trajectory "
                        f"{cleaned_source_indices[cleaned_index]})"
                    )
                grouped_cleaned_indices.setdefault(task_key, []).append(cleaned_index)
            grouped_items = list(grouped_cleaned_indices.items())
            unique_task_count = len(grouped_items)
            if unique_task_count == 0:
                raise ValueError(
                    "unique_tasks requires at least one cleaned trajectory"
                )
            if trajectory_subset_count % unique_task_count != 0:
                raise ValueError(
                    "trajectory_subset_count must be a multiple of the number "
                    f"of unique tasks: {trajectory_subset_count} % "
                    f"{unique_task_count} != 0"
                )
            per_task_count = trajectory_subset_count // unique_task_count
            effective_k = min(len(indices) for _, indices in grouped_items)
            if per_task_count > effective_k:
                raise ValueError(
                    "per-task unique_tasks quota exceeds available "
                    "trajectories per task: "
                    f"{per_task_count} > {effective_k}"
                )
            selected_task_keys = [task_key for task_key, _ in grouped_items]
            selected_cleaned_indices: list[int] = []
            for _task_key, indices in grouped_items:
                selected_cleaned_indices.extend(rng.sample(indices, per_task_count))
            selected_cleaned_indices.sort()
            selected_trajectory_indices = [
                cleaned_source_indices[index] for index in selected_cleaned_indices
            ]
            trajectory_results = [
                trajectory_results[index] for index in selected_cleaned_indices
            ]
            trajectory_selection_metadata = {
                "mode": "random_balanced_unique_tasks",
                "count": trajectory_subset_count,
                "effective_seed": effective_seed,
                "source_num_trajectories": source_num_trajectories,
                "source_num_unique_tasks": unique_task_count,
                "task_key_strategy": "task_name_or_task_index",
                "selected_task_keys": selected_task_keys,
                "selected_trajectory_indices": selected_trajectory_indices,
                "per_task_count": per_task_count,
                "effective_k": effective_k,
            }
            logger.info(
                "Selected %d trajectories across %d unique tasks "
                "(per_task=%d, effective_k=%d, seed=%d)",
                len(trajectory_results),
                unique_task_count,
                per_task_count,
                effective_k,
                effective_seed,
            )
        else:
            if trajectory_subset_count > clean_num_trajectories:
                raise ValueError(
                    "trajectory_subset_count exceeds available trajectories: "
                    f"{trajectory_subset_count} > {clean_num_trajectories}"
                )
            selected_cleaned_indices = sorted(
                rng.sample(
                    range(clean_num_trajectories),
                    trajectory_subset_count,
                )
            )
            selected_trajectory_indices = [
                cleaned_source_indices[index] for index in selected_cleaned_indices
            ]
            trajectory_results = [
                trajectory_results[index] for index in selected_cleaned_indices
            ]
            trajectory_selection_metadata = {
                "mode": "random_sample",
                "count": trajectory_subset_count,
                "effective_seed": effective_seed,
                "source_num_trajectories": source_num_trajectories,
                "selected_trajectory_indices": selected_trajectory_indices,
            }
            logger.info(
                "Selected %d/%d trajectories for resample (seed=%d)",
                len(trajectory_results),
                clean_num_trajectories,
                effective_seed,
            )
        if dropped_failed_trajectory_indices:
            trajectory_selection_metadata["dropped_failed_trajectory_indices"] = (
                dropped_failed_trajectory_indices
            )

    # Re-extract evaluation points
    points = collect_evaluation_points(trajectory_results, points_config)
    logger.info("Resampled %d evaluation points", len(points))

    # Apply max_trajectories_to_keep pruning (mirrors collect_dataset.py)
    max_keep = points_config.max_trajectories_to_keep
    final_trajectory_results = trajectory_results
    if max_keep is not None and points:
        traj_indices_seen: list[int] = []
        seen: set[int] = set()
        for pt in points:
            if pt.trajectory_index not in seen:
                seen.add(pt.trajectory_index)
                traj_indices_seen.append(pt.trajectory_index)
        kept_traj_indices = traj_indices_seen[:max_keep]
        if len(traj_indices_seen) > max_keep:
            logger.info(
                "Pruned trajectories: kept %d/%d (%d -> %d points)",
                max_keep,
                len(traj_indices_seen),
                len(points),
                sum(
                    1 for point in points
                    if point.trajectory_index in set(kept_traj_indices)
                ),
            )
        index_map = {
            old_index: new_index
            for new_index, old_index in enumerate(kept_traj_indices)
        }
        points = [
            replace(point, trajectory_index=index_map[point.trajectory_index])
            for point in points
            if point.trajectory_index in index_map
        ]
        final_trajectory_results = [
            trajectory_results[index] for index in kept_traj_indices
        ]

    # Recompute trajectory returns
    reward_signal_name = dataset.config.get("reward_signal_name")
    discount_factor = dataset.metadata.get("discount_factor", 1.0)
    traj_returns = [
        trajectory_return(tr.trajectory, reward_signal_name, discount_factor)
        for tr in final_trajectory_results
    ]

    # Build updated config
    new_config = dict(dataset.config)
    new_config["points_config"] = {
        "num_trajectories": len(final_trajectory_results),
        "max_points_per_trajectory": points_config.max_points_per_trajectory,
        "sampling_strategy": points_config.sampling_strategy,
        "max_trajectories_to_keep": points_config.max_trajectories_to_keep,
        "early_turns_to_discard": points_config.early_turns_to_discard,
        "late_turns_to_discard": points_config.late_turns_to_discard,
    }

    # Build updated metadata
    new_metadata = dict(dataset.metadata)
    if source_path is not None:
        new_metadata["resampled_from"] = str(source_path)
    new_metadata["resample_timestamp"] = datetime.now().astimezone().isoformat()
    final_trajectory_count = len(final_trajectory_results)
    final_task_indices = [
        getattr(result, "metadata", {}).get("task_index")
        for result in final_trajectory_results
    ]
    final_task_indices_are_ints = (
        bool(final_task_indices)
        and all(isinstance(index, int) for index in final_task_indices)
    )
    new_metadata["num_trajectories"] = final_trajectory_count
    new_metadata["collected_task_indices"] = final_task_indices
    if final_task_indices_are_ints:
        new_metadata["task_indices"] = list(final_task_indices)
    else:
        new_metadata.pop("task_indices", None)
    if "requested_trajectories" in new_metadata:
        new_metadata["requested_trajectories"] = final_trajectory_count
    if "retained_trajectories" in new_metadata:
        new_metadata["retained_trajectories"] = final_trajectory_count
    if "dropped_trajectories" in new_metadata:
        new_metadata["dropped_trajectories"] = 0
    if "num_attempted_trajectories" in new_metadata:
        new_metadata["num_attempted_trajectories"] = final_trajectory_count
    if "num_skipped_trajectories" in new_metadata:
        new_metadata["num_skipped_trajectories"] = 0
    if "drop_rate" in new_metadata:
        new_metadata["drop_rate"] = 0.0
    if "total_trajectories" in new_metadata:
        new_metadata["total_trajectories"] = final_trajectory_count
    if "num_tasks" in new_metadata:
        if final_task_indices_are_ints:
            new_metadata["num_tasks"] = len(set(final_task_indices))
        else:
            new_metadata.pop("num_tasks", None)
    if "samples_per_task" in new_metadata:
        if final_task_indices_are_ints:
            task_counts = Counter(final_task_indices)
            distinct_counts = set(task_counts.values())
            if len(distinct_counts) == 1:
                new_metadata["samples_per_task"] = next(iter(distinct_counts))
            else:
                new_metadata.pop("samples_per_task", None)
        else:
            new_metadata.pop("samples_per_task", None)
    if "max_steps" in new_metadata or "make_kwargs" in new_metadata:
        new_metadata["max_steps"] = resolve_effective_max_steps(
            new_metadata.get("max_steps"),
            new_metadata.get("make_kwargs"),
        )
    if dropped_failed_trajectory_count > 0:
        new_metadata["dropped_failed_trajectory_count"] = (
            dropped_failed_trajectory_count
        )
    if trajectory_selection_metadata is not None:
        new_metadata["trajectory_selection"] = trajectory_selection_metadata

    return Dataset(
        evaluation_points=points,
        trajectory_returns=traj_returns,
        config=new_config,
        metadata=new_metadata,
        ranking_points=[],
        trajectory_results=None if strip_trajectories else final_trajectory_results,
    )

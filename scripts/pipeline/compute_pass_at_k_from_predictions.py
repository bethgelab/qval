"""Compute Pass@k from the MC rollouts produced by ``predict.py``.

Walks the rollout store under ``<prediction-dir>/rollouts`` (without rewriting
manifests), maps every stored trajectory back to its source eval point or
ranking candidate, and emits two orthogonal task groupings:

- ``by_point``: every eval-point ``(trajectory_index, step_index)`` and every
  ranking candidate ``(trajectory_index, step_index, candidate_index)`` counts
  as its own "task". Rollouts across primitives and estimations are pooled —
  the signal type they were used to estimate is irrelevant for Pass@k.
- ``by_env_task``: every rollout aggregates under the underlying environment
  task (``task_name``).

ACTOR_PRIMARY ranking candidates normally reuse eval-point rollouts and are
thus excluded from ``by_point`` by default (no duplication). Pass
``--include-actor-primary-rankings`` to surface them as separate point-as-task
rows sharing storage with the eval point.

Usage::

    uv run python scripts/pipeline/compute_pass_at_k_from_predictions.py \\
        --prediction-dir path/to/predict_output/ \\
        --output-path path/to/report.json \\
        --k-values 1 5 10 20 \\
        --success-threshold 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from qval.data_cache import (
    Dataset,
    _task_name_for_point,
    load_dataset,
    resolve_dataset_path,
)
from qval.mc_rollout_store import (
    StoredTrajectory,
    dataset_fingerprint_for_file,
)
from qval.pass_at_k import (
    PassAtKAttempt,
    attempt_from_stored_trajectory,
    compute_pass_at_k,
    print_summary,
)
from qval.types import (
    EvaluationPoint,
    RankingCandidateSource,
    RankingPoint,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10, 20)

EVAL_POINTS_NAMESPACE = "evaluation_points"
RANKING_CANDIDATES_NAMESPACE = "ranking_candidates"


# ---------------------------------------------------------------------------
# Loaded rollout view
# ---------------------------------------------------------------------------


class _LoadedRollout:
    """A single stored trajectory + provenance, ready to become a Pass@k attempt."""

    __slots__ = ("trajectory", "namespace", "primitive", "generation_hash", "point_key")

    def __init__(
        self,
        *,
        trajectory: StoredTrajectory,
        namespace: str,
        primitive: str,
        generation_hash: str,
        point_key: tuple[int, ...],
    ) -> None:
        self.trajectory = trajectory
        self.namespace = namespace
        self.primitive = primitive
        self.generation_hash = generation_hash
        self.point_key = point_key


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def _find_sibling_dataset(prediction_dir: Path) -> Path | None:
    """Look for a Dataset pickle inside the prediction_dir (best-effort)."""
    for candidate in sorted(prediction_dir.glob("*.pkl")):
        # Skip rollout shard pickles — they live under `rollouts/`.
        if "rollouts" in candidate.parts:
            continue
        return candidate
    return None


def _load_prediction_dataset(
    prediction_dir: Path,
    explicit_dataset: str | Path | None,
) -> tuple[Dataset, Path]:
    """Resolve and load the dataset backing the prediction output.

    Prefers the explicit ``--dataset`` argument, then any ``*.pkl`` sitting in
    ``prediction_dir``. Raises ``FileNotFoundError`` with a clear message when
    no dataset can be located.
    """
    if explicit_dataset is not None:
        resolved = resolve_dataset_path(explicit_dataset)
        return load_dataset(resolved), resolved

    candidate = _find_sibling_dataset(prediction_dir)
    if candidate is None:
        raise FileNotFoundError(
            "No dataset pickle found inside prediction directory "
            f"{prediction_dir!s}. Pass --dataset to supply the path explicitly."
        )
    resolved = resolve_dataset_path(candidate)
    return load_dataset(resolved), resolved


# ---------------------------------------------------------------------------
# Rollout store walker (read-only; does not rewrite manifests)
# ---------------------------------------------------------------------------


def _parse_point_key(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(":") if part != "")


def _iter_generation_dirs(rollouts_root: Path) -> Iterable[Path]:
    """Yield every ``<namespace>/<primitive>/<sha>/<generation_hash>`` directory."""
    if not rollouts_root.is_dir():
        return
    for namespace_dir in sorted(p for p in rollouts_root.iterdir() if p.is_dir()):
        for primitive_dir in sorted(p for p in namespace_dir.iterdir() if p.is_dir()):
            for sha_dir in sorted(p for p in primitive_dir.iterdir() if p.is_dir()):
                for generation_dir in sorted(
                    p for p in sha_dir.iterdir() if p.is_dir()
                ):
                    yield generation_dir


def _walk_rollouts(
    rollouts_root: Path,
    *,
    expected_sha: str,
    skip_fingerprint_check: bool = False,
) -> tuple[list[_LoadedRollout], dict[str, Any]]:
    """Walk the rollout store tree and materialize every stored trajectory.

    Returns ``(rollouts, provenance)`` where provenance aggregates namespaces,
    primitives, and generation hashes seen during the walk.
    """
    rollouts: list[_LoadedRollout] = []
    namespaces: set[str] = set()
    primitives: set[str] = set()
    generation_hashes: set[str] = set()

    for generation_dir in _iter_generation_dirs(rollouts_root):
        # Path layout: rollouts/<namespace>/<primitive>/<sha>/<generation_hash>.
        generation_hash = generation_dir.name
        sha_dir = generation_dir.parent
        primitive = sha_dir.parent.name
        namespace = sha_dir.parent.parent.name
        dir_sha = sha_dir.name

        if not skip_fingerprint_check and dir_sha != expected_sha:
            raise ValueError(
                "Dataset fingerprint mismatch: rollout store path segment "
                f"{dir_sha} does not match loaded dataset sha {expected_sha}. "
                "Pass --skip-fingerprint-check only if you are certain the "
                "rollouts and dataset belong together."
            )

        shards_dir = generation_dir / "shards"
        if not shards_dir.is_dir():
            continue
        shard_files = sorted(shards_dir.glob("shard_*.pkl"))
        if not shard_files:
            continue

        for shard_path in shard_files:
            try:
                with shard_path.open("rb") as f:
                    payload = pickle.load(f)  # noqa: S301 — trusted local store
            except (pickle.UnpicklingError, EOFError, ValueError, OSError) as exc:
                warnings.warn(
                    f"Skipping corrupt rollout shard {shard_path}: {exc}",
                    stacklevel=2,
                )
                continue

            trajectories_map = payload.get("trajectories", {})
            for point_key_str, point_trajectories in trajectories_map.items():
                try:
                    point_key = _parse_point_key(point_key_str)
                except ValueError:
                    warnings.warn(
                        f"Skipping unparseable point key {point_key_str!r} in "
                        f"{shard_path}",
                        stacklevel=2,
                    )
                    continue
                for traj in point_trajectories:
                    rollouts.append(
                        _LoadedRollout(
                            trajectory=traj,
                            namespace=namespace,
                            primitive=primitive,
                            generation_hash=generation_hash,
                            point_key=point_key,
                        )
                    )
        namespaces.add(namespace)
        primitives.add(primitive)
        generation_hashes.add(generation_hash)

    provenance = {
        "namespaces_seen": sorted(namespaces),
        "primitives_seen": sorted(primitives),
        "generation_hashes_seen": sorted(generation_hashes),
        "total_rollouts_pooled": len(rollouts),
    }
    return rollouts, provenance


# ---------------------------------------------------------------------------
# Dataset indices
# ---------------------------------------------------------------------------


def _index_eval_points(
    dataset: Dataset,
) -> dict[tuple[int, int], EvaluationPoint]:
    return {
        (point.trajectory_index, point.step_index): point
        for point in dataset.evaluation_points
    }


def _index_ranking_points(
    dataset: Dataset,
) -> dict[tuple[int, int], RankingPoint]:
    return {
        (point.trajectory_index, point.step_index): point
        for point in dataset.ranking_points
    }


# ---------------------------------------------------------------------------
# Attempt assembly
# ---------------------------------------------------------------------------


def _resolve_reward_signal_name(dataset: Dataset) -> str | None:
    if "reward_signal_name" in dataset.config:
        return dataset.config["reward_signal_name"]
    if "reward_signal_name" in dataset.metadata:
        return dataset.metadata["reward_signal_name"]
    return None


def _attempts_from_rollouts(
    rollouts: list[_LoadedRollout],
    *,
    dataset: Dataset,
    reward_signal_name: str | None,
    success_threshold: float,
    include_actor_primary_rankings: bool,
) -> tuple[list[PassAtKAttempt], list[PassAtKAttempt]]:
    """Build attempt lists for both groupings in a single pass.

    Returns ``(by_point_attempts, by_env_task_attempts)``. A rollout may land
    in one, both, or neither list:

    - Eval-point rollouts: both groupings.
    - Alternative ranking-candidate rollouts: both groupings.
    - ACTOR_PRIMARY ranking rollouts: excluded from ``by_env_task`` (they were
      already counted via the eval-point path). Included in ``by_point`` only
      when the caller asks, pulling from the eval-point storage.
    """
    eval_by_key = _index_eval_points(dataset)
    ranking_by_key = _index_ranking_points(dataset)

    by_point: list[PassAtKAttempt] = []
    by_env_task: list[PassAtKAttempt] = []

    # Helper to drill task_name from either point type.
    def _env_task_name_for_eval(key: tuple[int, int]) -> str | None:
        point = eval_by_key.get(key)
        return _task_name_for_point(point) if point is not None else None

    def _env_task_name_for_ranking(key: tuple[int, int]) -> str | None:
        point = ranking_by_key.get(key)
        return _task_name_for_point(point) if point is not None else None

    # First pass: eval-point rollouts.
    eval_point_rollouts: dict[tuple[int, int], list[_LoadedRollout]] = defaultdict(list)
    for rollout in rollouts:
        if rollout.namespace != EVAL_POINTS_NAMESPACE:
            continue
        if len(rollout.point_key) != 2:
            warnings.warn(
                f"Ignoring eval-point rollout with unexpected key arity "
                f"{rollout.point_key}",
                stacklevel=2,
            )
            continue
        key = (int(rollout.point_key[0]), int(rollout.point_key[1]))
        eval_point_rollouts[key].append(rollout)
        if key not in eval_by_key:
            raise ValueError(
                f"Eval-point key {key} from rollout store not found in dataset."
            )

    for key, group in eval_point_rollouts.items():
        task_name = _env_task_name_for_eval(key)
        for rollout in group:
            by_point.append(
                attempt_from_stored_trajectory(
                    rollout.trajectory,
                    reward_signal_name=reward_signal_name,
                    success_threshold=success_threshold,
                    task_key=key,
                    task_name=task_name,
                    source="eval_point_rollout",
                    trajectory_index=key[0],
                    step_index=key[1],
                    primitive=rollout.primitive,
                )
            )
            by_env_task.append(
                attempt_from_stored_trajectory(
                    rollout.trajectory,
                    reward_signal_name=reward_signal_name,
                    success_threshold=success_threshold,
                    task_key=task_name if task_name is not None else ("env", key[0]),
                    task_name=task_name,
                    source="eval_point_rollout",
                    trajectory_index=key[0],
                    step_index=key[1],
                    primitive=rollout.primitive,
                )
            )

    # Second pass: ranking-candidate rollouts.
    ranking_rollouts: dict[tuple[int, int, int], list[_LoadedRollout]] = defaultdict(list)
    for rollout in rollouts:
        if rollout.namespace != RANKING_CANDIDATES_NAMESPACE:
            continue
        if len(rollout.point_key) != 3:
            warnings.warn(
                f"Ignoring ranking rollout with unexpected key arity "
                f"{rollout.point_key}",
                stacklevel=2,
            )
            continue
        key3 = (
            int(rollout.point_key[0]),
            int(rollout.point_key[1]),
            int(rollout.point_key[2]),
        )
        ranking_rollouts[key3].append(rollout)
        if (key3[0], key3[1]) not in ranking_by_key:
            raise ValueError(
                f"Ranking-point key {(key3[0], key3[1])} from rollout store "
                "not found in dataset."
            )

    for key3, group in ranking_rollouts.items():
        rp = ranking_by_key[(key3[0], key3[1])]
        cand_idx = key3[2]
        if cand_idx < 0 or cand_idx >= len(rp.candidates):
            raise ValueError(
                f"Ranking candidate index {cand_idx} out of range for "
                f"point {(key3[0], key3[1])} (has {len(rp.candidates)} candidates)."
            )
        candidate = rp.candidates[cand_idx]
        task_name = _env_task_name_for_ranking((key3[0], key3[1]))
        for rollout in group:
            by_point.append(
                attempt_from_stored_trajectory(
                    rollout.trajectory,
                    reward_signal_name=reward_signal_name,
                    success_threshold=success_threshold,
                    task_key=key3,
                    task_name=task_name,
                    source="ranking_candidate_rollout",
                    trajectory_index=key3[0],
                    step_index=key3[1],
                    candidate_index=cand_idx,
                    primitive=rollout.primitive,
                )
            )
            by_env_task.append(
                attempt_from_stored_trajectory(
                    rollout.trajectory,
                    reward_signal_name=reward_signal_name,
                    success_threshold=success_threshold,
                    task_key=task_name if task_name is not None else ("env", key3[0]),
                    task_name=task_name,
                    source="ranking_candidate_rollout",
                    trajectory_index=key3[0],
                    step_index=key3[1],
                    candidate_index=cand_idx,
                    primitive=rollout.primitive,
                )
            )

    # Optional: include ACTOR_PRIMARY candidates in by_point by borrowing eval-point rollouts.
    if include_actor_primary_rankings:
        for (ti, si), rp in ranking_by_key.items():
            eval_group = eval_point_rollouts.get((ti, si))
            if not eval_group:
                continue
            for cand_idx, candidate in enumerate(rp.candidates):
                if candidate.source != RankingCandidateSource.ACTOR_PRIMARY:
                    continue
                task_name = _env_task_name_for_ranking((ti, si))
                for rollout in eval_group:
                    by_point.append(
                        attempt_from_stored_trajectory(
                            rollout.trajectory,
                            reward_signal_name=reward_signal_name,
                            success_threshold=success_threshold,
                            task_key=(ti, si, cand_idx),
                            task_name=task_name,
                            source="ranking_candidate_rollout_primary",
                            trajectory_index=ti,
                            step_index=si,
                            candidate_index=cand_idx,
                            primitive=rollout.primitive,
                        )
                    )

    return by_point, by_env_task


# ---------------------------------------------------------------------------
# JSON shaping
# ---------------------------------------------------------------------------


def _jsonable_task_key(value: Any) -> Any:
    """Recursively convert tuples (and inner tuples) to lists for JSON."""
    if isinstance(value, tuple):
        return [_jsonable_task_key(v) for v in value]
    return value


def _jsonable_results(results: dict[str, Any]) -> dict[str, Any]:
    new_per_task = []
    for row in results["per_task"]:
        new_row = dict(row)
        new_row["task_key"] = _jsonable_task_key(row["task_key"])
        new_per_task.append(new_row)
    return {**results, "per_task": new_per_task}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_from_predictions(
    *,
    prediction_dir: str | Path,
    dataset_path: str | Path | None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    success_threshold: float = 1.0,
    include_actor_primary_rankings: bool = False,
    skip_fingerprint_check: bool = False,
    rollout_store_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Programmatic entry: build Pass@k report from a predict.py output dir."""
    prediction_dir = Path(prediction_dir)
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}"
        )

    dataset, resolved_dataset_path = _load_prediction_dataset(
        prediction_dir, dataset_path
    )
    dataset_fingerprint = dataset_fingerprint_for_file(resolved_dataset_path)

    rollouts_root = (
        Path(rollout_store_root)
        if rollout_store_root is not None
        else prediction_dir / "rollouts"
    )
    rollouts, provenance = _walk_rollouts(
        rollouts_root,
        expected_sha=dataset_fingerprint.sha256,
        skip_fingerprint_check=skip_fingerprint_check,
    )
    if not rollouts:
        raise ValueError(
            f"No rollouts found under {rollouts_root!s}. "
            "Prediction directory appears empty or malformed."
        )

    reward_signal_name = _resolve_reward_signal_name(dataset)
    logger.info(
        "Success criterion: cumulative %s >= %.4f (threshold: %.4f)",
        reward_signal_name
        if reward_signal_name is not None
        else "(all signals, excl. step_penalty)",
        success_threshold,
        success_threshold,
    )

    by_point_attempts, by_env_attempts = _attempts_from_rollouts(
        rollouts,
        dataset=dataset,
        reward_signal_name=reward_signal_name,
        success_threshold=success_threshold,
        include_actor_primary_rankings=include_actor_primary_rankings,
    )

    by_point_results = compute_pass_at_k(by_point_attempts, k_values)
    by_env_results = compute_pass_at_k(by_env_attempts, k_values)

    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source": "compute_pass_at_k_from_predictions",
        "prediction_dir": str(prediction_dir),
        "dataset_path": str(resolved_dataset_path),
        "dataset_fingerprint_sha256": dataset_fingerprint.sha256,
        "context_name": dataset.metadata.get("context_name"),
        "env_name": dataset.metadata.get("env_name"),
        "adapter": dataset.metadata.get("adapter"),
        "reward_signal_name": reward_signal_name,
        "success_threshold": success_threshold,
        "k_values": list(k_values),
        "include_actor_primary_rankings": include_actor_primary_rankings,
    }

    report = {
        "metadata": metadata,
        "rollout_provenance": provenance,
        "by_point": _jsonable_results(by_point_results),
        "by_env_task": _jsonable_results(by_env_results),
    }

    if output_path is not None:
        output_path = Path(output_path)
    else:
        ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_path = prediction_dir / f"pass_at_k_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(
        "Pass@k report written to %s (by_point=%d tasks, by_env_task=%d tasks)",
        output_path,
        len(by_point_results["per_task"]),
        len(by_env_results["per_task"]),
    )

    print_summary(
        by_point_results,
        metadata,
        title="Pass@k by datapoint",
        task_label="Point",
    )
    print_summary(
        by_env_results,
        metadata,
        title="Pass@k by env task",
        task_label="Task",
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_k_values(raw: list[str]) -> tuple[int, ...]:
    values = tuple(int(v) for v in raw)
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("--k-values must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Pass@k from the MC rollouts in a predict.py output dir."
    )
    parser.add_argument(
        "--prediction-dir",
        required=True,
        help="Path to the predict.py output directory (must contain rollouts/).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Override path to the Dataset pickle. Defaults to the *.pkl file "
            "inside the prediction directory."
        ),
    )
    parser.add_argument(
        "--rollout-store-root",
        default=None,
        help="Override the rollouts directory. Defaults to <prediction-dir>/rollouts.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the JSON report. Defaults to <prediction-dir>/pass_at_k_<ts>.json.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=None,
        help="K values to compute Pass@k for. Defaults to 1 5 10 20.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help=(
            "Minimum cumulative reward (under the dataset's reward_signal_name) "
            "for a trajectory to count as a Pass@k success. Default 1.0."
        ),
    )
    parser.add_argument(
        "--include-actor-primary-rankings",
        action="store_true",
        help=(
            "Include ACTOR_PRIMARY ranking candidates in the point-as-task grouping. "
            "Default excludes them to avoid double-counting the eval-point rollouts."
        ),
    )
    parser.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help=(
            "Skip validation that the rollout-store sha path segment matches "
            "the loaded dataset. Use only if you know what you're doing."
        ),
    )
    args = parser.parse_args()

    k_values = (
        _parse_k_values(args.k_values) if args.k_values is not None else DEFAULT_K_VALUES
    )
    run_from_predictions(
        prediction_dir=args.prediction_dir,
        dataset_path=args.dataset,
        rollout_store_root=args.rollout_store_root,
        k_values=k_values,
        success_threshold=args.success_threshold,
        include_actor_primary_rankings=args.include_actor_primary_rankings,
        skip_fingerprint_check=args.skip_fingerprint_check,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

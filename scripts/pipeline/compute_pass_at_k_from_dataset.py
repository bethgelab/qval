"""Compute Pass@k from a pre-collected Dataset pickle.

Reads trajectory_results out of a Dataset (produced by collect_dataset.py or
compute_pass_at_k.py) and emits the same per-task + aggregate Pass@k report
the live script would have, grouped by env task (``task_name``) / task_index.

Usage::

    uv run python scripts/pipeline/compute_pass_at_k_from_dataset.py \\
        --dataset path/to/dataset.pkl \\
        --output-path path/to/report.json \\
        --k-values 1 5 10 20 \\
        --success-threshold 1.0

The script performs no inference and no environment interaction.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llenvs.evaluation.runner import TrajectoryResult

from qval.data_cache import (
    Dataset,
    _task_name_for_result,
    load_dataset,
    resolve_dataset_path,
)
from qval.mc_rollout_store import dataset_fingerprint_for_file
from qval.pass_at_k import (
    PassAtKAttempt,
    attempt_from_trajectory_result,
    compute_pass_at_k,
    print_summary,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10, 20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_reward_signal_name(dataset: Dataset) -> str | None:
    """Pick the reward_signal_name the dataset was collected with.

    Searches ``config`` first (the authoritative place) and falls back to
    ``metadata``. Returns ``None`` if neither carries the field — callers
    interpret ``None`` as "sum all signals except step_penalty".
    """
    if "reward_signal_name" in dataset.config:
        return dataset.config["reward_signal_name"]
    if "reward_signal_name" in dataset.metadata:
        return dataset.metadata["reward_signal_name"]
    return None


def _requested_counts_from_metadata(
    dataset: Dataset,
    observed_counts: dict[int, int],
) -> dict[int, int]:
    """Merge observed attempt counts with any ``task_indices`` plan on the dataset.

    ``dataset.metadata["task_indices"]`` records the originally requested task
    sequence. Each occurrence represents one requested attempt, so we count
    occurrences to recover the full plan. If a task has more observations than
    entries in the list (shouldn't normally happen), the observed count wins —
    Pass@k should never under-count the data we actually have.
    """
    requested_list = dataset.metadata.get("task_indices")
    if not isinstance(requested_list, list):
        return dict(observed_counts)

    list_counts: dict[int, int] = defaultdict(int)
    for task_index in requested_list:
        if isinstance(task_index, int):
            list_counts[task_index] += 1

    merged: dict[int, int] = {}
    for key in set(observed_counts) | set(list_counts):
        merged[key] = max(observed_counts.get(key, 0), list_counts.get(key, 0))
    return merged


def _build_attempts(
    results: list[TrajectoryResult],
    *,
    reward_signal_name: str | None,
    success_threshold: float,
) -> list[PassAtKAttempt]:
    """Adapter: TrajectoryResult → PassAtKAttempt keyed by task_index."""
    attempts: list[PassAtKAttempt] = []
    for result in results:
        task_index = result.metadata.get("task_index")
        if not isinstance(task_index, int):
            raise ValueError(
                "Every trajectory_result must carry an integer "
                "metadata['task_index']; got: "
                f"{type(task_index).__name__}={task_index!r}"
            )
        attempts.append(
            attempt_from_trajectory_result(
                result,
                reward_signal_name=reward_signal_name,
                success_threshold=success_threshold,
                task_key=task_index,
                task_name=_task_name_for_result(result),
            )
        )
    return attempts


def _rename_task_key_to_task_index(results: dict[str, Any]) -> dict[str, Any]:
    """Mirror the old compute_pass_at_k.py JSON schema (``task_index`` field)."""
    per_task = []
    for row in results["per_task"]:
        renamed = {"task_index": row["task_key"]}
        for key, value in row.items():
            if key == "task_key":
                continue
            if key == "attempts":
                renamed["trajectories"] = [
                    {
                        "success": attempt["success"],
                        "total_reward": attempt["total_reward"],
                        "num_steps": attempt["num_steps"],
                    }
                    for attempt in value
                ]
            else:
                renamed[key] = value
        per_task.append(renamed)
    return {**results, "per_task": per_task}


def _build_report(
    dataset: Dataset,
    dataset_path: Path,
    *,
    k_values: tuple[int, ...],
    success_threshold: float,
) -> dict[str, Any]:
    if dataset.trajectory_results is None:
        raise ValueError(
            "Dataset.trajectory_results is None — the pickle has stripped "
            "trajectories, so per-trajectory Pass@k cannot be computed. "
            "Re-collect with trajectory storage enabled."
        )

    reward_signal_name = _resolve_reward_signal_name(dataset)
    logger.info(
        "Success criterion: cumulative %s >= %.4f",
        reward_signal_name
        if reward_signal_name is not None
        else "(all signals, excl. step_penalty)",
        success_threshold,
    )

    attempts = _build_attempts(
        dataset.trajectory_results,
        reward_signal_name=reward_signal_name,
        success_threshold=success_threshold,
    )

    observed_counts: dict[int, int] = defaultdict(int)
    for attempt in attempts:
        observed_counts[attempt.task_key] += 1
    requested_counts = _requested_counts_from_metadata(dataset, dict(observed_counts))

    results = compute_pass_at_k(
        attempts,
        k_values,
        requested_counts=requested_counts,
    )
    results = _rename_task_key_to_task_index(results)

    try:
        fingerprint = dataset_fingerprint_for_file(dataset_path).sha256
    except (OSError, ValueError):
        fingerprint = None

    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source": "compute_pass_at_k_from_dataset",
        "dataset_path": str(dataset_path),
        "dataset_fingerprint_sha256": fingerprint,
        "context_name": dataset.metadata.get("context_name"),
        "env_name": dataset.metadata.get("env_name"),
        "adapter": dataset.metadata.get("adapter"),
        "reward_signal_name": reward_signal_name,
        "success_threshold": success_threshold,
        "k_values": list(k_values),
        "num_tasks": len(results["per_task"]),
        "total_trajectories": len(attempts),
        "requested_trajectories": sum(requested_counts.values()),
        "retained_trajectories": len(attempts),
        "dropped_trajectories": sum(requested_counts.values()) - len(attempts),
    }
    return {"metadata": metadata, **results}


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def _default_output_path(dataset_path: Path) -> Path:
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return dataset_path.with_name(f"{dataset_path.stem}_pass_at_k_{ts}.json")


# ---------------------------------------------------------------------------
# Public entry point (used by tests)
# ---------------------------------------------------------------------------


def run_from_dataset(
    *,
    dataset_path: str | Path,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    success_threshold: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Programmatic entry: compute Pass@k for a dataset and write the JSON.

    Returns the report dict (same shape as the written JSON) so tests can
    inspect the summary without re-reading the file.
    """
    resolved_dataset_path = resolve_dataset_path(dataset_path)
    dataset = load_dataset(resolved_dataset_path)
    report = _build_report(
        dataset,
        resolved_dataset_path,
        k_values=tuple(k_values),
        success_threshold=success_threshold,
    )
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(resolved_dataset_path)
    )
    _write_report(report, resolved_output)
    logger.info(
        "Pass@k report written to %s (tasks=%d, trajectories=%d)",
        resolved_output,
        report["metadata"]["num_tasks"],
        report["metadata"]["total_trajectories"],
    )
    header = (
        f"Pass@k from dataset: {dataset.metadata.get('context_name', resolved_dataset_path.name)}"
    )
    print_summary(
        {
            "aggregate": report["aggregate"],
            "per_task": [
                {**row, "task_key": row["task_index"]} for row in report["per_task"]
            ],
        },
        report["metadata"],
        title=header,
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
        description="Compute Pass@k from a pre-collected Dataset pickle."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the Dataset pickle (supports timestamped siblings).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the JSON report. Defaults to <dataset_stem>_pass_at_k_<timestamp>.json.",
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
    args = parser.parse_args()

    k_values = (
        _parse_k_values(args.k_values) if args.k_values is not None else DEFAULT_K_VALUES
    )
    run_from_dataset(
        dataset_path=args.dataset,
        k_values=k_values,
        success_threshold=args.success_threshold,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

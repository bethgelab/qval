"""Resample evaluation points from a stored-trajectory dataset.

Loads a Dataset that contains raw TrajectoryResult objects, re-runs
``collect_evaluation_points()`` with new ``EvaluationPointsConfig``
parameters, recomputes trajectory returns, and saves a new Dataset pickle.

Ranking points are dropped because they depend on the specific
evaluation points and live environment interaction.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from qval.config import EvaluationPointsConfig
from qval.data_cache import (
    load_dataset,
    resample_dataset,
    resolve_dataset_path,
    save_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample evaluation points from a stored-trajectory dataset",
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to input dataset pickle",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same directory as input)",
    )
    parser.add_argument(
        "--sampling-strategy", default=None,
        choices=["first", "random", "uniform"],
        help="Point sampling strategy (default: inherit from source dataset)",
    )
    parser.add_argument(
        "--max-points-per-trajectory", type=int, default=None,
        help="Max points per trajectory (default: inherit from source dataset)",
    )
    parser.add_argument(
        "--max-trajectories-to-keep", type=int, default=None,
        help="Max trajectories to keep (default: inherit from source dataset)",
    )
    parser.add_argument(
        "--trajectory-subset-count", type=int, default=None,
        help=(
            "Randomly sample this many stored trajectories before point extraction "
            "(default: use all stored trajectories)"
        ),
    )
    parser.add_argument(
        "--trajectory-subset-seed", type=int, default=None,
        help=(
            "Seed for trajectory subsetting (default: source dataset metadata seed, "
            "then 42)"
        ),
    )
    parser.add_argument(
        "--unique-tasks",
        action="store_true",
        help=(
            "When trajectory subsetting is enabled, sample a balanced quota "
            "across all unique tasks instead of sampling raw trajectories"
        ),
    )
    parser.add_argument(
        "--early-turns-to-discard", type=int, default=None,
        help="Initial transitions to exclude per trajectory (default: inherit from source)",
    )
    parser.add_argument(
        "--late-turns-to-discard", type=int, default=None,
        help="Final transitions to exclude per trajectory (default: inherit from source)",
    )
    parser.add_argument(
        "--strip-trajectories", action="store_true",
        help="Do not carry trajectory_results into the output dataset",
    )
    args = parser.parse_args()

    # 1. Load source dataset
    resolved_path = resolve_dataset_path(args.dataset)
    dataset = load_dataset(resolved_path)

    logger.info(
        "Loaded dataset from %s: %d trajectories, %d existing points",
        resolved_path,
        len(dataset.trajectory_results) if dataset.trajectory_results else 0,
        len(dataset.evaluation_points),
    )

    # 2. Build EvaluationPointsConfig from source + CLI overrides
    source_pc = dataset.config.get("points_config", {})
    num_trajectories = (
        len(dataset.trajectory_results)
        if dataset.trajectory_results
        else source_pc.get("num_trajectories", 0)
    )
    config = EvaluationPointsConfig(
        num_trajectories=num_trajectories,
        max_points_per_trajectory=(
            args.max_points_per_trajectory
            if args.max_points_per_trajectory is not None
            else source_pc.get("max_points_per_trajectory")
        ),
        sampling_strategy=(
            args.sampling_strategy
            if args.sampling_strategy is not None
            else source_pc.get("sampling_strategy", "random")
        ),
        max_trajectories_to_keep=(
            args.max_trajectories_to_keep
            if args.max_trajectories_to_keep is not None
            else source_pc.get("max_trajectories_to_keep")
        ),
        early_turns_to_discard=(
            args.early_turns_to_discard
            if args.early_turns_to_discard is not None
            else source_pc.get("early_turns_to_discard", 1)
        ),
        late_turns_to_discard=(
            args.late_turns_to_discard
            if args.late_turns_to_discard is not None
            else source_pc.get("late_turns_to_discard", 1)
        ),
    )

    logger.info(
        "Resampling with: strategy=%s, max_points=%s, max_traj_keep=%s, "
        "trajectory_subset_count=%s, trajectory_subset_seed=%s, unique_tasks=%s",
        config.sampling_strategy,
        config.max_points_per_trajectory,
        config.max_trajectories_to_keep,
        args.trajectory_subset_count,
        args.trajectory_subset_seed,
        args.unique_tasks,
    )

    # 3. Resample
    result = resample_dataset(
        dataset,
        config,
        source_path=resolved_path,
        strip_trajectories=args.strip_trajectories,
        trajectory_subset_count=args.trajectory_subset_count,
        trajectory_subset_seed=args.trajectory_subset_seed,
        unique_tasks=args.unique_tasks,
    )

    # 4. Save
    output_dir = Path(args.output_dir) if args.output_dir else resolved_path.parent
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"dataset_{ts}.pkl"

    save_dataset(result, output_path)
    logger.info(
        "Saved resampled dataset to %s (%d points, %d trajectory returns)",
        output_path,
        len(result.evaluation_points),
        len(result.trajectory_returns),
    )


if __name__ == "__main__":
    main()

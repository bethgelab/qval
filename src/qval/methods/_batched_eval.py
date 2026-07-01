"""Shared helpers for grouped prompt evaluation."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from qval.types import EvaluationPoint


@dataclass(frozen=True)
class PromptGroupPlan:
    """Logical prompt-group plan for a set of evaluation points."""

    points: list[EvaluationPoint]
    original_indices: list[int]


def _stable_shuffle_key(group_key: str) -> int:
    return int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:16], 16)


def plan_prompt_groups(
    points: list[EvaluationPoint],
    *,
    prompt_batch_size: int | None,
    prompt_grouping: str,
    prompt_shuffle: str = "none",
) -> list[PromptGroupPlan]:
    """Plan grouped prompts with deterministic ordering behavior.

    When ``prompt_batch_size`` is ``None``, each logical cluster becomes a
    single un-chunked group: one group per trajectory for
    ``prompt_grouping="trajectory"``, one group spanning all points for
    ``"contiguous"`` and ``"random"``.
    """
    if prompt_grouping not in {"contiguous", "trajectory", "random"}:
        raise ValueError(f"Unknown prompt_grouping: {prompt_grouping!r}")
    if prompt_shuffle not in {"none", "deterministic_random"}:
        raise ValueError(f"Unknown prompt_shuffle: {prompt_shuffle!r}")
    if prompt_batch_size is not None and prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be >= 1 or None")

    indexed_points = list(enumerate(points))

    def _chunks(
        items: list[tuple[int, EvaluationPoint]],
    ) -> list[list[tuple[int, EvaluationPoint]]]:
        if prompt_batch_size is None:
            return [items] if items else []
        return [
            items[start: start + prompt_batch_size]
            for start in range(0, len(items), prompt_batch_size)
        ]

    if prompt_grouping == "random":
        shuffled = list(indexed_points)
        seed = _stable_shuffle_key(
            "random:" + ",".join(str(orig_idx) for orig_idx, _ in indexed_points)
        )
        random.Random(seed).shuffle(shuffled)
        return [
            PromptGroupPlan(
                points=[point for _, point in chunk],
                original_indices=[orig_idx for orig_idx, _ in chunk],
            )
            for chunk in _chunks(shuffled)
        ]

    if prompt_grouping == "trajectory":
        grouped: dict[int, list[tuple[int, EvaluationPoint]]] = {}
        for orig_idx, point in indexed_points:
            grouped.setdefault(point.trajectory_index, []).append((orig_idx, point))

        plans: list[PromptGroupPlan] = []
        for trajectory_index, grouped_points in grouped.items():
            grouped_points.sort(key=lambda item: (item[1].step_index, item[0]))
            for chunk in _chunks(grouped_points):
                plans.append(
                    _make_prompt_group_plan(
                        chunk,
                        prompt_shuffle=prompt_shuffle,
                        group_key=(
                            f"trajectory:{trajectory_index}:"
                            f"{','.join(str(orig_idx) for orig_idx, _ in chunk)}"
                        ),
                    )
                )
        return plans

    plans: list[PromptGroupPlan] = []
    for chunk_idx, chunk in enumerate(_chunks(indexed_points)):
        plans.append(
            _make_prompt_group_plan(
                chunk,
                prompt_shuffle=prompt_shuffle,
                group_key=f"contiguous:{chunk[0][0] if chunk else chunk_idx}",
            )
        )
    return plans


def _make_prompt_group_plan(
    indexed_points: list[tuple[int, EvaluationPoint]],
    *,
    prompt_shuffle: str,
    group_key: str,
) -> PromptGroupPlan:
    ordered_points = list(indexed_points)
    if len(ordered_points) > 1 and prompt_shuffle == "deterministic_random":
        random.Random(_stable_shuffle_key(group_key)).shuffle(ordered_points)
    return PromptGroupPlan(
        points=[point for _, point in ordered_points],
        original_indices=[orig_idx for orig_idx, _ in ordered_points],
    )


def aggregate_repeated_prediction_passes(
    *,
    total_points: int,
    num_passes: int,
    run_pass: Callable[
        [int, Callable[[int, int], None] | None],
        tuple[list[float], set[int]],
    ],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[float], set[int]]:
    """Run repeated prediction passes and average successful outputs."""
    if num_passes < 1:
        raise ValueError("num_passes must be >= 1")

    sums = [0.0] * total_points
    counts = [0] * total_points
    total_progress = total_points * num_passes
    progress_offset = 0

    for pass_index in range(num_passes):
        def _on_pass_progress(completed: int, _total: int) -> None:
            if progress_callback is not None:
                progress_callback(progress_offset + completed, total_progress)

        values, _aborted = run_pass(pass_index, _on_pass_progress)
        for idx, value in enumerate(values):
            if not math.isnan(value):
                sums[idx] += value
                counts[idx] += 1
        progress_offset += total_points

    results = [
        (sums[idx] / counts[idx]) if counts[idx] > 0 else math.nan
        for idx in range(total_points)
    ]
    aborted = {idx for idx, count in enumerate(counts) if count == 0}
    return results, aborted

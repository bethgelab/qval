"""Pass@k computation shared across the three pipeline scripts.

This module houses the analysis logic used by:

- ``scripts/pipeline/compute_pass_at_k.py`` — live actor rollouts.
- ``scripts/pipeline/compute_pass_at_k_from_dataset.py`` — stored Dataset pickles.
- ``scripts/pipeline/compute_pass_at_k_from_predictions.py`` — MC rollouts from
  ``predict.py`` output.

The three scripts reduce to: load artifact → build ``list[PassAtKAttempt]`` →
``compute_pass_at_k`` → ``print_summary`` → save JSON. Success is derived from a
configurable threshold on undiscounted cumulative reward, stripping
``step_penalty`` from the sum so it never changes the outcome classification.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llenvs.evaluation.metrics import compute_binary_statistics

__all__ = [
    "PassAtKAttempt",
    "attempt_from_stored_trajectory",
    "attempt_from_trajectory_result",
    "compute_pass_at_k",
    "compute_total_return",
    "is_success",
    "print_summary",
]


STEP_PENALTY_SIGNAL_NAME = "step_penalty"


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def compute_total_return(
    transitions: Iterable[Any],
    reward_signal_name: str | None,
) -> float:
    """Sum rewards across transitions, without discounting.

    When ``reward_signal_name`` is set, sum that signal's value across every
    transition. Transitions without the signal contribute ``0``.

    When ``reward_signal_name`` is ``None``, sum every signal on each transition
    except the one literally named ``"step_penalty"``. Rationale: step penalty is
    a control-cost term, not a task-success signal. Counting it would let long
    solutions flip from success to failure for no reason related to the outcome.

    Discounting is explicitly NOT applied — Pass@k is about environmental
    outcome, not policy return.
    """
    total = 0.0
    for transition in transitions:
        rewards = transition.rewards
        if reward_signal_name is None:
            step_total = 0.0
            for signal in rewards.signals:
                if signal.name == STEP_PENALTY_SIGNAL_NAME:
                    continue
                reward = signal.reward
                if reward is not None:
                    step_total += reward
            total += step_total
        else:
            signal = rewards.by_name(reward_signal_name, required=False)
            if signal is None:
                continue
            reward = signal.reward
            if reward is not None:
                total += reward
    return total


def is_success(total_return: float, success_threshold: float) -> bool:
    """Binary success: did the cumulative return meet the configured threshold?"""
    return total_return >= success_threshold


# ---------------------------------------------------------------------------
# Atomic unit of analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassAtKAttempt:
    """A single attempt at a task — the unit Pass@k is computed over.

    Attributes:
        task_key: Grouping key. Use ``task_index`` for env-task grouping; a
            ``(trajectory_index, step_index, ...)`` tuple for point-as-task
            grouping. Must be hashable.
        task_name: Human-readable label. ``None`` if unavailable.
        success: Whether the attempt cleared ``success_threshold``.
        total_reward: Undiscounted cumulative return (step_penalty stripped).
        num_steps: Number of transitions in the attempt's trajectory.
        source: Provenance tag — ``"actor"`` / ``"eval_point_rollout"`` /
            ``"ranking_candidate_rollout"`` / etc. Not used by the grouping math.
        trajectory_index: Source trajectory index, if applicable.
        step_index: Source step index, if applicable.
        candidate_index: Ranking candidate index, if applicable.
        primitive: MC rollout primitive (``"state"`` / ``"state_action"`` /
            ``"next_state"``), if applicable.
    """

    task_key: Any
    task_name: str | None
    success: bool
    total_reward: float
    num_steps: int
    source: str | None = None
    trajectory_index: int | None = None
    step_index: int | None = None
    candidate_index: int | None = None
    primitive: str | None = None


# ---------------------------------------------------------------------------
# Attempt builders
# ---------------------------------------------------------------------------


def attempt_from_trajectory_result(
    result: Any,
    *,
    reward_signal_name: str | None,
    success_threshold: float,
    task_key: Any,
    task_name: str | None,
) -> PassAtKAttempt:
    """Build a ``PassAtKAttempt`` from a live ``TrajectoryResult``.

    Success is derived from ``compute_total_return`` + ``is_success``, not from
    any pre-set ``result.success`` field. This keeps success semantics uniform
    across live and stored trajectories.
    """
    trajectory = result.trajectory
    total_reward = compute_total_return(trajectory.transitions, reward_signal_name)
    return PassAtKAttempt(
        task_key=task_key,
        task_name=task_name,
        success=is_success(total_reward, success_threshold),
        total_reward=total_reward,
        num_steps=len(trajectory.transitions),
        source="actor",
    )


def attempt_from_stored_trajectory(
    trajectory: Any,
    *,
    reward_signal_name: str | None,
    success_threshold: float,
    task_key: Any,
    task_name: str | None,
    source: str,
    trajectory_index: int | None,
    step_index: int | None,
    candidate_index: int | None = None,
    primitive: str | None = None,
) -> PassAtKAttempt:
    """Build a ``PassAtKAttempt`` from a ``StoredTrajectory`` (MC rollout)."""
    total_reward = compute_total_return(trajectory.transitions, reward_signal_name)
    return PassAtKAttempt(
        task_key=task_key,
        task_name=task_name,
        success=is_success(total_reward, success_threshold),
        total_reward=total_reward,
        num_steps=len(trajectory.transitions),
        source=source,
        trajectory_index=trajectory_index,
        step_index=step_index,
        candidate_index=candidate_index,
        primitive=primitive,
    )


# ---------------------------------------------------------------------------
# Grouping + Pass@k computation
# ---------------------------------------------------------------------------


def _stable_sort_keys(keys: Iterable[Any]) -> list[Any]:
    """Sort keys when comparable; otherwise preserve insertion order.

    Python 3 forbids comparing unrelated types (int < tuple raises). We try to
    sort and fall back to the original iteration order if that fails.
    """
    key_list = list(keys)
    try:
        return sorted(key_list)
    except TypeError:
        return key_list


def compute_pass_at_k(
    attempts: Sequence[PassAtKAttempt],
    k_values: Sequence[int],
    *,
    requested_counts: Mapping[Any, int] | None = None,
) -> dict[str, Any]:
    """Group attempts by ``task_key`` and compute per-task + aggregate Pass@k.

    Returns a dict with two top-level keys:

    - ``per_task``: list of per-task rows (``task_key``, ``task_name``, ``n``,
      ``successes``, ``success_rate``, ``mean_reward``, ``pass_at_k``,
      ``attempts``).
    - ``aggregate``: macro-averaged stats across tasks, sample stdev (N−1
      denominator).

    ``requested_counts`` records how many attempts were planned per task before
    drops. Tasks with zero attempts but nonzero planned count appear in the
    output with ``n=0``, ``pass_at_k=None``, so drops are visible in reports.
    """
    by_task: dict[Any, list[PassAtKAttempt]] = defaultdict(list)
    task_name_by_key: dict[Any, str | None] = {}

    for attempt in attempts:
        by_task[attempt.task_key].append(attempt)
        # First non-None name wins — keeps reporting deterministic.
        existing = task_name_by_key.get(attempt.task_key)
        if existing is None and attempt.task_name is not None:
            task_name_by_key[attempt.task_key] = attempt.task_name
        elif attempt.task_key not in task_name_by_key:
            task_name_by_key[attempt.task_key] = attempt.task_name

    all_task_keys: list[Any] = list(by_task.keys())
    if requested_counts is not None:
        for key in requested_counts:
            if key not in by_task:
                by_task[key] = []
            if key not in task_name_by_key:
                task_name_by_key[key] = None
            if key not in all_task_keys:
                all_task_keys.append(key)

    ordered_task_keys = _stable_sort_keys(all_task_keys)

    per_task: list[dict[str, Any]] = []
    for task_key in ordered_task_keys:
        group = by_task.get(task_key, [])
        binary_values = [1.0 if a.success else 0.0 for a in group]
        stats = compute_binary_statistics(binary_values) if binary_values else None
        n = stats.n if stats is not None else 0
        successes = stats.count if stats is not None else 0
        success_rate = stats.mean if stats is not None else None
        requested_n = (
            requested_counts.get(task_key, n) if requested_counts is not None else n
        )
        pass_at_k: dict[str, float | None] = {}
        for k in k_values:
            if stats is not None and k <= stats.n:
                pass_at_k[str(k)] = stats.pass_at_k(k)
            else:
                pass_at_k[str(k)] = None

        per_task.append(
            {
                "task_key": task_key,
                "task_name": task_name_by_key.get(task_key),
                "requested_n": requested_n,
                "n": n,
                "dropped": requested_n - n,
                "successes": successes,
                "success_rate": success_rate,
                "mean_reward": (
                    sum(a.total_reward for a in group) / len(group) if group else None
                ),
                "pass_at_k": pass_at_k,
                "attempts": [
                    {
                        "success": a.success,
                        "total_reward": a.total_reward,
                        "num_steps": a.num_steps,
                        "source": a.source,
                        "trajectory_index": a.trajectory_index,
                        "step_index": a.step_index,
                        "candidate_index": a.candidate_index,
                        "primitive": a.primitive,
                    }
                    for a in group
                ],
            }
        )

    agg_pass_at_k: dict[str, dict[str, Any]] = {}
    for k in k_values:
        sk = str(k)
        values = [
            row["pass_at_k"][sk]
            for row in per_task
            if row["pass_at_k"][sk] is not None
        ]
        if values:
            mean = sum(values) / len(values)
            variance_denom = max(len(values) - 1, 1)
            std = (sum((v - mean) ** 2 for v in values) / variance_denom) ** 0.5
            agg_pass_at_k[sk] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "num_tasks": len(values),
            }
        else:
            agg_pass_at_k[sk] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "num_tasks": 0,
            }

    total_successes = sum(1 for a in attempts if a.success)
    total_samples = len(attempts)
    requested_samples = (
        sum(requested_counts.values())
        if requested_counts is not None
        else total_samples
    )
    dropped_samples = requested_samples - total_samples
    drop_rate = dropped_samples / requested_samples if requested_samples > 0 else 0.0

    return {
        "aggregate": {
            "success_rate": total_successes / total_samples if total_samples else None,
            "total_successes": total_successes,
            "total_samples": total_samples,
            "requested_samples": requested_samples,
            "dropped_samples": dropped_samples,
            "drop_rate": drop_rate,
            "pass_at_k": agg_pass_at_k,
        },
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(
    results: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    title: str,
    task_label: str = "Task",
) -> None:
    """Print a human-readable Pass@k summary to stdout."""
    agg = results["aggregate"]
    k_values = sorted(int(k) for k in agg["pass_at_k"])

    print()
    print(title)
    print("-" * len(title))

    print("\nAggregate (macro-averaged over tasks):")
    for k in k_values:
        info = agg["pass_at_k"][str(k)]
        if info["mean"] is not None:
            print(f"  Pass@{k:<4d} {info['mean']:.4f}  (std={info['std']:.4f})")
        else:
            print(f"  Pass@{k:<4d} n/a (insufficient samples)")

    sr = agg["success_rate"]
    if sr is not None:
        print(
            f"\nOverall success rate: "
            f"{sr:.4f} ({agg['total_successes']}/{agg['total_samples']})"
        )
    else:
        print("\nNo attempts collected.")

    print(f"\nPer-{task_label.lower()} breakdown:")
    for row in results["per_task"]:
        identifier = row["task_name"] if row["task_name"] is not None else row["task_key"]
        parts = [
            f"  {task_label} {str(identifier):>20}: "
            f"{row['successes']:>3d}/{row['n']} successes"
        ]
        for k in k_values:
            v = row["pass_at_k"][str(k)]
            if v is not None:
                parts.append(f"Pass@{k}={v:.3f}")
        print(", ".join(parts))
    print()

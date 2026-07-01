"""Collect evaluation points from trajectory results."""

from __future__ import annotations

import logging
import random
from typing import Any

from llenvs.evaluation.runner import TrajectoryResult
from qval.config import EvaluationPointsConfig
from qval.methods.serialization import (
    action_text_for_display,
    extract_images,
    extract_thought,
    serialize_action,
    serialize_observation,
)
from qval.types import EvaluationPoint, HistoryTurn, ReplaySpec

logger = logging.getLogger(__name__)


def _build_history(
    transitions: Any,
    up_to: int,
) -> tuple[HistoryTurn, ...]:
    """Build history turns from prior transitions.

    For tool-calling actions, uses ``serialize_action`` which includes
    formatted tool calls. Otherwise uses three-tier priority:
    ``resolved_action`` (tier 3) → ``extracted_action`` (tier 2) →
    raw ``action.text`` with thinking stripped (tier 1).

    Also extracts ReAct-style thoughts from the raw generation
    (``serialize_action(t.action)``) when present.  The thought is
    always stored in the ``HistoryTurn``; display is controlled
    downstream by the ``include_history_thoughts`` flag.
    """
    turns: list[HistoryTurn] = []
    for t in transitions[:up_to]:
        state_text = serialize_observation(t.state)
        action_text = action_text_for_display(
            t.action,
            t.resolved_action,
            t.extracted_action,
        )
        thought = extract_thought(serialize_action(t.action))
        state_images = extract_images(t.state)
        turns.append(HistoryTurn(
            state_text=state_text,
            action_text=action_text,
            thought=thought,
            state_images=state_images,
        ))
    return tuple(turns)


def _select_indices(
    num_transitions: int,
    max_points: int | None,
    strategy: str,
    trajectory_index: int,
) -> list[int]:
    """Select which transition indices to evaluate.

    Args:
        num_transitions: Total transitions in the trajectory.
        max_points: Maximum points to sample (None = all).
        strategy: "first", "random", or "uniform".
        trajectory_index: Used as seed for "random" strategy.

    Returns:
        Sorted list of selected step indices.
    """
    if max_points is None or max_points >= num_transitions:
        return list(range(num_transitions))

    if strategy == "first":
        return list(range(max_points))

    if strategy == "random":
        rng = random.Random(trajectory_index)
        indices = rng.sample(range(num_transitions), max_points)
        return sorted(indices)

    if strategy == "uniform":
        if max_points == 1:
            return [0]
        return [
            round(i * (num_transitions - 1) / (max_points - 1))
            for i in range(max_points)
        ]

    raise ValueError(f"Unknown sampling_strategy: {strategy!r}")


def collect_evaluation_points(
    trajectory_results: list[TrajectoryResult],
    config: EvaluationPointsConfig | None = None,
) -> list[EvaluationPoint]:
    """Extract (state, action, next_state) evaluation points from trajectories.

    Args:
        trajectory_results: List of TrajectoryResult objects from llenvs.
        config: Configuration controlling how many points to collect.

    Returns:
        List of EvaluationPoint instances.
    """
    config = config or EvaluationPointsConfig()
    points: list[EvaluationPoint] = []

    for traj_idx, result in enumerate(trajectory_results):
        error = getattr(result, "metadata", {}).get("error")
        if error:
            logger.warning(
                "Skipping errored trajectory %d during evaluation-point extraction: %s",
                traj_idx,
                error,
            )
            continue
        transitions = result.trajectory.transitions
        if not transitions:
            continue

        # Compute eligible range by discarding early/late turns
        n = len(transitions)
        start = config.early_turns_to_discard
        end = n - config.late_turns_to_discard
        if start >= end:
            eligible = list(range(n))
        else:
            eligible = list(range(start, end))

        local_indices = _select_indices(
            len(eligible),
            config.max_points_per_trajectory,
            config.sampling_strategy,
            traj_idx,
        )
        selected = [eligible[i] for i in local_indices]

        for step_idx in selected:
            transition = transitions[step_idx]

            # Build replay_spec for non-pure environments (e.g., Harbor)
            replay_spec = None
            hidden = transition.state.hidden
            if hasattr(hidden, "trajectory") and hasattr(hidden, "task_name"):
                snapshot_ref = getattr(hidden, "snapshot_ref", None)
                replay_spec = ReplaySpec(
                    task_name=hidden.task_name,
                    task_index=hidden.task_index,
                    trajectory=hidden.trajectory,
                    restore_mode="snapshot_exact" if snapshot_ref is not None else "replay",
                    snapshot_ref=(
                        None if snapshot_ref is None else snapshot_ref.relative_path
                    ),
                    snapshot_runtime=(
                        None if snapshot_ref is None else snapshot_ref.runtime
                    ),
                    snapshot_options=(
                        None
                        if snapshot_ref is None
                        else {
                            "file_locks": snapshot_ref.options.file_locks,
                            "tcp_established": snapshot_ref.options.tcp_established,
                            "tcp_close": snapshot_ref.options.tcp_close,
                            "ignore_volumes": snapshot_ref.options.ignore_volumes,
                        }
                    ),
                    fs_restore_risk_now=getattr(hidden, "fs_restore_risk_now", False),
                    fs_restore_risk_ever=getattr(hidden, "fs_restore_risk_ever", False),
                    fs_restore_risk_reasons=getattr(hidden, "fs_restore_risk_reasons", ()),
                )

            point = EvaluationPoint(
                state=transition.state,
                action=transition.action,
                next_state=transition.next_state,
                trajectory_index=traj_idx,
                step_index=step_idx,
                extracted_action=transition.extracted_action,
                resolved_action=transition.resolved_action,
                history=_build_history(transitions, step_idx),
                current_thought=extract_thought(
                    serialize_action(transition.action)
                ),
                replay_spec=replay_spec,
            )
            points.append(point)

    # Log risk stats for Harbor points
    risk_points = [
        p for p in points
        if p.replay_spec and (p.replay_spec.fs_restore_risk_now or p.replay_spec.fs_restore_risk_ever)
    ]
    if risk_points:
        total_harbor = sum(1 for p in points if p.replay_spec is not None)
        reason_counts: dict[str, int] = {}
        for p in risk_points:
            for reason in p.replay_spec.fs_restore_risk_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        logger.info(
            "Filesystem restore risk: %d/%d Harbor points flagged",
            len(risk_points),
            total_harbor,
        )
        if reason_counts:
            logger.info(
                "Risk reasons: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items())),
            )

    return points

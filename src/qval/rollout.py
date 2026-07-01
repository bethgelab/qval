"""Rollout utilities.

Provides ``trajectory_return()`` for computing (discounted) cumulative
returns from llenvs Trajectory objects.
"""

from __future__ import annotations

from typing import Any


def trajectory_return(
    trajectory: Any,
    reward_signal_name: str | None,
    discount_factor: float = 1.0,
    start_index: int = 0,
) -> float:
    """Compute the (optionally discounted) cumulative return of a trajectory.

    With ``discount_factor=1.0`` (default), sums all rewards. With
    ``discount_factor < 1.0``, applies geometric discounting:
    ``return = Σ γ^t * r_t``.

    Args:
        trajectory: A Trajectory object from llenvs.
        reward_signal_name: Name of reward signal to use, or None for total.
        discount_factor: Discount factor γ (0.0–1.0). Default 1.0.
        start_index: Index of the first transition to include. Transitions
            before this index are skipped. Discounting starts fresh from
            the start index. Default 0 (include all).

    Returns:
        The (discounted) cumulative return (0.0 if no transitions).
    """
    transitions = trajectory.transitions
    if not transitions:
        return 0.0

    total = 0.0
    discount = 1.0
    for transition in transitions[start_index:]:
        if reward_signal_name is not None:
            signal = transition.rewards.by_name(reward_signal_name, required=True)
            if signal.reward is not None:
                total += discount * signal.reward
        else:
            total += discount * transition.rewards.total
        discount *= discount_factor
    return total

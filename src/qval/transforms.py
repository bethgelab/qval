"""Signal transform utilities for prediction post-processing."""

from __future__ import annotations

from qval.types import EvaluationPoint, SignalType


def sum_per_trajectory(
    predictions: list[float],
    points: list[EvaluationPoint],
    num_trajectories: int,
) -> list[float]:
    """Sum method predictions per trajectory.

    Groups predictions by ``trajectory_index`` and sums within each group.
    Returns one sum per trajectory, in trajectory order.

    NaN values propagate naturally: if any prediction in a trajectory is NaN,
    the trajectory sum will be NaN (Python float ``+=`` propagates NaN).
    """
    sums = [0.0] * num_trajectories
    for pred, point in zip(predictions, points):
        sums[point.trajectory_index] += pred
    return sums


def extract_potential_values(
    predictions: list[float],
    points: list[EvaluationPoint],
) -> list[float]:
    """Extract potential Phi values from per-transition F values via cumulative sum.

    For each trajectory, computes Phi(s_k) = sum_{i=0}^{k-1} F(s_i, a_i, s_{i+1})
    with Phi(s_0) = 0. Returns Phi values in the same order as the input points.

    This is used when the method produces shaped rewards F(s,a,s') but the
    assumption is POTENTIAL — we extract the implicit potential function.

    NaN values propagate naturally through the cumulative sum: once a NaN
    prediction is encountered, all subsequent Phi values in that trajectory
    will be NaN (Python float ``+=`` propagates NaN).
    """
    # Group points by trajectory
    traj_groups: dict[int, list[tuple[int, int, float]]] = {}
    for i, (pred, point) in enumerate(zip(predictions, points)):
        traj_idx = point.trajectory_index
        if traj_idx not in traj_groups:
            traj_groups[traj_idx] = []
        traj_groups[traj_idx].append((point.step_index, i, pred))

    result = [0.0] * len(predictions)
    for traj_idx, entries in traj_groups.items():
        # Sort by step_index
        entries.sort(key=lambda x: x[0])
        # Cumulative sum: Phi(s_0) = 0, Phi(s_k) = sum F(s_0..s_{k-1})
        cumsum = 0.0
        for step_idx, orig_idx, pred in entries:
            result[orig_idx] = cumsum
            cumsum += pred

    return result


def transform_predictions(
    predictions: list[float],
    points: list[EvaluationPoint],
    signal_type: SignalType,
    assumption: SignalType,
) -> list[float]:
    """Transform method predictions based on signal_type -> assumption mapping.

    When the method produces per-transition F values (signal_type=SHAPED_REWARD)
    but the assumption is POTENTIAL, extract Phi via cumulative sum. Otherwise
    return predictions unchanged.
    """
    if signal_type == SignalType.SHAPED_REWARD and assumption == SignalType.POTENTIAL:
        return extract_potential_values(predictions, points)
    return predictions

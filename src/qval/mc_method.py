"""Monte Carlo method for ground truth signal estimation.

Estimates V(s), Q(s,a), and A(s,a) via MC rollouts using the environment
and a rollout policy.

Uses lockstep batched rollouts via ``generate_chat_batch()``, flattening
all rollouts across all evaluation points into a single batch for maximum
GPU utilization.

Supports MEAN and MAX aggregation methods via the standalone
``aggregate_returns()`` function. Collection and aggregation are split:
``collect_returns()`` gathers raw rollout returns, and ``evaluate_batch()``
delegates to ``collect_returns()`` + ``aggregate_returns()``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from llenvs.core.environment import Environment
from llenvs.core.state import Action, State
from llenvs.evaluation.runner import TrajectoryRunner, TurnInfoConfig
from llenvs.inference.protocol import SamplingParams

from qval.dense_signal import DenseSignalMethod
from qval.error_handling import (
    is_recoverable_backend_error,
    is_recoverable_rollout_error,
)
from qval.estimator import aggregate_returns
from qval.rollout import trajectory_return as _trajectory_return
from qval.types import (
    AggregationMethod,
    EvaluationPoint,
    MethodContext,
    Policy,
    RawPointReturns,
    SignalType,
)

logger = logging.getLogger(__name__)


def _on_generation_error(exc: Exception) -> str:
    """Error callback for the runner: skip recoverable errors, raise others."""
    if is_recoverable_backend_error(exc):
        return "skip"
    return "raise"


def _on_environment_error(exc: Exception) -> str:
    """Error callback for rollout restore/step failures."""
    if is_recoverable_rollout_error(exc):
        return "skip"
    return "raise"


def _rollout_type(signal_type: SignalType) -> str:
    """Map signal type to rollout type for identity grouping.

    STATE_VALUE and POTENTIAL use the same V(s) rollouts.
    """
    if signal_type in (SignalType.STATE_VALUE, SignalType.POTENTIAL):
        return "v_s"
    elif signal_type == SignalType.Q_VALUE:
        return "q_sa"
    elif signal_type == SignalType.ADVANTAGE:
        return "advantage"
    else:
        raise ValueError(f"No rollout type for signal type: {signal_type}")


class MCMethod(DenseSignalMethod):
    """Estimates ground truth signal values via Monte Carlo rollouts.

    Supports STATE_VALUE, Q_VALUE, ADVANTAGE, and POTENTIAL estimation.
    All rollouts are executed in lockstep batches via ``generate_chat_batch()``.

    Owns a ``TrajectoryRunner`` constructed once for the lifetime of the
    method, avoiding per-call runner construction overhead.

    Attributes:
        runner: TrajectoryRunner for executing rollouts.
        env: The environment for rollouts.
        policy: The rollout policy (backend + sampling_params + system_prompt).
        num_rollouts: Number of rollouts per evaluation point.
        aggregation: How to aggregate rollout returns (MEAN or MAX).
        reward_signal_name: Which reward signal to use (None for total).
        batch_size: Maximum rollouts per lockstep batch.
    """

    def __init__(
        self,
        context: MethodContext,
        env: Environment[Any],
        policy: Policy,
        *,
        num_rollouts: int = 32,
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        reward_signal_name: str | None = "correctness",
        batch_size: int | None = None,
        turn_info: bool = False,
        discount_factor: float = 1.0,
        env_factory: Callable[[], Environment[Any]] | None = None,
        restore_fn: Callable[[Environment[Any], Any], Any] | None = None,
        history_fn: Any | None = None,
        prompt_budget: Any | None = None,
        format_reminder: str | None = None,
        step_penalty: float | None = None,
        generation_descriptor: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(context)
        self.env = env
        self.policy = policy
        self.num_rollouts = num_rollouts
        self.aggregation = aggregation
        self.reward_signal_name = reward_signal_name
        self.batch_size = batch_size
        self.turn_info = turn_info
        self.discount_factor = discount_factor
        self.step_penalty = step_penalty
        self.generation_descriptor = generation_descriptor
        self._env_factory = env_factory
        self._restore_fn = restore_fn
        self.last_rollout_failures: dict[int, tuple[dict[str, Any], ...]] = {}

        system_prompt = policy.system_prompt
        if turn_info:
            max_steps = env.spec.max_steps
            if system_prompt is not None and max_steps is not None:
                system_prompt = system_prompt + f"\n\nYou have a maximum of {max_steps} turns."

        runner_kwargs: dict[str, Any] = {
            "environment": env,
            "backend": policy.backend,
            "sampling_params": policy.sampling_params or SamplingParams(),
            "system_prompt": system_prompt,
            "turn_info": TurnInfoConfig(task_suffix="", task_suffix_no_max="") if turn_info else None,
            "env_factory": env_factory,
            "restore_fn": restore_fn,
            "format_reminder": format_reminder,
        }
        if prompt_budget is not None:
            runner_kwargs["prompt_budget"] = prompt_budget
        elif history_fn is not None:
            runner_kwargs["history_fn"] = history_fn
        self.runner = TrajectoryRunner(**runner_kwargs)

    # ------------------------------------------------------------------
    # DenseSignalMethod interface
    # ------------------------------------------------------------------

    def evaluate(self, point: EvaluationPoint) -> float:
        """Evaluate a single point."""
        return self.evaluate_batch([point])[0]

    def evaluate_batch(self, points: list[EvaluationPoint]) -> list[float]:
        """Evaluate multiple points via MC rollouts + aggregation."""
        raw, _aborted = self.collect_returns(points)
        estimates = aggregate_returns(
            raw,
            self.aggregation,
            is_advantage=self.context.signal_type == SignalType.ADVANTAGE,
        )
        return [e.value for e in estimates]

    # ------------------------------------------------------------------
    # Public API for rollout reuse
    # ------------------------------------------------------------------

    def collect_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
        v_s_cache: list[RawPointReturns] | None = None,
        v_next_cache: list[RawPointReturns] | None = None,
        *,
        num_rollouts: int | None = None,
    ) -> tuple[list[RawPointReturns], set[int]]:
        """Collect raw rollout returns without aggregating.

        All rollouts from all points are flattened into a single lockstep
        batch, maximizing GPU utilization. Recoverable backend errors
        cause individual points to be aborted rather than crashing.

        Returns:
            ``(raw_returns, aborted_indices)`` where ``aborted_indices``
            contains the indices of points that had rollout failures.
        """
        rollout_count = self.num_rollouts if num_rollouts is None else num_rollouts
        if self.context.signal_type in (SignalType.STATE_VALUE, SignalType.POTENTIAL):
            return self._collect_state_value_returns(
                points,
                progress_callback,
                rollout_count,
            )
        elif self.context.signal_type == SignalType.Q_VALUE:
            return self._collect_q_value_returns(points, progress_callback, rollout_count)
        elif self.context.signal_type == SignalType.ADVANTAGE:
            return self._collect_advantage_returns(
                points, progress_callback, v_s_cache, v_next_cache, rollout_count,
            )
        elif self.context.signal_type == SignalType.SHAPED_REWARD:
            raise ValueError("SHAPED_REWARD does not use rollout-based estimation")
        else:
            msg = f"Unknown signal type: {self.context.signal_type}"
            raise ValueError(msg)

    def rollout_identity(self) -> tuple:
        """Return a hashable key identifying the rollout configuration.

        Two MCMethod instances with the same rollout_identity() produce
        identical rollouts and can share raw returns. Aggregation is
        intentionally excluded — methods differing only in aggregation
        (mean vs max) share rollouts. Rollout count is also excluded, so
        a larger collection can be shared with methods that only need a
        prefix of the same samples. STATE_VALUE and POTENTIAL map to the
        same rollout type ("v_s") and can share rollouts. Different
        discount factors produce different returns, so they are included.
        """
        return (
            "mc",
            _rollout_type(self.context.signal_type),
            self.policy.sampling_params.temperature
            if hasattr(self.policy.sampling_params, "temperature")
            else None,
            self.policy.sampling_params.max_tokens
            if hasattr(self.policy.sampling_params, "max_tokens")
            else None,
            self.discount_factor,
            self._env_factory is not None,
        )

    def build_generation_spec(
        self,
        *,
        namespace: str,
        primitive: str,
        dataset_fingerprint: Any,
    ) -> Any:
        """Build a persistent rollout generation spec for a primitive family."""
        from qval.mc_rollout_store import RolloutGenerationSpec

        if self.generation_descriptor is None:
            raise ValueError("MCMethod has no generation descriptor for persistence")
        return RolloutGenerationSpec(
            method="mc",
            namespace=namespace,
            primitive=primitive,
            dataset_fingerprint=dataset_fingerprint,
            environment=dict(self.generation_descriptor["environment"]),
            policy=dict(self.generation_descriptor["policy"]),
            prompt=dict(self.generation_descriptor["prompt"]),
        )

    # ------------------------------------------------------------------
    # Collection methods by signal type
    # ------------------------------------------------------------------

    def _collect_state_value_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None,
        num_rollouts: int,
    ) -> tuple[list[RawPointReturns], set[int]]:
        """Collect V(s) returns for all points via batched rollouts.

        Returns:
            ``(raw_returns, aborted_indices)``.
        """
        n = num_rollouts
        point_trajectories, aborted = self.collect_state_trajectories(
            points,
            progress_callback=progress_callback,
            num_rollouts=n,
            on_generation_error=_on_generation_error,
        )

        # Unflatten and build RawPointReturns
        results: list[RawPointReturns] = []
        for i, (point, point_trajs) in enumerate(
            zip(points, point_trajectories, strict=True)
        ):
            returns = tuple(
                _trajectory_return(t, self.reward_signal_name, self.discount_factor)
                for t in point_trajs
            )
            step_counts = tuple(len(t.transitions) for t in point_trajs)
            results.append(
                RawPointReturns(
                    point=point,
                    returns=returns,
                    step_counts=step_counts,
                    aborted=i in aborted,
                )
            )

        return results, aborted

    def _collect_q_value_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None,
        num_rollouts: int,
    ) -> tuple[list[RawPointReturns], set[int]]:
        """Collect Q(s,a) returns for all points via batched rollouts.

        Returns:
            ``(raw_returns, aborted_indices)``.
        """
        n = num_rollouts
        point_trajectories, aborted = self.collect_state_action_trajectories(
            points,
            progress_callback=progress_callback,
            num_rollouts=n,
            on_generation_error=_on_generation_error,
        )

        # Unflatten and build RawPointReturns
        results: list[RawPointReturns] = []
        for i, (point, point_trajs) in enumerate(
            zip(points, point_trajectories, strict=True)
        ):
            returns = tuple(
                _trajectory_return(t, self.reward_signal_name, self.discount_factor)
                for t in point_trajs
            )
            step_counts = tuple(len(t.transitions) for t in point_trajs)
            continuation_returns = tuple(
                _trajectory_return(
                    t, self.reward_signal_name, self.discount_factor, start_index=1,
                )
                for t in point_trajs
            )
            continuation_step_counts = tuple(
                max(0, len(t.transitions) - 1) for t in point_trajs
            )
            results.append(
                RawPointReturns(
                    point=point,
                    returns=returns,
                    step_counts=step_counts,
                    next_state_returns=continuation_returns,
                    next_state_step_counts=continuation_step_counts,
                    aborted=i in aborted,
                )
            )

        return results, aborted

    def _collect_advantage_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None,
        v_s_cache: list[RawPointReturns] | None = None,
        v_next_cache: list[RawPointReturns] | None = None,
        num_rollouts: int | None = None,
    ) -> tuple[list[RawPointReturns], set[int]]:
        """Collect A(s,a) returns: V(s) and V(s') via batched rollouts.

        Returns:
            ``(raw_returns, aborted_indices)``.
        """
        n = self.num_rollouts if num_rollouts is None else num_rollouts
        def _returns_from_trajectories(
            trajectories: tuple[Any, ...],
        ) -> tuple[tuple[float, ...], tuple[int, ...]]:
            returns = tuple(
                _trajectory_return(t, self.reward_signal_name, self.discount_factor)
                for t in trajectories
            )
            step_counts = tuple(len(t.transitions) for t in trajectories)
            return returns, step_counts

        def _make_raw(
            point: EvaluationPoint,
            *,
            returns: tuple[float, ...] = (),
            step_counts: tuple[int, ...] = (),
            next_state_returns: tuple[float, ...] = (),
            next_state_step_counts: tuple[int, ...] = (),
            aborted: bool = False,
        ) -> RawPointReturns:
            return RawPointReturns(
                point=point,
                returns=returns,
                step_counts=step_counts,
                next_state_returns=next_state_returns,
                next_state_step_counts=next_state_step_counts,
                aborted=aborted,
            )

        cached_aborted: set[int] = set()
        if v_s_cache is not None:
            for i, r in enumerate(v_s_cache):
                if r.aborted:
                    cached_aborted.add(i)
        if v_next_cache is not None:
            for i, r in enumerate(v_next_cache):
                if r.aborted:
                    cached_aborted.add(i)

        if v_s_cache is not None and v_next_cache is not None:
            results: list[RawPointReturns] = []
            for i, point in enumerate(points):
                results.append(
                    _make_raw(
                        point,
                        returns=v_s_cache[i].returns,
                        step_counts=v_s_cache[i].step_counts,
                        next_state_returns=v_next_cache[i].next_state_returns,
                        next_state_step_counts=v_next_cache[i].next_state_step_counts,
                        aborted=i in cached_aborted,
                    )
                )
                if progress_callback:
                    progress_callback(i + 1, len(points))
            return results, cached_aborted

        if v_s_cache is not None:
            trajectories, new_aborted = self.collect_next_state_trajectories(
                points,
                progress_callback=progress_callback,
                num_rollouts=n,
                on_generation_error=_on_generation_error,
            )
            all_aborted = cached_aborted | new_aborted

            results = []
            for i, (point, v_next_trajs) in enumerate(
                zip(points, trajectories, strict=True)
            ):
                v_next_returns, v_next_step_counts = _returns_from_trajectories(v_next_trajs)
                results.append(
                    _make_raw(
                        point,
                        returns=v_s_cache[i].returns,
                        step_counts=v_s_cache[i].step_counts,
                        next_state_returns=v_next_returns,
                        next_state_step_counts=v_next_step_counts,
                        aborted=i in all_aborted,
                    )
                )
            return results, all_aborted

        if v_next_cache is not None:
            trajectories, new_aborted = self.collect_state_trajectories(
                points,
                progress_callback=progress_callback,
                num_rollouts=n,
                on_generation_error=_on_generation_error,
            )
            all_aborted = cached_aborted | new_aborted

            results = []
            for i, (point, v_s_trajs) in enumerate(
                zip(points, trajectories, strict=True)
            ):
                v_s_returns, v_s_step_counts = _returns_from_trajectories(v_s_trajs)
                results.append(
                    _make_raw(
                        point,
                        returns=v_s_returns,
                        step_counts=v_s_step_counts,
                        next_state_returns=v_next_cache[i].next_state_returns,
                        next_state_step_counts=v_next_cache[i].next_state_step_counts,
                        aborted=i in all_aborted,
                    )
                )
            return results, all_aborted

        state_trajectories, state_aborted = self.collect_state_trajectories(
            points,
            num_rollouts=n,
            on_generation_error=_on_generation_error,
        )
        next_state_trajectories, next_aborted = self.collect_next_state_trajectories(
            points,
            num_rollouts=n,
            progress_callback=progress_callback,
            on_generation_error=_on_generation_error,
        )
        all_aborted = state_aborted | next_aborted

        results = []
        for i, (point, v_s_trajs, v_next_trajs) in enumerate(
            zip(points, state_trajectories, next_state_trajectories, strict=True)
        ):
            v_s_returns, v_s_step_counts = _returns_from_trajectories(v_s_trajs)
            v_next_returns, v_next_step_counts = _returns_from_trajectories(v_next_trajs)
            results.append(
                _make_raw(
                    point,
                    returns=v_s_returns,
                    step_counts=v_s_step_counts,
                    next_state_returns=v_next_returns,
                    next_state_step_counts=v_next_step_counts,
                    aborted=i in all_aborted,
                )
            )

        return results, all_aborted

    def collect_state_trajectories(
        self,
        points: list[EvaluationPoint],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        num_rollouts: int | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
    ) -> tuple[list[tuple[Any, ...]], set[int]]:
        """Collect trajectories starting from ``point.state``.

        Returns:
            ``(grouped_trajectories, aborted_indices)``.
        """
        states = [point.state for point in points]
        return self._collect_state_like_trajectories(
            states,
            total_points=len(points),
            num_rollouts=num_rollouts,
            progress_callback=progress_callback,
            on_generation_error=on_generation_error,
        )

    def collect_next_state_trajectories(
        self,
        points: list[EvaluationPoint],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        num_rollouts: int | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
    ) -> tuple[list[tuple[Any, ...]], set[int]]:
        """Collect trajectories starting from ``point.next_state``.

        Returns:
            ``(grouped_trajectories, aborted_indices)``.
        """
        states = [point.next_state for point in points]
        return self._collect_state_like_trajectories(
            states,
            total_points=len(points),
            num_rollouts=num_rollouts,
            progress_callback=progress_callback,
            on_generation_error=on_generation_error,
        )

    def collect_state_action_trajectories(
        self,
        points: list[EvaluationPoint],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        num_rollouts: int | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
    ) -> tuple[list[tuple[Any, ...]], set[int]]:
        """Collect trajectories from forced ``(state, action)`` pairs.

        Returns:
            ``(grouped_trajectories, aborted_indices)``.
        """
        self.last_rollout_failures = {}
        n = self.num_rollouts if num_rollouts is None else num_rollouts
        all_states: list[State[Any]] = []
        all_actions: list[Action] = []
        for point in points:
            all_states.extend([point.state] * n)
            all_actions.extend([point.action] * n)

        # Point-aligned batch size
        effective_batch_size = self.batch_size
        if self.batch_size is not None and n > 0 and self.batch_size >= n:
            effective_batch_size = (self.batch_size // n) * n

        trajectories = self.runner.run_batch_from_state_actions(
            all_states,
            all_actions,
            batch_size=effective_batch_size,
            on_generation_error=on_generation_error,
            on_environment_error=_on_environment_error,
        )

        # Identify aborted points
        aborted: set[int] = set()
        point_failures: dict[int, tuple[dict[str, Any], ...]] = {}
        total_points = len(points)
        for i in range(total_points):
            chunk = trajectories[i * n : (i + 1) * n]
            failures = tuple(
                dict(self.runner.last_environment_errors[idx])
                for idx in range(i * n, (i + 1) * n)
                if idx in self.runner.last_environment_errors
            )
            if any(t is None for t in chunk):
                aborted.add(i)
                if failures:
                    point_failures[i] = failures

        grouped: list[tuple[Any, ...]] = []
        for i in range(total_points):
            grouped.append(
                tuple(
                    trajectory
                    for trajectory in trajectories[i * n : (i + 1) * n]
                    if trajectory is not None
                )
            )

        if progress_callback:
            for i in range(total_points):
                progress_callback(i + 1, total_points)

        if aborted:
            logger.warning(
                "MC collection (state-action): %d/%d points aborted",
                len(aborted), total_points,
            )

        self.last_rollout_failures = point_failures
        return grouped, aborted

    def _collect_state_like_trajectories(
        self,
        states: list[State[Any]],
        *,
        total_points: int,
        num_rollouts: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        on_generation_error: Callable[[Exception], str] | None = None,
    ) -> tuple[list[tuple[Any, ...]], set[int]]:
        """Collect trajectories for multiple points via batched rollouts.

        Returns:
            ``(grouped_trajectories, aborted_indices)`` where
            ``aborted_indices`` contains the indices of points that had
            any failed rollout.
        """
        self.last_rollout_failures = {}
        n = self.num_rollouts if num_rollouts is None else num_rollouts
        all_states: list[State[Any]] = []
        for state in states:
            all_states.extend([state] * n)

        # Point-aligned batch size: ensure chunk boundaries don't straddle
        # two points. Never exceeds the configured batch_size.
        effective_batch_size = self.batch_size
        if self.batch_size is not None and n > 0 and self.batch_size >= n:
            effective_batch_size = (self.batch_size // n) * n

        trajectories = self.runner.run_batch_from_states(
            all_states,
            batch_size=effective_batch_size,
            on_generation_error=on_generation_error,
            on_environment_error=_on_environment_error,
        )

        # Identify aborted points: any point with a None trajectory
        aborted: set[int] = set()
        point_failures: dict[int, tuple[dict[str, Any], ...]] = {}
        for i in range(total_points):
            chunk = trajectories[i * n : (i + 1) * n]
            failures = tuple(
                dict(self.runner.last_environment_errors[idx])
                for idx in range(i * n, (i + 1) * n)
                if idx in self.runner.last_environment_errors
            )
            if any(t is None for t in chunk):
                aborted.add(i)
                if failures:
                    point_failures[i] = failures

        # Group trajectories. Aborted points retain already completed rollouts.
        grouped: list[tuple[Any, ...]] = []
        for i in range(total_points):
            grouped.append(
                tuple(
                    trajectory
                    for trajectory in trajectories[i * n : (i + 1) * n]
                    if trajectory is not None
                )
            )

        if progress_callback:
            for i in range(total_points):
                progress_callback(i + 1, total_points)

        if aborted:
            logger.warning(
                "MC collection: %d/%d points aborted due to backend errors",
                len(aborted), total_points,
            )

        self.last_rollout_failures = point_failures
        return grouped, aborted

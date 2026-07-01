"""MCTS method for ground truth signal estimation.

Estimates V(s), Q(s,a), and A(s,a) via Monte Carlo Tree Search using the
environment and a rollout policy.

Uses a "wide expansion" variant optimized for batched LLM inference: when
expanding a node, generate W children via a single ``generate_chat_batch()``
call, then simulate all non-terminal children in a single
``run_batch_from_states()`` call. Each expansion round makes exactly
2 batched LLM calls regardless of W.

The budget is ``mcts_iterations`` = total number of simulation returns.
Each expansion round produces up to W returns. Rounds continue until the
budget is exhausted.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from llenvs.core.environment import Environment
from llenvs.core.state import Action, State
from llenvs.evaluation.runner import TrajectoryRunner, TurnInfoConfig
from llenvs.inference.protocol import SamplingParams

from qval.dense_signal import DenseSignalMethod
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


class MCTSNode:
    """Node in the MCTS search tree."""

    __slots__ = (
        "state",
        "action",
        "parent",
        "children",
        "visit_count",
        "total_value",
        "cumulative_reward",
        "is_terminal",
        "is_expanded",
        "depth",
    )

    def __init__(
        self,
        state: Any,
        action: Any | None = None,
        parent: MCTSNode | None = None,
        cumulative_reward: float = 0.0,
        is_terminal: bool = False,
        depth: int = 0,
    ) -> None:
        self.state = state
        self.action = action
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.cumulative_reward = cumulative_reward
        self.is_terminal = is_terminal
        self.is_expanded: bool = False
        self.depth = depth

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1_score(self, exploration_constant: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration


class MCTSMethod(DenseSignalMethod):
    """Estimates ground truth signal values via MCTS.

    Supports STATE_VALUE, Q_VALUE, and ADVANTAGE estimation. Uses wide
    expansion with batched LLM calls for efficiency.

    Owns a ``TrajectoryRunner`` constructed once for the lifetime of the
    method, avoiding per-call runner construction overhead.

    Attributes:
        runner: TrajectoryRunner for executing rollouts and building messages.
        env: The environment for rollouts.
        policy: The rollout policy (backend + sampling_params + system_prompt).
        mcts_iterations: Total simulation budget.
        mcts_expansion_width: Children per expansion.
        mcts_exploration_constant: The c_puct constant for UCB.
        aggregation: How to aggregate simulation values.
        reward_signal_name: Which reward signal to use (None for total).
        batch_size: Maximum rollouts per lockstep batch.
    """

    def __init__(
        self,
        context: MethodContext,
        env: Environment[Any],
        policy: Policy,
        *,
        mcts_iterations: int = 100,
        mcts_expansion_width: int = 5,
        mcts_exploration_constant: float = 1.41,
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        reward_signal_name: str | None = "correctness",
        batch_size: int | None = None,
        turn_info: bool = False,
        discount_factor: float = 1.0,
        history_fn: Any | None = None,
        prompt_budget: Any | None = None,
        format_reminder: str | None = None,
    ) -> None:
        super().__init__(context)
        self.env = env
        self.policy = policy
        self.mcts_iterations = mcts_iterations
        self.mcts_expansion_width = mcts_expansion_width
        self.mcts_exploration_constant = mcts_exploration_constant
        self.aggregation = aggregation
        self.reward_signal_name = reward_signal_name
        self.batch_size = batch_size
        self.turn_info = turn_info
        self.discount_factor = discount_factor

        runner_kwargs: dict[str, Any] = {
            "environment": env,
            "backend": policy.backend,
            "sampling_params": policy.sampling_params or SamplingParams(),
            "system_prompt": policy.system_prompt,
            "turn_info": TurnInfoConfig() if turn_info else None,
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
        """Evaluate multiple points via MCTS + aggregation."""
        from qval.types import SignalType as _ST

        raw = self.collect_returns(points)
        estimates = aggregate_returns(
            raw,
            self.aggregation,
            is_advantage=self.context.signal_type == _ST.ADVANTAGE,
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
    ) -> list[RawPointReturns]:
        """Collect raw MCTS returns without aggregating.

        Args:
            points: List of evaluation points.
            progress_callback: Optional callback(completed, total).
            v_s_cache: Optional pre-collected V(s) returns for ADVANTAGE
                estimation. When provided and signal type is ADVANTAGE,
                V(s) returns are taken from the cache and only V(s')
                MCTS runs are performed.
        """
        if self.context.signal_type in (SignalType.STATE_VALUE, SignalType.POTENTIAL):
            return self._collect_state_value_returns(points, progress_callback)
        elif self.context.signal_type == SignalType.Q_VALUE:
            return self._collect_q_value_returns(points, progress_callback)
        elif self.context.signal_type == SignalType.ADVANTAGE:
            return self._collect_advantage_returns(points, progress_callback, v_s_cache)
        elif self.context.signal_type == SignalType.SHAPED_REWARD:
            raise ValueError("SHAPED_REWARD does not use rollout-based estimation")
        else:
            raise ValueError(f"Unknown signal type: {self.context.signal_type}")

    def rollout_identity(self) -> tuple:
        """Return a hashable key identifying the rollout configuration.

        Two MCTSMethod instances with the same rollout_identity() produce
        identical rollouts and can share raw returns. Aggregation is
        intentionally excluded. STATE_VALUE and POTENTIAL map to the
        same rollout type and can share rollouts. Different discount
        factors produce different returns, so they are included.
        """
        from qval.mc_method import _rollout_type

        return (
            "mcts",
            _rollout_type(self.context.signal_type),
            self.mcts_iterations,
            self.mcts_expansion_width,
            self.mcts_exploration_constant,
            self.policy.sampling_params.temperature
            if hasattr(self.policy.sampling_params, "temperature")
            else None,
            self.policy.sampling_params.max_tokens
            if hasattr(self.policy.sampling_params, "max_tokens")
            else None,
            self.discount_factor,
        )

    # ------------------------------------------------------------------
    # Collection methods by signal type
    # ------------------------------------------------------------------

    def _collect_state_value_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[RawPointReturns]:
        """Collect V(s) returns for all points via MCTS."""
        results: list[RawPointReturns] = []
        for i, point in enumerate(points):
            returns, step_counts = self._run_mcts(point.state)
            results.append(
                RawPointReturns(
                    point=point,
                    returns=tuple(returns),
                    step_counts=tuple(step_counts),
                )
            )
            if progress_callback:
                progress_callback(i + 1, len(points))
        return results

    def _collect_q_value_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[RawPointReturns]:
        """Collect Q(s,a) returns: force action then MCTS from next state.

        The forced action reward is undiscounted (at time 0). MCTS returns
        from the next state are scaled by ``discount_factor`` (time offset 1).
        """
        results: list[RawPointReturns] = []
        gamma = self.discount_factor
        for i, point in enumerate(points):
            step_result = self.env.step(point.state, point.action)
            step_reward = self._get_step_reward(step_result.rewards)

            if step_result.done:
                returns = [step_reward] * self.mcts_iterations
                step_counts = [1] * self.mcts_iterations
            else:
                mcts_returns, mcts_step_counts = self._run_mcts(step_result.next_state)
                returns = [step_reward + gamma * r for r in mcts_returns]
                step_counts = [1 + count for count in mcts_step_counts]

            results.append(
                RawPointReturns(
                    point=point,
                    returns=tuple(returns),
                    step_counts=tuple(step_counts),
                )
            )
            if progress_callback:
                progress_callback(i + 1, len(points))
        return results

    def _collect_advantage_returns(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None,
        v_s_cache: list[RawPointReturns] | None = None,
    ) -> list[RawPointReturns]:
        """Collect A(s,a) returns: V(s) and V(s') via MCTS.

        When ``v_s_cache`` is provided, V(s) returns are taken from the
        cache and only V(s') MCTS runs are performed.
        """
        results: list[RawPointReturns] = []
        for i, point in enumerate(points):
            if v_s_cache is not None:
                vs_returns_tuple = v_s_cache[i].returns
                vs_step_counts_tuple = v_s_cache[i].step_counts
            else:
                vs_returns, vs_step_counts = self._run_mcts(point.state)
                vs_returns_tuple = tuple(vs_returns)
                vs_step_counts_tuple = tuple(vs_step_counts)
            vs_prime_returns, vs_prime_step_counts = self._run_mcts(point.next_state)
            results.append(
                RawPointReturns(
                    point=point,
                    returns=vs_returns_tuple,
                    step_counts=vs_step_counts_tuple,
                    next_state_returns=tuple(vs_prime_returns),
                    next_state_step_counts=tuple(vs_prime_step_counts),
                )
            )
            if progress_callback:
                progress_callback(i + 1, len(points))
        return results

    # ------------------------------------------------------------------
    # Core MCTS algorithm
    # ------------------------------------------------------------------

    def _run_mcts(self, state: State[Any]) -> tuple[list[float], list[int]]:
        """Run MCTS from a state and return per-simulation returns and lengths."""
        budget = self.mcts_iterations
        width = self.mcts_expansion_width
        c = self.mcts_exploration_constant

        is_terminal = hasattr(state, "metadata") and state.metadata.is_terminal
        root = MCTSNode(state=state, cumulative_reward=0.0, is_terminal=is_terminal)

        returns: list[float] = []
        step_counts: list[int] = []

        while len(returns) < budget:
            remaining = budget - len(returns)

            node = self._select(root, c)

            if node.is_terminal:
                returns.append(node.cumulative_reward)
                step_counts.append(node.depth)
                self._backpropagate(node, node.cumulative_reward)
                continue

            actual_width = min(width, remaining)
            children = self._expand(node, actual_width)

            terminal_children = [ch for ch in children if ch.is_terminal]
            active_children = [ch for ch in children if not ch.is_terminal]

            for tc in terminal_children:
                if len(returns) >= budget:
                    break
                returns.append(tc.cumulative_reward)
                step_counts.append(tc.depth)
                tc.visit_count = 1
                tc.total_value = tc.cumulative_reward
                self._backpropagate(tc, tc.cumulative_reward)

            limit = min(len(active_children), budget - len(returns))
            if limit > 0:
                sim_states = [ch.state for ch in active_children[:limit]]
                sim_depth = active_children[0].depth if active_children else 0
                sim_returns, sim_step_counts = self._simulate_batch(
                    sim_states,
                    depth=sim_depth,
                )

                for child, sim_ret, sim_steps in zip(
                    active_children[:limit],
                    sim_returns,
                    sim_step_counts,
                ):
                    total_ret = child.cumulative_reward + sim_ret
                    returns.append(total_ret)
                    step_counts.append(sim_steps)
                    child.visit_count = 1
                    child.total_value = total_ret
                    self._backpropagate(child, total_ret)

        return returns[:budget], step_counts[:budget]

    def _select(self, root: MCTSNode, c: float) -> MCTSNode:
        """Walk tree via UCB1 to find an unexpanded non-terminal leaf."""
        node = root
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            node = max(node.children, key=lambda ch: ch.ucb1_score(c))
        return node

    def _expand(self, node: MCTSNode, width: int) -> list[MCTSNode]:
        """Expand a node by generating actions and stepping the environment."""
        messages = self.runner.build_messages(node.state)
        messages_batch = [messages] * width
        results = self.runner.backend.generate_chat_batch(
            messages_batch,
            self.runner.sampling_params,
        )

        child_depth = node.depth + 1
        children: list[MCTSNode] = []
        for gen_result in results:
            action = Action(text=gen_result.text or "")
            step_result = self.env.step(node.state, action)

            step_reward = self._get_step_reward(step_result.rewards)
            discounted_step_reward = (self.discount_factor**node.depth) * step_reward
            child_cumulative = node.cumulative_reward + discounted_step_reward

            child = MCTSNode(
                state=step_result.next_state,
                action=action,
                parent=node,
                cumulative_reward=child_cumulative,
                is_terminal=step_result.done,
                depth=child_depth,
            )
            children.append(child)

        node.children = children
        node.is_expanded = True
        return children

    def _simulate_batch(
        self,
        states: list[State[Any]],
        depth: int = 0,
    ) -> tuple[list[float], list[int]]:
        """Run batch rollouts from states and return returns and lengths.

        Args:
            states: Starting states for simulations.
            depth: Tree depth of the simulation start nodes, used to
                offset discount factors in the trajectory return.
        """
        trajectories = self.runner.run_batch_from_states(
            states,
            batch_size=self.batch_size,
        )
        # Apply discounting: simulation returns start at depth `depth`
        depth_discount = self.discount_factor**depth
        returns = [
            depth_discount
            * _trajectory_return(t, self.reward_signal_name, self.discount_factor)
            for t in trajectories
        ]
        step_counts = [depth + len(t.transitions) for t in trajectories]
        return returns, step_counts

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update visit counts and total values from node up to root."""
        current: MCTSNode | None = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def _get_step_reward(self, rewards: Any) -> float:
        """Extract reward from a SignalBundle."""
        if self.reward_signal_name is not None:
            signal = rewards.by_name(self.reward_signal_name, required=True)
            return signal.reward if signal.reward is not None else 0.0
        return rewards.total

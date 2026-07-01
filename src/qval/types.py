"""Core types for qval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from llenvs.core.state import ObservationImages

if TYPE_CHECKING:
    from qval.trajectory_store import ObservableTrajectory


@dataclass(frozen=True)
class Policy:
    """A policy for generating actions from states.

    Bundles what's needed to produce actions: which model, how to
    sample, and what system instructions. Used by rollout-based
    methods (MC, MCTS) to define the behavior policy.

    Attributes:
        backend: A ModelBackend instance for generating actions.
        sampling_params: SamplingParams controlling generation (temperature, etc.).
        system_prompt: Optional system prompt prepended to all messages.
    """

    backend: Any  # ModelBackend
    sampling_params: Any  # SamplingParams
    system_prompt: str | None = None


class SignalType(Enum):
    """Type of dense signal function being evaluated."""

    STATE_VALUE = auto()
    Q_VALUE = auto()
    ADVANTAGE = auto()
    POTENTIAL = auto()
    SHAPED_REWARD = auto()


class CorrelationMethod(Enum):
    """Statistical correlation method."""

    PEARSON = auto()
    SPEARMAN = auto()
    SIGN_AGREEMENT = auto()
    KENDALL_TAU = auto()


class PolicyAssumption(Enum):
    """Policy assumption for value function estimation.

    Determines whose policy is used when interpreting value functions:
    - OPTIMAL: values reflect optimal play (e.g., MC-Max / MCTS GT).
    - SELF: values reflect the evaluator LLM's own policy.
    """

    OPTIMAL = auto()
    SELF = auto()


class AggregationMethod(Enum):
    """Method for aggregating MC rollout returns."""

    MEAN = auto()
    MAX = auto()


class RankingCandidateSource(Enum):
    """How a ranking candidate action was produced."""

    ACTOR_PRIMARY = auto()
    RANKING_MANUAL = auto()
    RANKING_LLM = auto()


@dataclass(frozen=True)
class HistoryTurn:
    """A single turn in the state-action history.

    Replaces raw ``(state_text, action_text)`` tuples with a named type
    that also carries the actor's ReAct-style thought (when available)
    and any images from the observation.

    Attributes:
        state_text: Serialized observation text for this turn.
        action_text: Best-available display text for the action taken.
        thought: The actor's ReAct-style reasoning for this action,
            extracted from the raw generation.  ``None`` when the
            trajectory was not collected with ReAct prompting or when
            extraction found no thought.
        state_images: Images from the observation at this turn, separated
            by source (task vs state).  Empty for text-only environments.
    """

    state_text: str
    action_text: str
    thought: str | None = None
    state_images: ObservationImages = ObservationImages()


@dataclass(frozen=True)
class ActionCandidate:
    """A candidate action within a ranking point.

    Attributes:
        action: The llenvs Action object (used for GT Q-value computation
            via ``env.step``).
        extracted_action: Extractor output text (tier 2).
        resolved_action: Formatted native action (tier 3), e.g. ``"right"``.
        source: Whether this candidate is the actor-sampled primary action
            or an auxiliary ranking-only action from manual/LLM sampling.
        thought: The actor's ReAct-style reasoning for this candidate
            action.  Populated for ``ACTOR_PRIMARY`` candidates from the
            raw generation; ``None`` for manual/LLM-generated candidates.
    """

    action: Any
    source: RankingCandidateSource
    extracted_action: str | None = None
    resolved_action: str | None = None
    next_state: Any = None
    thought: str | None = None


@dataclass(frozen=True)
class RankingPoint:
    """A state with k candidate actions to rank.

    The LLM must rank the candidates by expected Q-value. GT is computed
    by running MC rollouts for each candidate independently.

    Attributes:
        state: The environment state at this point.
        candidates: Candidate actions to rank (already shuffled). Exactly one
            candidate must have ``source=ACTOR_PRIMARY``.
        trajectory_index: Which trajectory this point came from.
        step_index: Which step within the trajectory.
        history: Pre-computed history turns for prior steps, identical in
            structure to EvaluationPoint.history.
    """

    state: Any
    candidates: tuple[ActionCandidate, ...]
    trajectory_index: int
    step_index: int
    history: tuple[HistoryTurn, ...] = ()


@dataclass(frozen=True)
class ReplaySpec:
    """Metadata needed to restore an environment to a saved state.

    Stored alongside ``EvaluationPoint`` for non-pure environments where
    state restoration requires replaying a command prefix.

    Attributes:
        task_name: The environment task identifier.
        task_index: Index into the task list.
        trajectory: Command history to replay.
        restore_mode: How this point should be restored for rollouts.
        snapshot_ref: Relative snapshot artifact path for exact restore.
        snapshot_runtime: Runtime that produced the snapshot artifact.
        snapshot_options: Runtime-specific restore flags.
        dataset_name: Optional dataset identifier for version tracking.
        dataset_version: Optional dataset version string.
        probe_outputs: Optional mapping of probe command → stdout captured
            from the replay-validation reference restore. Only populated when
            replay validation is enabled for collection.
    """

    task_name: str
    task_index: int
    trajectory: tuple[str, ...]
    restore_mode: str = "replay"
    snapshot_ref: str | None = None
    snapshot_runtime: str | None = None
    snapshot_options: dict[str, Any] | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    probe_outputs: dict[str, str] | None = None
    fs_restore_risk_now: bool = False
    fs_restore_risk_ever: bool = False
    fs_restore_risk_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationPoint:
    """A (state, action, next_state) triple to evaluate.

    Attributes:
        state: The environment state at this point.
        action: The action taken.
        next_state: The resulting state after the action.
        trajectory_index: Which trajectory this point came from.
        step_index: Which step within the trajectory.
        extracted_action: Extractor output text (tier 2). Always set when
            extraction succeeds (the ``raw`` extractor as composite fallback
            guarantees this). ``None`` when extraction failed entirely.
        resolved_action: Formatted native action (tier 3), e.g. ``"right"``
            for gym action 2. ``None`` when the action was invalid or the
            adapter doesn't produce resolved actions.
        history: Pre-computed history turns for prior steps in the
            trajectory, built from transitions at point collection time.
        current_thought: The actor's ReAct-style reasoning for the action
            at this evaluation point.  Extracted from the raw generation
            at collection time so it survives ``prepare_display_points()``.
            ``None`` when the trajectory was not collected with ReAct
            prompting or when extraction found no thought.
        replay_spec: Optional metadata for non-pure environments where
            state restoration requires trajectory replay.
    """

    state: Any
    action: Any
    next_state: Any
    trajectory_index: int
    step_index: int
    extracted_action: str | None = None
    resolved_action: str | None = None
    history: tuple[HistoryTurn, ...] = ()
    current_thought: str | None = None
    replay_spec: ReplaySpec | None = None


@dataclass(frozen=True)
class EnvironmentContext:
    """Environment-side context loaded from per-environment YAML.

    Contains only the information needed to construct environments, collect
    trajectories, and parse actor/evaluator outputs. Unlike ``MethodContext``,
    it does not carry any dense-signal-specific semantics.
    """

    task_description: str | None = None
    reward_description: str | None = None
    example_trajectories: list[ObservableTrajectory] | None = None
    evaluator_extractor: Any = None
    environment_extractor: Any = None
    include_actor_thinking: bool = False
    adapter: str | None = None
    env_name: str | None = None
    max_steps: int | None = None
    make_kwargs: dict[str, Any] | None = None
    step_penalty: float | None = None
    turn_info: bool | None = None
    reward_signal_name: str | None = "correctness"
    min_action_chars: int | None = None
    min_observation_chars: int | None = None
    min_current_observation_chars: int | None = None
    min_next_observation_chars: int | None = None
    min_ranking_candidate_action_chars: int | None = None
    min_ranking_next_observation_chars: int | None = None
    ranking_environment_extractor: Any = None
    prompting_scheme: str | None = None
    system_prompt_file: str | None = None
    early_turns_to_discard: int | None = None
    late_turns_to_discard: int | None = None
    invalid_action_text: str | None = None
    invalid_action_observation: str | None = None
    advance_on_invalid: str | None = None
    inject_task_text_in_prompts: bool = False
    """When True, qval prompt builders prepend ``obs.task.text`` at
    the top of evaluator/ranking user prompts. Used by WebShop to surface
    ``<goal>…</goal>`` in prompts that may omit or truncate history. Other
    environments leave this as ``False``; their ``obs.task.text`` content
    continues to be ignored by the prompt builders."""
    include_state_text_when_images: bool = True
    """When observations include images, whether prompt builders should also
    include the serialized state text. Vision-only contexts may set this to
    ``False``; text-only methods can still choose to ignore images explicitly."""
    include_images: bool = True
    """When False, qval prompt builders skip image blocks even on
    vision datasets. Lets us run text-only LLM evaluations on a vision
    pickle without rebuilding the dataset."""


@dataclass(frozen=True)
class RawPointReturns:
    """Raw rollout returns for a single evaluation point, before aggregation.

    Attributes:
        point: The evaluation point these returns are for.
        returns: Per-rollout cumulative returns (V(s) or Q(s,a)).
        step_counts: Per-rollout step counts aligned with ``returns``.
        next_state_returns: Per-rollout V(s') returns (only for ADVANTAGE).
        next_state_step_counts: Per-rollout step counts aligned with
            ``next_state_returns``.
        aborted: Whether rollout collection for this point aborted before all
            requested samples were gathered. Partial successful rollouts may
            still be present in ``returns`` / ``next_state_returns``.
    """

    point: EvaluationPoint
    returns: tuple[float, ...]
    step_counts: tuple[int, ...] = ()
    next_state_returns: tuple[float, ...] = ()
    next_state_step_counts: tuple[int, ...] = ()
    aborted: bool = False


@dataclass(frozen=True)
class SignalEstimate:
    """Result of estimating a ground-truth signal value at a point.

    Attributes:
        value: The estimated signal value.
        point: The evaluation point this estimate is for.
        num_rollouts: Number of MC rollouts used.
        raw_returns: Individual returns from each rollout.
    """

    value: float
    point: EvaluationPoint
    num_rollouts: int
    raw_returns: tuple[float, ...] = ()


@dataclass(frozen=True)
class CorrelationResult:
    """Result of computing correlation between ground truth and predicted signals.

    Attributes:
        method: Which correlation method was used.
        correlation: The correlation coefficient.
        p_value: Statistical p-value.
        num_points: Number of evaluation points used.
    """

    method: CorrelationMethod
    correlation: float
    p_value: float
    num_points: int


@dataclass(frozen=True)
class MethodContext:
    """Context provided to dense signal methods.

    Bundles the information sources that methods may use to produce signals.
    All fields except signal_type are optional, allowing the benchmark to
    study the effect of each information source by varying what is provided.

    Attributes:
        signal_type: Type of dense signal the method should produce.
        task_description: Natural language description of the environment and task.
        reward_description: Description or code of the ground truth sparse reward.
        example_trajectories: Example trajectories with states, actions, and rewards.
        evaluator_extractor: Extractor for parsing evaluator LLM output to get
            numeric signal values. Any object implementing the llenvs
            AnswerExtractor protocol (``extract(str) -> (str | None, dict)``).
        environment_extractor: Extractor for parsing actor LLM output into
            environment actions. Passed to the llenvs adapter's
            ``answer_extractor`` parameter. None uses the adapter's default.
        disclose_discount_factor: Controls prompt language for return phrases.
            When ``True`` (default), prompts reveal the exact γ value.
            When ``False``, prompts use neutral "cumulative reward" language.
        efficiency_guidance: Pre-resolved efficiency hint text to include in
            system prompts. ``None`` means no hint.
    """

    signal_type: SignalType
    task_description: str | None = None
    reward_description: str | None = None
    example_trajectories: list[ObservableTrajectory] | None = None
    evaluator_extractor: Any = None
    environment_extractor: Any = None
    include_actor_thinking: bool = False
    include_current_thoughts: bool = False
    include_history_thoughts: bool = False
    include_state_text_when_images: bool = True
    include_images: bool = True
    enable_thinking: bool = True
    environment_notes: list[str] | None = None
    policy_assumption: PolicyAssumption = PolicyAssumption.OPTIMAL
    discount_factor: float = 1.0
    disclose_discount_factor: bool = True
    efficiency_guidance: str | None = None
    include_next_state: bool = True
    adapter: str | None = None
    env_name: str | None = None
    max_steps: int | None = None
    make_kwargs: dict[str, Any] | None = None
    step_penalty: float | None = None
    turn_info: bool | None = None
    reward_signal_name: str | None = "correctness"
    min_action_chars: int | None = None
    min_observation_chars: int | None = None
    min_current_observation_chars: int | None = None
    min_next_observation_chars: int | None = None
    min_ranking_candidate_action_chars: int | None = None
    min_ranking_next_observation_chars: int | None = None
    max_history_turns: int | None = None
    prompting_scheme: str | None = None
    system_prompt_file: str | None = None
    invalid_action_text: str | None = None
    invalid_action_observation: str | None = None
    advance_on_invalid: str | None = None

"""Online test-time scaling helpers for guided action selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import math
import random
import re
from typing import Any, Protocol

from llenvs.core.cleaning import strip_special_tokens, strip_thinking_tokens
from llenvs.core.environment import Environment, StepResult
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, State
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.evaluation.runner import TrajectoryResult, TrajectoryRunner
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
)

from qval.error_handling import generate_batch_with_transient_retry
from qval.evaluation_points import _build_history
from qval.methods.serialization import action_text_for_display
from qval.types import (
    ActionCandidate,
    EvaluationPoint,
    RankingCandidateSource,
    RankingPoint,
)

logger = logging.getLogger(__name__)


class CandidateScorer(Protocol):
    """Scores candidate actions represented as temporary evaluation points."""

    include_next_state: bool

    def score_candidates(self, points: list[EvaluationPoint]) -> list[float]: ...

    def score_candidate_groups(
        self, groups: list[list[EvaluationPoint]]
    ) -> list[list[float]]:
        """Score several independent candidate groups (one per trajectory).

        Optional: scorers that can batch their LLM calls across groups override
        this. Callers that want the guaranteed per-group behavior for any scorer
        should dispatch through :func:`default_score_candidate_groups`.
        """
        ...


def default_score_candidate_groups(
    scorer: Any, groups: list[list[EvaluationPoint]]
) -> list[list[float]]:
    """Score each candidate group independently via ``score_candidates``.

    The correctness-preserving fallback for any scorer: per-group isolation by
    construction (votes/consensus never cross trajectories). The batched rollout
    engine prefers ``scorer.score_candidate_groups`` when present and falls back
    to this helper otherwise.
    """
    return [scorer.score_candidates(group) for group in groups]


@dataclass(frozen=True)
class DenseSignalCandidateScorer:
    """Adapter from a QVal dense-signal method to candidate scoring."""

    method: Any
    include_next_state: bool = True

    def score_candidates(self, points: list[EvaluationPoint]) -> list[float]:
        return [float(v) for v in self.method.evaluate_batch(points)]

    def score_candidate_groups(
        self, groups: list[list[EvaluationPoint]]
    ) -> list[list[float]]:
        # Per-point scoring is independent, so flatten every group into a single
        # ``evaluate_batch`` (which chunks internally by batch_size) and re-split
        # by group boundaries — B trajectories' candidates in one batched call.
        lengths = [len(group) for group in groups]
        flat = [point for group in groups for point in group]
        if not flat:
            return [[] for _ in groups]
        flat_scores = [float(v) for v in self.method.evaluate_batch(flat)]
        out: list[list[float]] = []
        offset = 0
        for length in lengths:
            out.append(flat_scores[offset : offset + length])
            offset += length
        return out


@dataclass(frozen=True)
class ConstantCandidateScorer:
    """Diagnostic scorer that assigns the same score to every candidate."""

    value: float = 0.0
    include_next_state: bool = False

    def score_candidates(self, points: list[EvaluationPoint]) -> list[float]:
        return [self.value for _ in points]


def _permutation_to_scores(ranking: list[int] | None, n: int) -> list[float]:
    """Convert a 1-indexed best-to-worst permutation into per-candidate scores.

    The candidate at rank position ``p`` (0-based, best first) scores ``n - p``,
    so the top-ranked candidate is the argmax. A ``None`` ranking (the ranking
    method failed to produce a valid permutation) degrades to all-NaN, which the
    guided actor handles by falling back to a random candidate. Candidates not
    referenced by the permutation remain NaN.
    """
    if ranking is None:
        return [math.nan] * n
    scores = [math.nan] * n
    for position, candidate_one_indexed in enumerate(ranking):
        idx = candidate_one_indexed - 1
        if 0 <= idx < n:
            scores[idx] = float(n - position)
    return scores


@dataclass(frozen=True)
class RankingCandidateScorer:
    """Adapter from a QVal ranking method to candidate scoring.

    Builds one :class:`RankingPoint` per trajectory group from the candidate
    evaluation points, calls the ranking method once across all groups, and
    converts the output into a per-candidate score vector the guided actor
    argmaxes over.

    Two output shapes are supported via ``mode``:

    * ``"permutation"`` — ``method.rank_batch`` returns a 1-indexed best-to-worst
      ordering per group (or ``None`` on failure). Converted via
      :func:`_permutation_to_scores`.
    * ``"scores"`` — ``method.score_batch`` returns per-candidate floats per
      group, used directly (``nan`` propagates).

    Groups with fewer than two candidates short-circuit to ``[1.0]`` (nothing to
    rank), and the ranking method is not called for them.
    """

    method: Any
    mode: str
    include_next_state: bool = True

    def score_candidates(self, points: list[EvaluationPoint]) -> list[float]:
        return self.score_candidate_groups([points])[0]

    def score_candidate_groups(
        self, groups: list[list[EvaluationPoint]]
    ) -> list[list[float]]:
        results: list[list[float]] = [[] for _ in groups]
        rankable_indices: list[int] = []
        for gi, group in enumerate(groups):
            if len(group) < 2:
                results[gi] = [1.0] * len(group)
            else:
                rankable_indices.append(gi)

        if not rankable_indices:
            return results

        ranking_points = [
            self._build_ranking_point(groups[gi]) for gi in rankable_indices
        ]
        if self.mode == "permutation":
            rankings = self.method.rank_batch(ranking_points)
            for local, gi in enumerate(rankable_indices):
                results[gi] = _permutation_to_scores(rankings[local], len(groups[gi]))
        elif self.mode == "scores":
            point_scores = self.method.score_batch(ranking_points)
            for local, gi in enumerate(rankable_indices):
                results[gi] = [float(s) for s in point_scores[local]]
        else:  # pragma: no cover - guarded at construction
            raise ValueError(f"Unknown RankingCandidateScorer mode: {self.mode!r}")
        return results

    @staticmethod
    def _build_ranking_point(group: list[EvaluationPoint]) -> RankingPoint:
        candidates = tuple(
            ActionCandidate(
                action=point.action,
                source=(
                    RankingCandidateSource.ACTOR_PRIMARY
                    if i == 0
                    else RankingCandidateSource.RANKING_LLM
                ),
                extracted_action=point.extracted_action,
                resolved_action=point.resolved_action,
                next_state=point.next_state,
                thought=point.current_thought,
            )
            for i, point in enumerate(group)
        )
        first = group[0]
        return RankingPoint(
            state=first.state,
            candidates=candidates,
            trajectory_index=first.trajectory_index,
            step_index=first.step_index,
            history=first.history,
        )


@dataclass(frozen=True)
class DiscreteVotingScorer:
    """Self-consistency scorer for small discrete action spaces.

    Scores each candidate by the number of candidates that resolved to the
    same action (its vote count). ``GuidedActor`` then selects the ``argmax``,
    breaking ties at random — so the executed action belongs to a most-voted
    group. Requires stepping (``include_next_state=True``) so each candidate's
    ``resolved_action`` (the canonical normalized action) is available; the
    vote key falls back to ``extracted_action`` then to the cleaned display
    text when an action could not be resolved.

    Intentionally has no ``score_candidate_groups`` override: vote counting is
    pure CPU (no model calls to batch) and counting votes across trajectories
    would be a correctness bug, so the engine's per-group fallback is exactly
    what we want.
    """

    include_next_state: bool = True

    def score_candidates(self, points: list[EvaluationPoint]) -> list[float]:
        keys = [self._vote_key(point) for point in points]
        counts = Counter(keys)
        return [float(counts[key]) for key in keys]

    @staticmethod
    def _vote_key(point: EvaluationPoint) -> str:
        return (
            point.resolved_action
            or point.extracted_action
            or action_text_for_display(
                point.action, point.resolved_action, point.extracted_action
            )
        )


_USC_ANCHORED_PATTERN = re.compile(
    r"most consistent response is\s+response\s*#?\s*(\d+)", re.IGNORECASE
)
_USC_BARE_PATTERN = re.compile(r"\bresponse\s*#?\s*(\d+)", re.IGNORECASE)


def _parse_usc_response(text: str, k: int) -> int | None:
    """Parse a 0-indexed selection from a USC selector response.

    Looks for the anchored phrase ``"most consistent response is Response N"``
    first, then any bare ``"Response N"``; the last in-range match wins.
    Thinking and special tokens are stripped before matching. Returns ``None``
    when no in-range index is found.
    """
    cleaned = strip_thinking_tokens(strip_special_tokens(text or ""))
    for pattern in (_USC_ANCHORED_PATTERN, _USC_BARE_PATTERN):
        for match in reversed(pattern.findall(cleaned)):
            index = int(match)
            if 1 <= index <= k:
                return index - 1
    return None


@dataclass(frozen=True)
class OpenEndedSelfConsistencyScorer:
    """Universal Self-Consistency (USC) scorer for open-ended action spaces.

    Presents each candidate's extracted action to a selector LLM and asks it
    to pick the most consistent one (Universal Self-Consistency, Chen et al.
    2023). Does not step the environment (``include_next_state=False``), so it
    works on restore-based environments where stepping every candidate is
    expensive. Returns a one-hot score vector for the selected candidate, or
    all-NaN when the selector response cannot be parsed — in which case
    ``GuidedActor`` falls back to a random candidate.

    When every candidate renders to the same action text the selection is
    outcome-invariant, so the selector call is skipped and the first candidate
    is chosen.
    """

    selector_backend: ModelBackend
    selector_sampling_params: SamplingParams
    extractor: Any | None = None
    max_candidate_chars: int = 4000
    include_next_state: bool = False

    def score_candidates(self, points: list[EvaluationPoint]) -> list[float]:
        n = len(points)
        if n == 0:
            return []
        if n == 1:
            return [1.0]
        candidate_texts = [self._candidate_text(point) for point in points]
        if len(set(candidate_texts)) == 1:
            # All candidates render identically — selection is outcome-invariant.
            return [1.0 if i == 0 else 0.0 for i in range(n)]
        messages = [
            ChatMessage(role="user", content=self._build_prompt(candidate_texts))
        ]
        # Route through the transient-retry helper (single-element batch) so a
        # dropped selector connection is retried, not propagated. An exhausted
        # retry yields a None slot, which degrades to all-NaN below — exactly
        # the unparseable-response fallback (GuidedActor picks at random).
        result = generate_batch_with_transient_retry(
            self.selector_backend, [messages], self.selector_sampling_params
        )[0]
        index = None if result is None else _parse_usc_response(result.text or "", n)
        if index is None:
            return [math.nan] * n
        return [1.0 if i == index else 0.0 for i in range(n)]

    def score_candidate_groups(
        self, groups: list[list[EvaluationPoint]]
    ) -> list[list[float]]:
        # One selector prompt per group, issued as a single batched call. Each
        # group keeps its own prompt — candidates are never merged across groups.
        results: list[list[float]] = [[] for _ in groups]
        prompt_indices: list[int] = []
        prompt_messages: list[list[ChatMessage]] = []
        for gi, points in enumerate(groups):
            n = len(points)
            if n == 1:
                results[gi] = [1.0]
            elif n >= 2:
                candidate_texts = [self._candidate_text(point) for point in points]
                if len(set(candidate_texts)) == 1:
                    # All candidates identical — selection is outcome-invariant.
                    results[gi] = [1.0 if i == 0 else 0.0 for i in range(n)]
                else:
                    prompt_indices.append(gi)
                    prompt_messages.append(
                        [
                            ChatMessage(
                                role="user",
                                content=self._build_prompt(candidate_texts),
                            )
                        ]
                    )
            # n == 0 keeps the empty default.
        if prompt_messages:
            # Transient-retry helper: a dropped slot is retried (only the failed
            # prompts are re-issued); a slot that exhausts retries comes back as
            # None and degrades that group to all-NaN, leaving the other groups
            # intact rather than crashing the whole run on a PartialBatchError.
            gen_results = generate_batch_with_transient_retry(
                self.selector_backend, prompt_messages, self.selector_sampling_params
            )
            for local, gi in enumerate(prompt_indices):
                n = len(groups[gi])
                gen_result = gen_results[local]
                index = (
                    None
                    if gen_result is None
                    else _parse_usc_response(gen_result.text or "", n)
                )
                if index is None:
                    results[gi] = [math.nan] * n
                else:
                    results[gi] = [1.0 if i == index else 0.0 for i in range(n)]
        return results

    def _candidate_text(self, point: EvaluationPoint) -> str:
        extracted: str | None = None
        if self.extractor is not None and point.action is not None:
            raw = getattr(point.action, "text", None) or ""
            value, _ = self.extractor.extract(raw)
            extracted = (value or "").strip() or None
        if extracted is None:
            extracted = point.extracted_action
        text = action_text_for_display(point.action, None, extracted)
        if len(text) > self.max_candidate_chars:
            text = text[: self.max_candidate_chars] + " …[truncated]"
        return text

    @staticmethod
    def _build_prompt(candidate_texts: list[str]) -> str:
        listing = "\n".join(
            f"Response {i + 1}: {text}" for i, text in enumerate(candidate_texts)
        )
        return (
            "I have generated the following candidate next actions for the "
            "current situation:\n\n"
            f"{listing}\n\n"
            "Select the response that is most consistent with the others — the "
            "single action that best represents the consensus among these "
            "candidates. You must choose exactly one. If there is no clear "
            "majority, choose the response that is most representative of the "
            "candidates overall. Reason briefly first if it helps (at most two "
            'or three sentences), then end your response with "The most '
            'consistent response is Response X", replacing X with that '
            "response's number."
        )


@dataclass(frozen=True)
class CandidateSelection:
    """Selected action and metadata from one guided-actor decision."""

    action: Action
    selected_index: int
    candidate_actions: tuple[Action, ...]
    candidate_texts: tuple[str, ...]
    scores: tuple[float, ...]
    fallback_reason: str | None
    used_scorer: bool
    selected_step_result: StepResult[Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "k": len(self.candidate_actions),
            "candidate_actions": list(self.candidate_texts),
            "scores": [
                None if math.isnan(score) else score for score in self.scores
            ],
            "selected_index": self.selected_index,
            "fallback_reason": self.fallback_reason,
            "used_scorer": self.used_scorer,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class GuidedActor:
    """Actor that samples candidate actions and optionally scores them."""

    actor_backend: ModelBackend
    actor_sampling_params: SamplingParams
    k: int = 1
    scorer: CandidateScorer | None = None
    rng: random.Random | None = None
    system_prompt: str | None = None
    history_fn: Any | None = None
    prompt_budget: Any | None = None
    turn_info: Any | None = None
    format_reminder: str | None = None
    env_factory: Any | None = None
    restore_fn: Any | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.rng is None:
            self.rng = random.Random()

    def select_action(
        self,
        *,
        env: Environment[Any],
        state: State[Any],
        trajectory: Trajectory[Any],
        trajectory_index: int,
    ) -> CandidateSelection:
        gen_results = self._generate_candidates(env, state, trajectory)
        actions = tuple(result.to_agent_action() for result in gen_results)
        prompt_tokens = sum(result.prompt_tokens for result in gen_results)
        completion_tokens = sum(result.completion_tokens for result in gen_results)

        if self.scorer is None:
            selected_index = 0 if len(actions) == 1 else self.rng.randrange(len(actions))
            return CandidateSelection(
                action=actions[selected_index],
                selected_index=selected_index,
                candidate_actions=actions,
                candidate_texts=tuple(_action_text(action) for action in actions),
                scores=(),
                fallback_reason=None,
                used_scorer=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        include_next_state = bool(getattr(self.scorer, "include_next_state", True))
        step_results: list[StepResult[Any] | None] = [None] * len(actions)
        if include_next_state:
            if not env.spec.pure_step and (
                self.env_factory is None or self.restore_fn is None
            ):
                raise ValueError(
                    "scorer include_next_state=True requires a pure_step "
                    "environment or restore support"
                )
            step_results = [
                self._step_candidate(env, state, action) for action in actions
            ]

        points = self._candidate_points(
            state=state,
            trajectory=trajectory,
            trajectory_index=trajectory_index,
            actions=actions,
            step_results=step_results,
        )
        raw_scores = self.scorer.score_candidates(points)
        if len(raw_scores) != len(actions):
            raise ValueError(
                "Candidate scorer returned "
                f"{len(raw_scores)} scores for {len(actions)} candidates"
            )
        scores = tuple(float(score) for score in raw_scores)
        selected_index, fallback_reason = self._select_index(scores)
        return CandidateSelection(
            action=actions[selected_index],
            selected_index=selected_index,
            candidate_actions=actions,
            candidate_texts=tuple(_action_text(action) for action in actions),
            scores=scores,
            fallback_reason=fallback_reason,
            used_scorer=True,
            selected_step_result=(
                step_results[selected_index] if env.spec.pure_step else None
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _generate_candidates(
        self,
        env: Environment[Any],
        state: State[Any],
        trajectory: Trajectory[Any],
    ) -> list[GenerationResult]:
        runner = TrajectoryRunner(
            environment=env,
            backend=self.actor_backend,
            sampling_params=self.actor_sampling_params,
            system_prompt=self.system_prompt,
            history_fn=self.history_fn,
            prompt_budget=self.prompt_budget,
            turn_info=self.turn_info,
            format_reminder=self.format_reminder,
        )
        messages = runner._build_messages(state, trajectory=trajectory)
        if self.k == 1:
            return [self.actor_backend.generate_chat(messages, self.actor_sampling_params)]
        return self.actor_backend.generate_chat_batch(
            [messages for _ in range(self.k)],
            self.actor_sampling_params,
        )

    def _step_candidate(
        self,
        env: Environment[Any],
        state: State[Any],
        action: Action,
    ) -> StepResult[Any]:
        if env.spec.pure_step:
            return env.step(state, action)
        step_env = self.env_factory()
        try:
            restored = self.restore_fn(step_env, state)
            return step_env.step(restored, action)
        finally:
            close = getattr(step_env, "close", None)
            if callable(close):
                close()

    def _candidate_points(
        self,
        *,
        state: State[Any],
        trajectory: Trajectory[Any],
        trajectory_index: int,
        actions: tuple[Action, ...],
        step_results: list[StepResult[Any] | None],
    ) -> list[EvaluationPoint]:
        step_index = len(trajectory.transitions)
        history = _build_history(trajectory.transitions, step_index)
        points: list[EvaluationPoint] = []
        for action, step_result in zip(actions, step_results, strict=True):
            points.append(
                EvaluationPoint(
                    state=state,
                    action=action,
                    next_state=(
                        None if step_result is None else step_result.next_state
                    ),
                    trajectory_index=trajectory_index,
                    step_index=step_index,
                    extracted_action=(
                        None if step_result is None else step_result.extracted_action
                    ),
                    resolved_action=(
                        None if step_result is None else step_result.resolved_action
                    ),
                    history=history,
                )
            )
        return points

    def _select_index(
        self, scores: tuple[float, ...], rng: random.Random | None = None
    ) -> tuple[int, str | None]:
        rng = rng or self.rng
        assert rng is not None
        finite = [
            (idx, score) for idx, score in enumerate(scores) if math.isfinite(score)
        ]
        if finite:
            max_score = max(score for _, score in finite)
            best = [idx for idx, score in finite if score == max_score]
            return rng.choice(best), None
        return rng.randrange(len(scores)), "all_scores_nan"

    # ------------------------------------------------------------------
    # Batched helpers (used by run_guided_actors_batch). These generalize the
    # single-trajectory path above so B trajectories share one generation /
    # scoring call per decision step. select_action / run_guided_actor are left
    # untouched and remain the B=1 path.
    # ------------------------------------------------------------------

    def _runner(self, env: Environment[Any]) -> TrajectoryRunner:
        return TrajectoryRunner(
            environment=env,
            backend=self.actor_backend,
            sampling_params=self.actor_sampling_params,
            system_prompt=self.system_prompt,
            history_fn=self.history_fn,
            prompt_budget=self.prompt_budget,
            turn_info=self.turn_info,
            format_reminder=self.format_reminder,
        )

    def _generate_candidate_batch(
        self,
        messages_per_traj: list[list[ChatMessage]],
        batch_size: int | None,
    ) -> list[list[GenerationResult | None]]:
        """Flatten ``active × k`` into one (chunked) batched generation.

        Returns one ``k``-length list of results per trajectory. A ``None`` slot
        is a candidate whose generation failed (transient retry exhausted or a
        too-long prompt); the caller drops it.
        """
        k = self.k
        flat = [messages for messages in messages_per_traj for _ in range(k)]
        # Point-aligned chunking so a chunk never straddles a trajectory's k
        # candidates (mirrors mc_method's effective batch size).
        effective = batch_size
        if batch_size is not None and k > 0 and batch_size >= k:
            effective = (batch_size // k) * k

        flat_results: list[GenerationResult | None] = []
        if not effective or effective <= 0:
            flat_results = generate_batch_with_transient_retry(
                self.actor_backend, flat, self.actor_sampling_params
            )
        else:
            for start in range(0, len(flat), effective):
                chunk = flat[start : start + effective]
                flat_results.extend(
                    generate_batch_with_transient_retry(
                        self.actor_backend, chunk, self.actor_sampling_params
                    )
                )
        return [
            flat_results[i * k : (i + 1) * k]
            for i in range(len(messages_per_traj))
        ]

    def _score_groups(
        self, groups: list[list[EvaluationPoint]]
    ) -> list[list[float]]:
        """Score per-trajectory candidate groups, batching across groups when the
        scorer supports it; validates the per-group score count."""
        assert self.scorer is not None
        group_fn = getattr(self.scorer, "score_candidate_groups", None)
        raw_groups = (
            group_fn(groups)
            if group_fn is not None
            else default_score_candidate_groups(self.scorer, groups)
        )
        out: list[list[float]] = []
        for scores, group in zip(raw_groups, groups, strict=True):
            if len(scores) != len(group):
                raise ValueError(
                    "Candidate scorer returned "
                    f"{len(scores)} scores for {len(group)} candidates"
                )
            out.append([float(score) for score in scores])
        return out

    def _build_selection(
        self,
        *,
        actions: list[Action],
        step_results: list[StepResult[Any] | None],
        scores: list[float] | None,
        pure_step: bool,
        rng: random.Random,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CandidateSelection:
        """Build a CandidateSelection from one trajectory's candidates, mirroring
        select_action's selection logic (no scorer -> random; scorer -> argmax)."""
        candidate_texts = tuple(_action_text(action) for action in actions)
        if scores is None:
            selected_index = 0 if len(actions) == 1 else rng.randrange(len(actions))
            return CandidateSelection(
                action=actions[selected_index],
                selected_index=selected_index,
                candidate_actions=tuple(actions),
                candidate_texts=candidate_texts,
                scores=(),
                fallback_reason=None,
                used_scorer=False,
                selected_step_result=(
                    step_results[selected_index] if pure_step else None
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        scores_tuple = tuple(float(score) for score in scores)
        selected_index, fallback_reason = self._select_index(scores_tuple, rng=rng)
        return CandidateSelection(
            action=actions[selected_index],
            selected_index=selected_index,
            candidate_actions=tuple(actions),
            candidate_texts=candidate_texts,
            scores=scores_tuple,
            fallback_reason=fallback_reason,
            used_scorer=True,
            # Pure-step: reuse the candidate's StepResult. Restore-based: None, so
            # the selected action is re-stepped on the persistent instance (see
            # _step_selected for why this re-step is required, not wasteful).
            selected_step_result=(
                step_results[selected_index] if pure_step else None
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def run_guided_actor(
    *,
    env: Environment[Any],
    actor: GuidedActor,
    task_index: int,
    trajectory_index: int,
    max_steps: int | None = None,
) -> TrajectoryResult:
    """Run one trajectory using a guided actor."""
    state, reset_info = env.reset(options={"task_index": task_index})
    trajectory: Trajectory[Any] = Trajectory.create(state)
    limit = max_steps or env.spec.max_steps or 100
    step_count = 0

    while not state.metadata.is_terminal and step_count < limit:
        selection = actor.select_action(
            env=env,
            state=state,
            trajectory=trajectory,
            trajectory_index=trajectory_index,
        )
        step_result = selection.selected_step_result
        if step_result is None:
            step_result = env.step(state, selection.action)

        transition = Transition(
            state=state,
            action=selection.action,
            next_state=step_result.next_state,
            rewards=step_result.rewards,
            extracted_action=step_result.extracted_action,
            resolved_action=step_result.resolved_action,
            info={
                "step": step_result.info,
                "test_time_scaling": selection.metadata(),
            },
        )
        trajectory.add_transition(transition)
        state = step_result.next_state
        step_count += 1
        if step_result.done:
            break

    success = False
    if trajectory.transitions:
        outcome_rewards = trajectory.transitions[-1].rewards.by_type(
            RewardType.OUTCOME
        )
        if outcome_rewards:
            success = outcome_rewards[-1].reward >= 1.0

    return TrajectoryResult(
        trajectory=trajectory,
        total_reward=trajectory.total_reward,
        success=success,
        metadata={
            "task_index": task_index,
            "num_steps": len(trajectory),
            "reset_info": reset_info,
        },
    )


@dataclass
class _BatchTrajectory:
    """Mutable per-trajectory state tracked during a batched rollout."""

    position: int
    task_index: int
    trajectory_index: int
    env_instance: Any
    state: State[Any]
    trajectory: Trajectory[Any]
    rng: random.Random
    reset_info: dict[str, Any]
    step_count: int = 0
    done: bool = False
    failed: bool = False
    error: Exception | None = None


def _is_active(traj: _BatchTrajectory, limit: int) -> bool:
    return (
        not traj.done
        and not traj.failed
        and not traj.state.metadata.is_terminal
        and traj.step_count < limit
    )


def _apply_step(
    traj: _BatchTrajectory,
    selection: CandidateSelection,
    step_result: StepResult[Any],
) -> None:
    transition = Transition(
        state=traj.state,
        action=selection.action,
        next_state=step_result.next_state,
        rewards=step_result.rewards,
        extracted_action=step_result.extracted_action,
        resolved_action=step_result.resolved_action,
        info={
            "step": step_result.info,
            "test_time_scaling": selection.metadata(),
        },
    )
    traj.trajectory.add_transition(transition)
    traj.state = step_result.next_state
    traj.step_count += 1
    if step_result.done:
        traj.done = True


def _finalize_batch_trajectory(traj: _BatchTrajectory) -> TrajectoryResult:
    success = False
    if traj.trajectory.transitions:
        outcome_rewards = traj.trajectory.transitions[-1].rewards.by_type(
            RewardType.OUTCOME
        )
        if outcome_rewards:
            success = outcome_rewards[-1].reward >= 1.0
    return TrajectoryResult(
        trajectory=traj.trajectory,
        total_reward=traj.trajectory.total_reward,
        success=success,
        metadata={
            "task_index": traj.task_index,
            "trajectory_index": traj.trajectory_index,
            "num_steps": len(traj.trajectory),
            "reset_info": traj.reset_info,
        },
    )


def _run_guided_tick(
    actor: GuidedActor,
    runner: TrajectoryRunner,
    env: Environment[Any],
    active: list[_BatchTrajectory],
    *,
    pure_step: bool,
    batch_size: int | None,
    restore_concurrency: int,
) -> None:
    """Advance every active trajectory by one decision step, batching the actor
    generation (and dense-signal scoring) across all of them."""
    # 1. One batched generation across active × k candidates.
    messages_per_traj = [
        runner._build_messages(traj.state, trajectory=traj.trajectory)
        for traj in active
    ]
    cand_results = actor._generate_candidate_batch(messages_per_traj, batch_size)

    # Drop failed candidate slots; a trajectory with no surviving candidate fails.
    per_traj_actions: list[tuple[list[Action], int, int] | None] = []
    for traj, results in zip(active, cand_results, strict=True):
        good = [result for result in results if result is not None]
        if not good:
            traj.failed = True
            traj.done = True
            traj.error = RuntimeError("all candidate generations failed")
            per_traj_actions.append(None)
            continue
        actions = [result.to_agent_action() for result in good]
        prompt_tokens = sum(result.prompt_tokens for result in good)
        completion_tokens = sum(result.completion_tokens for result in good)
        per_traj_actions.append((actions, prompt_tokens, completion_tokens))

    survivors = [
        (idx, traj)
        for idx, traj in enumerate(active)
        if per_traj_actions[idx] is not None
    ]
    if not survivors:
        return

    scorer = actor.scorer
    include_next_state = scorer is not None and bool(
        getattr(scorer, "include_next_state", True)
    )

    # 2. Optionally step every candidate (needed when the scorer wants next_state).
    step_results_by_idx: dict[int, list[StepResult[Any] | None]] = {
        idx: [None] * len(per_traj_actions[idx][0]) for idx, _ in survivors
    }
    if include_next_state:
        jobs: list[tuple[int, int, State[Any], Action]] = []
        for idx, traj in survivors:
            for cand_idx, action in enumerate(per_traj_actions[idx][0]):
                jobs.append((idx, cand_idx, traj.state, action))

        def _step_one(job: tuple[int, int, State[Any], Action]):
            idx, cand_idx, state, action = job
            try:
                return idx, cand_idx, actor._step_candidate(env, state, action)
            except Exception:
                # Isolate one candidate's restore/step failure from the rest of
                # the wave, but make it visible and let the scorer exclude it
                # (see the NaN-out below) instead of silently mis-scoring it.
                logger.warning(
                    "test_time_scaling: candidate step failed "
                    "(active_index=%s, candidate=%s); scoring it as NaN",
                    idx,
                    cand_idx,
                    exc_info=True,
                )
                return idx, cand_idx, None

        if pure_step:
            for job in jobs:
                idx, cand_idx, result = _step_one(job)
                step_results_by_idx[idx][cand_idx] = result
        else:
            with ThreadPoolExecutor(
                max_workers=max(1, restore_concurrency)
            ) as executor:
                for idx, cand_idx, result in executor.map(_step_one, jobs):
                    step_results_by_idx[idx][cand_idx] = result

    # 3. Build per-trajectory evaluation-point groups + score them in one batch.
    groups = [
        actor._candidate_points(
            state=traj.state,
            trajectory=traj.trajectory,
            trajectory_index=traj.trajectory_index,
            actions=tuple(per_traj_actions[idx][0]),
            step_results=step_results_by_idx[idx],
        )
        for idx, traj in survivors
    ]
    scores_per_group = None if scorer is None else actor._score_groups(groups)

    # A candidate whose step failed (None step-result under include_next_state)
    # cannot be meaningfully scored with its next state, so exclude it from the
    # argmax via NaN rather than trusting whatever the scorer returned for the
    # missing next_state. A group where every candidate failed becomes all-NaN,
    # which falls back to a random candidate. Only failures matter here: when the
    # scorer omits next state, candidates are never stepped (no None to confuse).
    if include_next_state and scores_per_group is not None:
        for order, (idx, _traj) in enumerate(survivors):
            step_results = step_results_by_idx[idx]
            scores_per_group[order] = [
                math.nan if step_result is None else score
                for score, step_result in zip(
                    scores_per_group[order], step_results, strict=True
                )
            ]

    # 4. Select per trajectory and step the chosen action.
    selections: list[tuple[int, _BatchTrajectory, CandidateSelection]] = []
    for order, (idx, traj) in enumerate(survivors):
        actions, prompt_tokens, completion_tokens = per_traj_actions[idx]
        selection = actor._build_selection(
            actions=actions,
            step_results=step_results_by_idx[idx],
            scores=None if scorer is None else scores_per_group[order],
            pure_step=pure_step,
            rng=traj.rng,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        selections.append((idx, traj, selection))

    def _step_selected(traj: _BatchTrajectory, selection: CandidateSelection):
        step_result = selection.selected_step_result
        if step_result is None:
            # Restore-based envs (selected_step_result is None below) intentionally
            # re-step the chosen action on the persistent instance: the transient
            # env that scored this candidate has already been closed, and its
            # StepResult cannot advance the persistent instance's own (container)
            # state. Pure-step envs skip this re-step by reusing the stored
            # StepResult. This extra restore only happens with include_next_state.
            step_result = traj.env_instance.step(traj.state, selection.action)
        return step_result

    if pure_step:
        for _idx, traj, selection in selections:
            try:
                step_result = _step_selected(traj, selection)
            except Exception as exc:
                traj.failed = True
                traj.done = True
                traj.error = exc
                continue
            _apply_step(traj, selection, step_result)
    else:
        def _do_selected(item):
            _idx, traj, selection = item
            try:
                return traj, selection, _step_selected(traj, selection), None
            except Exception as exc:  # noqa: BLE001
                return traj, selection, None, exc

        with ThreadPoolExecutor(max_workers=max(1, restore_concurrency)) as executor:
            stepped = list(executor.map(_do_selected, selections))
        for traj, selection, step_result, exc in stepped:
            if exc is not None:
                traj.failed = True
                traj.done = True
                traj.error = exc
                continue
            _apply_step(traj, selection, step_result)


def run_guided_actors_batch(
    *,
    env: Environment[Any],
    actor: GuidedActor,
    task_indices: Sequence[int],
    trajectory_index_base: int = 0,
    trajectory_indices: Sequence[int] | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    restore_concurrency: int = 16,
    base_seed: int = 0,
    on_trajectory_complete: (
        Callable[[int, TrajectoryResult | None, Exception | None], None] | None
    ) = None,
) -> list[tuple[TrajectoryResult | None, Exception | None]]:
    """Run a wave of trajectories concurrently, batching per-step generations.

    Advances all in-flight trajectories in lockstep: each decision step issues a
    single ``generate_chat_batch`` over ``active × k`` candidates and (for dense
    scorers) a single batched scoring call. Returns one ``(result, error)`` slot
    per input index, in order; a trajectory that errors yields ``(None, exc)``
    without aborting the others.

    For ``pure_step`` environments a single shared ``env`` serves every
    trajectory. For restore-based environments one persistent instance per
    trajectory is created from ``actor.env_factory`` (reset in parallel, closed
    at the end); transient per-candidate envs used for scoring are bounded by
    ``restore_concurrency``. Per-trajectory tie-break/fallback RNG is derived
    from ``base_seed + trajectory_index`` so results are independent of wave size.

    ``trajectory_indices`` assigns an explicit global index per slot (resume
    runs leave holes — completed indices are scattered, so a contiguous base
    cannot express them); it defaults to ``trajectory_index_base + position``.
    ``on_trajectory_complete`` fires the moment a trajectory finishes —
    eagerly, not at wave end — with ``(trajectory_index, result_or_None,
    error_or_None)``; callback exceptions are logged and never abort the wave.
    """
    indices = list(task_indices)
    n = len(indices)
    if trajectory_indices is not None:
        traj_index_for_pos = list(trajectory_indices)
        if len(traj_index_for_pos) != n:
            raise ValueError(
                "trajectory_indices must match task_indices length "
                f"({len(traj_index_for_pos)} != {n})"
            )
    else:
        traj_index_for_pos = [trajectory_index_base + pos for pos in range(n)]
    results: list[tuple[TrajectoryResult | None, Exception | None]] = [
        (None, None) for _ in range(n)
    ]
    if n == 0:
        return results

    def _notify(
        trajectory_index: int,
        result: TrajectoryResult | None,
        error: Exception | None,
    ) -> None:
        if on_trajectory_complete is None:
            return
        try:
            on_trajectory_complete(trajectory_index, result, error)
        except Exception:
            logger.warning(
                "on_trajectory_complete callback failed for trajectory %d",
                trajectory_index,
                exc_info=True,
            )

    finalized: set[int] = set()

    def _complete(traj: _BatchTrajectory) -> None:
        if traj.position in finalized:
            return
        finalized.add(traj.position)
        if traj.failed:
            results[traj.position] = (None, traj.error)
            _notify(traj.trajectory_index, None, traj.error)
        else:
            result = _finalize_batch_trajectory(traj)
            results[traj.position] = (result, None)
            _notify(traj.trajectory_index, result, None)

    pure_step = env.spec.pure_step
    if not pure_step:
        # env_factory is always needed on a non-pure env to build the persistent
        # per-trajectory instances. restore_fn is only needed when a scorer steps
        # each candidate (include_next_state) via a transient restored env — a
        # forward-only actor (no scorer, e.g. baseline_k1 / self_consistency)
        # never touches it. This mirrors GuidedActor.select_action's own guard.
        if actor.env_factory is None:
            raise ValueError(
                "batched rollout on a non-pure environment requires env_factory "
                "on the actor"
            )
        needs_restore = actor.scorer is not None and bool(
            getattr(actor.scorer, "include_next_state", True)
        )
        if needs_restore and actor.restore_fn is None:
            raise ValueError(
                "batched rollout on a non-pure environment with a next-state "
                "scorer requires restore_fn on the actor"
            )

    limit = max_steps or env.spec.max_steps or 100
    runner = actor._runner(env)

    trajs_by_pos: list[_BatchTrajectory | None] = [None] * n
    instances: list[Any] = [None] * n

    def _make_traj(pos: int, state: State[Any], reset_info: dict[str, Any], inst: Any):
        return _BatchTrajectory(
            position=pos,
            task_index=indices[pos],
            trajectory_index=traj_index_for_pos[pos],
            env_instance=inst,
            state=state,
            trajectory=Trajectory.create(state),
            rng=random.Random(base_seed + traj_index_for_pos[pos]),
            reset_info=reset_info,
        )

    try:
        # Allocate + reset one env instance per trajectory.
        if pure_step:
            for pos, task_index in enumerate(indices):
                try:
                    state, reset_info = env.reset(options={"task_index": task_index})
                except Exception as exc:  # noqa: BLE001
                    results[pos] = (None, exc)
                    _notify(traj_index_for_pos[pos], None, exc)
                    continue
                instances[pos] = env
                trajs_by_pos[pos] = _make_traj(pos, state, reset_info, env)
        else:
            for pos in range(n):
                instances[pos] = actor.env_factory()

            def _reset_job(pos: int):
                try:
                    state, reset_info = instances[pos].reset(
                        options={"task_index": indices[pos]}
                    )
                    return pos, state, reset_info, None
                except Exception as exc:  # noqa: BLE001
                    return pos, None, None, exc

            with ThreadPoolExecutor(
                max_workers=max(1, min(restore_concurrency, n))
            ) as executor:
                for pos, state, reset_info, exc in executor.map(
                    _reset_job, range(n)
                ):
                    if exc is not None:
                        results[pos] = (None, exc)
                        _notify(traj_index_for_pos[pos], None, exc)
                        continue
                    trajs_by_pos[pos] = _make_traj(
                        pos, state, reset_info, instances[pos]
                    )

        all_trajs = [traj for traj in trajs_by_pos if traj is not None]

        # Lockstep tick loop until every trajectory is done or failed.
        # Trajectories are finalized — and the completion callback fired — the
        # moment they leave the active set, not at wave end, so a crash mid-wave
        # only loses the trajectories still in flight.
        while True:
            for traj in all_trajs:
                if traj.position not in finalized and not _is_active(traj, limit):
                    _complete(traj)
            active = [traj for traj in all_trajs if _is_active(traj, limit)]
            if not active:
                break
            _run_guided_tick(
                actor,
                runner,
                env,
                active,
                pure_step=pure_step,
                batch_size=batch_size,
                restore_concurrency=restore_concurrency,
            )
    finally:
        if not pure_step:
            for inst in instances:
                if inst is None:
                    continue
                close = getattr(inst, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass

    # Defensive: a no-op on normal exit (every trajectory was completed inside
    # the loop), kept so no slot can ever be left unfilled.
    for traj in all_trajs:
        _complete(traj)
    return results


def _action_text(action: Action) -> str:
    return action_text_for_display(action, None, None)

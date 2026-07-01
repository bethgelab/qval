"""Offline Belief-RL-style ranking using success-belief change."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

from llenvs.inference import compose_system_prompt
from llenvs.inference.protocol import (
    ChatMessage,
    ModelBackend,
    ScoringResult,
)

from qval.error_handling import is_recoverable_backend_error
from qval.methods.serialization import (
    action_text_for_display,
    serialize_observation,
)
from qval.prompt_presets import PromptPreset, resolve_block
from qval.prompts import build_signal_type_fragment
from qval.types import MethodContext, PolicyAssumption, RankingPoint, SignalType

logger = logging.getLogger(__name__)

_DEFAULT_POSITIVE_VERBALIZERS: tuple[str, ...] = ("Yes",)
_DEFAULT_NEGATIVE_VERBALIZERS: tuple[str, ...] = ("No",)


def _history_turn_text(turn: Any) -> tuple[str, str]:
    if hasattr(turn, "state_text") and hasattr(turn, "action_text"):
        return str(turn.state_text), str(turn.action_text)
    if isinstance(turn, (tuple, list)) and len(turn) >= 2:
        return str(turn[0]), str(turn[1])
    raise TypeError(f"Unsupported history turn type: {type(turn).__name__}")


def _format_environment_notes(notes: list[str] | None) -> str | None:
    if not notes:
        return None
    if len(notes) == 1:
        return f"## Episode Configuration\n{notes[0]}"
    lines = "\n".join(f"- {note}" for note in notes)
    return f"## Episode Configuration\n{lines}"


def _build_history_block(
    history: list[Any] | None,
    *,
    history_truncated_from: int | None,
) -> str | None:
    if not history:
        return None

    history_lines = ["**State-Action History:**"]
    if history_truncated_from is not None and len(history) < history_truncated_from:
        history_lines.append(
            f"[Note: Only the last {len(history)} of "
            f"{history_truncated_from} turns are shown.]"
        )
    for i, turn in enumerate(history, 1):
        state_text, action_text = _history_turn_text(turn)
        history_lines.append(
            f"[Turn {i} - State]\n{state_text}\n[Turn {i} - Action]\n{action_text}"
        )
    return "\n\n".join(history_lines)


def _policy_success_clause(policy: PolicyAssumption) -> str:
    if policy == PolicyAssumption.OPTIMAL:
        return "assuming optimal future decisions thereafter"
    return "assuming the same agent will make all future decisions thereafter"


def _success_question(policy: PolicyAssumption) -> str:
    return (
        "Question: Given the task and reward function, is the agent likely to "
        "eventually achieve a successful overall outcome from here, "
        f"{_policy_success_clause(policy)}?\n"
        "Answer with exactly one word: Yes or No."
    )


def _build_pre_user_prompt(
    *,
    point: RankingPoint,
    history: list[Any] | None,
    history_truncated_from: int | None,
    policy_assumption: PolicyAssumption,
    state_text_override: str | None = None,
) -> str:
    parts = [
        "Assess the current situation before any candidate outcome is shown.",
    ]
    history_block = _build_history_block(
        history, history_truncated_from=history_truncated_from,
    )
    if history_block:
        parts.append(history_block)

    state_text = state_text_override or serialize_observation(point.state)
    parts.append(f"**Current State:**\n{state_text}")
    parts.append(
        "Do not answer directly. The assistant continuation being scored is the "
        "one-word answer to the success question below."
    )
    parts.append(_success_question(policy_assumption))
    return "\n\n".join(parts)


def _build_post_user_prompt(
    *,
    point: RankingPoint,
    candidate_index: int,
    history: list[Any] | None,
    history_truncated_from: int | None,
    policy_assumption: PolicyAssumption,
    include_next_state: bool,
    feedback_text: str | None,
    state_text_override: str | None = None,
) -> str:
    candidate = point.candidates[candidate_index]
    candidate_text = action_text_for_display(
        candidate.action, candidate.resolved_action, candidate.extracted_action,
    )

    parts = [
        "Re-assess the same situation after one specific candidate action has "
        "been attempted and its observed outcome is available.",
    ]
    history_block = _build_history_block(
        history, history_truncated_from=history_truncated_from,
    )
    if history_block:
        parts.append(history_block)

    state_text = state_text_override or serialize_observation(point.state)
    parts.append(f"**Current State:**\n{state_text}")
    parts.append(f"**Candidate Action:**\n{candidate_text}")

    if include_next_state:
        if feedback_text is not None:
            parts.append(f"**Observed Resulting State:**\n{feedback_text}")
        else:
            parts.append(
                "**Observed Resulting State:**\n"
                "[No explicit resulting-state feedback was available.]"
            )

    parts.append(
        "Do not answer directly. The assistant continuation being scored is the "
        "one-word answer to the success question below."
    )
    parts.append(_success_question(policy_assumption))
    return "\n\n".join(parts)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(v - max_value) for v in values))


class DeltaBeliefRankingMethod:
    """Rank candidates by post-minus-pre log success belief."""

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        *,
        prompt_preset: PromptPreset,
        batch_size: int | None = None,
        max_history_turns: int | None = None,
        estimate_tokens: Callable[[str], int] | None = None,
        max_prompt_tokens: int | None = None,
        min_observation_chars: int | None = None,
        min_current_observation_chars: int | None = None,
        min_next_observation_chars: int | None = None,
        positive_verbalizers: tuple[str, ...] = _DEFAULT_POSITIVE_VERBALIZERS,
        negative_verbalizers: tuple[str, ...] = _DEFAULT_NEGATIVE_VERBALIZERS,
    ) -> None:
        self._backend = backend
        self._scoring_backend = getattr(backend, "_backend", backend)
        self._context = context
        self._prompt_preset = prompt_preset
        self._batch_size = batch_size
        self._max_history_turns = max_history_turns
        self._estimate_tokens = estimate_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._min_observation_chars = min_observation_chars
        self._min_current_observation_chars = min_current_observation_chars
        self._min_next_observation_chars = min_next_observation_chars
        self._positive_verbalizers = positive_verbalizers
        self._negative_verbalizers = negative_verbalizers
        self._all_verbalizers = positive_verbalizers + negative_verbalizers
        self._system_content_tokens: int | None = None
        self.last_aborted_indices: set[int] = set()

        if not positive_verbalizers or not negative_verbalizers:
            raise ValueError("delta belief ranking requires non-empty verbalizer sets")
        if not self._scoring_backend.capabilities.supports_full_scoring:
            raise ValueError(
                "DeltaBeliefRankingMethod requires a backend with full scoring "
                "support (score_chat / score_chat_batch)."
            )

    @property
    def context(self) -> MethodContext:
        return self._context

    @property
    def positive_verbalizers(self) -> tuple[str, ...]:
        return self._positive_verbalizers

    @property
    def negative_verbalizers(self) -> tuple[str, ...]:
        return self._negative_verbalizers

    def score_batch(
        self,
        points: list[RankingPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not points:
            self.last_aborted_indices = set()
            return []

        total = len(points)
        point_scores: list[list[float]] = []
        aborted_points: set[int] = set()

        for point_idx, point in enumerate(points):
            try:
                scores, had_abort = self._score_point(point)
            except Exception:
                logger.exception(
                    "DeltaBeliefRanking: scoring failed for point (traj=%d, step=%d)",
                    point.trajectory_index,
                    point.step_index,
                )
                scores = [math.nan] * len(point.candidates)
                had_abort = True

            if had_abort:
                aborted_points.add(point_idx)
            point_scores.append(scores)

            if progress_callback:
                progress_callback(point_idx + 1, total)

        self.last_aborted_indices = aborted_points
        if aborted_points:
            logger.warning(
                "DeltaBeliefRanking: %d/%d points had one or more aborted "
                "scoring calls",
                len(aborted_points),
                total,
            )
        return point_scores

    def _score_point(self, point: RankingPoint) -> tuple[list[float], bool]:
        pre_messages, post_messages = self._build_messages(point)

        messages_batch: list[list[ChatMessage]] = []
        continuations: list[str] = []

        for verbalizer in self._all_verbalizers:
            messages_batch.append(pre_messages)
            continuations.append(verbalizer)

        for candidate_messages in post_messages:
            for verbalizer in self._all_verbalizers:
                messages_batch.append(candidate_messages)
                continuations.append(verbalizer)

        results, had_abort = self._score_batch_messages(messages_batch, continuations)

        num_verbalizers = len(self._all_verbalizers)
        pre_log_success = self._compute_log_success(results[:num_verbalizers])
        if math.isnan(pre_log_success):
            return [math.nan] * len(point.candidates), True

        scores: list[float] = []
        offset = num_verbalizers
        for _candidate in point.candidates:
            chunk = results[offset : offset + num_verbalizers]
            offset += num_verbalizers
            post_log_success = self._compute_log_success(chunk)
            if math.isnan(post_log_success):
                scores.append(math.nan)
                had_abort = True
                continue
            scores.append(post_log_success - pre_log_success)

        return scores, had_abort

    def _score_batch_messages(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> tuple[list[ScoringResult | None], bool]:
        if not messages_batch:
            return [], False

        total = len(messages_batch)
        chunk_size = self._batch_size or total
        results: list[ScoringResult | None] = [None] * total
        had_abort = False

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_indices = list(range(start, end))
            chunk_messages = [messages_batch[i] for i in chunk_indices]
            chunk_conts = [continuations[i] for i in chunk_indices]

            try:
                chunk_results = self._scoring_backend.score_chat_batch(
                    chunk_messages, chunk_conts,
                )
            except Exception as exc:
                if not is_recoverable_backend_error(exc):
                    raise
                had_abort = True
                offending = getattr(exc, "offending_indices", None)
                if offending and len(offending) < len(chunk_indices):
                    offending_set = set(offending)
                    retry_local = [
                        i for i in range(len(chunk_indices))
                        if i not in offending_set
                    ]
                    retry_messages = [chunk_messages[i] for i in retry_local]
                    retry_conts = [chunk_conts[i] for i in retry_local]
                    try:
                        retry_results = self._scoring_backend.score_chat_batch(
                            retry_messages, retry_conts,
                        )
                    except Exception as retry_exc:
                        if not is_recoverable_backend_error(retry_exc):
                            raise
                        continue
                    for local_idx, result in zip(
                        retry_local, retry_results, strict=True,
                    ):
                        abs_idx = chunk_indices[local_idx]
                        results[abs_idx] = result
                    continue
                continue

            for local_idx, result in enumerate(chunk_results):
                abs_idx = chunk_indices[local_idx]
                results[abs_idx] = result

        return results, had_abort

    def _compute_log_success(
        self,
        results: Sequence[ScoringResult | None],
    ) -> float:
        if len(results) != len(self._all_verbalizers):
            raise ValueError(
                "results length must match positive+negative verbalizer count"
            )

        seq_logps: list[float] = []
        for result in results:
            if result is None or not result.token_scores:
                return math.nan
            seq_logps.append(
                sum(float(token.logprob) for token in result.token_scores)
            )

        pos_logps = seq_logps[: len(self._positive_verbalizers)]
        total_logps = seq_logps
        return _logsumexp(pos_logps) - _logsumexp(total_logps)

    def _build_messages(
        self,
        point: RankingPoint,
    ) -> tuple[list[ChatMessage], list[list[ChatMessage]]]:
        system_content = self._build_system_prompt()
        history = list(point.history)
        history_truncated_from: int | None = None
        if (
            self._max_history_turns is not None
            and len(history) > self._max_history_turns
        ):
            history_truncated_from = len(history)
            history = (
                history[-self._max_history_turns:]
                if self._max_history_turns > 0 else []
            )

        state_text_override: str | None = None
        feedback_overrides: list[str | None] | None = None
        if (
            self._estimate_tokens is not None
            and self._max_prompt_tokens is not None
        ):
            history, state_text_override, feedback_overrides = (
                self._apply_budget_truncation(
                    history,
                    system_content,
                    point,
                    history_truncated_from,
                )
            )

        pre_messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(
                role="user",
                content=_build_pre_user_prompt(
                    point=point,
                    history=history,
                    history_truncated_from=history_truncated_from,
                    policy_assumption=self._context.policy_assumption,
                    state_text_override=state_text_override,
                ),
            ),
        ]

        post_messages: list[list[ChatMessage]] = []
        for idx, candidate in enumerate(point.candidates):
            feedback_override = (
                feedback_overrides[idx] if feedback_overrides is not None else None
            )
            feedback_text = None
            if self._context.include_next_state and candidate.next_state is not None:
                feedback_text = feedback_override or serialize_observation(
                    candidate.next_state,
                )
            post_messages.append([
                ChatMessage(role="system", content=system_content),
                ChatMessage(
                    role="user",
                    content=_build_post_user_prompt(
                        point=point,
                        candidate_index=idx,
                        history=history,
                        history_truncated_from=history_truncated_from,
                        policy_assumption=self._context.policy_assumption,
                        include_next_state=self._context.include_next_state,
                        feedback_text=feedback_text,
                        state_text_override=state_text_override,
                    ),
                ),
            ])

        return pre_messages, post_messages

    def _build_system_prompt(self) -> str:
        parts: list[object] = [
            (
                "You are an expert at judging how candidate actions change an "
                "agent's probability of eventual success in a sequential "
                "decision-making environment."
            ),
            build_signal_type_fragment(
                SignalType.Q_VALUE,
                self._context.policy_assumption,
                self._context.discount_factor,
                disclose_discount_factor=self._context.disclose_discount_factor,
            ),
            (
                "Interpret 'successful overall outcome' using the task and reward "
                "function below. For sparse-success tasks, this means eventually "
                "completing the task. For dense-reward tasks, this means achieving "
                "a strong overall return."
            ),
            (
                "Each candidate is scored by the change in log-probability of the "
                "one-word answer 'Yes' to a success question after the candidate's "
                "observed outcome is revealed."
            ),
        ]

        guidance = resolve_block(self._prompt_preset.approximation_guidance)
        if guidance is not None:
            parts.append(guidance)
        if self._context.efficiency_guidance:
            parts.append(self._context.efficiency_guidance)
        if self._context.task_description:
            parts.append(
                f"## Task Description\n{self._context.task_description.rstrip()}"
            )
        env_notes = _format_environment_notes(self._context.environment_notes)
        if env_notes:
            parts.append(env_notes)
        if self._context.reward_description:
            parts.append(
                f"## Reward Functions\n{self._context.reward_description.rstrip()}"
            )
        return compose_system_prompt(*parts)

    def _apply_budget_truncation(
        self,
        history: list[Any],
        system_content: str,
        point: RankingPoint,
        history_truncated_from: int | None,
    ) -> tuple[list[Any], str | None, list[str | None] | None]:
        from qval.token_budget import (
            budget_aware_truncation,
            truncate_text_to_token_savings,
        )

        assert self._estimate_tokens is not None
        assert self._max_prompt_tokens is not None

        if self._system_content_tokens is None:
            self._system_content_tokens = self._estimate_tokens(system_content)

        state_text = serialize_observation(point.state)
        feedback_texts: list[str | None] = []
        for candidate in point.candidates:
            if self._context.include_next_state and candidate.next_state is not None:
                feedback_texts.append(serialize_observation(candidate.next_state))
            else:
                feedback_texts.append(None)

        def _estimate_post_total(
            history_value: list[Any],
            *,
            candidate_index: int,
            state_override: str | None = None,
            feedback_override: str | None = None,
        ) -> int:
            user_content = _build_post_user_prompt(
                point=point,
                candidate_index=candidate_index,
                history=history_value,
                history_truncated_from=history_truncated_from,
                policy_assumption=self._context.policy_assumption,
                include_next_state=self._context.include_next_state,
                feedback_text=feedback_override,
                state_text_override=state_override,
            )
            return self._system_content_tokens + self._estimate_tokens(user_content)

        longest_feedback = max(
            (text for text in feedback_texts if text),
            key=len,
            default=None,
        )
        available = max(
            0,
            self._max_prompt_tokens - _estimate_post_total(
                [],
                candidate_index=0,
                state_override=state_text,
                feedback_override=longest_feedback,
            ),
        )
        if history:
            history = budget_aware_truncation(
                history,
                available,
                min_observation_chars=self._min_observation_chars,
                min_action_chars=None,
                estimate_tokens=self._estimate_tokens,
            )

        feedback_overrides: list[str | None] | None = None
        current_feedbacks = list(feedback_texts)
        point_totals = [
            _estimate_post_total(
                history,
                candidate_index=idx,
                state_override=state_text,
                feedback_override=feedback_text,
            )
            for idx, feedback_text in enumerate(current_feedbacks)
        ]

        if (
            point_totals
            and max(point_totals) > self._max_prompt_tokens
            and self._min_next_observation_chars is not None
        ):
            for idx, feedback_text in enumerate(current_feedbacks):
                if feedback_text is None:
                    continue
                excess = point_totals[idx] - self._max_prompt_tokens
                if excess <= 0:
                    continue
                truncated = truncate_text_to_token_savings(
                    feedback_text,
                    excess,
                    self._min_next_observation_chars,
                    self._estimate_tokens,
                )
                current_feedbacks[idx] = truncated
            feedback_overrides = current_feedbacks
            point_totals = [
                _estimate_post_total(
                    history,
                    candidate_index=idx,
                    state_override=state_text,
                    feedback_override=feedback_text,
                )
                for idx, feedback_text in enumerate(current_feedbacks)
            ]

        state_text_override: str | None = None
        if (
            point_totals
            and max(point_totals) > self._max_prompt_tokens
            and self._min_current_observation_chars is not None
        ):
            excess = max(point_totals) - self._max_prompt_tokens
            candidate_state_text = truncate_text_to_token_savings(
                state_text,
                excess,
                self._min_current_observation_chars,
                self._estimate_tokens,
            )
            if candidate_state_text != state_text:
                state_text_override = candidate_state_text

        return history, state_text_override, feedback_overrides

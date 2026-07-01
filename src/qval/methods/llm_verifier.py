"""Verifier-style direct method with score-bin decoding from token logprobs."""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from dataclasses import dataclass
from typing import Callable

from llenvs.inference.protocol import ChatMessage, GenerationResult, ModelBackend, SamplingParams, TokenLogprob

from qval.dense_signal import DenseSignalMethod
from qval.methods._batched_eval import (
    PromptGroupPlan,
    aggregate_repeated_prediction_passes,
    plan_prompt_groups,
)
from qval.prompt_presets import PromptPreset
from qval.prompts import (
    _task_text_for_injection,
    build_verifier_system_prompt,
    build_verifier_user_prompt_batch,
)
from qval.types import EvaluationPoint, HistoryTurn, MethodContext, SignalType
from qval.verifier_criteria import VerifierCriterion, get_verifier_criterion

_SCORE_LABELS = tuple(chr(ord("A") + i) for i in range(20))
_SCORE_BY_LABEL = {
    label: (len(_SCORE_LABELS) - 1 - idx) / (len(_SCORE_LABELS) - 1)
    for idx, label in enumerate(_SCORE_LABELS)
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedVerifierPoint:
    point: EvaluationPoint
    history: list[HistoryTurn]
    history_truncated_from: int | None
    state_text_override: str | None
    next_state_text_override: str | None


class LLMVerifierMethod(DenseSignalMethod):
    """Prompted verifier that decodes expected scores from discrete bins."""

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        sampling_params: SamplingParams | None = None,
        max_history_turns: int | None = None,
        *,
        prompt_preset: PromptPreset,
        batch_size: int | None = None,
        prompt_batch_size: int = 1,
        prompt_grouping: str = "random",
        criteria: tuple[str, ...] | None = None,
        num_verifications: int = 1,
        estimate_tokens: Callable[[str], int] | None = None,
        max_prompt_tokens: int | None = None,
        min_observation_chars: int | None = None,
        min_action_chars: int | None = None,
        min_current_observation_chars: int | None = None,
        min_next_observation_chars: int | None = None,
        inject_task_text: bool = False,
    ) -> None:
        super().__init__(context)
        if not backend.capabilities.supports_logprobs:
            raise ValueError("LLMVerifierMethod requires a backend with supports_logprobs=True")
        if prompt_grouping not in {"contiguous", "random", "trajectory"}:
            raise ValueError(f"Unknown prompt_grouping: {prompt_grouping!r}")
        if context.signal_type not in (SignalType.STATE_VALUE, SignalType.Q_VALUE):
            raise ValueError(
                "LLMVerifierMethod supports only state_value and q_value"
            )

        self._backend = backend
        base_params = sampling_params or SamplingParams()
        self._sampling_params = dataclasses.replace(
            base_params,
            logprobs=True,
            num_logprobs=20,
        )
        self._max_history_turns = max_history_turns
        self._prompt_preset = prompt_preset
        self._batch_size = batch_size
        self._prompt_batch_size = prompt_batch_size
        self._prompt_grouping = prompt_grouping
        self._criteria: tuple[VerifierCriterion, ...] = tuple(
            get_verifier_criterion(criterion_id) for criterion_id in (criteria or ())
        )
        self._num_verifications = num_verifications
        self._estimate_tokens = estimate_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._min_observation_chars = min_observation_chars
        self._min_action_chars = min_action_chars
        self._min_current_observation_chars = min_current_observation_chars
        self._min_next_observation_chars = min_next_observation_chars
        self._inject_task_text = inject_task_text
        self._system_content_tokens: dict[str, int] = {}
        self.last_aborted_indices: set[int] = set()

    def evaluate(self, point: EvaluationPoint) -> float:
        return self.evaluate_batch([point])[0]

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if not points:
            self.last_aborted_indices = set()
            return []

        prompt_groups = self._plan_prompt_groups(points)
        criteria = self._criteria or (None,)
        messages_by_criterion = {
            criterion.id if criterion is not None else "__none__": [
                self._build_messages_for_group(group.points, criterion)
                for group in prompt_groups
            ]
            for criterion in criteria
        }

        def _run_pass(
            pass_index: int,
            on_progress: Callable[[int, int], None] | None,
        ) -> tuple[list[float], set[int]]:
            criterion = criteria[pass_index % len(criteria)]
            key = criterion.id if criterion is not None else "__none__"
            return self._evaluate_groups_once(
                prompt_groups,
                messages_by_criterion[key],
                progress_callback=on_progress,
            )

        total_passes = self._num_verifications * len(criteria)
        results, aborted = aggregate_repeated_prediction_passes(
            total_points=len(points),
            num_passes=total_passes,
            run_pass=_run_pass,
            progress_callback=progress_callback,
        )
        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "LLMVerifier: %d/%d points aborted due to missing successful verifier scores",
                len(aborted), len(points),
            )
        return results

    def _build_messages_for_group(
        self,
        points: list[EvaluationPoint],
        criterion: VerifierCriterion | None,
    ) -> list[ChatMessage]:
        prepared = self._prepare_points_for_group(points, criterion)
        system = build_verifier_system_prompt(
            self.context,
            self._prompt_preset,
            num_points=len(prepared),
        )
        user = build_verifier_user_prompt_batch(
            [item.point for item in prepared],
            self.context.signal_type,
            histories=[item.history for item in prepared],
            history_truncated_froms=[item.history_truncated_from for item in prepared],
            include_next_state=self.context.include_next_state,
            include_current_thoughts=self.context.include_current_thoughts,
            include_history_thoughts=self.context.include_history_thoughts,
            criterion_name=criterion.name if criterion is not None else None,
            criterion_description=(
                criterion.description_for_signal(self.context.signal_type)
                if criterion is not None
                else None
            ),
            state_text_overrides=[item.state_text_override for item in prepared],
            next_state_text_overrides=[item.next_state_text_override for item in prepared],
            max_steps=self.context.max_steps,
            task_text_to_injects=(
                [_task_text_for_injection(item.point) for item in prepared]
                if self._inject_task_text
                else None
            ),
            include_state_text_when_images=self.context.include_state_text_when_images,
            include_images=self.context.include_images,
        )
        if user.has_images:
            user_message = ChatMessage(role="user", content_blocks=user.blocks)
        else:
            user_message = ChatMessage(role="user", content=user.text)
        return [
            ChatMessage(role="system", content=system),
            user_message,
        ]

    def _prepare_points_for_group(
        self,
        points: list[EvaluationPoint],
        criterion: VerifierCriterion | None,
    ) -> list[_PreparedVerifierPoint]:
        system_content = build_verifier_system_prompt(
            self.context,
            self._prompt_preset,
            num_points=len(points),
        )
        point_budget: int | None = None
        if self._max_prompt_tokens is not None and self._estimate_tokens is not None:
            system_tokens = self._estimate_system_tokens(system_content)
            shared_budget = max(0, self._max_prompt_tokens - system_tokens)
            point_budget = system_tokens + max(0, shared_budget // len(points))
        return [
            self._prepare_point(
                point,
                criterion=criterion,
                system_content=system_content,
                max_prompt_tokens=point_budget,
            )
            for point in points
        ]

    def _prepare_point(
        self,
        point: EvaluationPoint,
        *,
        criterion: VerifierCriterion | None,
        system_content: str | None = None,
        max_prompt_tokens: int | None = None,
    ) -> _PreparedVerifierPoint:
        if system_content is None:
            system_content = build_verifier_system_prompt(
                self.context,
                self._prompt_preset,
                num_points=1,
            )

        history = list(point.history)
        history_truncated_from: int | None = None
        if self._max_history_turns is not None and len(history) > self._max_history_turns:
            history_truncated_from = len(history)
            history = history[-self._max_history_turns:] if self._max_history_turns > 0 else []

        state_text_override: str | None = None
        next_state_text_override: str | None = None
        if self._max_prompt_tokens is not None and self._estimate_tokens is not None:
            history, state_text_override, next_state_text_override = self._apply_budget_truncation(
                history,
                system_content,
                point,
                history_truncated_from,
                criterion=criterion,
                max_prompt_tokens=max_prompt_tokens,
            )

        return _PreparedVerifierPoint(
            point=point,
            history=history,
            history_truncated_from=history_truncated_from,
            state_text_override=state_text_override,
            next_state_text_override=next_state_text_override,
        )

    def _estimate_system_tokens(self, system_content: str) -> int:
        if self._estimate_tokens is None:
            raise AssertionError("_estimate_system_tokens requires _estimate_tokens")
        if system_content not in self._system_content_tokens:
            self._system_content_tokens[system_content] = self._estimate_tokens(system_content)
        return self._system_content_tokens[system_content]

    def _apply_budget_truncation(
        self,
        history: list[HistoryTurn],
        system_content: str,
        point: EvaluationPoint,
        history_truncated_from: int | None,
        *,
        criterion: VerifierCriterion | None,
        max_prompt_tokens: int | None = None,
    ) -> tuple[list[HistoryTurn], str | None, str | None]:
        from qval.methods.serialization import serialize_observation
        from qval.token_budget import budget_aware_truncation, truncate_text_to_token_savings

        assert self._estimate_tokens is not None
        assert self._max_prompt_tokens is not None
        prompt_budget = max_prompt_tokens or self._max_prompt_tokens

        system_content_tokens = self._estimate_system_tokens(system_content)
        state_text = serialize_observation(point.state)
        next_state_text = serialize_observation(point.next_state) if point.next_state is not None else None

        state_text_override: str | None = None
        next_state_text_override: str | None = None

        def _estimate_total(
            history_value: list[HistoryTurn],
            *,
            state_override: str | None = None,
            next_state_override: str | None = None,
        ) -> int:
            user_content = build_verifier_user_prompt_batch(
                [point],
                self.context.signal_type,
                histories=[history_value],
                history_truncated_froms=[history_truncated_from],
                include_next_state=self.context.include_next_state,
                include_current_thoughts=self.context.include_current_thoughts,
                include_history_thoughts=self.context.include_history_thoughts,
                criterion_name=criterion.name if criterion is not None else None,
                criterion_description=(
                    criterion.description_for_signal(self.context.signal_type)
                    if criterion is not None
                    else None
                ),
                state_text_overrides=[state_override],
                next_state_text_overrides=[next_state_override],
                max_steps=self.context.max_steps,
                task_text_to_injects=(
                    [_task_text_for_injection(point)]
                    if self._inject_task_text
                    else None
                ),
                include_state_text_when_images=self.context.include_state_text_when_images,
                include_images=self.context.include_images,
            )
            return system_content_tokens + self._estimate_tokens(user_content.text)

        available = max(
            0,
            prompt_budget - _estimate_total(
                [],
                state_override=state_text,
                next_state_override=next_state_text,
            ),
        )

        if history:
            history = budget_aware_truncation(
                history,
                available,
                min_observation_chars=self._min_observation_chars,
                min_action_chars=self._min_action_chars,
                estimate_tokens=self._estimate_tokens,
            )

        total = _estimate_total(
            history,
            state_override=state_text,
            next_state_override=next_state_text,
        )
        excess = total - prompt_budget

        if (
            excess > 0
            and next_state_text is not None
            and self._min_next_observation_chars is not None
        ):
            candidate_text = truncate_text_to_token_savings(
                next_state_text,
                excess,
                self._min_next_observation_chars,
                self._estimate_tokens,
            )
            if candidate_text != next_state_text:
                next_state_text_override = candidate_text
                total = _estimate_total(
                    history,
                    state_override=state_text,
                    next_state_override=next_state_text_override,
                )
                excess = total - prompt_budget

        if excess > 0 and self._min_current_observation_chars is not None:
            candidate_text = truncate_text_to_token_savings(
                state_text,
                excess,
                self._min_current_observation_chars,
                self._estimate_tokens,
            )
            if candidate_text != state_text:
                state_text_override = candidate_text

        return history, state_text_override, next_state_text_override

    def _plan_prompt_groups(self, points: list[EvaluationPoint]) -> list[PromptGroupPlan]:
        return plan_prompt_groups(
            points,
            prompt_batch_size=self._prompt_batch_size,
            prompt_grouping=self._prompt_grouping,
        )

    def _evaluate_groups_once(
        self,
        prompt_groups: list[PromptGroupPlan],
        group_messages: list[list[ChatMessage]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[float], set[int]]:
        from qval.error_handling import generate_batch_with_transient_retry

        total = sum(len(group.points) for group in prompt_groups)
        results: list[float] = [math.nan] * total
        aborted: set[int] = set()
        chunk_size = self._batch_size or len(prompt_groups)
        points_completed = 0

        for start in range(0, len(prompt_groups), chunk_size):
            end = min(start + chunk_size, len(prompt_groups))
            chunk_indices = list(range(start, end))
            chunk_messages = [group_messages[i] for i in chunk_indices]
            gen_results = generate_batch_with_transient_retry(
                self._backend,
                chunk_messages,
                self._sampling_params,
            )
            for local_idx, generation in enumerate(gen_results):
                group = prompt_groups[chunk_indices[local_idx]]
                if generation is None:
                    for point_idx in group.original_indices:
                        aborted.add(point_idx)
                    continue
                parsed = self._parse_group_response(
                    generation,
                    expected_count=len(group.points),
                )
                for point_idx, value in zip(group.original_indices, parsed):
                    results[point_idx] = value
            points_completed += sum(len(prompt_groups[i].points) for i in chunk_indices)
            if progress_callback:
                progress_callback(points_completed, total)

        return results, aborted

    def _parse_group_response(
        self,
        result: GenerationResult,
        *,
        expected_count: int,
    ) -> list[float]:
        return [
            self._parse_score_tag(result, index)
            for index in range(1, expected_count + 1)
        ]

    def _parse_score_tag(self, result: GenerationResult, index: int) -> float:
        tag = f"<score_{index}>"
        expected = _expected_score_from_token_logprobs(result.token_logprobs, tag)
        if expected is not None:
            return expected

        label = _extract_score_label_from_text(result.text or "", index)
        if label is None:
            return math.nan
        return _SCORE_BY_LABEL[label]


def _normalize_label(token: str | None) -> str | None:
    if token is None:
        return None
    stripped = token.strip()
    if len(stripped) != 1:
        return None
    upper = stripped.upper()
    if upper not in _SCORE_BY_LABEL:
        return None
    return upper


def _expected_score_from_token_logprobs(
    token_logprobs: tuple[TokenLogprob, ...] | None,
    opening_tag: str,
) -> float | None:
    if not token_logprobs:
        return None

    text_so_far = ""
    for i, token in enumerate(token_logprobs):
        text_so_far += token.token
        if not text_so_far.rstrip().endswith(opening_tag):
            continue
        for next_token in token_logprobs[i + 1:]:
            score = _score_from_token_distribution(next_token)
            if score is not None:
                return score
            if next_token.token.strip().startswith("</score_"):
                break
        return None
    return None


def _score_from_token_distribution(token: TokenLogprob) -> float | None:
    raw_probs: dict[str, float] = {}
    actual = _normalize_label(token.token)
    if actual is not None:
        raw_probs[actual] = max(raw_probs.get(actual, 0.0), math.exp(token.logprob))
    for raw_token, logprob in (token.top_logprobs or {}).items():
        label = _normalize_label(raw_token)
        if label is not None:
            raw_probs[label] = max(raw_probs.get(label, 0.0), math.exp(logprob))
    if not raw_probs:
        return None
    total = sum(raw_probs.values())
    if total <= 0:
        return None
    return sum(_SCORE_BY_LABEL[label] * prob for label, prob in raw_probs.items()) / total


def _extract_score_label_from_text(text: str, index: int) -> str | None:
    match = re.search(
        rf"<score_{index}>\s*([A-Ta-t])\s*</score_{index}>",
        text,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).upper()

    # Some chat models obey the "one score" constraint but omit part of the
    # XML wrapper, especially at nonzero temperature. Treat these as usable
    # only when the response is still unambiguously a single score.
    match = re.search(
        rf"<score_{index}>\s*([A-Ta-t])\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).upper()

    match = re.fullmatch(r"\s*([A-Ta-t])\s*", text, flags=re.IGNORECASE)
    if match is not None:
        return match.group(1).upper()

    return None

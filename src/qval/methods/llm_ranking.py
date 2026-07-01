"""LLM-based method that ranks candidate actions by Q-value."""

from __future__ import annotations

import logging
import re
from typing import Callable

from llenvs.core.cleaning import strip_special_tokens, strip_thinking_tokens
from llenvs.core.extraction import TagBasedExtractor
from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams

from qval.prompt_presets import PromptPreset
from qval.prompts import (
    _task_text_for_injection,
    build_ranking_system_prompt,
    build_ranking_user_prompt,
)
from qval.types import HistoryTurn, MethodContext, RankingPoint

logger = logging.getLogger(__name__)

# Match sequences of comma/space separated integers
_RANKING_PATTERN = re.compile(r"\b(\d+(?:\s*,\s*\d+)*)\b")

_ANSWER_TAG_EXTRACTOR = TagBasedExtractor(tag_name="answer")


def _parse_ranking_from_text(text: str, k: int) -> list[int] | None:
    """Parse a ranking permutation from text (last valid match wins)."""
    matches = _RANKING_PATTERN.findall(text)
    if not matches:
        return None
    for match_str in reversed(matches):
        nums = [int(x.strip()) for x in match_str.split(",")]
        if len(nums) == k and sorted(nums) == list(range(1, k + 1)):
            return nums
    return None


def parse_ranking(
    response: str,
    k: int,
    *,
    use_answer_tags: bool = False,
) -> list[int] | None:
    """Parse a ranking from an LLM response.

    Searches for a valid permutation of ``[1..k]``.  When *use_answer_tags*
    is ``True``, checks ``<answer>`` tag content first before falling back
    to scanning the full (cleaned) response text.

    Args:
        response: Raw LLM response text.
        k: Expected number of items in the ranking.
        use_answer_tags: Try ``<answer>`` tag extraction before full-text scan.

    Returns:
        A list of action numbers (1-indexed, best-to-worst order),
        or ``None`` if no valid permutation was found.
    """
    cleaned = strip_thinking_tokens(strip_special_tokens(response))

    if use_answer_tags:
        tag_content, _ = _ANSWER_TAG_EXTRACTOR.extract(cleaned)
        if tag_content is not None:
            result = _parse_ranking_from_text(tag_content, k)
            if result is not None:
                return result

    return _parse_ranking_from_text(cleaned, k)


class LLMRankingMethod:
    """Prompts an LLM to rank candidate actions at each ranking point.

    For each ``RankingPoint``, builds a prompt listing the candidates and
    asks the LLM to output a best-to-worst ordering. Parses the response
    into a permutation of action indices.

    Not a ``DenseSignalMethod`` subclass since it operates on
    ``RankingPoint`` objects rather than ``EvaluationPoint`` objects.
    """

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        sampling_params: SamplingParams | None = None,
        max_history_turns: int | None = None,
        *,
        prompt_preset: PromptPreset,
        batch_size: int | None = None,
        estimate_tokens: Callable[[str], int] | None = None,
        max_prompt_tokens: int | None = None,
        min_observation_chars: int | None = None,
        min_action_chars: int | None = None,
        min_current_observation_chars: int | None = None,
        min_next_observation_chars: int | None = None,
        min_ranking_candidate_action_chars: int | None = None,
        min_ranking_next_observation_chars: int | None = None,
        inject_task_text: bool = False,
    ) -> None:
        self._backend = backend
        self._context = context
        self._sampling_params = sampling_params or SamplingParams()
        self._max_history_turns = max_history_turns
        self._prompt_preset = prompt_preset
        self._batch_size = batch_size
        self._estimate_tokens = estimate_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._min_observation_chars = min_observation_chars
        self._min_action_chars = min_action_chars
        self._min_current_observation_chars = min_current_observation_chars
        self._min_next_observation_chars = min_next_observation_chars
        self._min_ranking_candidate_action_chars = min_ranking_candidate_action_chars
        self._min_ranking_next_observation_chars = min_ranking_next_observation_chars
        self._inject_task_text = inject_task_text
        self._system_content_tokens: int | None = None  # cached
        self.last_aborted_indices: set[int] = set()

    @property
    def context(self) -> MethodContext:
        return self._context

    def rank(self, point: RankingPoint) -> list[int] | None:
        """Rank candidates at a single point.

        Returns:
            List of action numbers (1-indexed, best-to-worst), or ``None``
            if the response could not be parsed.
        """
        messages = self._build_messages(point)
        result = self._backend.generate_chat(messages, self._sampling_params)
        return parse_ranking(
            result.text or "", len(point.candidates),
            use_answer_tags=self._prompt_preset.uses_answer_tags,
        )

    def rank_batch(
        self,
        points: list[RankingPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[int] | None]:
        """Rank candidates for multiple points via batched generation."""
        if not points:
            self.last_aborted_indices: set[int] = set()
            return []

        from qval.error_handling import generate_batch_with_transient_retry

        total = len(points)
        all_messages = [self._build_messages(p) for p in points]
        k_per_point = [len(p.candidates) for p in points]
        aborted: set[int] = set()

        chunk_size = self._batch_size or total
        results: list[list[int] | None] = [None] * total
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_indices = list(range(start, end))
            chunk_msgs = [all_messages[i] for i in chunk_indices]
            chunk_ks = [k_per_point[i] for i in chunk_indices]

            gen_results = generate_batch_with_transient_retry(
                self._backend, chunk_msgs, self._sampling_params,
            )
            for local_idx, gr in enumerate(gen_results):
                abs_idx = chunk_indices[local_idx]
                if gr is None:
                    aborted.add(abs_idx)
                    continue
                results[abs_idx] = parse_ranking(
                    gr.text or "", chunk_ks[local_idx],
                    use_answer_tags=self._prompt_preset.uses_answer_tags,
                )
            if progress_callback:
                progress_callback(end, total)

        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "LLMRanking: %d/%d points aborted due to backend errors",
                len(aborted), total,
            )
        return results

    def _build_messages(self, point: RankingPoint) -> list[ChatMessage]:
        system_content = build_ranking_system_prompt(
            self._context, self._prompt_preset,
        )
        history = list(point.history)
        history_truncated_from: int | None = None
        if self._max_history_turns is not None and len(history) > self._max_history_turns:
            history_truncated_from = len(history)
            if self._max_history_turns > 0:
                history = history[-self._max_history_turns:]
            else:
                history = []

        # Budget-aware multi-phase truncation
        state_text_override: str | None = None
        candidate_ns_overrides: list[str | None] | None = None
        candidate_action_overrides: list[str | None] | None = None
        if (
            self._max_prompt_tokens is not None
            and self._estimate_tokens is not None
        ):
            history, state_text_override, candidate_ns_overrides, candidate_action_overrides = (
                self._apply_budget_truncation(
                    history,
                    system_content,
                    point,
                    history_truncated_from,
                )
            )

        prompt = build_ranking_user_prompt(
            point, history=history,
            history_truncated_from=history_truncated_from,
            include_next_state=self._context.include_next_state,
            include_current_thoughts=self._context.include_current_thoughts,
            include_history_thoughts=self._context.include_history_thoughts,
            state_text_override=state_text_override,
            candidate_next_state_overrides=candidate_ns_overrides,
            candidate_action_overrides=candidate_action_overrides,
            use_answer_tags=self._prompt_preset.uses_answer_tags,
            max_steps=self._context.max_steps,
            task_text_to_inject=(
                _task_text_for_injection(point)
                if self._inject_task_text
                else None
            ),
            include_state_text_when_images=self._context.include_state_text_when_images,
            include_images=self._context.include_images,
        )

        user_msg = (
            ChatMessage(role="user", content_blocks=prompt.blocks)
            if prompt.has_images
            else ChatMessage(role="user", content=prompt.text)
        )
        return [
            ChatMessage(role="system", content=system_content),
            user_msg,
        ]

    def _apply_budget_truncation(
        self,
        history: list[HistoryTurn],
        system_content: str,
        point: RankingPoint,
        history_truncated_from: int | None,
    ) -> tuple[list[HistoryTurn], str | None, list[str | None] | None, list[str | None] | None]:
        """Apply budget-aware multi-phase truncation.

        Returns:
            (truncated_history, state_text_override, candidate_ns_overrides,
            candidate_action_overrides). Overrides are ``None`` when no
            truncation was needed.
        """
        from qval.methods.serialization import serialize_observation
        from qval.token_budget import (
            budget_aware_truncation,
            truncate_text_to_token_savings,
        )

        assert self._estimate_tokens is not None
        assert self._max_prompt_tokens is not None

        # Cache system content token count
        if self._system_content_tokens is None:
            self._system_content_tokens = self._estimate_tokens(system_content)

        state_text = serialize_observation(point.state)

        # Serialize candidate next-states
        candidate_ns_texts: list[str | None] = []
        for candidate in point.candidates:
            if self._context.include_next_state and candidate.next_state is not None:
                candidate_ns_texts.append(serialize_observation(candidate.next_state))
            else:
                candidate_ns_texts.append(None)

        state_text_override: str | None = None
        candidate_ns_overrides: list[str | None] | None = None
        candidate_action_overrides: list[str | None] | None = None

        def _estimate_total(
            history_value: list[HistoryTurn],
            *,
            state_override: str | None = None,
            candidate_overrides: list[str | None] | None = None,
            action_overrides: list[str | None] | None = None,
        ) -> int:
            prompt = build_ranking_user_prompt(
                point,
                history=history_value,
                history_truncated_from=history_truncated_from,
                include_next_state=self._context.include_next_state,
                include_current_thoughts=self._context.include_current_thoughts,
                include_history_thoughts=self._context.include_history_thoughts,
                state_text_override=state_override,
                candidate_next_state_overrides=candidate_overrides,
                candidate_action_overrides=action_overrides,
                use_answer_tags=self._prompt_preset.uses_answer_tags,
                task_text_to_inject=(
                    _task_text_for_injection(point)
                    if self._inject_task_text
                    else None
                ),
                include_state_text_when_images=self._context.include_state_text_when_images,
                include_images=self._context.include_images,
            )
            return self._system_content_tokens + self._estimate_tokens(prompt.text)

        available = max(
            0,
            self._max_prompt_tokens - _estimate_total(
                [],
                state_override=state_text,
                candidate_overrides=candidate_ns_texts,
            ),
        )

        # Phase 1: truncate history
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
            candidate_overrides=candidate_ns_texts,
        )
        excess = total - self._max_prompt_tokens

        # Phase 2: truncate candidate next-states proportionally
        if excess > 0 and self._min_ranking_next_observation_chars is not None:
            current_candidate_texts = list(candidate_ns_texts)
            for _ in range(3):
                if excess <= 0:
                    break
                truncatable = [
                    (i, ns)
                    for i, ns in enumerate(current_candidate_texts)
                    if ns is not None and len(ns) > self._min_ranking_next_observation_chars
                ]
                if not truncatable:
                    break
                total_ns_len = sum(len(ns) for _, ns in truncatable)
                progressed = False
                for idx, ns in truncatable:
                    share = len(ns) / total_ns_len if total_ns_len > 0 else 1.0
                    needed = max(1, round(excess * share))
                    truncated = truncate_text_to_token_savings(
                        ns,
                        needed,
                        self._min_ranking_next_observation_chars,
                        self._estimate_tokens,
                    )
                    if truncated != ns:
                        current_candidate_texts[idx] = truncated
                        progressed = True
                if not progressed:
                    break
                candidate_ns_overrides = current_candidate_texts
                new_total = _estimate_total(
                    history,
                    state_override=state_text,
                    candidate_overrides=current_candidate_texts,
                )
                if new_total >= total:
                    break
                total = new_total
                excess = total - self._max_prompt_tokens

        # Phase 2b: truncate candidate actions proportionally
        if excess > 0 and self._min_ranking_candidate_action_chars is not None:
            from qval.methods.serialization import action_text_for_display

            current_action_texts = [
                action_text_for_display(
                    c.action, c.resolved_action, c.extracted_action,
                )
                for c in point.candidates
            ]
            for _ in range(3):
                if excess <= 0:
                    break
                truncatable = [
                    (i, at)
                    for i, at in enumerate(current_action_texts)
                    if len(at) > self._min_ranking_candidate_action_chars
                ]
                if not truncatable:
                    break
                total_at_len = sum(len(at) for _, at in truncatable)
                progressed = False
                for idx, at in truncatable:
                    share = len(at) / total_at_len if total_at_len > 0 else 1.0
                    needed = max(1, round(excess * share))
                    truncated = truncate_text_to_token_savings(
                        at,
                        needed,
                        self._min_ranking_candidate_action_chars,
                        self._estimate_tokens,
                    )
                    if truncated != at:
                        current_action_texts[idx] = truncated
                        progressed = True
                if not progressed:
                    break
                candidate_action_overrides = current_action_texts
                new_total = _estimate_total(
                    history,
                    state_override=state_text,
                    candidate_overrides=candidate_ns_overrides or candidate_ns_texts,
                    action_overrides=current_action_texts,
                )
                if new_total >= total:
                    break
                total = new_total
                excess = total - self._max_prompt_tokens

        # Phase 3: truncate current state if still over budget
        if excess > 0 and self._min_current_observation_chars is not None:
            candidate_text = truncate_text_to_token_savings(
                state_text,
                excess,
                self._min_current_observation_chars,
                self._estimate_tokens,
            )
            if candidate_text != state_text:
                state_text_override = candidate_text

        return history, state_text_override, candidate_ns_overrides, candidate_action_overrides

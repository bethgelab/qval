"""LLM-based method that prompts per evaluation point for a numeric value."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

from llenvs.core.cleaning import strip_special_tokens, strip_thinking_tokens
from llenvs.core.extraction import (
    CleanedExtractor,
    NumericExtractor,
    TagBasedExtractor,
)
from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams

from qval.dense_signal import DenseSignalMethod
from qval.methods._batched_eval import (
    PromptGroupPlan,
    aggregate_repeated_prediction_passes,
    plan_prompt_groups,
)
from qval.prompt_presets import PromptPreset
from qval.prompts import (
    _task_text_for_injection,
    build_direct_system_prompt,
    build_direct_user_prompt,
    build_direct_user_prompt_batch,
    build_direct_user_prompt_sequential_turn,
)
from qval.types import EvaluationPoint, HistoryTurn, MethodContext

logger = logging.getLogger(__name__)

_DEFAULT_EVALUATOR_EXTRACTOR = CleanedExtractor(
    inner=NumericExtractor(),
    pre_cleaners=[strip_thinking_tokens, strip_special_tokens],
)


class _AnswerTagNumericExtractor:
    """Extract numeric value from ``<answer>`` tags, with raw-numeric fallback.

    Pipeline:
    1. Strip thinking/special tokens.
    2. Try ``<answer>`` tag extraction; if found, extract number from tag content.
    3. Fallback: extract last number from full cleaned text.
    """

    def __init__(self) -> None:
        self._tag = TagBasedExtractor(tag_name="answer")
        self._numeric = NumericExtractor()
        self._pre_cleaners = [strip_thinking_tokens, strip_special_tokens]

    def extract(self, response: str) -> tuple[str | None, dict[str, Any]]:
        cleaned = response
        for cleaner in self._pre_cleaners:
            cleaned = cleaner(cleaned)

        # Try tag extraction first
        tag_content, tag_meta = self._tag.extract(cleaned)
        if tag_content is not None:
            # Parse number from tag content (handles "<answer>about 3.5</answer>")
            num_result, num_meta = self._numeric.extract(tag_content)
            if num_result is not None:
                return num_result, {"source": "answer_tag", **tag_meta, **num_meta}

        # Fallback: last number from full cleaned text
        num_result, num_meta = self._numeric.extract(cleaned)
        if num_result is not None:
            return num_result, {"source": "fallback_numeric", **num_meta}

        return None, {"source": "none", "found": False}


_ANSWER_TAG_EVALUATOR_EXTRACTOR = _AnswerTagNumericExtractor()
_ANSWER_TAG_EXTRACTOR = TagBasedExtractor(tag_name="answer")


@dataclass(frozen=True)
class _PreparedDirectPoint:
    point: EvaluationPoint
    history: list[HistoryTurn]
    history_truncated_from: int | None
    state_text_override: str | None
    next_state_text_override: str | None


@dataclass
class _SequentialDirectGroupState:
    prepared_points: list[_PreparedDirectPoint]
    original_indices: list[int]
    messages: list[ChatMessage]
    next_turn_index: int = 0

class LLMDirectMethod(DenseSignalMethod):
    """Prompts an LLM per evaluation point and parses a numeric signal value.

    For each point, builds a prompt describing the state, action, and next state,
    asks the LLM to output a numeric value for the requested signal type, and
    extracts the last number from the response (after stripping thinking tokens).

    Batched evaluation supports two prompt protocols:
    - ``prompt_batch_mode="packed"``: pack multiple datapoints into one user
      prompt and parse a comma-separated assistant response.
    - ``prompt_batch_mode="sequential"``: ask for one datapoint at a time while
      keeping prior numeric assistant replies in the conversation.

    Uses ``context.evaluator_extractor`` if provided, otherwise falls back to
    a default CleanedExtractor(NumericExtractor()) with thinking/special token
    stripping.
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
        prompt_batch_size: int | None = 1,
        prompt_batch_mode: str = "packed",
        num_verifications: int = 1,
        prompt_grouping: str = "contiguous",
        prompt_shuffle: str = "none",
        joint_prompting: bool = False,
        show_turn_indices: bool = True,
        estimate_tokens: Callable[[str], int] | None = None,
        max_prompt_tokens: int | None = None,
        min_observation_chars: int | None = None,
        min_action_chars: int | None = None,
        min_current_observation_chars: int | None = None,
        min_next_observation_chars: int | None = None,
        inject_task_text: bool = False,
    ) -> None:
        super().__init__(context)
        self._backend = backend
        self._sampling_params = sampling_params or SamplingParams()
        self._max_history_turns = max_history_turns
        self._prompt_preset = prompt_preset
        if context.evaluator_extractor is not None:
            self._extractor = context.evaluator_extractor
        elif prompt_preset.uses_answer_tags:
            self._extractor = _ANSWER_TAG_EVALUATOR_EXTRACTOR
        else:
            self._extractor = _DEFAULT_EVALUATOR_EXTRACTOR
        if prompt_grouping not in {"contiguous", "trajectory"}:
            raise ValueError(f"Unknown prompt_grouping: {prompt_grouping!r}")
        if prompt_batch_mode not in {"packed", "sequential"}:
            raise ValueError(f"Unknown prompt_batch_mode: {prompt_batch_mode!r}")
        if prompt_shuffle not in {"none", "deterministic_random"}:
            raise ValueError(f"Unknown prompt_shuffle: {prompt_shuffle!r}")
        if (
            prompt_batch_mode == "sequential"
            and estimate_tokens is not None
            and max_prompt_tokens is not None
        ):
            raise ValueError(
                "prompt_batch_mode='sequential' does not support prompt truncation"
            )
        self._batch_size = batch_size
        self._prompt_batch_size = prompt_batch_size
        self._prompt_batch_mode = prompt_batch_mode
        self._num_verifications = num_verifications
        self._prompt_grouping = prompt_grouping
        self._prompt_shuffle = prompt_shuffle
        self._joint_prompting = joint_prompting
        self._show_turn_indices = show_turn_indices
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
        if self._num_verifications == 1:
            messages = self._build_messages(point)
            result = self._backend.generate_chat(messages, self._sampling_params)
            answer, _meta = self._extractor.extract(result.text or "")
            value = float(answer) if answer is not None else math.nan
            self.last_aborted_indices = set() if not math.isnan(value) else {0}
            return value

        values: list[float] = []
        messages = self._build_messages(point)
        for _ in range(self._num_verifications):
            result = self._backend.generate_chat(messages, self._sampling_params)
            answer, _meta = self._extractor.extract(result.text or "")
            if answer is not None:
                values.append(float(answer))
        if not values:
            self.last_aborted_indices = {0}
            return math.nan
        self.last_aborted_indices = set()
        return sum(values) / len(values)

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if not points:
            self.last_aborted_indices: set[int] = set()
            return []

        total = len(points)
        prompt_groups = self._plan_prompt_groups(points)
        use_sequential = (
            self._prompt_batch_mode == "sequential"
            and any(len(group.points) > 1 for group in prompt_groups)
        )
        group_messages = (
            [self._build_messages_for_group(group.points) for group in prompt_groups]
            if not use_sequential
            else None
        )

        if self._num_verifications == 1:
            if use_sequential:
                results, aborted = self._evaluate_groups_once_sequential(
                    prompt_groups,
                    progress_callback=progress_callback,
                )
            else:
                assert group_messages is not None
                results, aborted = self._evaluate_groups_once(
                    prompt_groups,
                    group_messages,
                    progress_callback=progress_callback,
                )
        else:
            results, aborted = aggregate_repeated_prediction_passes(
                total_points=total,
                num_passes=self._num_verifications,
                run_pass=lambda _pass_index, on_progress: (
                    self._evaluate_groups_once_sequential(
                        prompt_groups,
                        progress_callback=on_progress,
                    )
                    if use_sequential
                    else self._evaluate_groups_once(
                        prompt_groups,
                        group_messages or [],
                        progress_callback=on_progress,
                    )
                ),
                progress_callback=progress_callback,
            )

        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "LLMDirect: %d/%d points aborted due to missing successful predictions",
                len(aborted), total,
            )
        return results

    def _evaluate_groups_once(
        self,
        prompt_groups: list[PromptGroupPlan],
        group_messages: list[list[ChatMessage]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[float], set[int]]:
        from qval.error_handling import generate_batch_with_transient_retry

        total = sum(len(group.points) for group in prompt_groups)
        aborted: set[int] = set()
        chunk_size = self._batch_size or len(prompt_groups)
        results: list[float] = [math.nan] * total
        points_completed = 0

        for start in range(0, len(prompt_groups), chunk_size):
            end = min(start + chunk_size, len(prompt_groups))
            chunk_indices = list(range(start, end))
            chunk_messages = [group_messages[i] for i in chunk_indices]

            gen_results = generate_batch_with_transient_retry(
                self._backend, chunk_messages, self._sampling_params,
            )
            for local_idx, gr in enumerate(gen_results):
                group_idx = chunk_indices[local_idx]
                group = prompt_groups[group_idx]
                point_indices = group.original_indices
                if gr is None:
                    for point_idx in point_indices:
                        aborted.add(point_idx)
                    continue
                parsed = self._parse_group_response(
                    gr.text or "",
                    expected_count=len(group.points),
                )
                if parsed is None:
                    continue
                for point_idx, value in zip(point_indices, parsed):
                    results[point_idx] = value
            points_completed += sum(len(prompt_groups[i].points) for i in chunk_indices)
            if progress_callback:
                progress_callback(points_completed, total)

        return results, aborted

    def _evaluate_groups_once_sequential(
        self,
        prompt_groups: list[PromptGroupPlan],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[float], set[int]]:
        from qval.error_handling import generate_batch_with_transient_retry

        total = sum(len(group.points) for group in prompt_groups)
        results: list[float] = [math.nan] * total
        aborted: set[int] = set()
        chunk_size = self._batch_size or len(prompt_groups)
        points_completed = 0
        active_group_indices = list(range(len(prompt_groups)))
        group_states = [
            self._build_sequential_group_state(group)
            for group in prompt_groups
        ]

        while active_group_indices:
            round_group_indices = list(active_group_indices)
            next_active_group_indices: list[int] = []
            for start in range(0, len(round_group_indices), chunk_size):
                chunk_group_indices = round_group_indices[start:start + chunk_size]
                chunk_messages = [
                    self._build_messages_for_sequential_turn(group_states[group_idx])
                    for group_idx in chunk_group_indices
                ]
                gen_results = generate_batch_with_transient_retry(
                    self._backend,
                    chunk_messages,
                    self._sampling_params,
                )
                for local_idx, gr in enumerate(gen_results):
                    group_idx = chunk_group_indices[local_idx]
                    group_state = group_states[group_idx]
                    remaining_indices = group_state.original_indices[group_state.next_turn_index:]
                    if gr is None:
                        aborted.update(remaining_indices)
                        points_completed += len(remaining_indices)
                        continue

                    parsed = self._parse_sequential_turn_response(gr.text or "")
                    if parsed is None:
                        aborted.update(remaining_indices)
                        points_completed += len(remaining_indices)
                        continue

                    canonical_reply, value = parsed
                    point_idx = group_state.original_indices[group_state.next_turn_index]
                    results[point_idx] = value
                    points_completed += 1

                    if len(group_state.prepared_points) > 1:
                        group_state.messages.append(
                            ChatMessage(
                                role="user",
                                content=chunk_messages[local_idx][-1].content,
                            )
                        )
                        group_state.messages.append(
                            ChatMessage(role="assistant", content=canonical_reply)
                        )

                    group_state.next_turn_index += 1
                    if group_state.next_turn_index < len(group_state.prepared_points):
                        next_active_group_indices.append(group_idx)

                if progress_callback is not None:
                    progress_callback(points_completed, total)
            active_group_indices = next_active_group_indices

        return results, aborted

    def _build_messages(self, point: EvaluationPoint) -> list[ChatMessage]:
        prepared = self._prepare_point(point)
        return self._build_messages_for_prepared_points([prepared])

    def _build_messages_for_group(
        self,
        points: list[EvaluationPoint],
    ) -> list[ChatMessage]:
        prepared_points = self._prepare_points_for_group(points)
        return self._build_messages_for_prepared_points(prepared_points)

    def _build_messages_for_prepared_points(
        self,
        points: list[_PreparedDirectPoint],
    ) -> list[ChatMessage]:
        if not points:
            raise ValueError("LLMDirectMethod requires at least one point")

        system_content = build_direct_system_prompt(
            self.context,
            self._prompt_preset,
            num_points=len(points),
            joint_trajectory_prompt=self._joint_prompting and len(points) > 1,
            batch_mode=self._prompt_batch_mode,
        )
        if self._prompt_batch_mode == "sequential" and len(points) > 1:
            user_content = self._build_sequential_user_turn_content(points, point_index=0)
        elif len(points) == 1:
            point = points[0]
            user_content = build_direct_user_prompt(
                point.point,
                self.context.signal_type,
                history=point.history,
                history_truncated_from=point.history_truncated_from,
                include_next_state=self.context.include_next_state,
                include_current_thoughts=self.context.include_current_thoughts,
                include_history_thoughts=self.context.include_history_thoughts,
                state_text_override=point.state_text_override,
                next_state_text_override=point.next_state_text_override,
                use_answer_tags=self._prompt_preset.uses_answer_tags,
                max_steps=self.context.max_steps,
                show_turn_labels=self._show_turn_indices,
                task_text_to_inject=(
                    _task_text_for_injection(point.point)
                    if self._inject_task_text
                    else None
                ),
                include_state_text_when_images=self.context.include_state_text_when_images,
                include_images=self.context.include_images,
            )
        else:
            user_content = build_direct_user_prompt_batch(
                [point.point for point in points],
                self.context.signal_type,
                histories=[point.history for point in points],
                history_truncated_froms=[
                    point.history_truncated_from for point in points
                ],
                include_next_state=self.context.include_next_state,
                include_current_thoughts=self.context.include_current_thoughts,
                include_history_thoughts=self.context.include_history_thoughts,
                state_text_overrides=[
                    point.state_text_override for point in points
                ],
                next_state_text_overrides=[
                    point.next_state_text_override for point in points
                ],
                use_answer_tags=self._prompt_preset.uses_answer_tags,
                max_steps=self.context.max_steps,
                joint_trajectory_prompt=self._joint_prompting,
                show_turn_labels=self._show_turn_indices,
                task_text_to_injects=(
                    [_task_text_for_injection(p.point) for p in points]
                    if self._inject_task_text
                    else None
                ),
                include_state_text_when_images=self.context.include_state_text_when_images,
                include_images=self.context.include_images,
            )

        if hasattr(user_content, "has_images") and user_content.has_images:
            user_msg = ChatMessage(role="user", content_blocks=user_content.blocks)
        elif hasattr(user_content, "text"):
            user_msg = ChatMessage(role="user", content=user_content.text)
        else:
            user_msg = ChatMessage(role="user", content=user_content)
        return [
            ChatMessage(role="system", content=system_content),
            user_msg,
        ]

    def _prepare_points_for_group(
        self,
        points: list[EvaluationPoint],
    ) -> list[_PreparedDirectPoint]:
        if len(points) == 1:
            return [self._prepare_point(points[0])]

        system_content = build_direct_system_prompt(
            self.context,
            self._prompt_preset,
            num_points=len(points),
            joint_trajectory_prompt=self._joint_prompting and len(points) > 1,
            batch_mode=self._prompt_batch_mode,
        )
        point_budget: int | None = None
        if self._max_prompt_tokens is not None and self._estimate_tokens is not None:
            system_tokens = self._estimate_system_tokens(system_content)
            shared_budget = max(0, self._max_prompt_tokens - system_tokens)
            point_budget = system_tokens + max(0, shared_budget // len(points))

        return [
            self._prepare_point(point, system_content=system_content, max_prompt_tokens=point_budget)
            for point in points
        ]

    def _prepare_point(
        self,
        point: EvaluationPoint,
        *,
        system_content: str | None = None,
        max_prompt_tokens: int | None = None,
    ) -> _PreparedDirectPoint:
        if system_content is None:
            system_content = build_direct_system_prompt(
                self.context,
                self._prompt_preset,
                num_points=1,
                batch_mode=self._prompt_batch_mode,
            )

        history = list(point.history)
        history_truncated_from: int | None = None
        if self._max_history_turns is not None and len(history) > self._max_history_turns:
            history_truncated_from = len(history)
            history = history[-self._max_history_turns:] if self._max_history_turns > 0 else []

        state_text_override: str | None = None
        next_state_text_override: str | None = None
        if self._max_prompt_tokens is not None and self._estimate_tokens is not None:
            history, state_text_override, next_state_text_override = (
                self._apply_budget_truncation(
                    history,
                    system_content,
                    point,
                    history_truncated_from,
                    max_prompt_tokens=max_prompt_tokens,
                )
            )

        return _PreparedDirectPoint(
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

    def _parse_group_response(
        self,
        response: str,
        *,
        expected_count: int,
    ) -> list[float] | None:
        if expected_count == 1:
            answer, _meta = self._extractor.extract(response)
            return [float(answer) if answer is not None else math.nan]

        cleaned = strip_special_tokens(strip_thinking_tokens(response))
        content = cleaned
        if self._prompt_preset.uses_answer_tags:
            tag_content, _meta = _ANSWER_TAG_EXTRACTOR.extract(cleaned)
            if tag_content is not None:
                content = tag_content

        items = [item.strip() for item in content.split(",")]
        if len(items) != expected_count or any(not item for item in items):
            return None

        values: list[float] = []
        for item in items:
            answer, _meta = self._extractor.extract(item)
            values.append(float(answer) if answer is not None else math.nan)
        return values

    def _parse_sequential_turn_response(
        self,
        response: str,
    ) -> tuple[str, float] | None:
        answer, _meta = self._extractor.extract(response)
        if answer is None:
            return None
        canonical_answer = answer.strip()
        if self._prompt_preset.uses_answer_tags:
            canonical_answer = (
                f"[thinking not shown]\n<answer>{canonical_answer}</answer>"
            )
        return canonical_answer, float(answer)

    def _build_sequential_group_state(
        self,
        group: PromptGroupPlan,
    ) -> _SequentialDirectGroupState:
        prepared_points = self._prepare_points_for_group(group.points)
        if len(prepared_points) <= 1:
            messages: list[ChatMessage] = []
        else:
            system_content = build_direct_system_prompt(
                self.context,
                self._prompt_preset,
                num_points=len(prepared_points),
                joint_trajectory_prompt=self._joint_prompting and len(prepared_points) > 1,
                batch_mode="sequential",
            )
            messages = [ChatMessage(role="system", content=system_content)]
        return _SequentialDirectGroupState(
            prepared_points=prepared_points,
            original_indices=list(group.original_indices),
            messages=messages,
        )

    def _build_messages_for_sequential_turn(
        self,
        group_state: _SequentialDirectGroupState,
    ) -> list[ChatMessage]:
        if len(group_state.prepared_points) == 1:
            return self._build_messages_for_prepared_points(group_state.prepared_points)
        user_content = self._build_sequential_user_turn_content(
            group_state.prepared_points,
            point_index=group_state.next_turn_index,
        )
        if user_content.has_images:
            user_msg = ChatMessage(role="user", content_blocks=user_content.blocks)
        else:
            user_msg = ChatMessage(role="user", content=user_content.text)
        return [*group_state.messages, user_msg]

    def _build_sequential_user_turn_content(
        self,
        points: list[_PreparedDirectPoint],
        *,
        point_index: int,
    ):
        point = points[point_index]
        return build_direct_user_prompt_sequential_turn(
            point.point,
            self.context.signal_type,
            point_index=point_index,
            num_points=len(points),
            history=point.history,
            history_truncated_from=point.history_truncated_from,
            include_next_state=self.context.include_next_state,
            include_current_thoughts=self.context.include_current_thoughts,
            include_history_thoughts=self.context.include_history_thoughts,
            state_text_override=point.state_text_override,
            next_state_text_override=point.next_state_text_override,
            use_answer_tags=self._prompt_preset.uses_answer_tags,
            max_steps=self.context.max_steps,
            show_turn_labels=self._show_turn_indices,
            task_text_to_inject=(
                _task_text_for_injection(point.point)
                if self._inject_task_text
                else None
            ),
            include_state_text_when_images=self.context.include_state_text_when_images,
            include_images=self.context.include_images,
        )

    def _apply_budget_truncation(
        self,
        history: list[HistoryTurn],
        system_content: str,
        point: EvaluationPoint,
        history_truncated_from: int | None,
        *,
        max_prompt_tokens: int | None = None,
    ) -> tuple[list[HistoryTurn], str | None, str | None]:
        """Apply budget-aware multi-phase truncation.

        Returns:
            (truncated_history, state_text_override, next_state_text_override).
            Text overrides are ``None`` when no truncation was needed for that
            component.
        """
        from qval.prompts import SIGNAL_TYPES_WITH_ACTION
        from qval.methods.serialization import serialize_observation
        from qval.token_budget import (
            budget_aware_truncation,
            truncate_text_to_token_savings,
        )

        assert self._estimate_tokens is not None
        assert self._max_prompt_tokens is not None
        prompt_budget = max_prompt_tokens or self._max_prompt_tokens

        # Cache system content token count
        system_content_tokens = self._estimate_system_tokens(system_content)

        state_text = serialize_observation(point.state)
        next_state_text: str | None = None
        if (
            self.context.signal_type in SIGNAL_TYPES_WITH_ACTION
            and self.context.include_next_state
            and point.next_state is not None
        ):
            next_state_text = serialize_observation(point.next_state)

        # Phase 1: truncate history
        state_text_override: str | None = None
        next_state_text_override: str | None = None

        def _estimate_total(
            history_value: list[HistoryTurn],
            *,
            state_override: str | None = None,
            next_state_override: str | None = None,
        ) -> int:
            prompt = build_direct_user_prompt(
                point,
                self.context.signal_type,
                history=history_value,
                history_truncated_from=history_truncated_from,
                include_next_state=self.context.include_next_state,
                include_current_thoughts=self.context.include_current_thoughts,
                include_history_thoughts=self.context.include_history_thoughts,
                state_text_override=state_override,
                next_state_text_override=next_state_override,
                use_answer_tags=self._prompt_preset.uses_answer_tags,
                show_turn_labels=self._show_turn_indices,
                task_text_to_inject=(
                    _task_text_for_injection(point)
                    if self._inject_task_text
                    else None
                ),
                include_state_text_when_images=self.context.include_state_text_when_images,
                include_images=self.context.include_images,
            )
            return system_content_tokens + self._estimate_tokens(prompt.text)

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

        # Phase 2: truncate next_state if still over budget
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

        return history, state_text_override, next_state_text_override

    def _plan_prompt_groups(
        self,
        points: list[EvaluationPoint],
    ) -> list[PromptGroupPlan]:
        return plan_prompt_groups(
            points,
            prompt_batch_size=self._prompt_batch_size,
            prompt_grouping=self._prompt_grouping,
            prompt_shuffle=self._prompt_shuffle,
        )

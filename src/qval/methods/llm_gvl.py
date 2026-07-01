"""Dedicated GVL-style direct method with full-trajectory shuffled context."""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Callable

from llenvs.core.state import ImageContent
from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams

from qval.dense_signal import DenseSignalMethod
from qval.methods.llm_direct import (
    _ANSWER_TAG_EVALUATOR_EXTRACTOR,
    _DEFAULT_EVALUATOR_EXTRACTOR,
)
from qval.methods.serialization import (
    action_text_for_display,
    extract_images,
    serialize_observation,
)
from qval.methods._batched_eval import aggregate_repeated_prediction_passes
from qval.prompt_presets import PromptPreset
from qval.prompts import (
    GVLContextTransitionText,
    _task_text_for_injection,
    build_gvl_system_prompt,
    build_gvl_user_prompt,
)
from qval.types import EvaluationPoint, MethodContext, SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrajectoryDisplayTransition:
    step_index: int
    state_text: str
    action_text: str
    next_state_text: str
    state_images: tuple[ImageContent, ...] = ()
    next_state_images: tuple[ImageContent, ...] = ()


@dataclass(frozen=True)
class _PreparedGVLPoint:
    point: EvaluationPoint
    context_transitions: list[GVLContextTransitionText]
    target_state_text: str
    target_next_state_text: str | None
    target_state_images: tuple[ImageContent, ...] = ()
    target_next_state_images: tuple[ImageContent, ...] = ()


class LLMGVLMethod(DenseSignalMethod):
    """GVL-style direct method using full-trajectory shuffled context."""

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        trajectory_results: list[Any] | None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_preset: PromptPreset,
        batch_size: int | None = None,
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
        if trajectory_results is None:
            raise ValueError(
                "LLMGVLMethod requires dataset trajectory_results. "
                "Saved-trajectory-JSON fallback is not implemented."
            )
        if context.include_current_thoughts:
            raise ValueError("LLMGVLMethod does not support include_current_thoughts")
        if context.include_history_thoughts:
            raise ValueError("LLMGVLMethod does not support include_history_thoughts")

        self._backend = backend
        self._sampling_params = sampling_params or SamplingParams()
        self._prompt_preset = prompt_preset
        self._batch_size = batch_size
        self._num_verifications = num_verifications
        self._estimate_tokens = estimate_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._min_observation_chars = min_observation_chars
        self._min_action_chars = min_action_chars
        self._min_current_observation_chars = min_current_observation_chars
        self._min_next_observation_chars = min_next_observation_chars
        self._inject_task_text = inject_task_text
        self._system_content = build_gvl_system_prompt(context, prompt_preset)
        self._system_tokens: int | None = None
        if context.evaluator_extractor is not None:
            self._extractor = context.evaluator_extractor
        elif prompt_preset.uses_answer_tags:
            self._extractor = _ANSWER_TAG_EVALUATOR_EXTRACTOR
        else:
            self._extractor = _DEFAULT_EVALUATOR_EXTRACTOR
        self._trajectory_lookup = self._build_trajectory_lookup(trajectory_results)
        self.last_aborted_indices: set[int] = set()

    def evaluate(self, point: EvaluationPoint) -> float:
        values = self.evaluate_batch([point])
        return values[0] if values else math.nan

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if not points:
            self.last_aborted_indices = set()
            return []

        prepared = [self._prepare_point(point) for point in points]
        messages = [self._build_messages(item) for item in prepared]

        if self._num_verifications == 1:
            results, aborted = self._evaluate_once(
                messages,
                progress_callback=progress_callback,
            )
        else:
            results, aborted = aggregate_repeated_prediction_passes(
                total_points=len(points),
                num_passes=self._num_verifications,
                run_pass=lambda _pass_index, on_progress: self._evaluate_once(
                    messages,
                    progress_callback=on_progress,
                ),
                progress_callback=progress_callback,
            )

        self.last_aborted_indices = aborted
        if aborted:
            logger.warning(
                "LLMGVL: %d/%d points aborted due to missing successful predictions",
                len(aborted), len(points),
            )
        return results

    def _build_trajectory_lookup(
        self,
        trajectory_results: list[Any],
    ) -> dict[int, tuple[_TrajectoryDisplayTransition, ...]]:
        lookup: dict[int, tuple[_TrajectoryDisplayTransition, ...]] = {}
        for trajectory_index, result in enumerate(trajectory_results):
            transitions = getattr(getattr(result, "trajectory", None), "transitions", None)
            if transitions is None:
                raise ValueError(
                    "LLMGVLMethod requires trajectory_results with trajectory.transitions"
                )
            lookup[trajectory_index] = tuple(
                _TrajectoryDisplayTransition(
                    step_index=step_index,
                    state_text=serialize_observation(transition.state),
                    action_text=action_text_for_display(
                        transition.action,
                        transition.resolved_action,
                        transition.extracted_action,
                    ),
                    next_state_text=serialize_observation(transition.next_state),
                    state_images=extract_images(transition.state).all,
                    next_state_images=extract_images(transition.next_state).all,
                )
                for step_index, transition in enumerate(transitions)
            )
        return lookup

    def _prepare_point(self, point: EvaluationPoint) -> _PreparedGVLPoint:
        trajectory = self._trajectory_lookup.get(point.trajectory_index)
        if trajectory is None:
            raise ValueError(
                "Missing stored trajectory for GVL point: "
                f"trajectory_index={point.trajectory_index}"
            )
        if not (0 <= point.step_index < len(trajectory)):
            raise ValueError(
                "GVL point step_index is outside stored trajectory bounds: "
                f"trajectory_index={point.trajectory_index}, step_index={point.step_index}, "
                f"num_steps={len(trajectory)}"
            )

        target = trajectory[point.step_index]
        context_transitions = [
            GVLContextTransitionText(
                state_text=item.state_text,
                action_text=item.action_text,
                step_index=item.step_index,
                state_images=item.state_images,
            )
            for item in trajectory
            if item.step_index != point.step_index
        ]
        target_state_text = target.state_text
        include_target_next = (
            self.context.signal_type in {
                SignalType.Q_VALUE,
                SignalType.ADVANTAGE,
                SignalType.SHAPED_REWARD,
            }
            and self.context.include_next_state
        )
        target_next_state_text = target.next_state_text if include_target_next else None
        target_state_images = target.state_images
        target_next_state_images = (
            target.next_state_images if include_target_next else ()
        )

        if self._max_prompt_tokens is not None and self._estimate_tokens is not None:
            (
                context_transitions,
                target_state_text,
                target_next_state_text,
            ) = self._fit_to_budget(
                point,
                context_transitions,
                target_state_text,
                target_next_state_text,
                target_state_images,
                target_next_state_images,
            )

        shuffled_context = self._shuffle_context(context_transitions, point)
        return _PreparedGVLPoint(
            point=point,
            context_transitions=shuffled_context,
            target_state_text=target_state_text,
            target_next_state_text=target_next_state_text,
            target_state_images=target_state_images,
            target_next_state_images=target_next_state_images,
        )

    def _build_messages(self, prepared: _PreparedGVLPoint) -> list[ChatMessage]:
        user_prompt = build_gvl_user_prompt(
            prepared.point,
            self.context.signal_type,
            context_transitions=prepared.context_transitions,
            target_state_text=prepared.target_state_text,
            target_next_state_text=prepared.target_next_state_text,
            target_state_images=prepared.target_state_images,
            target_next_state_images=prepared.target_next_state_images,
            include_next_state=self.context.include_next_state,
            use_answer_tags=self._prompt_preset.uses_answer_tags,
            task_text_to_inject=(
                _task_text_for_injection(prepared.point)
                if self._inject_task_text
                else None
            ),
            include_state_text_when_images=self.context.include_state_text_when_images,
            include_images=self.context.include_images,
        )
        if user_prompt.has_images:
            user_message = ChatMessage(role="user", content_blocks=user_prompt.blocks)
        else:
            user_message = ChatMessage(role="user", content=user_prompt.text)
        return [
            ChatMessage(role="system", content=self._system_content),
            user_message,
        ]

    def _evaluate_once(
        self,
        messages: list[list[ChatMessage]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[float], set[int]]:
        from qval.error_handling import generate_batch_with_transient_retry

        total = len(messages)
        results: list[float] = [math.nan] * total
        aborted: set[int] = set()
        chunk_size = self._batch_size or len(messages)
        completed = 0

        for start in range(0, len(messages), chunk_size):
            chunk = messages[start:start + chunk_size]
            gen_results = generate_batch_with_transient_retry(
                self._backend,
                chunk,
                self._sampling_params,
            )
            for local_idx, gr in enumerate(gen_results):
                point_idx = start + local_idx
                if gr is None:
                    aborted.add(point_idx)
                    continue
                answer, _meta = self._extractor.extract(gr.text or "")
                if answer is None:
                    aborted.add(point_idx)
                    continue
                results[point_idx] = float(answer)
            completed = min(total, start + chunk_size)
            if progress_callback is not None:
                progress_callback(completed, total)

        return results, aborted

    def _shuffle_context(
        self,
        context_transitions: list[GVLContextTransitionText],
        point: EvaluationPoint,
    ) -> list[GVLContextTransitionText]:
        shuffled = list(context_transitions)
        if len(shuffled) <= 1:
            return shuffled
        seed = int(
            hashlib.sha256(
                f"{point.trajectory_index}:{point.step_index}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def _fit_to_budget(
        self,
        point: EvaluationPoint,
        context_transitions: list[GVLContextTransitionText],
        target_state_text: str,
        target_next_state_text: str | None,
        target_state_images: tuple[ImageContent, ...] = (),
        target_next_state_images: tuple[ImageContent, ...] = (),
    ) -> tuple[list[GVLContextTransitionText], str, str | None]:
        from qval.token_budget import truncate_text_to_token_savings

        assert self._estimate_tokens is not None
        assert self._max_prompt_tokens is not None

        working_context = list(context_transitions)
        working_target_state = target_state_text
        working_target_next = target_next_state_text

        def _excess() -> int:
            return (
                self._estimate_total_tokens(
                    point,
                    working_context,
                    working_target_state,
                    working_target_next,
                    target_state_images,
                    target_next_state_images,
                )
                - self._max_prompt_tokens
            )

        excess = _excess()
        if excess <= 0:
            return working_context, working_target_state, working_target_next

        prioritized_indices = sorted(
            range(len(working_context)),
            key=lambda idx: (
                -abs(working_context[idx].step_index - point.step_index),
                working_context[idx].step_index,
            ),
        )
        for idx in prioritized_indices:
            if excess <= 0:
                break
            item = working_context[idx]
            if self._min_observation_chars is not None:
                new_state = truncate_text_to_token_savings(
                    item.state_text,
                    excess,
                    self._min_observation_chars,
                    self._estimate_tokens,
                )
                if new_state != item.state_text:
                    working_context[idx] = replace(item, state_text=new_state)
                    excess = _excess()
            if excess <= 0:
                break
            item = working_context[idx]
            if self._min_action_chars is not None:
                new_action = truncate_text_to_token_savings(
                    item.action_text,
                    excess,
                    self._min_action_chars,
                    self._estimate_tokens,
                )
                if new_action != item.action_text:
                    working_context[idx] = replace(item, action_text=new_action)
                    excess = _excess()

        if (
            excess > 0
            and working_target_next is not None
            and self._min_next_observation_chars is not None
        ):
            candidate = truncate_text_to_token_savings(
                working_target_next,
                excess,
                self._min_next_observation_chars,
                self._estimate_tokens,
            )
            if candidate != working_target_next:
                working_target_next = candidate
                excess = _excess()

        if excess > 0 and self._min_current_observation_chars is not None:
            candidate = truncate_text_to_token_savings(
                working_target_state,
                excess,
                self._min_current_observation_chars,
                self._estimate_tokens,
            )
            if candidate != working_target_state:
                working_target_state = candidate
                excess = _excess()

        while excess > 0 and working_context:
            drop_idx = min(
                range(len(working_context)),
                key=lambda idx: (
                    -abs(working_context[idx].step_index - point.step_index),
                    working_context[idx].step_index,
                ),
            )
            del working_context[drop_idx]
            excess = _excess()

        return working_context, working_target_state, working_target_next

    def _estimate_total_tokens(
        self,
        point: EvaluationPoint,
        context_transitions: list[GVLContextTransitionText],
        target_state_text: str,
        target_next_state_text: str | None,
        target_state_images: tuple[ImageContent, ...] = (),
        target_next_state_images: tuple[ImageContent, ...] = (),
    ) -> int:
        if self._estimate_tokens is None:
            raise AssertionError("_estimate_total_tokens requires _estimate_tokens")
        user_content = build_gvl_user_prompt(
            point,
            self.context.signal_type,
            context_transitions=context_transitions,
            target_state_text=target_state_text,
            target_next_state_text=target_next_state_text,
            target_state_images=target_state_images,
            target_next_state_images=target_next_state_images,
            include_next_state=self.context.include_next_state,
            use_answer_tags=self._prompt_preset.uses_answer_tags,
            task_text_to_inject=(
                _task_text_for_injection(point)
                if self._inject_task_text
                else None
            ),
            include_state_text_when_images=self.context.include_state_text_when_images,
            include_images=self.context.include_images,
        )
        return self._estimate_system_tokens() + self._estimate_tokens(user_content.text)

    def _estimate_system_tokens(self) -> int:
        if self._estimate_tokens is None:
            raise AssertionError("_estimate_system_tokens requires _estimate_tokens")
        if self._system_tokens is None:
            self._system_tokens = self._estimate_tokens(self._system_content)
        return self._system_tokens

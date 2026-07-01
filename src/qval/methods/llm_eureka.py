"""Eureka-style iterative code search with LLM judging over candidate outputs."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Callable

from llenvs.inference import compose_system_prompt
from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams

from qval.dense_signal import DenseSignalMethod
from qval.error_handling import (
    generate_batch_with_transient_retry,
    is_recoverable_backend_error,
    is_transient_api_error,
    retry_on_transient,
)
from qval.methods.code_parsing import extract_code, validate_code
from qval.methods.llm_codegen import LLMCodeGenMethod
from qval.methods.llm_direct import LLMDirectMethod
from qval.methods.serialization import serialize_action, serialize_observation
from qval.prompt_presets import PromptPreset, resolve_block
from qval.prompts import (
    SIGNAL_TYPE_NAMES,
    _build_direct_prompt_sections,
    _format_environment_notes,
    _format_example_trajectories,
    _task_text_for_injection,
    build_codegen_system_prompt,
    build_signal_type_fragment,
    codegen_param_names,
)
from qval.types import EvaluationPoint, MethodContext

logger = logging.getLogger(__name__)

_WINNER_TAG = re.compile(r"<winner>\s*(\d+)\s*</winner>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CandidateStats:
    finite_count: int
    nan_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    collapsed: bool


@dataclass(frozen=True)
class CandidateComponentStats:
    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    collapsed: bool


@dataclass
class _CandidateResult:
    code: str | None
    signal_fn: Callable[..., float] | None
    predictions: list[float]
    stats: CandidateStats
    component_stats: dict[str, CandidateComponentStats]
    invalid_reason: str | None = None
    runtime_error: str | None = None
    from_previous: bool = False
    judge_label: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.signal_fn is not None and self.code is not None


@dataclass(frozen=True)
class JudgeDecision:
    winner: int
    rationale: str
    feedback: str


def _stats(values: list[float]) -> CandidateStats:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return CandidateStats(
            finite_count=0,
            nan_count=len(values),
            minimum=None,
            maximum=None,
            mean=None,
            std=None,
            collapsed=False,
        )
    minimum = min(finite)
    maximum = max(finite)
    mean = sum(finite) / len(finite)
    if len(finite) > 1:
        variance = sum((value - mean) ** 2 for value in finite) / len(finite)
        std = variance ** 0.5
    else:
        std = 0.0
    return CandidateStats(
        finite_count=len(finite),
        nan_count=len(values) - len(finite),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        collapsed=(maximum - minimum) < 1e-9 if len(finite) > 1 else False,
    )


def _component_stats(values: list[float]) -> CandidateComponentStats:
    if not values:
        return CandidateComponentStats(0, None, None, None, None, False)
    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = variance ** 0.5
    else:
        std = 0.0
    return CandidateComponentStats(
        count=len(values),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        collapsed=(maximum - minimum) < 1e-9 if len(values) > 1 else False,
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _extract_tag(text: str, tag_name: str) -> str:
    pattern = re.compile(
        rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_judge_decision(text: str) -> JudgeDecision | None:
    winner_match = _WINNER_TAG.search(text)
    if winner_match is None:
        return None
    rationale = _extract_tag(text, "rationale")
    feedback = _extract_tag(text, "feedback")
    return JudgeDecision(
        winner=int(winner_match.group(1)),
        rationale=rationale,
        feedback=feedback,
    )


def build_eureka_generation_system_prompt(
    context: MethodContext,
    preset: PromptPreset,
) -> str:
    base = build_codegen_system_prompt(context, preset)
    iterative = (
        "## Iterative Search\n"
        "You are participating in an iterative code-search loop. Each proposed "
        "candidate will be executed on benchmark datapoints, judged from its "
        "predicted values, and the winning code plus feedback may be shown back "
        "to you in the next iteration. Improve usefulness of the signal based on "
        "behavior, not code aesthetics."
    )
    return compose_system_prompt(base, iterative)


def build_eureka_generation_user_prompt(
    signal_type,
    preset: PromptPreset,
    *,
    include_next_state: bool = True,
    enable_thinking: bool = True,
    iteration: int,
    total_iterations: int,
    previous_winner_code: str | None = None,
    reflection: str | None = None,
) -> str:
    signal_name = SIGNAL_TYPE_NAMES[signal_type]
    signature = ", ".join(
        f"{name}: str" for name in codegen_param_names(signal_type, include_next_state)
    )
    definition = f"def signal_function({signature}):"
    parts = [
        f"## Search Iteration\nIteration {iteration} of {total_iterations}. "
        f"Write one candidate Python function for estimating the {signal_name}.",
        (
            "## Function Contract\n"
            f"The function definition must start with:\n```python\n{definition}\n```\n\n"
            "The function may return either:\n"
            "1. a single float, which is the actual signal value\n"
            "2. a tuple ``(float, dict[str, float])`` where the first float is "
            "the actual signal value and the dictionary contains named additive "
            "components used only for search-time feedback.\n\n"
            "If you return a dictionary, use short stable snake_case keys and "
            "make the scalar total equal to the sum of the component values. "
            "Build the components dictionary with explicit literal keys — do "
            "NOT use `locals()`, `globals()`, or `vars()` (they are unavailable "
            "in the sandbox and raise NameError)."
        ),
        (
            "## Return Example\n"
            "```python\n"
            f"{definition}\n"
            "    progress_reward = 0.4\n"
            "    safety_penalty = -0.1\n"
            "    total = progress_reward + safety_penalty\n"
            "    return total, {\n"
            "        \"progress_reward\": progress_reward,\n"
            "        \"safety_penalty\": safety_penalty,\n"
            "    }\n"
            "```"
        ),
    ]

    if previous_winner_code is not None:
        parts.append(
            "## Current Best Code\n"
            "This is the current best candidate from the previous iteration. "
            "Use it as a starting point and improve it.\n\n"
            f"```python\n{previous_winner_code}\n```"
        )
    if reflection:
        parts.append(f"## Judge Feedback\n{reflection}")

    constraints = resolve_block(preset.codegen_constraints)
    if constraints is not None:
        parts.append(f"## Constraints\n{constraints}")

    closing = resolve_block(preset.codegen_closing)
    if closing is not None:
        if not enable_thinking and closing.startswith("After any reasoning, "):
            closing = closing[len("After any reasoning, "):]
            closing = closing[:1].upper() + closing[1:]
        parts.append(closing)

    return "\n\n".join(parts)


def build_eureka_judge_system_prompt(
    context: MethodContext,
) -> str:
    parts: list[str] = [
        "You are selecting the most useful dense-signal candidate for a "
        "reinforcement learning environment.",
        build_signal_type_fragment(
            context.signal_type,
            context.policy_assumption,
            context.discount_factor,
            disclose_discount_factor=context.disclose_discount_factor,
        ).content,
        (
            "Judge candidate behavior from predicted values and diagnostics. "
            "Do not infer quality from code style because candidate code is not shown."
        ),
    ]
    if context.task_description:
        parts.append(f"## Task Description\n{context.task_description.rstrip()}")
    if context.environment_notes:
        parts.append(_format_environment_notes(context.environment_notes))
    if context.reward_description:
        parts.append(f"## Reward Functions\n{context.reward_description.rstrip()}")
    if context.example_trajectories:
        parts.append(_format_example_trajectories(context.example_trajectories))
    return compose_system_prompt(*parts)


def build_eureka_judge_user_prompt(
    *,
    prepared_points: list[Any],
    signal_type,
    valid_candidates: list[_CandidateResult],
    invalid_candidates: list[_CandidateResult],
    include_next_state: bool,
    include_current_thoughts: bool,
    include_history_thoughts: bool,
    max_steps: int | None,
    inject_task_text: bool,
) -> str:
    signal_name = SIGNAL_TYPE_NAMES[signal_type]
    parts: list[str] = [
        "## Selection Rule\n"
        f"Choose the candidate whose predictions are the most appropriate and useful "
        f"{signal_name} estimates. Prefer candidates that provide discriminative, "
        "stable, non-collapsed values and that match the intended semantics of the "
        "signal on the sampled datapoints."
    ]

    parts.append("## Candidate Summaries")
    for candidate in valid_candidates:
        summary_lines = [
            f"### Candidate {candidate.judge_label}",
            f"- Finite predictions: {candidate.stats.finite_count}",
            f"- NaN predictions: {candidate.stats.nan_count}",
            f"- Min: {_format_float(candidate.stats.minimum)}",
            f"- Max: {_format_float(candidate.stats.maximum)}",
            f"- Mean: {_format_float(candidate.stats.mean)}",
            f"- Std: {_format_float(candidate.stats.std)}",
            f"- Collapsed: {'yes' if candidate.stats.collapsed else 'no'}",
        ]
        if candidate.component_stats:
            summary_lines.append("- Components:")
            for key in sorted(candidate.component_stats):
                comp = candidate.component_stats[key]
                summary_lines.append(
                    f"  - {key}: count={comp.count}, min={_format_float(comp.minimum)}, "
                    f"max={_format_float(comp.maximum)}, mean={_format_float(comp.mean)}, "
                    f"std={_format_float(comp.std)}, collapsed={'yes' if comp.collapsed else 'no'}"
                )
        parts.append("\n".join(summary_lines))

    if invalid_candidates:
        rejected = ["## Rejected Candidates"]
        for idx, candidate in enumerate(invalid_candidates, start=1):
            rejected.append(
                f"### Rejected Candidate {idx}\n- Status: invalid\n- Reason: {candidate.invalid_reason}"
            )
        parts.append("\n\n".join(rejected))

    parts.append("## Sampled Datapoints")
    for point_idx, prepared in enumerate(prepared_points, start=1):
        point_parts = _build_direct_prompt_sections(
            prepared.point,
            signal_type,
            history=prepared.history,
            history_truncated_from=prepared.history_truncated_from,
            include_next_state=include_next_state,
            include_current_thoughts=include_current_thoughts,
            include_history_thoughts=include_history_thoughts,
            state_text_override=prepared.state_text_override,
            next_state_text_override=prepared.next_state_text_override,
            max_steps=max_steps,
            show_turn_labels=True,
            task_text_to_inject=(
                _task_text_for_injection(prepared.point) if inject_task_text else None
            ),
        )
        prediction_lines = ["### Candidate Predictions"]
        for candidate in valid_candidates:
            prediction_lines.append(
                f"Candidate {candidate.judge_label}: "
                f"{_format_float(candidate.predictions[point_idx - 1])}"
            )
        point_parts.append("\n".join(prediction_lines))
        parts.append(f"### Datapoint {point_idx}\n\n" + "\n\n".join(point_parts))

    parts.append(
        "## Output Format\n"
        "Respond with:\n"
        "<winner>INTEGER</winner>\n"
        "<rationale>Short explanation of why the winner is best.</rationale>\n"
        "<feedback>Concrete guidance for improving the next iteration.</feedback>"
    )
    return "\n\n".join(parts)


class LLMEurekaMethod(DenseSignalMethod):
    """Iteratively searches over generated signal functions with LLM judging."""

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_preset: PromptPreset,
        num_candidates: int = 1,
        search_iterations: int = 3,
        judge_num_points: int = 8,
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
        self._judge_sampling_params = replace(
            self._sampling_params,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        self._prompt_preset = prompt_preset
        self._num_candidates = num_candidates
        self._search_iterations = search_iterations
        self._judge_num_points = judge_num_points
        self._estimate_tokens = estimate_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._generated_code: str | None = None
        self._signal_fn: Callable[..., float] | None = None
        self.last_aborted_indices: set[int] = set()
        self.last_runtime_error: str | None = None
        self._param_names = codegen_param_names(
            context.signal_type, context.include_next_state,
        )
        self._point_preparer = LLMDirectMethod(
            backend=backend,
            context=context,
            sampling_params=sampling_params,
            max_history_turns=context.max_history_turns,
            prompt_preset=prompt_preset,
            batch_size=None,
            prompt_batch_size=1,
            estimate_tokens=estimate_tokens,
            max_prompt_tokens=max_prompt_tokens,
            min_observation_chars=min_observation_chars,
            min_action_chars=min_action_chars,
            min_current_observation_chars=min_current_observation_chars,
            min_next_observation_chars=min_next_observation_chars,
            inject_task_text=inject_task_text,
        )
        self._inject_task_text = inject_task_text

    @property
    def generated_code(self) -> str | None:
        return self._generated_code

    def evaluate(self, point: EvaluationPoint) -> float:
        if self._signal_fn is None:
            values = self.evaluate_batch([point])
            return values[0] if values else math.nan
        return self._evaluate_point_with_fn(point, self._signal_fn)[0]

    def evaluate_batch(
        self,
        points: list[EvaluationPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if not points:
            self.last_aborted_indices = set()
            self.last_runtime_error = None
            return []

        if self._signal_fn is None:
            try:
                winner = self._search(points)
            except Exception as exc:
                if not (
                    is_recoverable_backend_error(exc)
                    or is_transient_api_error(exc)
                ):
                    raise
                logger.warning(
                    "Recoverable backend failure during Eureka search; returning NaN "
                    "for %d evaluation points",
                    len(points),
                    exc_info=True,
                )
                self.last_aborted_indices = set(range(len(points)))
                self.last_runtime_error = None
                if progress_callback is not None:
                    progress_callback(len(points), len(points))
                return [math.nan] * len(points)

            if winner is None or winner.signal_fn is None or winner.code is None:
                self.last_aborted_indices = set(range(len(points)))
                self.last_runtime_error = None
                if progress_callback is not None:
                    progress_callback(len(points), len(points))
                return [math.nan] * len(points)

            self._signal_fn = winner.signal_fn
            self._generated_code = winner.code
            self.last_runtime_error = winner.runtime_error
            predicted = winner.predictions
        else:
            predicted = [
                self._evaluate_point_with_fn(point, self._signal_fn)[0]
                for point in points
            ]

        self.last_aborted_indices = {
            idx for idx, value in enumerate(predicted) if math.isnan(value)
        }
        if progress_callback is not None:
            progress_callback(len(points), len(points))
        return predicted

    def _search(self, points: list[EvaluationPoint]) -> _CandidateResult | None:
        incumbent: _CandidateResult | None = None
        reflection: str | None = None
        method_key = ",".join(
            f"{point.trajectory_index}:{point.step_index}" for point in points
        )

        for iteration in range(1, self._search_iterations + 1):
            logger.info(
                "Eureka search iteration %d/%d with %d candidates",
                iteration,
                self._search_iterations,
                self._num_candidates,
            )
            messages = self._build_generation_messages(
                iteration=iteration,
                previous_winner_code=incumbent.code if incumbent is not None else None,
                reflection=reflection,
            )
            results = generate_batch_with_transient_retry(
                self._backend,
                [messages] * self._num_candidates,
                self._sampling_params,
            )

            new_candidates = [
                self._compile_generation_result(result)
                for result in results
            ]

            valid_new = [candidate for candidate in new_candidates if candidate.is_valid]
            if not valid_new:
                if incumbent is None:
                    return None
                logger.warning(
                    "Iteration %d produced no valid new candidates; keeping incumbent.",
                    iteration,
                )
                break

            for candidate in valid_new:
                predictions, component_stats, runtime_error = self._evaluate_candidate(
                    points, candidate.signal_fn,
                )
                candidate.predictions = predictions
                candidate.stats = _stats(predictions)
                candidate.component_stats = component_stats
                candidate.runtime_error = runtime_error

            judge_candidates = list(valid_new)
            if incumbent is not None:
                judge_candidates.append(incumbent)

            shuffled = self._shuffle_candidates(judge_candidates, method_key, iteration)
            label_to_candidate: dict[int, _CandidateResult] = {}
            judge_views: list[_CandidateResult] = []
            for judge_label, candidate in enumerate(shuffled, start=1):
                candidate.judge_label = judge_label
                label_to_candidate[judge_label] = candidate

            invalid_candidates = [
                candidate for candidate in new_candidates if not candidate.is_valid
            ]

            sample_indices = self._sample_point_indices(
                len(points), method_key, iteration,
            )
            sampled_points = [points[idx] for idx in sample_indices]
            for candidate in shuffled:
                judge_views.append(
                    self._build_judge_view(candidate, sample_indices)
                )
            prepared_points = self._prepare_points_for_judge(
                sampled_points,
                valid_candidates=judge_views,
                invalid_candidates=invalid_candidates,
            )

            judge_messages = self._build_judge_messages(
                prepared_points=prepared_points,
                valid_candidates=judge_views,
                invalid_candidates=invalid_candidates,
            )
            try:
                judge_result = retry_on_transient(
                    lambda: self._backend.generate_chat(
                        judge_messages, self._judge_sampling_params,
                    ),
                    description="eureka judge",
                )
            except Exception as exc:
                if incumbent is not None and (
                    is_recoverable_backend_error(exc) or is_transient_api_error(exc)
                ):
                    logger.warning(
                        "Judge backend failure in later Eureka iteration; keeping incumbent.",
                        exc_info=True,
                    )
                    break
                raise

            decision = _parse_judge_decision(judge_result.text or "")
            if decision is None:
                if incumbent is not None:
                    logger.warning(
                        "Malformed judge response in later Eureka iteration; keeping incumbent."
                    )
                    break
                return None

            winner = label_to_candidate.get(decision.winner)
            if winner is None:
                if incumbent is not None:
                    logger.warning(
                        "Judge selected invalid candidate id %s; keeping incumbent.",
                        decision.winner,
                    )
                    break
                return None

            reflection = self._build_reflection(winner, decision)
            incumbent = winner
            (
                incumbent.predictions,
                incumbent.component_stats,
                incumbent.runtime_error,
            ) = self._evaluate_candidate(points, incumbent.signal_fn)
            incumbent.stats = _stats(incumbent.predictions)
            incumbent.from_previous = True

        return incumbent

    @staticmethod
    def _build_judge_view(
        candidate: _CandidateResult,
        sample_indices: list[int],
    ) -> _CandidateResult:
        return _CandidateResult(
            code=candidate.code,
            signal_fn=candidate.signal_fn,
            predictions=[candidate.predictions[idx] for idx in sample_indices],
            stats=candidate.stats,
            component_stats=candidate.component_stats,
            invalid_reason=candidate.invalid_reason,
            runtime_error=candidate.runtime_error,
            from_previous=candidate.from_previous,
            judge_label=candidate.judge_label,
        )

    def _build_generation_messages(
        self,
        *,
        iteration: int,
        previous_winner_code: str | None,
        reflection: str | None,
    ) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=build_eureka_generation_system_prompt(
                    self.context, self._prompt_preset,
                ),
            ),
            ChatMessage(
                role="user",
                content=build_eureka_generation_user_prompt(
                    self.context.signal_type,
                    self._prompt_preset,
                    include_next_state=self.context.include_next_state,
                    enable_thinking=self.context.enable_thinking,
                    iteration=iteration,
                    total_iterations=self._search_iterations,
                    previous_winner_code=previous_winner_code,
                    reflection=reflection,
                ),
            ),
        ]

    def _compile_generation_result(
        self,
        result: Any | None,
    ) -> _CandidateResult:
        if result is None:
            return _CandidateResult(
                code=None,
                signal_fn=None,
                predictions=[],
                stats=_stats([]),
                component_stats={},
                invalid_reason="backend generation failure",
            )
        try:
            code = extract_code(result.text).code
            validate_code(code)
            fn = LLMCodeGenMethod._compile_code(code, len(self._param_names))
        except ValueError as exc:
            return _CandidateResult(
                code=None,
                signal_fn=None,
                predictions=[],
                stats=_stats([]),
                component_stats={},
                invalid_reason=str(exc),
            )
        return _CandidateResult(
            code=code,
            signal_fn=fn,
            predictions=[],
            stats=_stats([]),
            component_stats={},
        )

    def _evaluate_candidate(
        self,
        points: list[EvaluationPoint],
        signal_fn: Callable[..., float] | None,
    ) -> tuple[list[float], dict[str, CandidateComponentStats], str | None]:
        if signal_fn is None:
            return [math.nan] * len(points), {}, None
        predictions: list[float] = []
        components: dict[str, list[float]] = {}
        first_error: str | None = None
        for point in points:
            value, point_components, error = self._evaluate_point_with_fn(point, signal_fn)
            predictions.append(value)
            if error is not None and first_error is None:
                first_error = error
            for key, component_value in point_components.items():
                components.setdefault(key, []).append(component_value)
        component_stats = {
            key: _component_stats(values)
            for key, values in components.items()
        }
        return predictions, component_stats, first_error

    def _evaluate_point_with_fn(
        self,
        point: EvaluationPoint,
        signal_fn: Callable[..., float],
    ) -> tuple[float, dict[str, float], str | None]:
        args: list[str] = []
        for name in self._param_names:
            if name == "state":
                args.append(serialize_observation(point.state))
            elif name == "action":
                args.append(serialize_action(point.action))
            elif name == "next_state":
                args.append(serialize_observation(point.next_state))
        try:
            raw = signal_fn(*args)
            if isinstance(raw, tuple):
                if len(raw) != 2 or not isinstance(raw[1], dict):
                    raise ValueError("signal_function tuple return must be (float, dict[str, float])")
                total = float(raw[0])
                component_values = {
                    str(key): float(value)
                    for key, value in raw[1].items()
                }
                return total, component_values, None
            return float(raw), {}, None
        except Exception as exc:
            logger.warning(
                "Runtime error while evaluating Eureka candidate",
                exc_info=True,
            )
            return math.nan, {}, f"{type(exc).__name__}: {exc}"

    def _shuffle_candidates(
        self,
        candidates: list[_CandidateResult],
        method_key: str,
        iteration: int,
    ) -> list[_CandidateResult]:
        shuffled = list(candidates)
        seed = self._stable_seed(f"candidates:{method_key}:{iteration}:{len(candidates)}")
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def _sample_point_indices(
        self,
        total_points: int,
        method_key: str,
        iteration: int,
    ) -> list[int]:
        sample_size = min(self._judge_num_points, total_points)
        seed = self._stable_seed(f"points:{method_key}:{iteration}:{total_points}")
        rng = random.Random(seed)
        return sorted(rng.sample(range(total_points), sample_size))

    @staticmethod
    def _stable_seed(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)

    def _build_judge_messages(
        self,
        *,
        prepared_points: list[Any],
        valid_candidates: list[_CandidateResult],
        invalid_candidates: list[_CandidateResult],
    ) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=build_eureka_judge_system_prompt(self.context),
            ),
            ChatMessage(
                role="user",
                content=build_eureka_judge_user_prompt(
                    prepared_points=prepared_points,
                    signal_type=self.context.signal_type,
                    valid_candidates=valid_candidates,
                    invalid_candidates=invalid_candidates,
                    include_next_state=self.context.include_next_state,
                    include_current_thoughts=self.context.include_current_thoughts,
                    include_history_thoughts=self.context.include_history_thoughts,
                    max_steps=self.context.max_steps,
                    inject_task_text=self._inject_task_text,
                ),
            ),
        ]

    def _build_reflection(
        self,
        winner: _CandidateResult,
        decision: JudgeDecision,
    ) -> str:
        parts = []
        if decision.rationale:
            parts.append(f"### Judge Rationale\n{decision.rationale}")
        if decision.feedback:
            parts.append(f"### Improvement Feedback\n{decision.feedback}")
        parts.append(
            "### Winner Diagnostics\n"
            f"- Finite predictions: {winner.stats.finite_count}\n"
            f"- NaN predictions: {winner.stats.nan_count}\n"
            f"- Min: {_format_float(winner.stats.minimum)}\n"
            f"- Max: {_format_float(winner.stats.maximum)}\n"
            f"- Mean: {_format_float(winner.stats.mean)}\n"
            f"- Std: {_format_float(winner.stats.std)}\n"
            f"- Collapsed: {'yes' if winner.stats.collapsed else 'no'}"
        )
        if winner.runtime_error is not None:
            parts.append(
                "### Runtime Error\n"
                "The selected candidate raised the following exception while "
                "evaluating at least one datapoint. Fix the cause in the next "
                "iteration:\n"
                f"```\n{winner.runtime_error}\n```"
            )
        if winner.component_stats:
            component_lines = ["### Component Diagnostics"]
            for key in sorted(winner.component_stats):
                comp = winner.component_stats[key]
                component_lines.append(
                    f"- {key}: count={comp.count}, min={_format_float(comp.minimum)}, "
                    f"max={_format_float(comp.maximum)}, mean={_format_float(comp.mean)}, "
                    f"std={_format_float(comp.std)}, collapsed={'yes' if comp.collapsed else 'no'}"
                )
            parts.append("\n".join(component_lines))
        return "\n\n".join(parts)

    def _prepare_points_for_judge(
        self,
        sampled_points: list[EvaluationPoint],
        *,
        valid_candidates: list[_CandidateResult],
        invalid_candidates: list[_CandidateResult],
    ) -> list[Any]:
        prepared_points = self._point_preparer._prepare_points_for_group(sampled_points)
        if self._estimate_tokens is None or self._max_prompt_tokens is None:
            return prepared_points

        judge_system_content = build_eureka_judge_system_prompt(self.context)
        initial_tokens = self._estimate_judge_prompt_tokens(
            judge_system_content,
            prepared_points=prepared_points,
            valid_candidates=valid_candidates,
            invalid_candidates=invalid_candidates,
        )
        if initial_tokens <= self._max_prompt_tokens:
            return prepared_points

        best_over_budget = (initial_tokens, prepared_points)
        best_fit: list[Any] | None = None
        low = 1
        high = max(1, self._max_prompt_tokens)

        while low <= high:
            point_budget = (low + high) // 2
            candidate_prepared = [
                self._point_preparer._prepare_point(
                    point,
                    system_content=judge_system_content,
                    max_prompt_tokens=point_budget,
                )
                for point in sampled_points
            ]
            candidate_tokens = self._estimate_judge_prompt_tokens(
                judge_system_content,
                prepared_points=candidate_prepared,
                valid_candidates=valid_candidates,
                invalid_candidates=invalid_candidates,
            )
            if candidate_tokens < best_over_budget[0]:
                best_over_budget = (candidate_tokens, candidate_prepared)
            if candidate_tokens <= self._max_prompt_tokens:
                best_fit = candidate_prepared
                low = point_budget + 1
            else:
                high = point_budget - 1

        return best_fit if best_fit is not None else best_over_budget[1]

    def _estimate_judge_prompt_tokens(
        self,
        judge_system_content: str,
        *,
        prepared_points: list[Any],
        valid_candidates: list[_CandidateResult],
        invalid_candidates: list[_CandidateResult],
    ) -> int:
        if self._estimate_tokens is None:
            raise AssertionError("_estimate_judge_prompt_tokens requires _estimate_tokens")
        judge_user_content = build_eureka_judge_user_prompt(
            prepared_points=prepared_points,
            signal_type=self.context.signal_type,
            valid_candidates=valid_candidates,
            invalid_candidates=invalid_candidates,
            include_next_state=self.context.include_next_state,
            include_current_thoughts=self.context.include_current_thoughts,
            include_history_thoughts=self.context.include_history_thoughts,
            max_steps=self.context.max_steps,
            inject_task_text=self._inject_task_text,
        )
        return (
            self._estimate_tokens(judge_system_content)
            + self._estimate_tokens(judge_user_content)
        )

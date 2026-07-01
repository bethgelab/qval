#!/usr/bin/env python3
"""Generate prompt previews for all methods without running experiments.

Creates a directory of prompt files showing exactly what each method
(`llm_direct`, `llm_direct_gvl`, `llm_verifier`, `llm_codegen`, `llm_eureka`,
`llm_ranking`) and the collection step would send to an LLM, using real
environment observations from a short trajectory with fixed actions.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path
from types import SimpleNamespace

from llenvs.core.state import Action

from qval.contexts import load_context, load_environment_context
from qval.methods.llm_direct import LLMDirectMethod
from qval.methods.llm_gvl import LLMGVLMethod
from qval.methods.llm_eureka import (
    CandidateComponentStats,
    CandidateStats,
    _CandidateResult,
    build_eureka_generation_system_prompt,
    build_eureka_generation_user_prompt,
    build_eureka_judge_system_prompt,
    build_eureka_judge_user_prompt,
)
from qval.methods.serialization import (
    action_text_for_display,
    serialize_observation,
)
from qval.ranking_collection import build_ranking_sampling_messages
from qval.prompt_presets import get_preset
from qval.prompts import (
    _task_text_for_injection,
    build_codegen_system_prompt,
    build_codegen_user_prompt,
    build_direct_system_prompt,
    build_direct_user_prompt,
    build_direct_user_prompt_batch,
    build_ranking_system_prompt,
    build_ranking_user_prompt,
    build_verifier_system_prompt,
    build_verifier_user_prompt_batch,
)
from qval.script_utils import (
    create_adapter_env,
    resolve_ranking_system_prompt,
    resolve_system_prompt,
)
from qval.types import (
    ActionCandidate,
    EvaluationPoint,
    HistoryTurn,
    RankingCandidateSource,
    RankingPoint,
    SignalType,
)
from qval.verifier_criteria import get_verifier_criterion


# -- Helpers ----------------------------------------------------------------

_DIRECT_BATCH_PREVIEW_SIZE = 4
_DEFAULT_DIRECT_PROMPT_BATCH_SIZES = (1, _DIRECT_BATCH_PREVIEW_SIZE)
_DEFAULT_DIRECT_PROMPT_BATCH_MODES = ("packed",)
_DEFAULT_DIRECT_NUM_VERIFICATIONS = (1, 3)

def _write(output_dir: Path, *parts: str, content) -> Path:
    """Write content to ``output_dir / parts`` and return the path.

    Accepts ``str``, ``None``, or a ``PromptWithImages`` (in which case the
    text rendition is used).
    """
    path = output_dir.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        text = ""
    elif hasattr(content, "text") and not isinstance(content, str):
        text = content.text
    else:
        text = content
    path.write_text(text)
    return path


def _parse_int_csv(value: str, *, label: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{label} must be a comma-separated list of integers")

    parsed: list[int] = []
    for part in parts:
        try:
            item = int(part)
        except ValueError as exc:
            raise ValueError(
                f"{label} must contain only integers, got {part!r}"
            ) from exc
        if item < 1:
            raise ValueError(f"{label} values must be >= 1, got {item}")
        if item not in parsed:
            parsed.append(item)
    return tuple(parsed)


def _parse_choice_csv(
    value: str,
    *,
    label: str,
    choices: set[str],
) -> tuple[str, ...]:
    parts = [part.strip().lower() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{label} must be a comma-separated list of values")

    parsed: list[str] = []
    for part in parts:
        if part not in choices:
            raise ValueError(
                f"{label} must contain only {sorted(choices)}, got {part!r}"
            )
        if part not in parsed:
            parsed.append(part)
    return tuple(parsed)


def _collect_trajectory(env, seed: int, actions: list[str]):
    """Step through *env* with fixed *actions*, returning per-step dicts."""
    state, _ = env.reset(seed=seed)
    trajectory = []
    for act_text in actions:
        action = Action(text=act_text)
        result = env.step(state, action)
        trajectory.append({
            "state": state,
            "obs_text": serialize_observation(state),
            "action": action,
            "action_text": action_text_for_display(
                action, result.resolved_action, result.extracted_action,
            ),
            "resolved_action": result.resolved_action,
            "extracted_action": result.extracted_action,
            "next_state": result.next_state,
            "next_obs_text": serialize_observation(result.next_state),
        })
        state = result.next_state
    return trajectory


def _history_turns(trajectory, up_to: int) -> list[HistoryTurn]:
    return [
        HistoryTurn(state_text=t["obs_text"], action_text=t["action_text"])
        for t in trajectory[:up_to]
    ]


def _eval_point(trajectory, idx: int, history=None) -> EvaluationPoint:
    t = trajectory[idx]
    return EvaluationPoint(
        state=t["state"],
        action=t["action"],
        next_state=t["next_state"],
        trajectory_index=0,
        step_index=idx,
        extracted_action=t["extracted_action"],
        resolved_action=t["resolved_action"],
        history=tuple(history) if history else (),
    )


def _preview_trajectory_result(trajectory):
    transitions = [
        SimpleNamespace(
            state=turn["state"],
            action=turn["action"],
            next_state=turn["next_state"],
            extracted_action=turn["extracted_action"],
            resolved_action=turn["resolved_action"],
        )
        for turn in trajectory
    ]
    return SimpleNamespace(
        trajectory=SimpleNamespace(transitions=transitions),
        metadata={},
    )


def _ranking_candidates(env, state, action_texts):
    """Build ActionCandidate tuple by stepping from *state* with each action."""
    candidates = []
    for i, act_text in enumerate(action_texts):
        act = Action(text=act_text)
        result = env.step(state, act)
        source = (
            RankingCandidateSource.ACTOR_PRIMARY if i == 0
            else RankingCandidateSource.RANKING_LLM
        )
        candidates.append(ActionCandidate(
            action=act,
            source=source,
            extracted_action=result.extracted_action,
            resolved_action=result.resolved_action,
            next_state=result.next_state,
        ))
    return tuple(candidates)


class _PreviewDirectPreparer:
    def __init__(self, *, context, preset, inject_task_text: bool) -> None:
        self._helper = LLMDirectMethod(
            backend=object(),
            context=context,
            prompt_preset=preset,
            max_history_turns=context.max_history_turns,
            inject_task_text=inject_task_text,
        )

    def prepare(self, points):
        return self._helper._prepare_points_for_group(points)


# -- Generators per category ------------------------------------------------

_ROLE_LABELS = {"system": "=== SYSTEM ===", "user": "=== USER ===",
                "assistant": "=== ASSISTANT ==="}


def _format_messages(msgs) -> str:
    """Render a list of ChatMessages into the preview text format."""
    return "\n\n".join(
        f"{_ROLE_LABELS.get(m.role, m.role)}\n{m.content}" for m in msgs
    )


def _gen_collection(out, actor_prompt, ranking_prompt, trajectory, *,
                    format_reminder: str | None = None,
                    max_steps: int | None = None,
                    inject_task_text: bool = False):
    files = []
    files.append(_write(out, "collection", "actor_system_prompt.txt",
                        content=actor_prompt))

    last = len(trajectory) - 1
    for turn in (0, min(2, last)):
        t = trajectory[turn]
        point = _eval_point(trajectory, turn,
                            _history_turns(trajectory, turn) if turn > 0 else None)
        seen_actions = [t['action_text']]

        msgs = build_ranking_sampling_messages(
            point,
            seen_actions=seen_actions,
            strategy="avoid_seen",
            system_prompt=ranking_prompt,
            format_reminder=format_reminder,
            max_steps=max_steps,
            task_text_to_inject=(
                _task_text_for_injection(point) if inject_task_text else None
            ),
        )

        files.append(_write(out, "collection",
                            f"ranking_collection_turn_{turn}.txt",
                            content=_format_messages(msgs)))
    return files


def _gen_direct(
    out,
    trajectory,
    env_name,
    preset,
    discount,
    disclose,
    *,
    prompt_batch_sizes: tuple[int, ...] = _DEFAULT_DIRECT_PROMPT_BATCH_SIZES,
    prompt_batch_modes: tuple[str, ...] = _DEFAULT_DIRECT_PROMPT_BATCH_MODES,
    num_verification_values: tuple[int, ...] = _DEFAULT_DIRECT_NUM_VERIFICATIONS,
    inject_task_text: bool = False,
):
    files = []
    hist = _history_turns(trajectory, 2)
    pt_first = _eval_point(trajectory, 0)
    pt_third = _eval_point(trajectory, 2, hist)
    pt_third_bare = _eval_point(trajectory, 2)  # no history on point

    def _inject(pt):
        return _task_text_for_injection(pt) if inject_task_text else None

    for st in (SignalType.STATE_VALUE, SignalType.Q_VALUE, SignalType.ADVANTAGE):
        name = st.name.lower()
        ctx = load_context(env_name, st)
        ctx = dataclasses.replace(ctx, discount_factor=discount,
                                  disclose_discount_factor=disclose)
        ctx0 = dataclasses.replace(ctx, max_history_turns=0)

        files.append(_write(out, "direct", name, "system.txt",
                            content=build_direct_system_prompt(ctx, preset)))
        files.append(_write(out, "direct", name, "system_no_history.txt",
                            content=build_direct_system_prompt(ctx0, preset)))

        files.append(_write(out, "direct", name, "turn_0_user.txt",
                            content=build_direct_user_prompt(
                                pt_first, st,
                                include_next_state=ctx.include_next_state,
                                use_answer_tags=preset.uses_answer_tags,
                                max_steps=ctx.max_steps,
                                task_text_to_inject=_inject(pt_first))))

        files.append(_write(out, "direct", name, "turn_2_user.txt",
                            content=build_direct_user_prompt(
                                pt_third, st,
                                history=list(hist),
                                include_next_state=ctx.include_next_state,
                                use_answer_tags=preset.uses_answer_tags,
                                max_steps=ctx.max_steps,
                                task_text_to_inject=_inject(pt_third))))

        files.append(_write(out, "direct", name, "turn_2_no_history_user.txt",
                            content=build_direct_user_prompt(
                                pt_third_bare, st,
                                include_next_state=ctx.include_next_state,
                                use_answer_tags=preset.uses_answer_tags,
                                max_steps=ctx.max_steps,
                                task_text_to_inject=_inject(pt_third_bare))))

        batch_points = []
        batch_histories = []
        batch_count = min(_DIRECT_BATCH_PREVIEW_SIZE, len(trajectory))
        for idx in range(batch_count):
            point_history = _history_turns(trajectory, idx)
            batch_points.append(
                _eval_point(
                    trajectory,
                    idx,
                    point_history if point_history else None,
                )
            )
            batch_histories.append(point_history)

        files.append(_write(
            out,
            "direct",
            name,
            f"batch_{batch_count}_system.txt",
            content=build_direct_system_prompt(
                ctx,
                preset,
                num_points=batch_count,
            ),
        ))
        files.append(_write(
            out,
            "direct",
            name,
            f"batch_{batch_count}_user.txt",
            content=build_direct_user_prompt_batch(
                batch_points,
                st,
                histories=batch_histories,
                include_next_state=ctx.include_next_state,
                use_answer_tags=preset.uses_answer_tags,
                max_steps=ctx.max_steps,
                task_text_to_injects=(
                    [_inject(p) for p in batch_points]
                    if inject_task_text else None
                ),
            ),
        ))

        all_points = []
        for idx in range(len(trajectory)):
            point_history = _history_turns(trajectory, idx)
            all_points.append(
                _eval_point(
                    trajectory,
                    idx,
                    point_history if point_history else None,
                )
            )

        for prompt_batch_mode in prompt_batch_modes:
            for prompt_batch_size in prompt_batch_sizes:
                helper = LLMDirectMethod(
                    backend=object(),
                    context=ctx,
                    prompt_preset=preset,
                    max_history_turns=ctx.max_history_turns,
                    prompt_batch_size=prompt_batch_size,
                    prompt_batch_mode=prompt_batch_mode,
                    inject_task_text=inject_task_text,
                )
                prompt_groups = helper._plan_prompt_groups(all_points)
                prepared_groups = [
                    helper._prepare_points_for_group(group.points)
                    for group in prompt_groups
                ]
                group_sizes = ", ".join(str(len(group.points)) for group in prompt_groups)
                for num_verifications in num_verification_values:
                    variant_parts = (
                        "direct",
                        name,
                        "variants",
                        (
                            f"prompt_batch_mode_{prompt_batch_mode}_"
                            f"prompt_batch_size_{prompt_batch_size}_"
                            f"num_verifications_{num_verifications}"
                        ),
                    )
                    metadata_lines = [
                        "LLMDirect variant preview",
                        f"signal_type: {name}",
                        f"trajectory_points: {len(all_points)}",
                        f"prompt_batch_size: {prompt_batch_size}",
                        f"prompt_batch_mode: {prompt_batch_mode}",
                        f"num_verifications: {num_verifications}",
                        f"prompt_group_count: {len(prompt_groups)}",
                        f"prompt_group_sizes: {group_sizes}",
                        "",
                        "Runtime behavior:",
                        "- prompt_batch_size controls how many evaluation points are grouped together before prediction.",
                        f"- num_verifications repeats the same prompt groups {num_verifications} times and averages predictions per original datapoint.",
                        "- The same prompt groups are reused on each verification pass.",
                    ]
                    if prompt_batch_mode == "sequential":
                        metadata_lines.extend([
                            "- prompt_batch_mode=sequential evaluates each multi-point prompt group as a conversation with one datapoint per user turn.",
                            "- runtime inserts canonicalized assistant replies between successive user turns within the same prompt group.",
                            "- The prompt files in this directory show the system prompt once and the per-turn user prompts reused on each verification pass.",
                        ])
                    else:
                        metadata_lines.extend([
                            "- prompt_batch_mode=packed packs all datapoints in a prompt group into a single user message.",
                            "- The prompt files in this directory are the unique prompt groups reused on each verification pass.",
                        ])
                    files.append(_write(
                        out,
                        *variant_parts,
                        "metadata.txt",
                        content="\n".join(metadata_lines) + "\n",
                    ))
                    for group_idx, prepared_points in enumerate(prepared_groups):
                        if prompt_batch_mode == "sequential" and len(prepared_points) > 1:
                            files.append(_write(
                                out,
                                *variant_parts,
                                f"prompt_{group_idx}_system.txt",
                                content=build_direct_system_prompt(
                                    ctx,
                                    preset,
                                    num_points=len(prepared_points),
                                    batch_mode="sequential",
                                ),
                            ))
                            for turn_idx in range(len(prepared_points)):
                                files.append(_write(
                                    out,
                                    *variant_parts,
                                    f"prompt_{group_idx}_user_turn_{turn_idx}.txt",
                                    content=helper._build_sequential_user_turn_content(
                                        prepared_points,
                                        point_index=turn_idx,
                                    ),
                                ))
                            continue

                        messages = helper._build_messages_for_prepared_points(prepared_points)
                        files.append(_write(
                            out,
                            *variant_parts,
                            f"prompt_{group_idx}_system.txt",
                            content=messages[0].content,
                        ))
                        files.append(_write(
                            out,
                            *variant_parts,
                            f"prompt_{group_idx}_user.txt",
                            content=messages[1].content,
                        ))
    return files


def _gen_direct_gvl(
    out,
    trajectory,
    env_name,
    preset,
    discount,
    disclose,
    *,
    inject_task_text: bool = False,
):
    files = []
    if not trajectory:
        return files

    preview_indices = sorted({0, min(2, len(trajectory) - 1)})
    trajectory_result = _preview_trajectory_result(trajectory)

    for st in (SignalType.STATE_VALUE, SignalType.Q_VALUE, SignalType.ADVANTAGE):
        name = st.name.lower()
        ctx = load_context(env_name, st)
        ctx = dataclasses.replace(
            ctx,
            discount_factor=discount,
            disclose_discount_factor=disclose,
            max_history_turns=0,
            include_current_thoughts=False,
            include_history_thoughts=False,
        )
        helper = LLMGVLMethod(
            backend=object(),
            context=ctx,
            trajectory_results=[trajectory_result],
            prompt_preset=preset,
            inject_task_text=inject_task_text,
        )

        files.append(_write(
            out,
            "direct_gvl",
            name,
            "system.txt",
            content=helper._system_content,
        ))

        for step_index in preview_indices:
            point = _eval_point(trajectory, step_index)
            prepared = helper._prepare_point(point)
            messages = helper._build_messages(prepared)
            files.append(_write(
                out,
                "direct_gvl",
                name,
                f"target_{step_index}_user.txt",
                content=messages[1].content,
            ))
    return files


def _gen_verifier(
    out,
    trajectory,
    env_name,
    preset,
    discount,
    disclose,
    *,
    inject_task_text: bool = False,
):
    files = []
    batch_count = min(_DIRECT_BATCH_PREVIEW_SIZE, len(trajectory))
    criteria = [
        None,
        get_verifier_criterion("terminal_bench_correctness"),
    ]

    for st in (SignalType.STATE_VALUE, SignalType.Q_VALUE):
        name = st.name.lower()
        ctx = load_context(env_name, st)
        ctx = dataclasses.replace(
            ctx,
            discount_factor=discount,
            disclose_discount_factor=disclose,
        )
        batch_points = []
        batch_histories = []
        for idx in range(batch_count):
            point_history = _history_turns(trajectory, idx)
            batch_points.append(
                _eval_point(
                    trajectory,
                    idx,
                    point_history if point_history else None,
                )
            )
            batch_histories.append(point_history)

        files.append(_write(
            out,
            "verifier",
            name,
            "system.txt",
            content=build_verifier_system_prompt(
                ctx,
                preset,
                num_points=1,
            ),
        ))
        for criterion in criteria:
            suffix = (
                "no_criteria"
                if criterion is None
                else f"criterion_{criterion.id}"
            )
            files.append(_write(
                out,
                "verifier",
                name,
                f"batch_{batch_count}_{suffix}_user.txt",
                content=build_verifier_user_prompt_batch(
                    batch_points,
                    st,
                    histories=batch_histories,
                    include_next_state=ctx.include_next_state,
                    criterion_name=criterion.name if criterion is not None else None,
                    criterion_description=(
                        criterion.description_for_signal(st)
                        if criterion is not None
                        else None
                    ),
                    max_steps=ctx.max_steps,
                    task_text_to_injects=(
                        [_task_text_for_injection(p) for p in batch_points]
                        if inject_task_text else None
                    ),
                ),
            ))
    return files


def _gen_codegen(out, env_name, preset, discount, disclose):
    files = []
    for st in (SignalType.STATE_VALUE, SignalType.Q_VALUE, SignalType.ADVANTAGE):
        name = st.name.lower()
        ctx = load_context(env_name, st)
        ctx = dataclasses.replace(ctx, discount_factor=discount,
                                  disclose_discount_factor=disclose)

        files.append(_write(out, "codegen", name, "system.txt",
                            content=build_codegen_system_prompt(ctx, preset)))
        files.append(_write(out, "codegen", name, "user.txt",
                            content=build_codegen_user_prompt(
                                st, preset,
                                include_next_state=ctx.include_next_state,
                                enable_thinking=ctx.enable_thinking)))
    return files


def _gen_eureka(out, trajectory, env_name, preset, discount, disclose, *, inject_task_text: bool = False):
    files = []
    batch_count = min(_DIRECT_BATCH_PREVIEW_SIZE, len(trajectory))
    for st in (SignalType.STATE_VALUE, SignalType.Q_VALUE, SignalType.ADVANTAGE):
        name = st.name.lower()
        ctx = load_context(env_name, st)
        ctx = dataclasses.replace(ctx, discount_factor=discount,
                                  disclose_discount_factor=disclose)
        batch_points = []
        batch_histories = []
        for idx in range(batch_count):
            point_history = _history_turns(trajectory, idx)
            batch_points.append(
                _eval_point(
                    trajectory,
                    idx,
                    point_history if point_history else None,
                )
            )
            batch_histories.append(point_history)

        helper = _PreviewDirectPreparer(
            context=ctx,
            preset=preset,
            inject_task_text=inject_task_text,
        )
        prepared_points = helper.prepare(batch_points)
        valid_candidates = [
            _CandidateResult(
                code="def signal_function(...): return 0.1",
                signal_fn=lambda *args: 0.1,
                predictions=[0.1] * batch_count,
                stats=CandidateStats(
                    finite_count=batch_count,
                    nan_count=0,
                    minimum=0.1,
                    maximum=0.1,
                    mean=0.1,
                    std=0.0,
                    collapsed=True,
                ),
                component_stats={},
            ),
            _CandidateResult(
                code="def signal_function(...): return 0.8",
                signal_fn=lambda *args: 0.8,
                predictions=[0.8] * batch_count,
                stats=CandidateStats(
                    finite_count=batch_count,
                    nan_count=0,
                    minimum=0.8,
                    maximum=0.8,
                    mean=0.8,
                    std=0.0,
                    collapsed=True,
                ),
                component_stats={
                    "progress_reward": CandidateComponentStats(
                        count=batch_count,
                        minimum=0.8,
                        maximum=0.8,
                        mean=0.8,
                        std=0.0,
                        collapsed=True,
                    ),
                },
            ),
        ]
        for judge_label, candidate in enumerate(valid_candidates, start=1):
            candidate.judge_label = judge_label

        files.append(_write(
            out, "eureka", name, "generation_initial_system.txt",
            content=build_eureka_generation_system_prompt(ctx, preset),
        ))
        files.append(_write(
            out, "eureka", name, "generation_initial_user.txt",
            content=build_eureka_generation_user_prompt(
                st, preset,
                include_next_state=ctx.include_next_state,
                enable_thinking=ctx.enable_thinking,
                iteration=1,
                total_iterations=3,
            ),
        ))
        files.append(_write(
            out, "eureka", name, "judge_system.txt",
            content=build_eureka_judge_system_prompt(ctx),
        ))
        files.append(_write(
            out, "eureka", name, "judge_user.txt",
            content=build_eureka_judge_user_prompt(
                prepared_points=prepared_points,
                signal_type=st,
                valid_candidates=valid_candidates,
                invalid_candidates=[],
                include_next_state=ctx.include_next_state,
                include_current_thoughts=ctx.include_current_thoughts,
                include_history_thoughts=ctx.include_history_thoughts,
                max_steps=ctx.max_steps,
                inject_task_text=inject_task_text,
            ),
        ))
        files.append(_write(
            out, "eureka", name, "generation_iterative_user.txt",
            content=build_eureka_generation_user_prompt(
                st, preset,
                include_next_state=ctx.include_next_state,
                enable_thinking=ctx.enable_thinking,
                iteration=2,
                total_iterations=3,
                previous_winner_code=(
                    "def signal_function(state: str, action: str, next_state: str):\n"
                    "    progress_reward = 0.8\n"
                    "    return progress_reward, {\"progress_reward\": progress_reward}"
                ),
                reflection=(
                    "### Judge Rationale\nCandidate 2 separated promising states better.\n\n"
                    "### Improvement Feedback\nKeep the stronger scale but avoid flat predictions."
                ),
            ),
        ))
    return files


def _gen_ranking(out, env, trajectory, env_name, preset, discount, disclose,
                 *, inject_task_text: bool = False):
    files = []
    st = SignalType.Q_VALUE
    ctx = load_context(env_name, st)
    ctx = dataclasses.replace(ctx, discount_factor=discount,
                              disclose_discount_factor=disclose)
    ctx0 = dataclasses.replace(ctx, max_history_turns=0)

    files.append(_write(out, "ranking", "system.txt",
                        content=build_ranking_system_prompt(ctx, preset)))
    files.append(_write(out, "ranking", "system_no_history.txt",
                        content=build_ranking_system_prompt(ctx0, preset)))

    def _rinject(pt):
        return _task_text_for_injection(pt) if inject_task_text else None

    hist = _history_turns(trajectory, 2)

    def _candidates_for_turn(turn_idx: int) -> list[str]:
        # Primary = action actually taken at turn_idx (ACTOR_PRIMARY);
        # alternatives = actions taken at the other turns (RANKING_LLM).
        # Keeps the preview env-agnostic at the cost of some state-dependent
        # candidates resolving to invalid-action fallbacks (e.g. webshop
        # click-only actions at pre-search turns).
        primary = trajectory[turn_idx]["action_text"]
        others = [
            t["action_text"]
            for i, t in enumerate(trajectory)
            if i != turn_idx
        ]
        return [primary] + others

    # Turn 0
    cands0 = _ranking_candidates(env, trajectory[0]["state"], _candidates_for_turn(0))
    rp0 = RankingPoint(state=trajectory[0]["state"], candidates=cands0,
                       trajectory_index=0, step_index=0)
    files.append(_write(out, "ranking", "turn_0_user.txt",
                        content=build_ranking_user_prompt(
                            rp0,
                            include_next_state=ctx.include_next_state,
                            use_answer_tags=preset.uses_answer_tags,
                            max_steps=ctx.max_steps,
                            task_text_to_inject=_rinject(rp0))))

    # Turn 2 with history
    cands2 = _ranking_candidates(env, trajectory[2]["state"], _candidates_for_turn(2))
    rp2 = RankingPoint(state=trajectory[2]["state"], candidates=cands2,
                       trajectory_index=0, step_index=2,
                       history=tuple(hist))
    files.append(_write(out, "ranking", "turn_2_user.txt",
                        content=build_ranking_user_prompt(
                            rp2,
                            history=list(hist),
                            include_next_state=ctx.include_next_state,
                            use_answer_tags=preset.uses_answer_tags,
                            max_steps=ctx.max_steps,
                            task_text_to_inject=_rinject(rp2))))

    # Turn 2 without history
    rp2_bare = RankingPoint(state=trajectory[2]["state"], candidates=cands2,
                            trajectory_index=0, step_index=2)
    files.append(_write(out, "ranking", "turn_2_no_history_user.txt",
                        content=build_ranking_user_prompt(
                            rp2_bare,
                            include_next_state=ctx.include_next_state,
                            use_answer_tags=preset.uses_answer_tags,
                            max_steps=ctx.max_steps,
                            task_text_to_inject=_rinject(rp2_bare))))
    return files


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate prompt previews for all methods.",
    )
    parser.add_argument("--env-name",
                        default="craftax-classic_50ms_react_tags",
                        help="Environment config name")
    parser.add_argument("--output-dir", default="prompt_previews",
                        help="Output directory")
    parser.add_argument("--preset", default="v2",
                        help="Prompt preset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", default="left,right,interact",
                        help="Comma-separated actions to step with")
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--disclose-discount", action="store_true",
                        help="Disclose discount factor to methods (off by default)")
    parser.add_argument(
        "--direct-prompt-batch-sizes",
        default=",".join(str(v) for v in _DEFAULT_DIRECT_PROMPT_BATCH_SIZES),
        help=(
            "Comma-separated LLMDirect prompt_batch_size variants to preview. "
            "Each value generates a separate variant directory."
        ),
    )
    parser.add_argument(
        "--direct-prompt-batch-modes",
        default=",".join(_DEFAULT_DIRECT_PROMPT_BATCH_MODES),
        help=(
            "Comma-separated LLMDirect prompt_batch_mode variants to preview. "
            "Supported values: packed, sequential."
        ),
    )
    parser.add_argument(
        "--direct-num-verifications",
        default=",".join(str(v) for v in _DEFAULT_DIRECT_NUM_VERIFICATIONS),
        help=(
            "Comma-separated LLMDirect num_verifications variants to preview. "
            "Each value generates a separate variant directory."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    actions = [a.strip() for a in args.actions.split(",")]
    preset = get_preset(args.preset)
    disclose = args.disclose_discount
    try:
        direct_prompt_batch_sizes = _parse_int_csv(
            args.direct_prompt_batch_sizes,
            label="--direct-prompt-batch-sizes",
        )
        direct_prompt_batch_modes = _parse_choice_csv(
            args.direct_prompt_batch_modes,
            label="--direct-prompt-batch-modes",
            choices={"packed", "sequential"},
        )
        direct_num_verifications = _parse_int_csv(
            args.direct_num_verifications,
            label="--direct-num-verifications",
        )
    except ValueError as exc:
        parser.error(str(exc))

    min_actions = 3
    if len(actions) < min_actions:
        parser.error(
            f"Need at least {min_actions} actions for the preview scenarios",
        )

    # Load env context and create environment
    env_ctx = load_environment_context(args.env_name)
    env, adapter_system_prompt = create_adapter_env(
        adapter=env_ctx.adapter,
        env_name=env_ctx.env_name,
        env_size=1,
        seed=args.seed,
        max_steps=env_ctx.max_steps,
        answer_extractor=env_ctx.environment_extractor,
        make_kwargs=env_ctx.make_kwargs,
        invalid_action_text=getattr(env_ctx, "invalid_action_text", None),
        invalid_action_observation=getattr(env_ctx, "invalid_action_observation", None),
        advance_on_invalid=getattr(env_ctx, "advance_on_invalid", None),
    )
    actor_system_prompt = resolve_system_prompt(
        override_system_prompt=None,
        adapter=env_ctx.adapter,
        prompting_scheme=env_ctx.prompting_scheme,
        adapter_system_prompt=adapter_system_prompt,
        make_kwargs=env_ctx.make_kwargs,
        env_name=env_ctx.env_name,
    )

    # Collect trajectory
    print(f"Stepping through {env_ctx.env_name} with actions: {actions}")
    trajectory = _collect_trajectory(env, args.seed, actions)
    print(f"Collected {len(trajectory)} steps")

    # Prepare output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    all_files: list[Path] = []

    from qval.system_prompts import RANKING_SCHEME_INSTRUCTIONS

    ranking_scheme = env_ctx.prompting_scheme or "answer_tags"
    ranking_format_reminder = RANKING_SCHEME_INSTRUCTIONS.get(ranking_scheme)
    ranking_system_prompt = resolve_ranking_system_prompt(
        override_system_prompt=None,
        adapter=env_ctx.adapter,
        prompting_scheme=env_ctx.prompting_scheme,
        adapter_system_prompt=adapter_system_prompt,
        make_kwargs=env_ctx.make_kwargs,
        env_name=env_ctx.env_name,
    )

    inject_task_text = env_ctx.inject_task_text_in_prompts

    print("Generating collection prompts...")
    all_files.extend(_gen_collection(output_dir, actor_system_prompt,
                                     ranking_system_prompt, trajectory,
                                     format_reminder=ranking_format_reminder,
                                     max_steps=env_ctx.max_steps,
                                     inject_task_text=inject_task_text))

    print("Generating direct method prompts...")
    all_files.extend(_gen_direct(output_dir, trajectory, args.env_name, preset,
                                 args.discount_factor, disclose,
                                 prompt_batch_sizes=direct_prompt_batch_sizes,
                                 prompt_batch_modes=direct_prompt_batch_modes,
                                 num_verification_values=direct_num_verifications,
                                 inject_task_text=inject_task_text))

    print("Generating llm_direct_gvl prompts...")
    all_files.extend(_gen_direct_gvl(output_dir, trajectory, args.env_name, preset,
                                     args.discount_factor, disclose,
                                     inject_task_text=inject_task_text))

    print("Generating verifier method prompts...")
    all_files.extend(_gen_verifier(output_dir, trajectory, args.env_name, preset,
                                   args.discount_factor, disclose,
                                   inject_task_text=inject_task_text))

    print("Generating codegen method prompts...")
    all_files.extend(_gen_codegen(output_dir, args.env_name, preset,
                                  args.discount_factor, disclose))

    print("Generating Eureka method prompts...")
    all_files.extend(_gen_eureka(output_dir, trajectory, args.env_name, preset,
                                 args.discount_factor, disclose,
                                 inject_task_text=inject_task_text))

    print("Generating ranking method prompts...")
    all_files.extend(_gen_ranking(output_dir, env, trajectory, args.env_name,
                                  preset, args.discount_factor, disclose,
                                  inject_task_text=inject_task_text))

    # Summary
    print(f"\n{len(all_files)} prompt files written to {output_dir}/\n")
    for f in sorted(all_files):
        rel = f.relative_to(output_dir)
        size = f.stat().st_size
        print(f"  {rel}  ({size:,} bytes)")


if __name__ == "__main__":
    main()

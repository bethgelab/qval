"""Shared construction for benchmark eval methods.

This module is the single source of truth for turning an
:class:`~qval.config.EvalMethodConfig` into a constructed method
instance. Both the prediction pipeline (``scripts/pipeline/predict.py``) and the
test-time scaling experiment (``scripts/pipeline/test_time_scaling.py``) build
methods through here, so a method behaves identically wherever it is used.

The construction logic is a verbatim port of what previously lived inline in
``predict.py``; its behavior must remain bit-for-bit identical to that pipeline.

Two families of methods are produced:

* **Individual-prediction** methods (``DenseSignalMethod`` subclasses) via
  :func:`build_regular_method` — they expose ``evaluate_batch(points)``.
* **Ranking-based** methods via :func:`build_ranking_method` — they expose
  ``rank_batch``/``score_batch`` over ``RankingPoint`` inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from qval.config import EvalMethodConfig
from qval.methods.baseline_random import RandomBaselineMethod
from qval.methods.delta_belief_ranking import DeltaBeliefRankingMethod
from qval.methods.liv import build_liv_method, get_liv_encoder
from qval.methods.llm_codegen import LLMCodeGenMethod
from qval.methods.llm_direct import LLMDirectMethod
from qval.methods.llm_eureka import LLMEurekaMethod
from qval.methods.llm_gvl import LLMGVLMethod
from qval.methods.llm_ranking import LLMRankingMethod
from qval.methods.llm_verifier import LLMVerifierMethod
from qval.methods.sdpo_ranking import (
    SDPORankingMethod,
    SDPOShortExpertContinuationRankingMethod,
)
from qval.methods.vip import build_vip_method, get_vip_encoder
from qval.methods.vle import build_vle_method, get_vle_encoder
from qval.prompt_presets import PromptPreset, get_preset, resolve_block
from qval.script_utils import resolve_backend_name, sampling_params_from_config
from qval.types import MethodContext

# Method types that operate on RankingPoints (rank_batch / score_batch).
RANKING_METHOD_TYPES = frozenset(
    {
        "llm_ranking",
        "sdpo_ranking",
        "sdpo_short_expert_continuation",
        "delta_belief_ranking",
    }
)

# Vision methods that consume image observations / goal images.
VISION_METHOD_TYPES = frozenset({"vlm_rm", "vlm_sor", "vip", "liv_img", "liv_txt"})


@dataclasses.dataclass(frozen=True)
class MethodBuildContext:
    """Run-level dependencies shared by every per-method construction.

    Carries the data a method needs plus ``get_backend`` (a callable, so the
    caller's per-run backend cache and CLI overrides are reused rather than
    duplicated here).

    Attributes:
        context: Base environment :class:`MethodContext` to specialize per method.
        backends_config: ``BackendsRegistryConfig`` (``.backends``, ``.default_backend``).
        get_backend: Callable mapping a backend config to a (cached) backend.
        eval_backend_name: Fallback backend name when ``mc.backend_name`` is unset.
        default_max_history_turns: Default ``max_history_turns`` when ``mc`` omits it.
        step_penalty: Collection/prediction step penalty (for disclosure prompts).
        discount_factor: Discount factor passed through to constructed methods.
        batch_size: Generation batch size.
        env_context: Environment context (for ``inject_task_text_in_prompts``).
        dataset: ``Dataset`` (needed by GVL trajectory_results and vision goal
            images). ``None`` when no dataset is available (e.g. test-time scaling).
    """

    context: MethodContext
    backends_config: Any
    get_backend: Callable[[Any], Any]
    eval_backend_name: str | None
    default_max_history_turns: int | None
    step_penalty: float | None
    discount_factor: float
    batch_size: int
    env_context: Any
    dataset: Any = None


def _build_method_context(
    base_context: MethodContext,
    mc: EvalMethodConfig,
    step_penalty: float | None,
    discount_factor: float,
    preset: PromptPreset,
    enable_thinking: bool = True,
    max_history_turns: int | None = None,
) -> MethodContext:
    """Build per-method context with disclosure and efficiency guidance."""
    ctx = dataclasses.replace(
        base_context,
        signal_type=mc.signal_type,
        policy_assumption=mc.policy_assumption,
        include_next_state=mc.include_next_state,
        include_current_thoughts=mc.include_current_thoughts,
        include_history_thoughts=mc.include_history_thoughts,
        include_state_text_when_images=(
            mc.include_state_text_when_images
            if mc.include_state_text_when_images is not None
            else base_context.include_state_text_when_images
        ),
        include_images=(
            mc.include_images
            if mc.include_images is not None
            else base_context.include_images
        ),
        enable_thinking=enable_thinking,
        max_history_turns=max_history_turns,
    )
    # Step penalty disclosure: append to reward description
    if mc.disclose_step_penalty and step_penalty is not None and ctx.reward_description:
        ctx = dataclasses.replace(
            ctx,
            reward_description=(
                ctx.reward_description
                + f"\nAdditionally, a penalty of -{step_penalty} is applied "
                "at every step to encourage shorter solutions."
            ),
        )
    # Discount factor disclosure: control prompt language (no longer override value)
    if not mc.disclose_discount_factor:
        ctx = dataclasses.replace(ctx, disclose_discount_factor=False)
    # Efficiency guidance: vague hint when settings are active but not disclosed
    needs_hint = (
        step_penalty is not None and step_penalty > 0 and not mc.disclose_step_penalty
    ) or (discount_factor < 1.0 and not mc.disclose_discount_factor)
    if needs_hint:
        efficiency_block = resolve_block(preset.efficiency_guidance)
        if efficiency_block is not None:
            ctx = dataclasses.replace(ctx, efficiency_guidance=efficiency_block)
    return ctx


def _resolve_truncation_param(
    method_config: Any,
    field: str,
    context_value: int | None,
) -> int | None:
    """Method-level override wins; falls back to environment context default."""
    method_value = getattr(method_config, field, None)
    return method_value if method_value is not None else context_value


def _nullify_if_truncation_disabled(
    method_config: Any,
    r_min_action: int | None,
    r_min_obs: int | None,
    r_min_cur_obs: int | None,
    r_min_next_obs: int | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """When ``disable_prompt_truncation`` is set, null all resolved ``min_*_chars``
    so the caller's truncation gate never activates — required e.g. for
    ``llm_direct`` with ``prompt_batch_mode='sequential'``."""
    if getattr(method_config, "disable_prompt_truncation", False):
        return None, None, None, None
    return r_min_action, r_min_obs, r_min_cur_obs, r_min_next_obs


def build_regular_method(
    mc: EvalMethodConfig,
    deps: MethodBuildContext,
) -> tuple[Any, str | None]:
    """Construct a non-ranking (individual-prediction) eval method.

    Returns ``(method, backend_name)``. ``backend_name`` is ``None`` for
    ``baseline_random`` (which needs no backend).
    """
    context = deps.context
    backends_config = deps.backends_config

    if mc.type == "baseline_random":
        baseline_context = MethodContext(
            signal_type=mc.signal_type,
            policy_assumption=mc.policy_assumption,
            discount_factor=deps.discount_factor,
            include_next_state=mc.include_next_state,
        )
        return RandomBaselineMethod(context=baseline_context, seed=mc.seed), None
    if mc.type in {"vlm_rm", "vlm_sor"}:
        vle_backend_name = resolve_backend_name(
            mc.backend_name,
            backends_config.backends,
            backends_config.default_backend,
        )
        vle_backend_cfg = backends_config.backends[vle_backend_name]
        vle_context = dataclasses.replace(
            context,
            signal_type=mc.signal_type,
            policy_assumption=mc.policy_assumption,
            include_next_state=mc.include_next_state,
        )
        encoder = get_vle_encoder(vle_backend_cfg)
        method = build_vle_method(
            encoder=encoder,
            context=vle_context,
            method_type=mc.type,
            baseline_prompt=mc.vle_baseline_prompt,
            alpha=mc.vle_alpha,
            negative_goals=mc.vle_negative_goals,
            temperature=mc.vle_temperature,
            beta=mc.vle_beta,
            image_source=mc.vle_image_source,
            multi_image_strategy=mc.vle_multi_image_strategy,
            goal_text=mc.vle_goal_text,
            goal_per_point=mc.vle_goal_per_point,
        )
        return method, vle_backend_name
    if mc.type == "vip":
        vip_backend_name = resolve_backend_name(
            mc.backend_name,
            backends_config.backends,
            backends_config.default_backend,
        )
        vip_backend_cfg = backends_config.backends[vip_backend_name]
        vip_context = dataclasses.replace(
            context,
            signal_type=mc.signal_type,
            policy_assumption=mc.policy_assumption,
            include_next_state=mc.include_next_state,
        )
        vip_encoder = get_vip_encoder(vip_backend_cfg)
        method = build_vip_method(
            encoder=vip_encoder,
            context=vip_context,
            goal_source=mc.vip_goal_source,
            env_goal_image_path=mc.vip_env_goal_image_path,
            multi_image_strategy=mc.vip_multi_image_strategy,
            dataset=deps.dataset,
        )
        return method, vip_backend_name
    if mc.type in {"liv_img", "liv_txt"}:
        liv_backend_name = resolve_backend_name(
            mc.backend_name,
            backends_config.backends,
            backends_config.default_backend,
        )
        liv_backend_cfg = backends_config.backends[liv_backend_name]
        liv_context = dataclasses.replace(
            context,
            signal_type=mc.signal_type,
            policy_assumption=mc.policy_assumption,
            include_next_state=mc.include_next_state,
        )
        liv_encoder = get_liv_encoder(liv_backend_cfg)
        method = build_liv_method(
            encoder=liv_encoder,
            context=liv_context,
            method_type=mc.type,
            similarity=mc.liv_similarity,
            goal_source=mc.liv_goal_source,
            env_goal_image_path=mc.liv_env_goal_image_path,
            goal_text=mc.liv_goal_text,
            goal_per_point=mc.liv_goal_per_point,
            multi_image_strategy=mc.liv_multi_image_strategy,
            dataset=deps.dataset,
        )
        return method, liv_backend_name

    preset = get_preset(mc.prompt_preset)
    method_backend_name = resolve_backend_name(
        mc.backend_name or deps.eval_backend_name,
        backends_config.backends,
        backends_config.default_backend,
    )
    method_backend_config = backends_config.backends[method_backend_name]
    if mc.type == "llm_direct_gvl":
        r_max_turns = 0 if mc.max_history_turns is None else mc.max_history_turns
    else:
        r_max_turns = (
            mc.max_history_turns
            if mc.max_history_turns is not None
            else deps.default_max_history_turns
        )
    method_context = _build_method_context(
        context,
        mc,
        deps.step_penalty,
        deps.discount_factor,
        preset,
        enable_thinking=method_backend_config.enable_thinking,
        max_history_turns=r_max_turns,
    )
    method_backend = deps.get_backend(method_backend_config)
    method_sampling_params = sampling_params_from_config(
        method_backend_config.sampling,
        backend_type=method_backend_config.type,
        provider_preferences=method_backend_config.openrouter_provider,
    )
    r_min_action = _resolve_truncation_param(
        mc,
        "min_action_chars",
        context.min_action_chars,
    )
    r_min_obs = _resolve_truncation_param(
        mc,
        "min_observation_chars",
        context.min_observation_chars,
    )
    r_min_cur_obs = _resolve_truncation_param(
        mc,
        "min_current_observation_chars",
        context.min_current_observation_chars,
    )
    r_min_next_obs = _resolve_truncation_param(
        mc,
        "min_next_observation_chars",
        context.min_next_observation_chars,
    )
    r_min_action, r_min_obs, r_min_cur_obs, r_min_next_obs = (
        _nullify_if_truncation_disabled(
            mc,
            r_min_action,
            r_min_obs,
            r_min_cur_obs,
            r_min_next_obs,
        )
    )
    if mc.type == "llm_direct":
        # Resolve budget params for this eval backend
        eval_estimate_tokens = None
        eval_max_prompt_tokens = None
        if any(
            v is not None
            for v in (
                r_min_action,
                r_min_obs,
                r_min_cur_obs,
                r_min_next_obs,
            )
        ):
            from qval.token_budget import (
                make_token_estimator,
                resolve_max_prompt_tokens,
            )

            eval_estimate_tokens = make_token_estimator(method_backend)
            eval_max_prompt_tokens = resolve_max_prompt_tokens(
                method_backend,
                method_backend_config,
            )

        method = LLMDirectMethod(
            backend=method_backend,
            context=method_context,
            sampling_params=method_sampling_params,
            max_history_turns=r_max_turns,
            prompt_preset=preset,
            batch_size=deps.batch_size,
            prompt_batch_size=(
                None if mc.type == "llm_direct_gvl" else mc.prompt_batch_size
            ),
            prompt_batch_mode=(
                mc.prompt_batch_mode if mc.type == "llm_direct" else "packed"
            ),
            num_verifications=mc.num_verifications,
            prompt_grouping=(
                "trajectory" if mc.type == "llm_direct_gvl" else "contiguous"
            ),
            prompt_shuffle=(
                "deterministic_random" if mc.type == "llm_direct_gvl" else "none"
            ),
            joint_prompting=(mc.type == "llm_direct_gvl"),
            show_turn_indices=(mc.type != "llm_direct_gvl"),
            estimate_tokens=eval_estimate_tokens,
            max_prompt_tokens=eval_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_action_chars=r_min_action,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
            inject_task_text=deps.env_context.inject_task_text_in_prompts,
        )
        return method, method_backend_name
    elif mc.type == "llm_direct_gvl":
        eval_estimate_tokens = None
        eval_max_prompt_tokens = None
        if any(
            v is not None
            for v in (
                r_min_action,
                r_min_obs,
                r_min_cur_obs,
                r_min_next_obs,
            )
        ):
            from qval.token_budget import (
                make_token_estimator,
                resolve_max_prompt_tokens,
            )

            eval_estimate_tokens = make_token_estimator(method_backend)
            eval_max_prompt_tokens = resolve_max_prompt_tokens(
                method_backend,
                method_backend_config,
            )

        method = LLMGVLMethod(
            backend=method_backend,
            context=method_context,
            trajectory_results=deps.dataset.trajectory_results,
            sampling_params=method_sampling_params,
            prompt_preset=preset,
            batch_size=deps.batch_size,
            num_verifications=mc.num_verifications,
            estimate_tokens=eval_estimate_tokens,
            max_prompt_tokens=eval_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_action_chars=r_min_action,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
            inject_task_text=deps.env_context.inject_task_text_in_prompts,
        )
        return method, method_backend_name
    elif mc.type == "llm_verifier":
        eval_estimate_tokens = None
        eval_max_prompt_tokens = None
        if not getattr(mc, "disable_prompt_truncation", False):
            from qval.token_budget import (
                make_token_estimator,
                resolve_max_prompt_tokens,
            )

            eval_estimate_tokens = make_token_estimator(method_backend)
            eval_max_prompt_tokens = resolve_max_prompt_tokens(
                method_backend,
                method_backend_config,
            )

        method = LLMVerifierMethod(
            backend=method_backend,
            context=method_context,
            sampling_params=method_sampling_params,
            max_history_turns=r_max_turns,
            prompt_preset=preset,
            batch_size=deps.batch_size,
            prompt_batch_size=mc.prompt_batch_size,
            prompt_grouping=mc.prompt_grouping or "random",
            criteria=mc.criteria,
            num_verifications=mc.num_verifications,
            estimate_tokens=eval_estimate_tokens,
            max_prompt_tokens=eval_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_action_chars=r_min_action,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
            inject_task_text=deps.env_context.inject_task_text_in_prompts,
        )
        return method, method_backend_name
    elif mc.type == "llm_codegen":
        method = LLMCodeGenMethod(
            backend=method_backend,
            context=method_context,
            sampling_params=method_sampling_params,
            prompt_preset=preset,
        )
        return method, method_backend_name
    elif mc.type == "llm_eureka":
        eval_estimate_tokens = None
        eval_max_prompt_tokens = None
        if not getattr(mc, "disable_prompt_truncation", False):
            from qval.token_budget import (
                make_token_estimator,
                resolve_max_prompt_tokens,
            )

            eval_estimate_tokens = make_token_estimator(method_backend)
            eval_max_prompt_tokens = resolve_max_prompt_tokens(
                method_backend,
                method_backend_config,
            )

        method = LLMEurekaMethod(
            backend=method_backend,
            context=method_context,
            sampling_params=method_sampling_params,
            prompt_preset=preset,
            num_candidates=mc.num_samples,
            search_iterations=mc.search_iterations,
            judge_num_points=mc.judge_num_points,
            estimate_tokens=eval_estimate_tokens,
            max_prompt_tokens=eval_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_action_chars=r_min_action,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
            inject_task_text=deps.env_context.inject_task_text_in_prompts,
        )
        return method, method_backend_name

    raise ValueError(f"Unsupported eval method type: {mc.type!r}")


def build_ranking_method(
    mc: EvalMethodConfig,
    deps: MethodBuildContext,
) -> tuple[Any, str | None]:
    """Construct a ranking-based eval method.

    Returns ``(method, backend_name)``. The returned method exposes
    ``rank_batch`` (``llm_ranking``) or ``score_batch`` (the SDPO / delta-belief
    families) over ``RankingPoint`` inputs.
    """
    context = deps.context
    backends_config = deps.backends_config

    preset = get_preset(mc.prompt_preset)
    method_backend_name = resolve_backend_name(
        mc.backend_name or deps.eval_backend_name,
        backends_config.backends,
        backends_config.default_backend,
    )
    method_backend_config = backends_config.backends[method_backend_name]
    r_max_turns = (
        mc.max_history_turns
        if mc.max_history_turns is not None
        else deps.default_max_history_turns
    )
    method_context = _build_method_context(
        context,
        mc,
        deps.step_penalty,
        deps.discount_factor,
        preset,
        enable_thinking=method_backend_config.enable_thinking,
        max_history_turns=r_max_turns,
    )
    method_backend = deps.get_backend(method_backend_config)
    method_sampling_params = sampling_params_from_config(
        method_backend_config.sampling,
        backend_type=method_backend_config.type,
        provider_preferences=method_backend_config.openrouter_provider,
    )

    # Resolve per-method truncation overrides for ranking
    r_min_action = _resolve_truncation_param(
        mc, "min_action_chars", context.min_action_chars
    )
    r_min_obs = _resolve_truncation_param(
        mc, "min_observation_chars", context.min_observation_chars
    )
    r_min_cur_obs = _resolve_truncation_param(
        mc,
        "min_current_observation_chars",
        context.min_current_observation_chars,
    )
    r_min_next_obs = _resolve_truncation_param(
        mc, "min_next_observation_chars", context.min_next_observation_chars
    )
    r_min_action, r_min_obs, r_min_cur_obs, r_min_next_obs = (
        _nullify_if_truncation_disabled(
            mc,
            r_min_action,
            r_min_obs,
            r_min_cur_obs,
            r_min_next_obs,
        )
    )
    rank_ctx_min_cand_action = (
        None
        if getattr(mc, "disable_prompt_truncation", False)
        else context.min_ranking_candidate_action_chars
    )
    rank_ctx_min_next_obs = (
        None
        if getattr(mc, "disable_prompt_truncation", False)
        else context.min_ranking_next_observation_chars
    )

    # Resolve budget params for ranking eval backend
    rank_estimate_tokens = None
    rank_max_prompt_tokens = None
    if any(
        v is not None
        for v in (
            r_min_action,
            r_min_obs,
            r_min_cur_obs,
            r_min_next_obs,
            rank_ctx_min_cand_action,
            rank_ctx_min_next_obs,
        )
    ):
        from qval.token_budget import (
            make_token_estimator,
            resolve_max_prompt_tokens,
        )

        rank_estimate_tokens = make_token_estimator(method_backend)
        rank_max_prompt_tokens = resolve_max_prompt_tokens(
            method_backend,
            method_backend_config,
        )

    if mc.type == "llm_ranking":
        method = LLMRankingMethod(
            backend=method_backend,
            context=method_context,
            sampling_params=method_sampling_params,
            max_history_turns=r_max_turns,
            prompt_preset=preset,
            batch_size=deps.batch_size,
            estimate_tokens=rank_estimate_tokens,
            max_prompt_tokens=rank_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_action_chars=r_min_action,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
            min_ranking_candidate_action_chars=context.min_ranking_candidate_action_chars,
            min_ranking_next_observation_chars=context.min_ranking_next_observation_chars,
            inject_task_text=deps.env_context.inject_task_text_in_prompts,
        )
    elif mc.type == "sdpo_ranking":
        method = SDPORankingMethod(
            backend=method_backend,
            context=method_context,
            prompt_preset=preset,
            batch_size=deps.batch_size,
            max_history_turns=r_max_turns,
            estimate_tokens=rank_estimate_tokens,
            max_prompt_tokens=rank_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
            max_feedback_chars=mc.max_feedback_chars,
        )
    elif mc.type == "sdpo_short_expert_continuation":
        if mc.expert_rollout_store is None:
            raise ValueError(
                "sdpo_short_expert_continuation requires expert_rollout_store"
            )
        method = SDPOShortExpertContinuationRankingMethod(
            backend=method_backend,
            context=method_context,
            prompt_preset=preset,
            expert_rollout_store=mc.expert_rollout_store,
            max_expert_continuation_steps=mc.max_expert_continuation_steps,
            batch_size=deps.batch_size,
            max_history_turns=r_max_turns,
            estimate_tokens=rank_estimate_tokens,
            max_prompt_tokens=rank_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
        )
    elif mc.type == "delta_belief_ranking":
        method = DeltaBeliefRankingMethod(
            backend=method_backend,
            context=method_context,
            prompt_preset=preset,
            batch_size=deps.batch_size,
            max_history_turns=r_max_turns,
            estimate_tokens=rank_estimate_tokens,
            max_prompt_tokens=rank_max_prompt_tokens,
            min_observation_chars=r_min_obs,
            min_current_observation_chars=r_min_cur_obs,
            min_next_observation_chars=r_min_next_obs,
        )
    else:
        raise ValueError(f"Unsupported ranking eval method type: {mc.type!r}")

    return method, method_backend_name

"""Run online test-time scaling with guided action selection."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from qval import (
    BackendsRegistryConfig,
    PipelineTestTimeScalingConfig,
    load_context,
    load_environment_context,
)
from qval.config import EvalMethodConfig
from qval.method_factory import (
    RANKING_METHOD_TYPES,
    VISION_METHOD_TYPES,
    MethodBuildContext,
    build_ranking_method,
    build_regular_method,
)
from qval.data_cache import Dataset, save_dataset
from qval.error_handling import check_abort_threshold
from qval.experiment_logging import (
    ExperimentLogger,
    LoggingBackend,
    compute_stats,
    summarize_trajectory_results,
)
from qval.quota_retry_backend import wrap_with_quota_retry
from qval.resource_monitor import ResourceMonitor
from qval.rollout import trajectory_return
from qval.run_checkpoint import (
    RESUME_FINGERPRINT_SCHEMA_VERSION,
    ResumeState,
    backend_content_descriptor,
    build_resume_fingerprint,
    checkpoint_path,
    content_make_kwargs,
    environment_context_descriptor,
    load_rolling_checkpoint,
    save_rolling_checkpoint,
)
from qval.script_utils import (
    ReplayCache,
    _apply_harbor_env_overrides,
    apply_cli_backend_tmux_prompt_hardening,
    attach_backend_logging_metadata,
    backend_cache_key,
    build_env_factory,
    build_history_fn,
    build_prompt_budget,
    chat_template_kwargs_from_config,
    create_adapter_env,
    create_backend_from_config,
    create_harbor_env_factory,
    inspect_harbor_runtime_eligibility,
    resolve_backend_name,
    resolve_env_params,
    resolve_system_prompt,
    resolve_task_indices,
    sampling_params_from_config,
)
from qval.second_elicitation_backend import wrap_with_second_elicitation
from qval.test_time_scaling import (
    CandidateScorer,
    DenseSignalCandidateScorer,
    DiscreteVotingScorer,
    GuidedActor,
    OpenEndedSelfConsistencyScorer,
    RankingCandidateScorer,
    run_guided_actors_batch,
)
from qval.trajectory_store import from_trajectory_result, save_trajectories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Throttle per-strategy progress lines to roughly one every this many percent of
# completed trajectories (plus the first and final update). Coarse on purpose —
# enough to eyeball how far along each strategy is, not a fine-grained tracker.
_PROGRESS_LOG_STEP_PCT = 10.0


def _timestamped_path(path: str | None, stem: str, suffix: str) -> Path | None:
    if path is None:
        return None
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    p = Path(path)
    return p.with_name(f"{p.stem}_{ts}{p.suffix or suffix}")


def _validate_vision_scorer(mc: EvalMethodConfig) -> None:
    """Reject vision scorers that depend on a dataset-sourced goal image.

    Test-time scaling has no ``Dataset``, so a vision method's goal must come
    from an explicit image path or text rather than ``goal_source='trajectory_end'``.
    """
    goal_source = None
    if mc.type == "vip":
        goal_source = mc.vip_goal_source
    elif mc.type in {"liv_img", "liv_txt"}:
        goal_source = mc.liv_goal_source
    if goal_source == "trajectory_end":
        raise ValueError(
            f"test_time_scaling: vision scorer {mc.type!r} cannot use "
            "goal_source='trajectory_end' (no dataset is available at test "
            "time). Provide an explicit goal via vip_env_goal_image_path / "
            "liv_env_goal_image_path / liv_goal_text instead."
        )


def _build_candidate_scorer(
    mc: EvalMethodConfig,
    deps: MethodBuildContext,
) -> CandidateScorer:
    """Build a candidate scorer for any benchmarked method.

    Individual-prediction methods are wrapped in :class:`DenseSignalCandidateScorer`
    so the guided actor ranks their per-candidate values; ranking-based methods are
    wrapped in :class:`RankingCandidateScorer` so their ordering / per-candidate
    scores drive selection directly. Construction is delegated to the shared
    factory so a method behaves identically here and in the prediction pipeline.
    """
    if mc.type in RANKING_METHOD_TYPES:
        method, _ = build_ranking_method(mc, deps)
        if mc.type == "llm_ranking":
            return RankingCandidateScorer(
                method=method,
                mode="permutation",
                include_next_state=mc.include_next_state,
            )
        # Score-based ranking methods (SDPO / delta-belief) read the candidate's
        # next_state for feedback, so candidates must be stepped regardless of the
        # config flag.
        return RankingCandidateScorer(
            method=method,
            mode="scores",
            include_next_state=True,
        )

    if mc.type in VISION_METHOD_TYPES:
        _validate_vision_scorer(mc)
    method, _ = build_regular_method(mc, deps)
    return DenseSignalCandidateScorer(
        method=method,
        include_next_state=mc.include_next_state,
    )


def _compute_summary(
    results_by_strategy: dict[str, list[Any]],
    *,
    reward_signal_name: str | None,
    success_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy, results in results_by_strategy.items():
        returns = [
            trajectory_return(result.trajectory, reward_signal_name)
            for result in results
        ]
        successes = [value >= success_threshold for value in returns]
        fallbacks = 0
        scorer_steps = 0
        candidate_counts: list[int] = []
        by_task: dict[int, list[bool]] = defaultdict(list)
        for result, success in zip(results, successes, strict=True):
            by_task[int(result.metadata["task_index"])].append(success)
            for transition in result.trajectory.transitions:
                meta = transition.info.get("test_time_scaling", {})
                candidate_counts.append(int(meta.get("k", 1)))
                if meta.get("used_scorer"):
                    scorer_steps += 1
                if meta.get("fallback_reason"):
                    fallbacks += 1
        summary[strategy] = {
            "num_trajectories": len(results),
            "success_rate": (
                sum(successes) / len(successes) if successes else 0.0
            ),
            "mean_return": sum(returns) / len(returns) if returns else 0.0,
            "mean_steps": (
                sum(len(result.trajectory) for result in results) / len(results)
                if results
                else 0.0
            ),
            "mean_candidate_count": (
                sum(candidate_counts) / len(candidate_counts)
                if candidate_counts
                else 0.0
            ),
            "scorer_steps": scorer_steps,
            "fallback_steps": fallbacks,
            "per_task": [
                {
                    "task_index": task_index,
                    "num_trajectories": len(values),
                    "success_rate": sum(values) / len(values) if values else 0.0,
                }
                for task_index, values in sorted(by_task.items())
            ],
        }
    return summary


def _resolve_actor_restore_fn(
    adapter: str,
    env_name: str,
    factory_result: Any,
    *,
    harbor_replay_cache: bool,
) -> Any:
    """Resolve the restore_fn the guided actor uses to step candidates.

    ``build_env_factory`` only populates ``factory_result.restore_fn`` for the
    gymnasium+use_images case; Harbor (and open_apps) must wire their own
    restore_fn separately — the same contract ``predict.py`` follows. A
    next-state scorer on a non-pure env (e.g. Harbor/TerminalBench) needs this
    to probe each candidate without advancing the live trajectory.

    Test-time scaling is a fresh run with no pre-captured snapshots, so
    ``restore_mode='replay'`` is the only viable mode (``snapshot_exact`` needs a
    prior collection's artifacts). When ``harbor_replay_cache`` is set on an
    apptainer runtime, the replay is wrapped in a ``ReplayCache`` so the ``k``
    candidates restored from the same state at a step share one replay.
    """
    restore_fn = factory_result.restore_fn
    if adapter != "harbor":
        return restore_fn

    harbor_kwargs = dict(factory_result.harbor_kwargs or {})
    _, restore_fn = create_harbor_env_factory(
        env_name, restore_mode="replay", **harbor_kwargs
    )
    environment_type = harbor_kwargs.get("environment_type", "docker")
    is_apptainer = environment_type in ("apptainer-hpc", "singularity-hpc")
    if harbor_replay_cache and is_apptainer:
        effective_kwargs = _apply_harbor_env_overrides(dict(harbor_kwargs))
        trials_dir = effective_kwargs.get("trials_dir") or os.environ.get(
            "VB_TRIALS_DIR", "trials"
        )
        cache_dir = Path(trials_dir) / "fs_restore_cache"
        restore_fn = ReplayCache(cache_dir, env_name=env_name).wrap(restore_fn)
        logger.info("Harbor replay cache enabled: %s", cache_dir)
    return restore_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run test-time scaling guided action selection"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-aborted-points", type=float, default=None)
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore (and remove) any existing resume checkpoint and start over",
    )
    args = parser.parse_args()

    from qval.error_handling import install_sigfpe_handler

    install_sigfpe_handler()

    config = PipelineTestTimeScalingConfig.from_yaml(args.config)
    if args.max_aborted_points is not None:
        config = dataclasses.replace(config, max_aborted_points=args.max_aborted_points)
    if args.success_threshold is not None:
        config = dataclasses.replace(config, success_threshold=args.success_threshold)
    if args.fresh:
        config = dataclasses.replace(config, resume=False)
    backends_config = BackendsRegistryConfig.from_yaml(config.backends_config_path)

    if config.debug:
        for name in ("qval", "llenvs", __name__):
            logging.getLogger(name).setLevel(logging.DEBUG)

    if not config.context_name:
        raise ValueError("context_name must be specified")

    env_context = load_environment_context(
        env_name=config.context_name,
        contexts_dir=config.contexts_dir,
    )
    params = resolve_env_params(env_context, make_kwargs_override=config.make_kwargs)
    env_name = params.env_name
    adapter = params.adapter
    max_steps = params.max_steps
    make_kwargs = params.make_kwargs
    env_make_kwargs = params.env_make_kwargs
    step_penalty = params.step_penalty
    extra_rewards = params.extra_rewards
    reward_signal_name = params.reward_signal_name

    explicit_task_indices = config.task_indices
    runtime_filter_report = None
    if adapter == "harbor":
        _, runtime_filter_report = inspect_harbor_runtime_eligibility(
            env_name,
            dataset_path=env_make_kwargs.get("dataset_path"),
            environment_type=str(env_make_kwargs.get("environment_type", "docker")),
            sif_cache_dir=env_make_kwargs.get("sif_cache_dir"),
            task_indices=explicit_task_indices,
            num_trajectories=None,
            difficulties=env_make_kwargs.get("difficulties"),
        )

    if runtime_filter_report is not None:
        env_size = len(runtime_filter_report["selected_task_indices"])
    elif explicit_task_indices is not None:
        env_size = max(explicit_task_indices) + 1
    elif config.num_tasks is not None:
        env_size = config.num_tasks
    else:
        env_size = 1000

    env, adapter_system_prompt = create_adapter_env(
        adapter,
        env_name,
        env_size=env_size,
        seed=config.seed,
        max_steps=max_steps,
        answer_extractor=env_context.environment_extractor,
        make_kwargs=env_make_kwargs,
        extra_rewards=extra_rewards,
        invalid_action_text=getattr(env_context, "invalid_action_text", None),
        invalid_action_observation=getattr(
            env_context, "invalid_action_observation", None
        ),
        advance_on_invalid=getattr(env_context, "advance_on_invalid", None),
    )

    system_prompt = resolve_system_prompt(
        override_system_prompt=config.system_prompt,
        adapter=getattr(env_context, "adapter", None),
        prompting_scheme=getattr(env_context, "prompting_scheme", None),
        context_system_prompt_file=getattr(env_context, "system_prompt_file", None),
        adapter_system_prompt=adapter_system_prompt,
        make_kwargs=env_make_kwargs,
        env_name=getattr(env_context, "env_name", None),
    )

    actor_backend_name = resolve_backend_name(
        config.actor_backend_name,
        backends_config.backends,
        backends_config.default_backend,
    )
    actor_backend_config = backends_config.backends[actor_backend_name]
    system_prompt = apply_cli_backend_tmux_prompt_hardening(
        system_prompt,
        enabled=config.cli_backend_tmux_prompt_hardening,
        adapter=adapter,
        make_kwargs=make_kwargs,
        backend_config=actor_backend_config,
    )

    exp_logger = ExperimentLogger()
    output_path = config.output_path
    if args.output_dir:
        output_path = str(Path(args.output_dir) / f"test_time_scaling_{config.context_name}.json")
    final_path = _timestamped_path(output_path, "test_time_scaling", ".json")
    if final_path is not None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        exp_logger.attach_jsonl_sink(final_path.with_suffix(".logs.jsonl"))

    backend_cache: dict[Any, Any] = {}

    def get_backend(backend_config: Any) -> Any:
        key = backend_cache_key(backend_config)
        if key not in backend_cache:
            raw = create_backend_from_config(
                backend_config,
                chat_template_kwargs=chat_template_kwargs_from_config(backend_config),
                tensor_parallel_size_override=args.tensor_parallel_size,
            )
            attach_backend_logging_metadata(raw, backend_config)
            wrapped = wrap_with_quota_retry(
                raw,
                policy=backend_config.quota_retry_policy,
            )
            backend_cache[key] = wrap_with_second_elicitation(
                LoggingBackend(wrapped, exp_logger)
            )
        return backend_cache[key]

    actor_backend = get_backend(actor_backend_config)
    actor_sampling_params = sampling_params_from_config(
        actor_backend_config.sampling,
        backend_type=actor_backend_config.type,
        provider_preferences=actor_backend_config.openrouter_provider,
    )

    factory_result = build_env_factory(
        adapter,
        env_name,
        env_make_kwargs,
        max_steps=max_steps,
        extra_rewards=extra_rewards,
        answer_extractor=env_context.environment_extractor,
    )
    actor_restore_fn = _resolve_actor_restore_fn(
        adapter,
        env_name,
        factory_result,
        harbor_replay_cache=config.harbor_replay_cache,
    )
    history_fn = build_history_fn(
        env_context,
        max_history_turns=config.max_history_turns,
    )
    prompt_budget = build_prompt_budget(
        actor_backend,
        actor_backend_config,
        env_context,
        max_history_turns=config.max_history_turns,
    )

    scorer = None
    method_backend_config = None
    if config.guided_actor.scorer is not None:
        mc = config.guided_actor.scorer
        if mc.type != "baseline_random":
            assert mc.backend_name is not None
            method_backend_name = resolve_backend_name(
                mc.backend_name,
                backends_config.backends,
                backends_config.default_backend,
            )
            method_backend_config = backends_config.backends[method_backend_name]
        method_base_context = load_context(
            config.context_name,
            mc.signal_type,
            contexts_dir=config.contexts_dir,
        )
        # Construction is delegated to the shared factory (same path as the
        # prediction pipeline). Test-time scaling has no Dataset and pins the
        # discount factor to 1.0; ``eval_backend_name=None`` because each scorer
        # declares its own backend explicitly.
        scorer = _build_candidate_scorer(
            mc,
            MethodBuildContext(
                context=method_base_context,
                backends_config=backends_config,
                get_backend=get_backend,
                eval_backend_name=None,
                default_max_history_turns=config.max_history_turns,
                step_penalty=step_penalty,
                discount_factor=1.0,
                batch_size=config.batch_size,
                env_context=env_context,
                dataset=None,
            ),
        )

    if runtime_filter_report is not None:
        all_task_indices = list(runtime_filter_report["selected_task_indices"])
        if config.num_tasks is not None:
            all_task_indices = all_task_indices[: config.num_tasks]
    elif explicit_task_indices is not None:
        all_task_indices = list(explicit_task_indices)
    else:
        all_task_indices = resolve_task_indices(
            env,
            adapter=adapter,
            num_tasks=config.num_tasks,
            shuffle=config.shuffle_tasks,
            seed=config.seed,
            overflow="cap",
            logger=logger,
        )

    repeated_indices = [
        task_idx
        for task_idx in all_task_indices
        for _ in range(config.samples_per_task)
    ]

    actors = {
        "baseline_k1": GuidedActor(
            actor_backend=actor_backend,
            actor_sampling_params=actor_sampling_params,
            k=1,
            scorer=None,
            rng=random.Random(config.seed),
            system_prompt=system_prompt,
            history_fn=history_fn,
            prompt_budget=prompt_budget,
            turn_info=env_context.turn_info or False,
            env_factory=factory_result.env_factory,
            restore_fn=actor_restore_fn,
        )
    }
    if config.guided_actor.k != 1 or scorer is not None:
        actors["guided"] = GuidedActor(
            actor_backend=actor_backend,
            actor_sampling_params=actor_sampling_params,
            k=config.guided_actor.k,
            scorer=scorer,
            rng=random.Random(config.seed + 1),
            system_prompt=system_prompt,
            history_fn=history_fn,
            prompt_budget=prompt_budget,
            turn_info=env_context.turn_info or False,
            env_factory=factory_result.env_factory,
            restore_fn=actor_restore_fn,
        )

    selector_backend_config = None
    if config.self_consistency is not None:
        sc = config.self_consistency
        if sc.mode == "open_ended":
            assert sc.backend_name is not None
            selector_backend_name = resolve_backend_name(
                sc.backend_name,
                backends_config.backends,
                backends_config.default_backend,
            )
            selector_backend_config = backends_config.backends[selector_backend_name]
            sc_scorer: Any = OpenEndedSelfConsistencyScorer(
                selector_backend=get_backend(selector_backend_config),
                selector_sampling_params=sampling_params_from_config(
                    selector_backend_config.sampling,
                    backend_type=selector_backend_config.type,
                    provider_preferences=selector_backend_config.openrouter_provider,
                ),
                extractor=env_context.environment_extractor,
                max_candidate_chars=sc.max_candidate_chars,
            )
        else:
            sc_scorer = DiscreteVotingScorer()
        actors["self_consistency"] = GuidedActor(
            actor_backend=actor_backend,
            actor_sampling_params=actor_sampling_params,
            k=config.guided_actor.k,
            scorer=sc_scorer,
            rng=random.Random(config.seed + 2),
            system_prompt=system_prompt,
            history_fn=history_fn,
            prompt_budget=prompt_budget,
            turn_info=env_context.turn_info or False,
            env_factory=factory_result.env_factory,
            restore_fn=actor_restore_fn,
        )

    # Per-strategy base seed (matches each actor's rng offset above). The batched
    # rollout derives a per-trajectory rng as base_seed + trajectory_index, so
    # tie-break/fallback outcomes are independent of the wave size.
    base_seeds = {
        "baseline_k1": config.seed,
        "guided": config.seed + 1,
        "self_consistency": config.seed + 2,
    }

    # Resume fingerprints cover every content-affecting input: anything that
    # changes trajectory content or the trajectory_index -> task mapping.
    # Transport/parallelism/summary-only knobs (batch_size,
    # max_concurrent_trajectories, restore_concurrency, max_aborted_points,
    # success_threshold, output paths) are deliberately excluded. The resolved
    # repeated_indices list captures num_tasks/task_indices/shuffle_tasks and
    # any harbor runtime filtering.
    common_fingerprint_descriptor = {
        "schema_version": RESUME_FINGERPRINT_SCHEMA_VERSION,
        "pipeline": "test_time_scaling",
        "context_name": config.context_name,
        "env": {
            "env_name": env_name,
            "adapter": adapter,
            "max_steps": max_steps,
            "step_penalty": step_penalty,
            "reward_signal_name": reward_signal_name,
            "make_kwargs": content_make_kwargs(make_kwargs),
        },
        "env_context": environment_context_descriptor(env_context),
        "system_prompt": system_prompt,
        "seed": config.seed,
        "samples_per_task": config.samples_per_task,
        "repeated_indices": repeated_indices,
        "max_history_turns": config.max_history_turns,
        "actor": {
            "backend_name": actor_backend_name,
            "config": backend_content_descriptor(actor_backend_config),
        },
    }
    strategy_extras: dict[str, dict[str, Any]] = {
        "baseline_k1": {},
        "guided": {
            "k": config.guided_actor.k,
            "scorer": config.guided_actor.scorer,
            "scorer_backend": (
                backend_content_descriptor(method_backend_config)
                if method_backend_config is not None
                else None
            ),
        },
        "self_consistency": {
            "k": config.guided_actor.k,
            "self_consistency": config.self_consistency,
            "selector_backend": (
                backend_content_descriptor(selector_backend_config)
                if selector_backend_config is not None
                else None
            ),
        },
    }
    out_base = Path(output_path) if output_path is not None else None
    if out_base is None:
        logger.warning(
            "No output_path configured; resume checkpointing is disabled"
        )
    strategy_fingerprints: dict[str, str] = {}
    strategy_ckpt_paths: dict[str, Path] = {}
    for strategy in actors:
        fingerprint = build_resume_fingerprint(
            {**common_fingerprint_descriptor, "strategy": strategy, **strategy_extras[strategy]}
        )
        strategy_fingerprints[strategy] = fingerprint
        if out_base is not None:
            strategy_ckpt_paths[strategy] = checkpoint_path(
                out_base.parent, f"{out_base.stem}_{strategy}", fingerprint
            )

    resource_monitor = ResourceMonitor(watch_dirs=[])
    resource_monitor.start()
    start = time.time()
    results_by_strategy: dict[str, list[Any]] = {name: [] for name in actors}
    skipped_by_strategy: dict[str, int] = {name: 0 for name in actors}
    resumed_by_strategy: dict[str, int] = {name: 0 for name in actors}

    wave_size = config.max_concurrent_trajectories
    try:
        for strategy, guided_actor in actors.items():
            exp_logger.set_phase(f"test_time_scaling:{strategy}")
            ckpt_path = strategy_ckpt_paths.get(strategy)
            fingerprint = strategy_fingerprints[strategy]
            state = ResumeState()
            if ckpt_path is not None:
                if config.resume:
                    state = ResumeState.from_checkpoint(
                        load_rolling_checkpoint(
                            ckpt_path, expected_fingerprint=fingerprint
                        )
                    )
                    if state.resumed_count:
                        logger.info(
                            "Resuming %s: %d/%d trajectories loaded from %s",
                            strategy,
                            state.resumed_count,
                            len(repeated_indices),
                            ckpt_path,
                        )
                        exp_logger.log_event(
                            "resume",
                            {
                                "strategy": strategy,
                                "resumed_trajectories": state.resumed_count,
                                "checkpoint": str(ckpt_path),
                            },
                        )
                else:
                    ckpt_path.unlink(missing_ok=True)
            resumed_by_strategy[strategy] = state.resumed_count

            total_trajectories = len(repeated_indices)
            progress = exp_logger.make_progress_reporter(
                name=f"test_time_scaling:{strategy}",
                logger_name=__name__,
                min_step_pct=_PROGRESS_LOG_STEP_PCT,
            )
            # Counts every finished trajectory this run (success or failure) on
            # top of any resumed from a checkpoint. Wrapped in a list so the
            # completion callback can mutate it.
            progress_done = [state.resumed_count]
            progress(progress_done[0], total_trajectories)

            def _on_complete(
                trajectory_index: int,
                result: Any,
                exc: Exception | None,
                *,
                _state: ResumeState = state,
                _ckpt_path: Path | None = ckpt_path,
                _fingerprint: str = fingerprint,
                _strategy: str = strategy,
                _progress: Any = progress,
                _progress_done: list[int] = progress_done,
                _total: int = total_trajectories,
            ) -> None:
                _progress_done[0] += 1
                _progress(_progress_done[0], _total)
                if result is None:
                    return  # failures are retried on resume, never checkpointed
                _state.record(trajectory_index, result)
                if _ckpt_path is None:
                    return
                completed = _state.completed_trajectory_indices()
                save_rolling_checkpoint(
                    _ckpt_path,
                    trajectory_results=_state.ordered_results(),
                    config_dict=dataclasses.asdict(config),
                    metadata={
                        "resume_fingerprint": _fingerprint,
                        "strategy": _strategy,
                        "context_name": config.context_name,
                        "env_name": env_name,
                        "adapter": adapter,
                        "actor_backend_name": actor_backend_name,
                        "completed_trajectory_indices": completed,
                        "num_trajectories_so_far": len(completed),
                    },
                )

            pending = state.remaining(repeated_indices)
            processed = 0
            for wave_start in range(0, len(pending), wave_size):
                wave = pending[wave_start : wave_start + wave_size]
                slots = run_guided_actors_batch(
                    env=env,
                    actor=guided_actor,
                    task_indices=[task_index for _, task_index in wave],
                    trajectory_indices=[
                        trajectory_index for trajectory_index, _ in wave
                    ],
                    max_steps=max_steps,
                    batch_size=config.batch_size,
                    restore_concurrency=config.restore_concurrency,
                    base_seed=base_seeds[strategy],
                    on_trajectory_complete=_on_complete,
                )
                for offset, (result, exc) in enumerate(slots):
                    trajectory_index, task_index = wave[offset]
                    if result is None:
                        skipped_by_strategy[strategy] += 1
                        exp_logger.log_event(
                            "skipped_trajectory",
                            {
                                "strategy": strategy,
                                "task_index": task_index,
                                "trajectory_index": trajectory_index,
                                "error": str(exc),
                            },
                        )
                        logger.warning(
                            "Skipping %s task %s: %s",
                            strategy,
                            task_index,
                            exc,
                        )
                    else:
                        # Idempotent vs. the completion callback; also covers a
                        # callback whose checkpoint flush failed.
                        state.record(trajectory_index, result)
                    processed += 1
                    check_abort_threshold(
                        skipped_by_strategy[strategy],
                        processed,
                        config.max_aborted_points,
                        f"test_time_scaling:{strategy}",
                    )
            results_by_strategy[strategy] = state.ordered_results()
    finally:
        resource_summary = resource_monitor.stop()
        for backend in reversed(list(backend_cache.values())):
            try:
                backend.close()
            except Exception as exc:
                logger.warning("Backend cleanup failed: %s", exc)

    elapsed = time.time() - start
    for strategy, results in results_by_strategy.items():
        exp_logger.log_event(
            "trajectory_summary",
            {
                "strategy": strategy,
                **summarize_trajectory_results(results),
            },
        )
        stats = compute_stats(exp_logger, f"test_time_scaling:{strategy}")
        exp_logger.log_event("stats", {"strategy": strategy, **stats})
    if resource_summary:
        exp_logger.log_event("resource_summary", resource_summary)

    summary = _compute_summary(
        results_by_strategy,
        reward_signal_name=reward_signal_name,
        success_threshold=config.success_threshold,
    )
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "context_name": config.context_name,
        "env_name": env_name,
        "adapter": adapter,
        "actor_backend_name": actor_backend_name,
        "actor_model": actor_backend_config.model,
        "seed": config.seed,
        "samples_per_task": config.samples_per_task,
        "num_tasks": len(all_task_indices),
        "success_threshold": config.success_threshold,
        "reward_signal_name": reward_signal_name,
        "elapsed_seconds": round(elapsed, 2),
        "skipped_by_strategy": skipped_by_strategy,
        "resumed_by_strategy": resumed_by_strategy,
        "resume_fingerprints": strategy_fingerprints,
    }
    output = {
        "metadata": metadata,
        "summary": summary,
        "config": dataclasses.asdict(config),
    }
    print(json.dumps(output, indent=2, default=str))

    if final_path is not None:
        with final_path.open("w") as f:
            json.dump(output, f, indent=2, default=str)
        for strategy, results in results_by_strategy.items():
            traj_returns = [
                trajectory_return(result.trajectory, reward_signal_name)
                for result in results
            ]
            dataset = Dataset(
                evaluation_points=[],
                trajectory_returns=traj_returns,
                trajectory_results=results,
                config=dataclasses.asdict(config),
                metadata={**metadata, "strategy": strategy},
            )
            save_dataset(dataset, final_path.with_name(f"{final_path.stem}_{strategy}.pkl"))
            save_trajectories(
                [from_trajectory_result(result, reward_signal_name) for result in results],
                str(final_path.with_name(f"{final_path.stem}_{strategy}_trajectories.json")),
                metadata={**metadata, "strategy": strategy},
            )
        # Remove resume checkpoints only after every strategy's final artifacts
        # exist — the per-strategy pkls above are the first durable copies, so
        # deleting any checkpoint earlier would orphan completed strategies if a
        # later strategy crashed.
        for ckpt_path in strategy_ckpt_paths.values():
            ckpt_path.unlink(missing_ok=True)
        logger.info("Saved test-time scaling results to %s", final_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

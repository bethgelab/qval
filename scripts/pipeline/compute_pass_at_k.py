"""Step 0: Compute Pass@k — evaluate model success rate on environment tasks.

Runs N independent trajectories per task, groups results by task, and
computes Pass@k using the combinatorial formula from llenvs. Outputs a
JSON report and console summary.

Usage::

    uv run python scripts/pipeline/compute_pass_at_k.py \
        --config shared/configs/pass_at_k/<env>/<config>.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from llenvs.evaluation.runner import TrajectoryResult

from qval import (
    BackendsRegistryConfig,
    PipelinePassAtKConfig,
    load_environment_context,
)
from qval.benchmark import collect_trajectories
from qval.config import BenchmarkConfig, EvaluationPointsConfig
from qval.data_cache import Dataset, save_dataset
from qval.error_handling import check_abort_threshold
from qval.experiment_logging import (
    ExperimentLogger,
    LoggingBackend,
    compute_stats,
    summarize_trajectory_results,
)
from qval.pass_at_k import (
    PassAtKAttempt,
    attempt_from_trajectory_result,
    compute_pass_at_k,
    print_summary,
)
from qval.quota_retry_backend import wrap_with_quota_retry
from qval.resource_monitor import ResourceMonitor
from qval.rollout import trajectory_return
from qval.second_elicitation_backend import wrap_with_second_elicitation
from qval.trajectory_store import from_trajectory_result, save_trajectories
from qval.script_utils import (
    apply_cli_backend_tmux_prompt_hardening,
    attach_backend_logging_metadata,
    build_env_factory,
    build_history_fn,
    build_prompt_budget,
    chat_template_kwargs_from_config,
    create_adapter_env,
    create_backend_from_config,
    inspect_harbor_runtime_eligibility,
    resolve_backend_name,
    resolve_effective_max_steps,
    resolve_env_params,
    resolve_system_prompt,
    resolve_task_indices,
    sampling_params_from_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result computation
# ---------------------------------------------------------------------------


# Re-exported helper — used by tests that patch this symbol.
from qval.data_cache import _task_name_for_result as _task_name_for_result


def _attempts_from_results(
    all_results: list[TrajectoryResult],
    *,
    reward_signal_name: str | None,
    success_threshold: float,
) -> list[PassAtKAttempt]:
    """Adapt a list of TrajectoryResults to Pass@k attempts keyed by task_index."""
    attempts: list[PassAtKAttempt] = []
    for r in all_results:
        task_index = r.metadata["task_index"]
        attempts.append(
            attempt_from_trajectory_result(
                r,
                reward_signal_name=reward_signal_name,
                success_threshold=success_threshold,
                task_key=task_index,
                task_name=_task_name_for_result(r),
            )
        )
    return attempts


def _rename_task_key_to_task_index(results: dict[str, Any]) -> dict[str, Any]:
    """Keep the historical JSON schema: per-task rows use ``task_index``.

    The shared ``compute_pass_at_k`` emits ``task_key`` (neutral). This script
    always uses integer task indices as keys, so we rename the field for
    backward compatibility with downstream consumers of the JSON output.
    """
    per_task = []
    for row in results["per_task"]:
        renamed = {"task_index": row["task_key"]}
        for field_name, field_value in row.items():
            if field_name == "task_key":
                continue
            if field_name == "attempts":
                renamed["trajectories"] = [
                    {
                        "success": attempt["success"],
                        "total_reward": attempt["total_reward"],
                        "num_steps": attempt["num_steps"],
                    }
                    for attempt in field_value
                ]
            else:
                renamed[field_name] = field_value
        per_task.append(renamed)
    return {**results, "per_task": per_task}


def _compute_results(
    all_results: list[TrajectoryResult],
    k_values: tuple[int, ...],
    *,
    requested_counts: dict[int, int] | None = None,
    requested_task_names: dict[int, str | None] | None = None,
    reward_signal_name: str | None = None,
    success_threshold: float = 1.0,
) -> dict[str, Any]:
    """Group TrajectoryResults by task and compute per-task + aggregate Pass@k.

    Thin adapter around the shared :func:`compute_pass_at_k`. Preserves the
    historical JSON field name ``task_index`` (the shared field is the neutral
    ``task_key``) and overlays names from ``requested_task_names`` for tasks
    whose attempts all dropped.
    """
    attempts = _attempts_from_results(
        all_results,
        reward_signal_name=reward_signal_name,
        success_threshold=success_threshold,
    )
    results = compute_pass_at_k(
        attempts,
        k_values,
        requested_counts=requested_counts,
    )
    if requested_task_names:
        for row in results["per_task"]:
            if row["task_name"] is None:
                row["task_name"] = requested_task_names.get(row["task_key"])
    return _rename_task_key_to_task_index(results)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(
    results: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Print human-readable Pass@k summary to console.

    Adapter around the shared :func:`print_summary` that renders with the
    traditional header used by this script.
    """
    header = (
        f"Pass@k Results: {metadata['context_name']} "
        f"({metadata['backend_name']}, "
        f"N={metadata['samples_per_task']}, "
        f"{len(results['per_task'])} tasks)"
    )
    # The JSON has been renamed to ``task_index``; rebuild a ``task_key``
    # view for the shared printer.
    view = {
        "aggregate": results["aggregate"],
        "per_task": [
            {**row, "task_key": row["task_index"]} for row in results["per_task"]
        ],
    }
    print_summary(view, metadata, title=header, task_label="Task")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 0: Compute Pass@k for a model on an environment"
    )
    parser.add_argument("--config", required=True, help="Path to pass_at_k YAML config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config output_path dir)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override tensor_parallel_size for all vLLM backends",
    )
    parser.add_argument(
        "--max-aborted-points",
        type=float,
        default=None,
        help="Abort if the fraction of dropped trajectories exceeds this threshold",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help=(
            "Minimum cumulative reward_signal_name return for a trajectory to "
            "count as a Pass@k success. Overrides the YAML setting. "
            "Defaults to 1.0 for binary-reward envs."
        ),
    )
    args = parser.parse_args()

    from qval.error_handling import install_sigfpe_handler

    install_sigfpe_handler()

    # 1. Load config
    config_path = Path(args.config)
    with config_path.open() as f:
        raw_config = yaml.safe_load(f) or {}
    config = PipelinePassAtKConfig.from_dict(raw_config)
    if args.max_aborted_points is not None:
        config = dataclasses.replace(config, max_aborted_points=args.max_aborted_points)
    if args.success_threshold is not None:
        config = dataclasses.replace(config, success_threshold=args.success_threshold)
    backends_config = BackendsRegistryConfig.from_yaml(config.backends_config_path)

    if config.debug:
        for name in ("qval", "llenvs", __name__):
            logging.getLogger(name).setLevel(logging.DEBUG)

    # 2. Load environment context
    context_name = config.context_name
    if not context_name:
        raise ValueError("context_name must be specified in config")
    context = load_environment_context(
        env_name=context_name,
        contexts_dir=config.contexts_dir,
    )

    params = resolve_env_params(context, make_kwargs_override=config.make_kwargs)
    env_name = params.env_name
    adapter = params.adapter
    max_steps = params.max_steps
    make_kwargs = params.make_kwargs
    env_make_kwargs = params.env_make_kwargs
    step_penalty = params.step_penalty
    extra_rewards = params.extra_rewards
    reward_signal_name = params.reward_signal_name

    if step_penalty is not None:
        logger.info("Step penalty: %.4f", step_penalty)

    logger.info(
        "Success criterion: cumulative %s >= %.4f",
        reward_signal_name
        if reward_signal_name is not None
        else "(all signals, excl. step_penalty)",
        config.success_threshold,
    )

    # 3. Harbor runtime-eligibility filtering
    explicit_task_indices = config.task_indices
    requested_num_tasks = config.num_tasks
    runtime_filter_report: dict[str, Any] | None = None

    if adapter == "harbor":
        environment_type = str(env_make_kwargs.get("environment_type", "docker"))
        sif_cache_dir = env_make_kwargs.get("sif_cache_dir")
        _, runtime_filter_report = inspect_harbor_runtime_eligibility(
            env_name,
            dataset_path=env_make_kwargs.get("dataset_path"),
            environment_type=environment_type,
            sif_cache_dir=sif_cache_dir,
            task_indices=explicit_task_indices,
            num_trajectories=None,  # get full eligible pool
            difficulties=env_make_kwargs.get("difficulties"),
        )
        if runtime_filter_report["filtered_count"] > 0:
            logger.info(
                "Harbor runtime eligibility (%s): %d/%d eligible, filtered %d",
                environment_type,
                runtime_filter_report["eligible_count"],
                runtime_filter_report["total_tasks"],
                runtime_filter_report["filtered_count"],
            )

    # 4. Create environment
    if runtime_filter_report is not None:
        env_size = len(runtime_filter_report["selected_task_indices"])
    elif explicit_task_indices is not None:
        env_size = max(explicit_task_indices) + 1
    elif requested_num_tasks is not None:
        env_size = requested_num_tasks
    else:
        env_size = None

    env, adapter_system_prompt = create_adapter_env(
        adapter,
        env_name,
        env_size=env_size if env_size is not None else 1000,
        seed=config.seed,
        max_steps=max_steps,
        answer_extractor=context.environment_extractor,
        make_kwargs=env_make_kwargs,
        extra_rewards=extra_rewards,
        invalid_action_text=getattr(context, "invalid_action_text", None),
        invalid_action_observation=getattr(context, "invalid_action_observation", None),
        advance_on_invalid=getattr(context, "advance_on_invalid", None),
    )

    system_prompt = resolve_system_prompt(
        override_system_prompt=config.system_prompt,
        adapter=getattr(context, "adapter", None),
        prompting_scheme=getattr(context, "prompting_scheme", None),
        context_system_prompt_file=getattr(context, "system_prompt_file", None),
        adapter_system_prompt=adapter_system_prompt,
        make_kwargs=env_make_kwargs,
        env_name=getattr(context, "env_name", None),
    )
    from qval.system_prompts import SCHEME_INSTRUCTIONS

    actor_scheme = getattr(context, "prompting_scheme", None) or "answer_tags"
    actor_format_reminder = SCHEME_INSTRUCTIONS.get(actor_scheme)

    factory_result = build_env_factory(
        adapter,
        env_name,
        env_make_kwargs,
        max_steps=max_steps,
        extra_rewards=extra_rewards,
        answer_extractor=context.environment_extractor,
    )
    env_factory = factory_result.env_factory

    # 5. Determine task indices (unique tasks, then repeated samples_per_task times)
    if runtime_filter_report is not None:
        eligible_pool = list(runtime_filter_report["selected_task_indices"])
    else:
        eligible_pool = None

    if explicit_task_indices is not None:
        all_task_indices = list(explicit_task_indices)
    elif eligible_pool is not None:
        # Harbor: select from eligible pool
        all_task_indices = list(eligible_pool)
        if config.shuffle_tasks:
            random.Random(config.seed).shuffle(all_task_indices)
        if requested_num_tasks is not None:
            all_task_indices = all_task_indices[:requested_num_tasks]
    else:
        # Pass@k caps (does not cycle) on overflow so per-task sample counts
        # stay even across the bounded task pool.
        all_task_indices = resolve_task_indices(
            env,
            adapter=adapter,
            num_tasks=requested_num_tasks,
            shuffle=config.shuffle_tasks,
            seed=config.seed,
            overflow="cap",
            logger=logger,
        )

    logger.info(
        "Task selection: %d tasks, samples_per_task=%d, total trajectories=%d",
        len(all_task_indices),
        config.samples_per_task,
        len(all_task_indices) * config.samples_per_task,
    )

    # 6. Build repeated indices: each task repeated N times
    repeated_indices: list[int] = []
    for task_idx in all_task_indices:
        repeated_indices.extend([task_idx] * config.samples_per_task)

    # 7. Create backend
    backend_name = resolve_backend_name(
        config.backend_name,
        backends_config.backends,
        backends_config.default_backend,
    )
    backend_config = backends_config.backends[backend_name]
    system_prompt = apply_cli_backend_tmux_prompt_hardening(
        system_prompt,
        enabled=config.cli_backend_tmux_prompt_hardening,
        adapter=adapter,
        make_kwargs=make_kwargs,
        backend_config=backend_config,
    )
    chat_template_kwargs = chat_template_kwargs_from_config(backend_config)
    raw_backend = create_backend_from_config(
        backend_config,
        chat_template_kwargs=chat_template_kwargs,
        tensor_parallel_size_override=args.tensor_parallel_size,
    )
    attach_backend_logging_metadata(raw_backend, backend_config)

    exp_logger = ExperimentLogger()

    # Attach JSONL sink early so logs survive mid-collection crashes
    output_path = config.output_path
    if args.output_dir:
        output_path = str(Path(args.output_dir) / f"pass_at_k_{context_name}.json")
    log_path: Path | None = None
    final_path: Path | None = None
    if output_path:
        ts_early = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        p = Path(output_path)
        final_path = p.with_name(f"{p.stem}_{ts_early}{p.suffix}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = final_path.with_suffix(".logs.jsonl")
        exp_logger.attach_jsonl_sink(log_path)

    quota_wrapped = wrap_with_quota_retry(
        raw_backend, policy=backend_config.quota_retry_policy
    )
    backend = wrap_with_second_elicitation(LoggingBackend(quota_wrapped, exp_logger))

    # Resolve turn_info from context
    turn_info = context.turn_info if context.turn_info is not None else False

    # 8. Build truncation strategy (prompt_budget takes precedence over history_fn)
    prompt_budget = build_prompt_budget(
        backend,
        backend_config,
        context,
        max_history_turns=config.max_history_turns,
    )
    history_fn = None
    if prompt_budget is None:
        history_fn = build_history_fn(
            context, max_history_turns=config.max_history_turns
        )

    # 9. Collect trajectories in chunks
    sampling_params = sampling_params_from_config(
        backend_config.sampling,
        backend_type=backend_config.type,
        provider_preferences=backend_config.openrouter_provider,
    )

    # Start resource monitoring
    import os

    trials_dir = os.environ.get("VB_TRIALS_DIR", "trials")
    if adapter == "harbor" and factory_result.harbor_kwargs:
        from qval.script_utils import _apply_harbor_env_overrides

        effective_harbor_kwargs = _apply_harbor_env_overrides(
            factory_result.harbor_kwargs
        )
        trials_dir = effective_harbor_kwargs.get("trials_dir", trials_dir)
    resource_monitor = ResourceMonitor(watch_dirs=[trials_dir])
    resource_monitor.start()

    all_trajectory_results: list[TrajectoryResult] = []
    chunk_size = config.batch_size
    total_trajectories = len(repeated_indices)
    attempted_trajectories = 0
    skipped_trajectory_count = 0
    attempted_by_task: defaultdict[int, int] = defaultdict(int)
    attempted_task_names: dict[int, str | None] = {}
    start_time = time.time()
    resource_summary: dict[str, Any] | None = None

    exp_logger.set_phase("pass_at_k_collection")
    inner_progress = exp_logger.make_progress_reporter(
        name="Pass@k collection",
        logger_name=__name__,
    )
    chunk_offset = 0

    def progress(completed: int, total: int) -> None:
        del total
        inner_progress(chunk_offset + completed, total_trajectories)

    num_batches = (total_trajectories + chunk_size - 1) // chunk_size

    def _build_dataset_config(num_trajectories: int) -> dict[str, Any]:
        return {
            "reward_signal_name": reward_signal_name,
            "batch_size": config.batch_size,
            "max_aborted_points": config.max_aborted_points,
            "points_config": {
                "num_trajectories": num_trajectories,
                "sampling_strategy": "random",
                "early_turns_to_discard": 1,
                "late_turns_to_discard": 1,
            },
        }

    def _build_dataset_metadata(
        num_trajectories: int,
        *,
        num_requested: int,
        elapsed_seconds: float | None = None,
        checkpoint: bool = False,
        partial: bool = False,
        partial_reason: str | None = None,
    ) -> dict[str, Any]:
        dropped_trajectories = max(num_requested - num_trajectories, 0)
        effective_max_steps = resolve_effective_max_steps(max_steps, make_kwargs)
        metadata = {
            "env_name": env_name,
            "adapter": adapter,
            "seed": config.seed,
            "backend_names": [backend_name],
            "num_trajectories": num_trajectories,
            "requested_trajectories": num_requested,
            "retained_trajectories": num_trajectories,
            "dropped_trajectories": dropped_trajectories,
            "drop_rate": (
                dropped_trajectories / num_requested if num_requested > 0 else 0.0
            ),
            "task_indices": all_task_indices,
            "system_prompt": system_prompt,
            "max_steps": effective_max_steps,
            "make_kwargs": make_kwargs,
            "context_name": context_name,
            "contexts_dir": config.contexts_dir,
            "step_penalty": step_penalty,
            "turn_info": turn_info,
            "source": "compute_pass_at_k",
            "samples_per_task": config.samples_per_task,
        }
        if elapsed_seconds is not None:
            metadata["collection_time_seconds"] = round(elapsed_seconds, 1)
            metadata["collection_timestamp"] = datetime.now().astimezone().isoformat()
        if checkpoint:
            metadata["checkpoint"] = True
            metadata["num_trajectories_so_far"] = num_trajectories
            metadata["checkpoint_timestamp"] = datetime.now().astimezone().isoformat()
        if partial:
            metadata["partial"] = True
        if partial_reason is not None:
            metadata["partial_reason"] = partial_reason
        return metadata

    def _on_skipped(skipped: dict[str, Any]) -> None:
        nonlocal skipped_trajectory_count
        skipped_trajectory_count += 1
        task_idx = skipped.get("task_index")
        task_name = skipped.get("task_name")
        if isinstance(task_idx, int) and task_idx not in attempted_task_names:
            attempted_task_names[task_idx] = (
                task_name if isinstance(task_name, str) and task_name else None
            )
        exp_logger.log_event("skipped_trajectory", skipped)

    try:
        for batch_idx, chunk_start in enumerate(
            range(0, total_trajectories, chunk_size)
        ):
            chunk_indices = repeated_indices[chunk_start : chunk_start + chunk_size]
            chunk_offset = chunk_start
            chunk_config = BenchmarkConfig(
                points_config=EvaluationPointsConfig(
                    task_indices=tuple(chunk_indices),
                    num_trajectories=len(chunk_indices),
                ),
                batch_size=config.batch_size,
            )
            attempted_trajectories += len(chunk_indices)
            for task_idx in chunk_indices:
                attempted_by_task[task_idx] += 1
            results = collect_trajectories(
                env,
                backend,
                chunk_config,
                sampling_params,
                system_prompt,
                sampling_params.extra,
                turn_info=turn_info,
                env_factory=env_factory,
                progress_callback=progress,
                history_fn=history_fn,
                prompt_budget=prompt_budget,
                format_reminder=actor_format_reminder,
                on_skipped_trajectory=_on_skipped,
            )
            for result in results:
                task_idx = result.metadata["task_index"]
                if (
                    task_idx not in attempted_task_names
                    or attempted_task_names[task_idx] is None
                ):
                    attempted_task_names[task_idx] = _task_name_for_result(result)
            all_trajectory_results.extend(results)
            if final_path is not None:
                try:
                    checkpoint_returns = [
                        trajectory_return(tr.trajectory, reward_signal_name)
                        for tr in all_trajectory_results
                    ]
                    checkpoint = Dataset(
                        evaluation_points=[],
                        trajectory_returns=checkpoint_returns,
                        trajectory_results=all_trajectory_results,
                        config=_build_dataset_config(len(all_trajectory_results)),
                        metadata=_build_dataset_metadata(
                            len(all_trajectory_results),
                            num_requested=attempted_trajectories,
                            checkpoint=True,
                        ),
                    )
                    save_dataset(checkpoint, final_path.with_suffix(".checkpoint.pkl"))
                except Exception:
                    logger.warning("Failed to save checkpoint", exc_info=True)
            check_abort_threshold(
                skipped_trajectory_count,
                attempted_trajectories,
                config.max_aborted_points,
                "pass@k collection",
            )
            done = batch_idx + 1
            logger.info(
                "Pass@k batches: %d/%d (%.1f%%)",
                done,
                num_batches,
                done / num_batches * 100,
            )
    except BaseException:
        exc = sys.exc_info()[1]
        if attempted_trajectories > 0 and final_path is not None:
            try:
                partial_results = _compute_results(
                    all_trajectory_results,
                    config.k_values,
                    requested_counts=dict(attempted_by_task),
                    requested_task_names=attempted_task_names,
                    reward_signal_name=reward_signal_name,
                    success_threshold=config.success_threshold,
                )
                partial_metadata = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "context_name": context_name,
                    "env_name": env_name,
                    "adapter": adapter,
                    "backend_name": backend_name,
                    "model": backend_config.model,
                    "seed": config.seed,
                    "samples_per_task": config.samples_per_task,
                    "k_values": list(config.k_values),
                    "success_threshold": config.success_threshold,
                    "reward_signal_name": reward_signal_name,
                    "num_tasks": len(all_task_indices),
                    "total_trajectories": len(all_trajectory_results),
                    "requested_trajectories": attempted_trajectories,
                    "retained_trajectories": len(all_trajectory_results),
                    "dropped_trajectories": skipped_trajectory_count,
                    "drop_rate": (
                        skipped_trajectory_count / attempted_trajectories
                        if attempted_trajectories > 0
                        else 0.0
                    ),
                    "partial": True,
                    "partial_reason": str(exc),
                }
                _print_summary(partial_results, partial_metadata)
                with final_path.open("w") as f:
                    json.dump(
                        {
                            "metadata": partial_metadata,
                            **partial_results,
                            "config": dataclasses.asdict(config),
                        },
                        f,
                        indent=2,
                        default=str,
                    )
                logger.info("Partial results saved to %s", final_path)
            except Exception:
                logger.exception("Failed to save partial Pass@k results")

            try:
                traj_returns = [
                    trajectory_return(tr.trajectory, reward_signal_name)
                    for tr in all_trajectory_results
                ]
                partial_dataset = Dataset(
                    evaluation_points=[],
                    trajectory_returns=traj_returns,
                    config=_build_dataset_config(len(all_trajectory_results)),
                    metadata=_build_dataset_metadata(
                        len(all_trajectory_results),
                        num_requested=attempted_trajectories,
                        partial=True,
                        partial_reason=str(exc),
                    ),
                    trajectory_results=all_trajectory_results,
                )
                save_dataset(partial_dataset, final_path.with_suffix(".pkl"))
                logger.info(
                    "Partial dataset saved to %s", final_path.with_suffix(".pkl")
                )
            except Exception:
                logger.exception("Failed to save partial dataset")

            try:
                observable = [
                    from_trajectory_result(tr, reward_signal_name)
                    for tr in all_trajectory_results
                ]
                trajectories_path = final_path.with_name(
                    final_path.stem + "_trajectories.json"
                )
                save_trajectories(
                    observable,
                    str(trajectories_path),
                    metadata={
                        "environment": env_name,
                        "source": "compute_pass_at_k",
                        "partial": True,
                    },
                )
                logger.info(
                    "Partial trajectories saved to %s (%d)",
                    trajectories_path,
                    len(observable),
                )
            except Exception:
                logger.exception("Failed to save partial trajectories")

        if log_path is not None:
            try:
                exp_logger.save_logs(log_path)
            except Exception:
                logger.exception("Failed to save logs on crash")
        raise
    finally:
        try:
            backend.close()
        except Exception as exc:
            logger.warning("Backend cleanup failed: %s", exc)
        resource_summary = resource_monitor.stop()

    elapsed = time.time() - start_time
    logger.info(
        "Collection complete: %d trajectories in %.1fs",
        len(all_trajectory_results),
        elapsed,
    )

    # Log trajectory summary and compute token stats
    if all_trajectory_results:
        summary = summarize_trajectory_results(all_trajectory_results)
        exp_logger.log_event("trajectory_summary", summary)

    token_stats = compute_stats(exp_logger, "pass_at_k_collection")
    exp_logger.log_event("stats", token_stats)
    if resource_summary:
        exp_logger.log_event("resource_summary", resource_summary)
    if token_stats["truncated_generations_pct"] > 10:
        logger.warning(
            "High truncation rate: %d/%d generations truncated (%.1f%%)",
            token_stats["truncated_generations"],
            token_stats["total_generations"],
            token_stats["truncated_generations_pct"],
        )

    # 10. Compute Pass@k
    pass_at_k_results = _compute_results(
        all_trajectory_results,
        config.k_values,
        requested_counts=dict(attempted_by_task),
        requested_task_names=attempted_task_names,
        reward_signal_name=reward_signal_name,
        success_threshold=config.success_threshold,
    )

    # 11. Build metadata
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "context_name": context_name,
        "env_name": env_name,
        "adapter": adapter,
        "backend_name": backend_name,
        "model": backend_config.model,
        "seed": config.seed,
        "samples_per_task": config.samples_per_task,
        "k_values": list(config.k_values),
        "success_threshold": config.success_threshold,
        "reward_signal_name": reward_signal_name,
        "num_tasks": len(all_task_indices),
        "total_trajectories": len(all_trajectory_results),
        "requested_trajectories": attempted_trajectories,
        "retained_trajectories": len(all_trajectory_results),
        "dropped_trajectories": skipped_trajectory_count,
        "drop_rate": (
            skipped_trajectory_count / attempted_trajectories
            if attempted_trajectories > 0
            else 0.0
        ),
        "elapsed_seconds": round(elapsed, 2),
        "token_usage": {
            "total_prompt_tokens": token_stats["total_prompt_tokens"],
            "total_completion_tokens": token_stats["total_completion_tokens"],
            "total_generations": token_stats["total_generations"],
            "truncated_generations": token_stats["truncated_generations"],
            "truncated_generations_pct": token_stats["truncated_generations_pct"],
        },
    }

    # 12. Print console summary
    _print_summary(pass_at_k_results, metadata)

    # 13. Save all artifacts
    if final_path is not None:
        # 12a. Pass@k JSON report
        output = {
            "metadata": metadata,
            **pass_at_k_results,
            "config": dataclasses.asdict(config),
        }
        with final_path.open("w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Results saved to %s", final_path)

        # 12b. Dataset pickle (for reuse with resample_points.py)
        traj_returns = [
            trajectory_return(tr.trajectory, reward_signal_name)
            for tr in all_trajectory_results
        ]
        dataset = Dataset(
            evaluation_points=[],
            trajectory_returns=traj_returns,
            config=_build_dataset_config(len(all_trajectory_results)),
            metadata=_build_dataset_metadata(
                len(all_trajectory_results),
                num_requested=attempted_trajectories,
                elapsed_seconds=elapsed,
            ),
            trajectory_results=all_trajectory_results,
        )
        dataset_path = final_path.with_suffix(".pkl")
        save_dataset(dataset, dataset_path)
        logger.info(
            "Dataset saved to %s (%d trajectories)",
            dataset_path,
            len(all_trajectory_results),
        )

        # 12c. Observable trajectories JSON
        observable = [
            from_trajectory_result(tr, reward_signal_name)
            for tr in all_trajectory_results
        ]
        trajectories_path = final_path.with_name(final_path.stem + "_trajectories.json")
        save_trajectories(
            observable,
            str(trajectories_path),
            metadata={"environment": env_name, "source": "compute_pass_at_k"},
        )
        logger.info("Trajectories saved to %s (%d)", trajectories_path, len(observable))

        checkpoint_path = final_path.with_suffix(".checkpoint.pkl")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Removed checkpoint %s", checkpoint_path)
    else:
        logger.warning(
            "No output_path configured; results printed to console only. "
            "Trajectories and dataset NOT saved."
        )


if __name__ == "__main__":
    main()

"""Step 1: Collect dataset — trajectories and evaluation points.

Runs the environment to collect trajectories, extracts evaluation points,
and saves a Dataset pickle. GT estimation is deferred to ``predict.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import dataclasses
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from qval import (
    BackendsRegistryConfig,
    PipelineCollectionConfig,
    load_environment_context,
)
from qval.benchmark import collect_trajectories
from qval.data_cache import Dataset, save_dataset
from qval.error_handling import check_abort_threshold
from qval.evaluation_points import collect_evaluation_points
from qval.experiment_logging import (
    ExperimentLogger,
    LoggingBackend,
    compute_stats,
    summarize_trajectory_results,
)
from qval.quota_retry_backend import wrap_with_quota_retry
from qval.rollout import trajectory_return
from qval.run_checkpoint import (
    RESUME_FINGERPRINT_SCHEMA_VERSION,
    backend_content_descriptor,
    build_resume_fingerprint,
    checkpoint_path as resume_checkpoint_path,
    collection_completed_counts,
    content_make_kwargs,
    environment_context_descriptor,
    load_rolling_checkpoint,
    save_rolling_checkpoint,
    subtract_completed_tasks,
)
from qval.second_elicitation_backend import wrap_with_second_elicitation
from qval.script_utils import (
    attach_backend_logging_metadata,
    build_env_factory,
    build_history_fn,
    build_prompt_budget,
    chat_template_kwargs_from_config,
    create_adapter_env,
    create_backend_from_config,
    inspect_harbor_runtime_eligibility,
    inspect_harbor_snapshot_filter,
    resolve_backend_name,
    resolve_effective_max_steps,
    resolve_env_params,
    resolve_ranking_system_prompt,
    resolve_system_prompt,
    resolve_task_indices,
    serialize_harbor_snapshot_filter_report,
    sampling_params_from_config,
)
from qval.trajectory_store import from_trajectory_result, save_trajectories
from qval.types import EvaluationPoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_MISSING = object()


def _configure_collection_debug_logging(enabled: bool) -> None:
    """Enable verbose collection debug logs for local packages."""
    if not enabled:
        return
    for name in ("qval", "llenvs", __name__):
        logging.getLogger(name).setLevel(logging.DEBUG)
    logger.info("Collection debug logging enabled")


def _harbor_hidden_risk(hidden: Any) -> tuple[bool, tuple[str, ...]]:
    """Return whether a Harbor hidden state carries filesystem-restore risk."""
    if hidden is None:
        return False, ()
    risk_now = bool(getattr(hidden, "fs_restore_risk_now", False))
    risk_ever = bool(getattr(hidden, "fs_restore_risk_ever", False))
    reasons = tuple(
        str(reason)
        for reason in getattr(hidden, "fs_restore_risk_reasons", ())
        if str(reason)
    )
    return (risk_now or risk_ever), reasons


def _find_risky_harbor_trajectories(
    trajectory_results: list[Any],
) -> list[dict[str, Any]]:
    """Collect per-trajectory runtime-probe risk details from Harbor results."""
    risky: list[dict[str, Any]] = []
    for trajectory_index, result in enumerate(trajectory_results):
        transitions = getattr(getattr(result, "trajectory", None), "transitions", ())
        reasons: set[str] = set()
        task_index = None
        task_name = None

        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            task_index = metadata.get("task_index")
            task_name = metadata.get("task_name")

        for transition in transitions:
            for state_like in (
                getattr(transition, "state", None),
                getattr(transition, "next_state", None),
            ):
                hidden = getattr(state_like, "hidden", None)
                if task_index is None:
                    task_index = getattr(hidden, "task_index", None)
                if task_name is None:
                    task_name = getattr(hidden, "task_name", None)
                has_risk, hidden_reasons = _harbor_hidden_risk(hidden)
                if has_risk:
                    reasons.update(hidden_reasons)

        if reasons:
            risky.append(
                {
                    "trajectory_index": trajectory_index,
                    "task_index": task_index,
                    "task_name": task_name,
                    "transition_count": len(transitions),
                    "reasons": tuple(sorted(reasons)),
                }
            )
    return risky


# Runtime knobs that don't affect dataset content — comparing them across
# resume would block legitimate ranking-resume runs after a timeout bump.
_RESUME_MAKE_KWARGS_IGNORE = frozenset({"browsergym_call_timeout"})


def _normalize_resume_make_kwargs(value: Any) -> Any:
    """Normalize stored make kwargs for resume-compatibility checks."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {k: v for k, v in value.items() if k not in _RESUME_MAKE_KWARGS_IGNORE}
    return value


def _validate_resume_dataset_compatibility(
    metadata: dict[str, Any],
    *,
    env_name: str,
    adapter: str,
    max_steps: int | None,
    make_kwargs: dict[str, Any] | None,
) -> None:
    """Fail fast if a ranking-resume dataset targets a different environment setup."""
    mismatches: list[str] = []
    expected_max_steps = resolve_effective_max_steps(max_steps, make_kwargs)
    if "max_steps" in metadata or "make_kwargs" in metadata:
        actual_max_steps = resolve_effective_max_steps(
            metadata.get("max_steps"),
            metadata.get("make_kwargs"),
        )
        if actual_max_steps != expected_max_steps:
            mismatches.append(
                "max_steps: "
                f"resume={actual_max_steps!r}, current={expected_max_steps!r}"
            )
    expected_by_key = {
        "env_name": env_name,
        "adapter": adapter,
        "make_kwargs": _normalize_resume_make_kwargs(make_kwargs),
    }

    for key, expected in expected_by_key.items():
        actual = metadata.get(key, _MISSING)
        if actual is _MISSING:
            continue
        if key == "make_kwargs":
            actual = _normalize_resume_make_kwargs(actual)
        if actual != expected:
            mismatches.append(f"{key}: resume={actual!r}, current={expected!r}")

    if mismatches:
        raise ValueError(
            "resume_ranking_from dataset is incompatible with current "
            f"collection config: {'; '.join(mismatches)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Collect trajectories and evaluation points"
    )
    parser.add_argument(
        "--config", required=True, help="Path to collection YAML config"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory path (overrides config dataset_path dir)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override tensor_parallel_size for all vLLM backends (e.g. from SLURM GPU count)",
    )
    parser.add_argument(
        "--max-aborted-points",
        type=float,
        default=None,
        help="Abort if the fraction of dropped trajectories exceeds this threshold",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore (and remove) any existing resume checkpoint and start over",
    )

    args = parser.parse_args()

    from qval.error_handling import install_sigfpe_handler

    install_sigfpe_handler()

    # 1. Load config
    config_path = Path(args.config)
    with config_path.open() as f:
        raw_config = yaml.safe_load(f) or {}

    config = PipelineCollectionConfig.from_dict(raw_config)
    if args.max_aborted_points is not None:
        config = dataclasses.replace(config, max_aborted_points=args.max_aborted_points)
    if args.fresh:
        config = dataclasses.replace(config, resume=False)
    backends_config = BackendsRegistryConfig.from_yaml(config.backends_config_path)

    collection_config = config.collection
    _configure_collection_debug_logging(collection_config.debug)
    # 2. Load context and create env
    context_name = collection_config.context_name or collection_config.env_name
    if not context_name:
        raise ValueError("Context name or environment name must be specified in config")
    context = load_environment_context(
        env_name=context_name,
        contexts_dir=collection_config.contexts_dir,
    )

    params = resolve_env_params(
        context,
        env_name_override=collection_config.env_name,
        adapter_override=collection_config.adapter,
        max_steps_override=collection_config.max_steps,
        make_kwargs_override=collection_config.make_kwargs,
        step_penalty_override=collection_config.step_penalty,
    )
    env_name = params.env_name
    adapter = params.adapter
    max_steps = params.max_steps
    make_kwargs = params.make_kwargs
    env_make_kwargs = params.env_make_kwargs
    step_penalty = params.step_penalty
    extra_rewards = params.extra_rewards
    reward_signal_name = params.reward_signal_name

    if step_penalty is not None:
        logger.info(
            "Step penalty: %.4f (reward_signal_name overridden to None)",
            step_penalty,
        )

    logger.info("Config: batch_size=%d", config.batch_size)

    # Apply per-environment turn-discard overrides to points config
    _discard_overrides: dict[str, int] = {}
    if context.early_turns_to_discard is not None:
        _discard_overrides["early_turns_to_discard"] = context.early_turns_to_discard
    if context.late_turns_to_discard is not None:
        _discard_overrides["late_turns_to_discard"] = context.late_turns_to_discard
    if _discard_overrides:
        config = dataclasses.replace(
            config,
            points_config=dataclasses.replace(
                config.points_config, **_discard_overrides
            ),
        )

    explicit_task_indices = config.points_config.task_indices
    requested_num_trajectories = config.points_config.num_trajectories

    base_path = config.dataset_path
    if args.output_dir:
        base_path = str(Path(args.output_dir) / "dataset.pkl")
    dataset_path = _timestamped_path(
        base_path,
        "dataset",
        suffix=".pkl",
    )
    snapshot_artifact_root: Path | None = None
    harbor_snapshot_filter_report: dict[str, Any] | None = None
    if (
        adapter == "harbor"
        and collection_config.harbor_state_capture == "snapshot_exact"
    ):
        snapshot_artifact_root = Path(dataset_path).with_name(
            f"{Path(dataset_path).stem}_snapshots"
        )
        environment_type = str(env_make_kwargs.get("environment_type", "docker"))
        harbor_tasks, _eligibility, snapshot_report = inspect_harbor_snapshot_filter(
            env_name,
            dataset_path=env_make_kwargs.get("dataset_path"),
            environment_type=environment_type,
            task_indices=explicit_task_indices,
            num_trajectories=requested_num_trajectories,
            difficulties=env_make_kwargs.get("difficulties"),
        )
        harbor_snapshot_filter_report = serialize_harbor_snapshot_filter_report(
            snapshot_report
        )
        env_make_kwargs["tasks"] = harbor_tasks
        env_make_kwargs["state_capture_mode"] = "snapshot_exact"
        env_make_kwargs["snapshot_artifact_root"] = str(snapshot_artifact_root)
        logger.info(
            "Harbor snapshot eligibility (%s): kept %d/%d tasks, filtered %d",
            environment_type,
            len(snapshot_report.selected_task_indices),
            snapshot_report.total_tasks,
            len(snapshot_report.filtered_task_indices),
        )
        if snapshot_report.reason_counts:
            logger.info(
                "Harbor snapshot filter reasons: %s",
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in snapshot_report.reason_counts.items()
                ),
            )

    # Runtime eligibility filter (e.g., apptainer-hpc cannot run compose tasks)
    runtime_filter_report: dict[str, Any] | None = None
    if adapter == "harbor" and harbor_snapshot_filter_report is None:
        environment_type = str(env_make_kwargs.get("environment_type", "docker"))
        sif_cache_dir = env_make_kwargs.get("sif_cache_dir")
        runtime_selected, runtime_filter_report = inspect_harbor_runtime_eligibility(
            env_name,
            dataset_path=env_make_kwargs.get("dataset_path"),
            environment_type=environment_type,
            sif_cache_dir=sif_cache_dir,
            task_indices=explicit_task_indices,
            num_trajectories=requested_num_trajectories,
            difficulties=env_make_kwargs.get("difficulties"),
        )
        if runtime_filter_report["filtered_count"] > 0:
            logger.info(
                "Harbor runtime eligibility (%s): kept %d/%d tasks, filtered %d",
                environment_type,
                runtime_filter_report["eligible_count"],
                runtime_filter_report["total_tasks"],
                runtime_filter_report["filtered_count"],
            )
            if runtime_filter_report["reason_counts"]:
                logger.info(
                    "Harbor runtime filter reasons: %s",
                    ", ".join(
                        f"{reason}={count}"
                        for reason, count in runtime_filter_report[
                            "reason_counts"
                        ].items()
                    ),
                )

    # Compute env_size from config
    if adapter == "harbor" and harbor_snapshot_filter_report is not None:
        env_size = len(harbor_snapshot_filter_report["selected_task_indices"])
    elif adapter == "harbor" and runtime_filter_report is not None:
        env_size = len(runtime_filter_report["selected_task_indices"])
    elif explicit_task_indices is not None:
        env_size = max(explicit_task_indices) + 1
    else:
        env_size = requested_num_trajectories

    env, adapter_system_prompt = create_adapter_env(
        adapter,
        env_name,
        env_size=env_size,
        seed=collection_config.seed,
        max_steps=max_steps,
        answer_extractor=context.environment_extractor,
        make_kwargs=env_make_kwargs,
        extra_rewards=extra_rewards,
        invalid_action_text=getattr(context, "invalid_action_text", None),
        invalid_action_observation=getattr(context, "invalid_action_observation", None),
        advance_on_invalid=getattr(context, "advance_on_invalid", None),
    )
    system_prompt = resolve_system_prompt(
        override_system_prompt=collection_config.system_prompt,
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

    ranking_env_factory = None
    ranking_restore_fn = None
    factory_result = build_env_factory(
        adapter,
        env_name,
        env_make_kwargs,
        max_steps=max_steps,
        extra_rewards=extra_rewards,
        answer_extractor=context.environment_extractor,
        runtime_probing=collection_config.runtime_probing,
    )
    env_factory = factory_result.env_factory
    harbor_kwargs = factory_result.harbor_kwargs
    harbor_preflight = factory_result.harbor_preflight
    # Generic restore_fn (currently only populated for gymnasium+use_images).
    # Harbor wires its own via create_harbor_env_factory below.
    env_restore_fn = factory_result.restore_fn

    logger.info(
        "Environment: %s (adapter=%s)",
        env_name,
        adapter,
    )

    # Compute task indices
    shuffle_tasks = config.points_config.shuffle_tasks
    if harbor_snapshot_filter_report is not None:
        all_task_indices = list(harbor_snapshot_filter_report["selected_task_indices"])
    elif runtime_filter_report is not None:
        all_task_indices = list(runtime_filter_report["selected_task_indices"])
    elif explicit_task_indices is not None:
        all_task_indices = list(explicit_task_indices)
    else:
        # Collection wants exactly ``requested_num_trajectories`` trajectories,
        # so bounded adapters cycle through the task pool on overflow.
        all_task_indices = resolve_task_indices(
            env,
            adapter=adapter,
            num_tasks=requested_num_trajectories,
            shuffle=shuffle_tasks,
            seed=collection_config.seed,
            overflow="cycle",
            logger=logger,
        )
        logger.info(
            "Selected %d task indices (shuffle=%s, seed=%s)",
            len(all_task_indices),
            shuffle_tasks,
            collection_config.seed,
        )

    # Build history function: content truncation (per-turn) + turn dropping
    history_fn = build_history_fn(
        context, max_history_turns=collection_config.max_history_turns
    )

    # Scripted-policy collection: when ``scripted_policy_name`` is set,
    # use a single ``ScriptedBackend`` (with optional epsilon-random mixing)
    # in place of any LLM/VLM actor. Skip the backends.yaml lookup entirely.
    use_scripted_policy = collection_config.scripted_policy_name is not None

    backend_names: list[str]
    if use_scripted_policy:
        # One synthetic "policy" name for logging/metadata. The actual
        # backend is constructed below in the policy loop.
        backend_names = [
            f"scripted:{collection_config.scripted_policy_name}"
            f"@eps={collection_config.scripted_policy_epsilon}"
        ]
    elif collection_config.backend_names:
        backend_names = list(collection_config.backend_names)
    else:
        backend_names = [
            resolve_backend_name(
                None, backends_config.backends, backends_config.default_backend
            )
        ]

    if not use_scripted_policy:
        for name in backend_names:
            resolve_backend_name(
                name, backends_config.backends, backends_config.default_backend
            )
    if collection_config.ranking_backend_names:
        for name in collection_config.ranking_backend_names:
            resolve_backend_name(
                name, backends_config.backends, backends_config.default_backend
            )

    # Partition task indices among policies
    num_policies = len(backend_names)
    per_policy = len(all_task_indices) // num_policies
    remainder = len(all_task_indices) % num_policies

    policies_meta: list[dict[str, Any]] = []
    all_trajectory_results = []
    attempted_trajectory_count = 0
    skipped_trajectory_count = 0
    turn_info = collection_config.turn_info

    exp_logger = ExperimentLogger()
    log_path = Path(dataset_path).with_suffix(".logs.jsonl")
    exp_logger.attach_jsonl_sink(log_path)
    if adapter == "harbor":
        exp_logger.log_event("harbor_preflight", harbor_preflight)
        if harbor_kwargs and harbor_kwargs.get("text_exec_mode") == "tmux_session":
            from qval.script_utils import preflight_harbor_tmux_sessions

            tmux_preflight = preflight_harbor_tmux_sessions(
                env_name,
                task_indices=tuple(all_task_indices),
                **harbor_kwargs,
            )
            exp_logger.log_event("harbor_tmux_preflight", tmux_preflight)
    start = time.time()

    # Start resource monitoring
    from qval.resource_monitor import ResourceMonitor

    trials_dir = os.environ.get("VB_TRIALS_DIR", "trials")
    if adapter == "harbor":
        from qval.script_utils import _apply_harbor_env_overrides

        effective_harbor_kwargs = _apply_harbor_env_overrides(harbor_kwargs)
        trials_dir = effective_harbor_kwargs.get("trials_dir", trials_dir)
    resource_monitor = ResourceMonitor(watch_dirs=[trials_dir])
    resource_monitor.start()

    # Build config_dict early so it's available for partial saves on crash
    config_dict = {
        "correlation_methods": [m.name.lower() for m in config.correlation_methods],
        "reward_signal_name": reward_signal_name,
        "batch_size": config.batch_size,
        "include_actor_thinking": config.include_actor_thinking,
        "max_aborted_points": config.max_aborted_points,
        "resume_ranking_subset": (
            None
            if config.resume_ranking_subset is None
            else {
                "count": config.resume_ranking_subset.count,
                "seed": config.resume_ranking_subset.seed,
            }
        ),
        "points_config": {
            "num_trajectories": config.points_config.num_trajectories,
            "max_points_per_trajectory": config.points_config.max_points_per_trajectory,
            "task_indices": list(config.points_config.task_indices)
            if config.points_config.task_indices
            else None,
            "sampling_strategy": config.points_config.sampling_strategy,
            "early_turns_to_discard": config.points_config.early_turns_to_discard,
            "late_turns_to_discard": config.points_config.late_turns_to_discard,
        },
    }
    resume_ranking_subset_config = config_dict["resume_ranking_subset"]

    def _on_skipped(
        skipped: dict[str, Any],
        phase: str,
        backend_name: str,
    ) -> None:
        nonlocal skipped_trajectory_count
        _log_skipped_trajectory_event(
            exp_logger,
            phase=phase,
            backend_name=backend_name,
            skipped=skipped,
        )
        skipped_trajectory_count += 1

    points: list[Any] = []
    traj_returns: list[float] = []
    ranking_points: list[Any] = []
    ranking_k = collection_config.ranking_actions
    ranking_sampling_backend_name: str | None = None
    ranking_sampling_backend_names: list[str] | None = None
    ranking_input_points: list[Any] = []
    ranking_point_selection_metadata: dict[str, Any] | None = None
    resume_path: Path | None = None
    existing: Dataset | None = None

    def _drop_rate() -> float:
        if attempted_trajectory_count <= 0:
            return 0.0
        return skipped_trajectory_count / attempted_trajectory_count

    def _discard_risky_harbor_trajectories(
        results: list[Any],
        *,
        trajectory_offset: int,
    ) -> list[Any]:
        nonlocal skipped_trajectory_count
        if not (
            adapter == "harbor"
            and collection_config.runtime_probing
            and collection_config.discard_runtime_probe_risky_trajectories
            and results
        ):
            return results

        risky_trajectories = _find_risky_harbor_trajectories(results)
        if not risky_trajectories:
            return results

        risky_indices = {item["trajectory_index"] for item in risky_trajectories}
        for item in risky_trajectories:
            logger.warning(
                "Discarding Harbor trajectory %d (task_index=%s, task_name=%s, transitions=%d) "
                "due to runtime probe risk: %s",
                trajectory_offset + item["trajectory_index"],
                item["task_index"],
                item["task_name"],
                item["transition_count"],
                ", ".join(item["reasons"]),
            )

        kept_results = [
            result for idx, result in enumerate(results) if idx not in risky_indices
        ]
        skipped_trajectory_count += len(risky_trajectories)
        logger.warning(
            "Discarded %d Harbor trajectories due to runtime probe risk; retained %d",
            len(risky_trajectories),
            len(kept_results),
        )
        return kept_results

    def _compute_trajectory_returns(
        trajectory_results: list[Any],
        evaluation_points: list[Any],
    ) -> list[float]:
        returns = [
            trajectory_return(
                tr.trajectory,
                reward_signal_name,
                config.discount_factor,
            )
            for tr in trajectory_results
        ]
        if (
            config.points_config.max_trajectories_to_keep is not None
            and evaluation_points
        ):
            kept_traj = {pt.trajectory_index for pt in evaluation_points}
            returns = [r for i, r in enumerate(returns) if i in kept_traj]
        return returns

    def _build_dataset_metadata(
        *,
        partial: bool,
        elapsed_seconds: float | None = None,
        partial_reason: str | None = None,
    ) -> dict[str, Any]:
        metadata_timestamp = datetime.now().astimezone().isoformat()
        collected_task_indices = [
            tr.metadata.get("task_index") for tr in all_trajectory_results
        ]

        if config.resume_ranking_from:
            metadata = dict(existing.metadata) if existing is not None else {}
            if "num_trajectories" not in metadata:
                metadata["num_trajectories"] = len(all_trajectory_results)
            if "collected_task_indices" not in metadata:
                metadata["collected_task_indices"] = collected_task_indices
            if resume_path is not None:
                metadata["resumed_ranking_from"] = str(resume_path)
            if elapsed_seconds is not None:
                metadata["ranking_collection_time_seconds"] = round(elapsed_seconds, 1)
            metadata["ranking_collection_timestamp"] = metadata_timestamp
        else:
            metadata = {
                "env_name": env_name,
                "adapter": adapter,
                "seed": collection_config.seed,
                "policies": policies_meta,
                "backend_names": backend_names,
                "num_attempted_trajectories": attempted_trajectory_count,
                "num_trajectories": len(all_trajectory_results),
                "num_skipped_trajectories": skipped_trajectory_count,
                "drop_rate": _drop_rate(),
                "task_indices": all_task_indices,
                "collected_task_indices": collected_task_indices,
                "system_prompt": system_prompt,
                "max_steps": resolve_effective_max_steps(max_steps, make_kwargs),
                "make_kwargs": make_kwargs,
                "context_name": context_name,
                "contexts_dir": collection_config.contexts_dir,
                "discount_factor": config.discount_factor,
                "step_penalty": step_penalty,
                "turn_info": turn_info,
                "harbor_state_capture": collection_config.harbor_state_capture,
                "resumed_trajectories": resumed_trajectory_count,
                "resume_fingerprint": resume_fingerprint,
            }
            if elapsed_seconds is not None:
                metadata["collection_time_seconds"] = round(elapsed_seconds, 1)
            metadata["collection_timestamp"] = metadata_timestamp

        if (
            not config.resume_ranking_from
            or "max_steps" in metadata
            or "make_kwargs" in metadata
        ):
            metadata["max_steps"] = resolve_effective_max_steps(
                metadata.get("max_steps"),
                metadata.get("make_kwargs"),
            )

        if harbor_snapshot_filter_report is not None:
            metadata["harbor_snapshot_filter"] = harbor_snapshot_filter_report
        if snapshot_artifact_root is not None:
            metadata["snapshot_artifact_dir"] = snapshot_artifact_root.name
        if ranking_point_selection_metadata is not None:
            metadata["ranking_point_selection"] = ranking_point_selection_metadata
        if ranking_k is not None:
            metadata["ranking_actions"] = ranking_k
            metadata["ranking_sampling_mode"] = collection_config.ranking_sampling_mode
            metadata["ranking_sampling_strategy"] = (
                collection_config.ranking_sampling_strategy
            )
            if collection_config.ranking_sampler_name:
                metadata["ranking_sampler_name"] = (
                    collection_config.ranking_sampler_name
                )
            if ranking_sampling_backend_name is not None:
                metadata["ranking_sampling_backend_name"] = (
                    ranking_sampling_backend_name
                )
            if ranking_sampling_backend_names is not None:
                metadata["ranking_sampling_backend_names"] = (
                    ranking_sampling_backend_names
                )
            if collection_config.ranking_sampling is not None:
                metadata["ranking_sampling"] = dataclasses.asdict(
                    collection_config.ranking_sampling
                )
        if partial:
            metadata["partial"] = True
            if partial_reason is not None:
                metadata["partial_reason"] = partial_reason
        return metadata

    def _build_dataset(
        *,
        evaluation_points: list[Any],
        trajectory_returns: list[float],
        partial: bool,
        partial_reason: str | None = None,
        elapsed_seconds: float | None = None,
        ranking_points_value: list[Any] | None = None,
    ) -> Dataset:
        return Dataset(
            evaluation_points=evaluation_points,
            trajectory_returns=trajectory_returns,
            config=config_dict,
            metadata=_build_dataset_metadata(
                partial=partial,
                elapsed_seconds=elapsed_seconds,
                partial_reason=partial_reason,
            ),
            ranking_points=ranking_points
            if ranking_points_value is None
            else ranking_points_value,
            trajectory_results=all_trajectory_results,
        )

    offset = 0
    created_backends: list[LoggingBackend] = []
    backend: LoggingBackend | None = None
    backend_config = None
    resume_ckpt_path: Path | None = None
    resume_fingerprint: str | None = None
    resumed_trajectory_count = 0
    completed_by_policy: dict[int, Counter[int]] = {}
    try:
        if config.resume_ranking_from:
            from qval.data_cache import load_dataset, resolve_dataset_path

            resume_path = resolve_dataset_path(config.resume_ranking_from)
            existing = load_dataset(resume_path)
            _validate_resume_dataset_compatibility(
                existing.metadata,
                env_name=env_name,
                adapter=adapter,
                max_steps=max_steps,
                make_kwargs=make_kwargs,
            )

            points = existing.evaluation_points
            all_trajectory_results = existing.trajectory_results or []
            traj_returns = existing.trajectory_returns
            policies_meta = existing.metadata.get("policies", [])
            backend_names = existing.metadata.get("backend_names", backend_names)
            config_dict = dict(existing.config)
            existing_points_config = config_dict.get("points_config")
            if isinstance(existing_points_config, dict):
                config_dict["points_config"] = dict(existing_points_config)
            config_dict["resume_ranking_subset"] = resume_ranking_subset_config

            if not points:
                raise ValueError(
                    f"Resumed dataset has no evaluation points: {resume_path}"
                )

            logger.info(
                "Resumed from %s: %d points, %d trajectories",
                resume_path,
                len(points),
                len(all_trajectory_results),
            )
            ranking_input_points = points
            if config.resume_ranking_subset is not None:
                subset = config.resume_ranking_subset
                effective_seed = (
                    subset.seed if subset.seed is not None else collection_config.seed
                )
                if subset.count > len(points):
                    raise ValueError(
                        "resume_ranking_subset.count exceeds available evaluation "
                        f"points: {subset.count} > {len(points)}"
                    )
                selected_point_indices = sorted(
                    random.Random(effective_seed).sample(
                        range(len(points)),
                        subset.count,
                    )
                )
                ranking_input_points = [points[i] for i in selected_point_indices]
                ranking_point_selection_metadata = {
                    "mode": "random_sample",
                    "count": subset.count,
                    "effective_seed": effective_seed,
                    "source_num_points": len(points),
                    "selected_point_indices": selected_point_indices,
                }
                logger.info(
                    "Selected %d/%d resumed evaluation points for ranking (seed=%d)",
                    len(ranking_input_points),
                    len(points),
                    effective_seed,
                )

            # For Harbor snapshot mode, fix snapshot_artifact_root to point
            # at the original dataset's snapshots (not the new dataset_path).
            if snapshot_artifact_root is not None:
                resumed_snap_dir = existing.metadata.get("snapshot_artifact_dir")
                if resumed_snap_dir:
                    snapshot_artifact_root = resume_path.parent / resumed_snap_dir

            # If no explicit ranking backends, create one from config so
            # ranking has a backend (normally the last collection backend
            # is reused). Skip when the resumed dataset was collected with
            # a scripted policy: its synthetic actor name
            # (scripted:<name>@eps=...) isn't in backends.yaml, and ranking
            # in this case uses a manual sampler or explicit ranking
            # backends rather than the actor.
            if not collection_config.ranking_backend_names and not use_scripted_policy:
                _fallback_name = resolve_backend_name(
                    backend_names[0] if backend_names else None,
                    backends_config.backends,
                    backends_config.default_backend,
                )
                backend_config = backends_config.backends[_fallback_name]
                chat_template_kwargs = chat_template_kwargs_from_config(backend_config)
                raw_backend = create_backend_from_config(
                    backend_config,
                    chat_template_kwargs=chat_template_kwargs,
                    tensor_parallel_size_override=args.tensor_parallel_size,
                )
                attach_backend_logging_metadata(raw_backend, backend_config)
                quota_wrapped = wrap_with_quota_retry(
                    raw_backend, policy=backend_config.quota_retry_policy
                )
                backend = wrap_with_second_elicitation(
                    LoggingBackend(quota_wrapped, exp_logger)
                )
                created_backends.append(backend)
        else:
            # Resume fingerprint over content-affecting inputs only: anything
            # that changes trajectory content or the task partition. Transport
            # and chunking knobs (batch_size, restore_concurrency,
            # max_aborted_points, output paths) are deliberately excluded. The
            # resolved all_task_indices captures task_indices/num_trajectories/
            # shuffle/cycling and any harbor filtering.
            resolved_turn_info = (
                turn_info
                if turn_info is not None
                else (context.turn_info if context.turn_info is not None else False)
            )
            policy_descriptors: list[dict[str, Any]] = []
            for name in backend_names:
                if use_scripted_policy:
                    policy_descriptors.append(
                        {
                            "kind": "scripted",
                            "name": collection_config.scripted_policy_name,
                            "epsilon": collection_config.scripted_policy_epsilon,
                            "seed": collection_config.scripted_policy_seed,
                        }
                    )
                else:
                    resolved_policy_name = resolve_backend_name(
                        name,
                        backends_config.backends,
                        backends_config.default_backend,
                    )
                    policy_descriptors.append(
                        {
                            "kind": "backend",
                            "name": resolved_policy_name,
                            "config": backend_content_descriptor(
                                backends_config.backends[resolved_policy_name]
                            ),
                        }
                    )
            resume_fingerprint = build_resume_fingerprint(
                {
                    "schema_version": RESUME_FINGERPRINT_SCHEMA_VERSION,
                    "pipeline": "collection",
                    "context_name": context_name,
                    "env": {
                        "env_name": env_name,
                        "adapter": adapter,
                        "max_steps": resolve_effective_max_steps(
                            max_steps, make_kwargs
                        ),
                        "step_penalty": step_penalty,
                        "reward_signal_name": reward_signal_name,
                        "make_kwargs": content_make_kwargs(make_kwargs),
                        "harbor_state_capture": (
                            collection_config.harbor_state_capture
                        ),
                        "runtime_probing": collection_config.runtime_probing,
                        "discard_runtime_probe_risky_trajectories": (
                            collection_config.discard_runtime_probe_risky_trajectories
                        ),
                    },
                    "env_context": environment_context_descriptor(context),
                    "system_prompt": system_prompt,
                    "seed": collection_config.seed,
                    "turn_info": resolved_turn_info,
                    "max_history_turns": collection_config.max_history_turns,
                    "all_task_indices": all_task_indices,
                    "policies": policy_descriptors,
                }
            )
            base = Path(base_path)
            resume_ckpt_path = resume_checkpoint_path(
                base.parent, base.stem, resume_fingerprint
            )
            if config.resume:
                loaded = load_rolling_checkpoint(
                    resume_ckpt_path, expected_fingerprint=resume_fingerprint
                )
                if loaded is not None:
                    counts = collection_completed_counts(loaded.trajectory_results)
                    if counts is None:
                        logger.warning(
                            "Resume checkpoint %s lacks per-trajectory "
                            "provenance; starting fresh",
                            resume_ckpt_path,
                        )
                    else:
                        all_trajectory_results = list(loaded.trajectory_results)
                        completed_by_policy = counts
                        resumed_trajectory_count = len(all_trajectory_results)
                        logger.info(
                            "Resumed %d collected trajectories from %s",
                            resumed_trajectory_count,
                            resume_ckpt_path,
                        )
                        exp_logger.log_event(
                            "resume",
                            {
                                "resumed_trajectories": resumed_trajectory_count,
                                "checkpoint": str(resume_ckpt_path),
                            },
                        )
            else:
                resume_ckpt_path.unlink(missing_ok=True)

            for policy_idx, backend_name in enumerate(backend_names):
                # Distribute tasks: first `remainder` policies get one extra
                count = per_policy + (1 if policy_idx < remainder else 0)
                task_indices = all_task_indices[offset : offset + count]
                offset += count

                if not task_indices:
                    continue

                completed = completed_by_policy.get(policy_idx)
                remaining_indices = (
                    subtract_completed_tasks(task_indices, completed)
                    if completed
                    else list(task_indices)
                )
                full_task_indices = task_indices
                if not remaining_indices:
                    # Everything for this policy was loaded from the resume
                    # checkpoint — skip it entirely, including backend creation.
                    resolved_name = (
                        backend_name
                        if use_scripted_policy
                        else resolve_backend_name(
                            backend_name,
                            backends_config.backends,
                            backends_config.default_backend,
                        )
                    )
                    policies_meta.append(
                        {
                            "backend_name": resolved_name,
                            "num_trajectories": sum(completed.values()),
                            "task_indices": full_task_indices,
                            "resumed": True,
                        }
                    )
                    logger.info(
                        "Policy %d/%d fully resumed from checkpoint (%d trajectories)",
                        policy_idx + 1,
                        num_policies,
                        sum(completed.values()),
                    )
                    continue
                task_indices = remaining_indices

                # Close previous backend before creating a new one to free GPU memory
                if backend is not None:
                    try:
                        backend.close()
                    except Exception as exc:
                        logger.warning("Backend cleanup failed: %s", exc)

                logger.info(
                    "Policy %d/%d: tasks=%d (indices %d-%d)",
                    policy_idx + 1,
                    num_policies,
                    len(task_indices),
                    task_indices[0],
                    task_indices[-1],
                )

                if use_scripted_policy:
                    from qval.optimal_policies import (
                        ScriptedBackend,
                        get_policy,
                        make_stochastic_policy,
                    )

                    if collection_config.scripted_policy_epsilon > 0.0:
                        policy_cfg = make_stochastic_policy(
                            collection_config.scripted_policy_name,
                            epsilon=collection_config.scripted_policy_epsilon,
                            seed=collection_config.scripted_policy_seed,
                        )
                    else:
                        policy_cfg = get_policy(
                            collection_config.scripted_policy_name
                        )
                    raw_backend = ScriptedBackend(
                        policy_fn=policy_cfg.fn,
                        metadata=policy_cfg.metadata,
                    )
                    backend_config = None
                    backend = LoggingBackend(raw_backend, exp_logger)
                    created_backends.append(backend)
                else:
                    backend_name = resolve_backend_name(
                        backend_name,
                        backends_config.backends,
                        backends_config.default_backend,
                    )
                    backend_config = backends_config.backends[backend_name]
                    chat_template_kwargs = chat_template_kwargs_from_config(backend_config)
                    raw_backend = create_backend_from_config(
                        backend_config,
                        chat_template_kwargs=chat_template_kwargs,
                        tensor_parallel_size_override=args.tensor_parallel_size,
                    )
                    attach_backend_logging_metadata(raw_backend, backend_config)
                    quota_wrapped = wrap_with_quota_retry(
                        raw_backend, policy=backend_config.quota_retry_policy
                    )
                    backend = wrap_with_second_elicitation(
                        LoggingBackend(quota_wrapped, exp_logger)
                    )
                    created_backends.append(backend)

                # Override task indices in config for this policy
                policy_config = dataclasses.replace(
                    config,
                    points_config=dataclasses.replace(
                        config.points_config,
                        task_indices=tuple(task_indices),
                        num_trajectories=len(task_indices),
                    ),
                )

                exp_logger.set_phase(f"trajectory_collection:{backend_name}")
                if backend_config is not None:
                    sampling_params = sampling_params_from_config(
                        backend_config.sampling,
                        backend_type=backend_config.type,
                        provider_preferences=backend_config.openrouter_provider,
                    )
                else:
                    # ScriptedBackend ignores sampling params; pass defaults.
                    from llenvs.inference.protocol import SamplingParams
                    sampling_params = SamplingParams()
                progress = exp_logger.make_progress_reporter(
                    name=f"Trajectory collection {backend_name}",
                    logger_name=__name__,
                )

                if turn_info is None:
                    turn_info = (
                        context.turn_info if context.turn_info is not None else False
                    )

                # Try budget-aware truncation for this collection backend.
                # ScriptedBackend has no token budget — skip it.
                if backend_config is not None:
                    collection_prompt_budget = build_prompt_budget(
                        backend,
                        backend_config,
                        context,
                        max_history_turns=collection_config.max_history_turns,
                    )
                else:
                    collection_prompt_budget = None

                # Collect in per-chunk calls so all_trajectory_results
                # accumulates incrementally (survives mid-collection crashes).
                policy_results: list = []
                chunk_size = config.batch_size
                for chunk_start in range(0, len(task_indices), chunk_size):
                    chunk_indices = task_indices[chunk_start : chunk_start + chunk_size]
                    chunk_config = dataclasses.replace(
                        policy_config,
                        points_config=dataclasses.replace(
                            policy_config.points_config,
                            task_indices=tuple(chunk_indices),
                            num_trajectories=len(chunk_indices),
                        ),
                    )
                    attempted_trajectory_count += len(chunk_indices)
                    results = collect_trajectories(
                        env,  # type: ignore[arg-type]
                        backend,  # type: ignore[arg-type]
                        chunk_config,  # type: ignore[arg-type]
                        sampling_params,
                        system_prompt,
                        sampling_params.extra,
                        turn_info=turn_info,
                        env_factory=env_factory,
                        progress_callback=progress,
                        history_fn=history_fn,
                        prompt_budget=collection_prompt_budget,
                        format_reminder=actor_format_reminder,
                        on_skipped_trajectory=lambda skipped, phase=(f"trajectory_collection:{backend_name}"), backend_name=backend_name: (
                            _on_skipped(
                                skipped,
                                phase,
                                backend_name,
                            )
                        ),
                    )
                    results = _discard_risky_harbor_trajectories(
                        results,
                        trajectory_offset=len(all_trajectory_results),
                    )
                    # Provenance for resume: which policy produced each result.
                    for result in results:
                        result_metadata = getattr(result, "metadata", None)
                        if isinstance(result_metadata, dict):
                            result_metadata.setdefault(
                                "collection_policy_index", policy_idx
                            )
                            result_metadata.setdefault(
                                "collection_backend_name", backend_name
                            )
                    all_trajectory_results.extend(results)
                    policy_results.extend(results)

                    # Incremental checkpoint: persist accumulated trajectories
                    # so that OOM kills / SIGKILL don't lose all progress and a
                    # rerun with the same config resumes from here. Points are
                    # left empty — they are sampled fresh after collection (or
                    # via resample_points.py for manual salvage).
                    save_rolling_checkpoint(
                        resume_ckpt_path,
                        trajectory_results=all_trajectory_results,
                        config_dict=config_dict,
                        metadata={
                            "resume_fingerprint": resume_fingerprint,
                            "num_trajectories_so_far": len(all_trajectory_results),
                            "resumed_trajectories": resumed_trajectory_count,
                            "backend_name": backend_name,
                            "env_name": env_name,
                            "adapter": adapter,
                            "discount_factor": config.discount_factor,
                            "reward_signal_name": reward_signal_name,
                            "step_penalty": step_penalty,
                            "turn_info": turn_info,
                            "context_name": context_name,
                        },
                    )
                    check_abort_threshold(
                        skipped_trajectory_count,
                        attempted_trajectory_count,
                        config.max_aborted_points,
                        f"trajectory_collection:{backend_name}",
                    )

                _log_trajectory_summary_event(
                    exp_logger,
                    phase=f"trajectory_collection:{backend_name}",
                    event_type="trajectory_summary",
                    results=policy_results,
                    extra={"backend_name": backend_name},
                )

                resumed_for_policy = sum(completed.values()) if completed else 0
                policies_meta.append(
                    {
                        "backend_name": backend_name,
                        "num_trajectories": len(policy_results) + resumed_for_policy,
                        "task_indices": full_task_indices,
                        **({"resumed": True} if resumed_for_policy else {}),
                    }
                )

                logger.info("  Collected %d trajectories", len(policy_results))

            # A fully-resumed run never created a collection backend, but LLM
            # ranking sampling reuses "the last collection backend" — create one
            # from the last policy (mirrors the resume_ranking_from fallback).
            if (
                backend is None
                and ranking_k is not None
                and ranking_k >= 2
                and collection_config.ranking_sampling_mode != "manual"
                and not collection_config.ranking_backend_names
                and not use_scripted_policy
                and backend_names
            ):
                _fallback_name = resolve_backend_name(
                    backend_names[-1],
                    backends_config.backends,
                    backends_config.default_backend,
                )
                backend_config = backends_config.backends[_fallback_name]
                chat_template_kwargs = chat_template_kwargs_from_config(backend_config)
                raw_backend = create_backend_from_config(
                    backend_config,
                    chat_template_kwargs=chat_template_kwargs,
                    tensor_parallel_size_override=args.tensor_parallel_size,
                )
                attach_backend_logging_metadata(raw_backend, backend_config)
                quota_wrapped = wrap_with_quota_retry(
                    raw_backend, policy=backend_config.quota_retry_policy
                )
                backend = wrap_with_second_elicitation(
                    LoggingBackend(quota_wrapped, exp_logger)
                )
                created_backends.append(backend)
                logger.info(
                    "Created fallback ranking backend %s "
                    "(collection fully resumed from checkpoint)",
                    _fallback_name,
                )

            # Log stats per policy
            for meta in policies_meta:
                bname = meta["backend_name"]
                stats = compute_stats(exp_logger, f"trajectory_collection:{bname}")
                exp_logger.set_phase(f"trajectory_collection:{bname}")
                exp_logger.log_event("stats", stats)
                if stats["truncated_generations_pct"] > 10:
                    logger.warning(
                        "High truncation rate for %s: %d/%d generations truncated (%.1f%%)",
                        bname,
                        stats["truncated_generations"],
                        stats["total_generations"],
                        stats["truncated_generations_pct"],
                    )
                else:
                    logger.info(
                        "Truncation stats for %s: %d/%d generations truncated (%.1f%%)",
                        bname,
                        stats["truncated_generations"],
                        stats["total_generations"],
                        stats["truncated_generations_pct"],
                    )

            # 5. Save observable trajectories if requested
            if collection_config.trajectories_path and all_trajectory_results:
                observable = [
                    from_trajectory_result(tr, reward_signal_name)
                    for tr in all_trajectory_results
                ]
                save_trajectories(
                    observable,
                    collection_config.trajectories_path,
                    metadata={"environment": env_name},
                )
                logger.info(
                    "Trajectories saved to %s (%d)",
                    collection_config.trajectories_path,
                    len(observable),
                )

            # 6. Extract evaluation points
            points = collect_evaluation_points(
                all_trajectory_results, config.points_config
            )
            logger.info("Extracted %d evaluation points", len(points))

            # 6a. Optional post-collection replay validation for Harbor replay datasets
            if (
                adapter == "harbor"
                and collection_config.harbor_state_capture == "replay"
                and points
                and config.replay_validation.enabled
            ):
                from qval.script_utils import create_harbor_env_factory

                harbor_kwargs = {"max_steps": max_steps} if max_steps else {}
                if make_kwargs:
                    harbor_kwargs.update(make_kwargs)
                if context.environment_extractor is not None:
                    harbor_kwargs["answer_extractor"] = context.environment_extractor
                probe_factory, _ = create_harbor_env_factory(env_name, **harbor_kwargs)

                probe_cmds = (
                    "find /app /home /etc -type f 2>/dev/null | sort | md5sum",
                    "dpkg -l 2>/dev/null | awk '{print $2, $3}' | md5sum",
                )

                # Deduplicate by (task_index, trajectory) to minimize container replays
                from qval.types import ReplaySpec

                unique_states: dict[tuple, list[int]] = {}
                for i, pt in enumerate(points):
                    if pt.replay_spec is not None:
                        key = (pt.replay_spec.task_index, pt.replay_spec.trajectory)
                        unique_states.setdefault(key, []).append(i)
                replay_validation = config.replay_validation

                if unique_states:
                    logger.info(
                        "Capturing replay probe baselines for %d unique states (%d points) with replay validation",
                        len(unique_states),
                        sum(len(v) for v in unique_states.values()),
                    )

                    discarded_indices: set[int] = set()
                    discard_count = 0

                    from llenvs.adapters.harbor import (
                        capture_replay_probe_outputs,
                        validate_replay_consistency,
                    )

                    for (task_idx, trajectory), point_indices in unique_states.items():
                        try:
                            probe_outputs = capture_replay_probe_outputs(
                                env_factory=probe_factory,
                                task_index=task_idx,
                                trajectory=trajectory,
                                probe_commands=probe_cmds,
                            )

                            validation = validate_replay_consistency(
                                env_factory=probe_factory,
                                task_index=task_idx,
                                trajectory=trajectory,
                                probe_commands=probe_cmds,
                                reference_probes=probe_outputs,
                                num_trials=1,
                            )
                            if not validation["consistent"]:
                                logger.warning(
                                    "Replay inconsistent for task %d (trajectory len %d): %s",
                                    task_idx,
                                    len(trajectory),
                                    validation.get("mismatched_commands", ""),
                                )
                                discarded_indices.update(point_indices)
                                discard_count += len(point_indices)
                                if discard_count > replay_validation.max_discards:
                                    logger.error(
                                        "Exceeded max_discards (%d): %d points discarded",
                                        replay_validation.max_discards,
                                        discard_count,
                                    )
                                    # Save what we have before erroring
                                    break
                                continue

                            # Update replay_spec with probe outputs on each point
                            for idx in point_indices:
                                old = points[idx]
                                points[idx] = EvaluationPoint(
                                    state=old.state,
                                    action=old.action,
                                    next_state=old.next_state,
                                    trajectory_index=old.trajectory_index,
                                    step_index=old.step_index,
                                    extracted_action=old.extracted_action,
                                    resolved_action=old.resolved_action,
                                    history=old.history,
                                    replay_spec=ReplaySpec(
                                        task_name=old.replay_spec.task_name,
                                        task_index=old.replay_spec.task_index,
                                        trajectory=old.replay_spec.trajectory,
                                        probe_outputs=probe_outputs,
                                    ),
                                )
                        except Exception as e:
                            logger.warning(
                                "Probe capture failed for task %d: %s", task_idx, e
                            )

                    # Filter out discarded points
                    if discarded_indices:
                        validated_count = len(points) - len(discarded_indices)
                        points = [
                            p
                            for i, p in enumerate(points)
                            if i not in discarded_indices
                        ]
                        logger.info(
                            "Validated %d points, discarded %d (%d unique states)",
                            validated_count,
                            len(discarded_indices),
                            sum(
                                1
                                for key, idxs in unique_states.items()
                                if any(i in discarded_indices for i in idxs)
                            ),
                        )
                        if discard_count > replay_validation.max_discards:
                            raise RuntimeError(
                                f"Replay validation exceeded max_discards "
                                f"({discard_count} > {replay_validation.max_discards})"
                            )
                    else:
                        logger.info(
                            "All %d points passed replay validation", len(points)
                        )

                    if len(points) < replay_validation.target_points:
                        logger.warning(
                            "Only %d validated points (target: %d)",
                            len(points),
                            replay_validation.target_points,
                        )

                    logger.info("Probe capture complete")

            # 6a-ii. Trajectory pruning (max_trajectories_to_keep)
            max_keep = config.points_config.max_trajectories_to_keep
            if max_keep is not None and points:
                traj_indices_seen: list[int] = []
                seen: set[int] = set()
                for pt in points:
                    if pt.trajectory_index not in seen:
                        seen.add(pt.trajectory_index)
                        traj_indices_seen.append(pt.trajectory_index)
                if len(traj_indices_seen) > max_keep:
                    keep_set = set(traj_indices_seen[:max_keep])
                    before = len(points)
                    points = [p for p in points if p.trajectory_index in keep_set]
                    logger.info(
                        "Pruned trajectories: kept %d/%d trajectories (%d -> %d points)",
                        max_keep,
                        len(traj_indices_seen),
                        before,
                        len(points),
                    )
            ranking_input_points = points

        # 6b. Optionally collect ranking points
        if ranking_k is not None and ranking_k >= 2:
            from qval.ranking_collection import collect_ranking_points

            ranking_mode = collection_config.ranking_sampling_mode
            sampler_name = collection_config.ranking_sampler_name
            ranking_extractor = (
                getattr(context, "ranking_environment_extractor", None)
                or context.environment_extractor
            )
            ranking_system_prompt = resolve_ranking_system_prompt(
                override_system_prompt=collection_config.system_prompt,
                adapter=getattr(context, "adapter", None),
                prompting_scheme=getattr(context, "prompting_scheme", None),
                context_system_prompt_file=getattr(context, "system_prompt_file", None),
                adapter_system_prompt=adapter_system_prompt,
                make_kwargs=env_make_kwargs,
                env_name=getattr(context, "env_name", None),
            )
            from qval.system_prompts import RANKING_SCHEME_INSTRUCTIONS

            ranking_scheme = getattr(context, "prompting_scheme", None) or "answer_tags"
            ranking_format_reminder = RANKING_SCHEME_INSTRUCTIONS.get(ranking_scheme)

            if adapter == "harbor":
                from qval.script_utils import create_harbor_ranking_env_factory

                assert harbor_kwargs is not None
                ranking_harbor_kwargs = dict(harbor_kwargs)
                if ranking_extractor is not None:
                    ranking_harbor_kwargs["answer_extractor"] = ranking_extractor
                ranking_env_factory, ranking_restore_fn = (
                    create_harbor_ranking_env_factory(
                        env_name,
                        restore_mode=collection_config.harbor_state_capture,
                        artifact_root=(
                            str(snapshot_artifact_root)
                            if snapshot_artifact_root
                            else None
                        ),
                        **ranking_harbor_kwargs,
                    )
                )
            elif env_factory is not None and env_restore_fn is not None:
                # Non-Harbor adapters that provide both env_factory and a
                # restore_fn (currently: gymnasium + use_images for FL).
                # Reuse them for ranking.
                ranking_env_factory = env_factory
                ranking_restore_fn = env_restore_fn

            if ranking_mode == "manual":
                if not sampler_name:
                    raise ValueError(
                        "ranking_sampler_name is required when ranking_sampling_mode='manual'"
                    )
                # For non-pure envs without a restore_fn (currently
                # OpenApps and similar browser-driven adapters), env-based
                # next_state validation is impossible; fall back to
                # env=None so the manual sampler emits candidates with no
                # next_state. Ranking GT recomputes next_state during MC
                # rollouts so this only affects evaluation methods that
                # render candidate next_states in their prompts.
                ranking_env = env
                if (
                    env is not None
                    and not env.spec.pure_step
                    and (ranking_env_factory is None or ranking_restore_fn is None)
                ):
                    logger.warning(
                        "Manual ranking on non-pure env without restore_fn: "
                        "skipping candidate next_state validation. "
                        "ActionCandidate.next_state will be None for "
                        "non-primary candidates."
                    )
                    ranking_env = None
                ranking_points = collect_ranking_points(
                    points=ranking_input_points,
                    k=ranking_k,
                    mode="manual",
                    sampler_name=sampler_name,
                    env=ranking_env,
                    restore_concurrency=collection_config.restore_concurrency,
                    seed=collection_config.seed,
                    env_factory=ranking_env_factory,
                    restore_fn=ranking_restore_fn,
                    inject_task_text=context.inject_task_text_in_prompts,
                )
            else:
                # LLM sampling: use explicit ranking backends when configured,
                # otherwise reuse the last collection backend.
                ranking_backends: list[LoggingBackend] = []
                ranking_sampling = collection_config.ranking_sampling

                # Resolve ranking backend configs so we can thread provider
                # preferences through the shared ranking sampling params.
                # ``collect_ranking_points`` uses a single SamplingParams across
                # all ranking backends, so openrouter_provider must be uniform.
                if collection_config.ranking_backend_names:
                    ranking_backend_configs = [
                        backends_config.backends[
                            resolve_backend_name(
                                n,
                                backends_config.backends,
                                backends_config.default_backend,
                            )
                        ]
                        for n in collection_config.ranking_backend_names
                    ]
                else:
                    ranking_backend_configs = [backend_config]

                provider_prefs_seen = {
                    cfg.openrouter_provider for cfg in ranking_backend_configs
                }
                if len(provider_prefs_seen) > 1:
                    raise ValueError(
                        "Ranking backends have conflicting openrouter_provider "
                        "settings; ranking sampling uses a single shared "
                        "SamplingParams across all ranking backends. Unify "
                        "openrouter_provider across ranking backends."
                    )
                uniform_provider_prefs = next(iter(provider_prefs_seen))
                uniform_backend_type = ranking_backend_configs[0].type

                if (
                    uniform_provider_prefs is not None
                    and ranking_sampling is None
                ):
                    raise ValueError(
                        "Ranking backend has openrouter_provider set but "
                        "collection.ranking_sampling is not configured. Set "
                        "collection.ranking_sampling to enable provider "
                        "routing during ranking sampling."
                    )

                ranking_sp = (
                    sampling_params_from_config(
                        ranking_sampling,
                        backend_type=uniform_backend_type,
                        provider_preferences=uniform_provider_prefs,
                    )
                    if ranking_sampling is not None
                    else None
                )
                if collection_config.ranking_backend_names:
                    for created_backend in reversed(created_backends):
                        try:
                            created_backend.close()
                        except Exception as exc:
                            logger.warning("Backend cleanup failed: %s", exc)
                    created_backends.clear()

                    ranking_sampling_backend_names = list(
                        collection_config.ranking_backend_names
                    )
                    for ranking_backend_name in ranking_sampling_backend_names:
                        resolved_ranking_backend_name = resolve_backend_name(
                            ranking_backend_name,
                            backends_config.backends,
                            backends_config.default_backend,
                        )
                        ranking_backend_config = backends_config.backends[
                            resolved_ranking_backend_name
                        ]
                        chat_template_kwargs = chat_template_kwargs_from_config(
                            ranking_backend_config
                        )
                        raw_backend = create_backend_from_config(
                            ranking_backend_config,
                            chat_template_kwargs=chat_template_kwargs,
                            tensor_parallel_size_override=args.tensor_parallel_size,
                        )
                        attach_backend_logging_metadata(
                            raw_backend, ranking_backend_config
                        )
                        quota_wrapped = wrap_with_quota_retry(
                            raw_backend,
                            policy=ranking_backend_config.quota_retry_policy,
                        )
                        ranking_backends.append(
                            wrap_with_second_elicitation(
                                LoggingBackend(quota_wrapped, exp_logger)
                            )
                        )
                else:
                    if backend is None:
                        raise ValueError(
                            "LLM ranking sampling requires a collection backend"
                        )
                    if policies_meta:
                        ranking_sampling_backend_name = str(
                            policies_meta[-1]["backend_name"]
                        )
                    ranking_backends = [backend]

                # Build prompt budget for ranking backends
                ranking_prompt_budget = None
                if (
                    context.min_action_chars is not None
                    or context.min_observation_chars is not None
                    or context.min_current_observation_chars is not None
                ):
                    from qval.token_budget import make_prompt_budget

                    # Use the ranking backend config if available, else
                    # fall back to the last collection backend config.
                    budget_backend_raw = (
                        ranking_backends[0]._backend
                        if hasattr(ranking_backends[0], "_backend")
                        else ranking_backends[0]
                    )
                    budget_backend_config = (
                        ranking_backend_config
                        if collection_config.ranking_backend_names
                        else backend_config
                    )
                    ranking_prompt_budget = make_prompt_budget(
                        budget_backend_raw,
                        budget_backend_config,
                        min_observation_chars=context.min_observation_chars,
                        min_action_chars=context.min_action_chars,
                        min_current_observation_chars=context.min_current_observation_chars,
                        max_history_turns=collection_config.max_history_turns,
                    )

                # Swap the env's extractor to the ranking extractor so that
                # env.step() uses the correct extraction for ranking LLM
                # outputs (avoids double-extraction with mismatched tags).
                original_extractor = env.answer_extractor
                env.answer_extractor = ranking_extractor

                try:
                    ranking_points = collect_ranking_points(
                        points=ranking_input_points,
                        k=ranking_k,
                        mode="llm",
                        env=env,
                        backend=ranking_backends[0]
                        if len(ranking_backends) == 1
                        else None,
                        backends=ranking_backends,
                        system_prompt=ranking_system_prompt,
                        ranking_sampling_params=ranking_sp,
                        ranking_sampling_strategy=(
                            collection_config.ranking_sampling_strategy
                        ),
                        batch_size=config.batch_size,
                        restore_concurrency=collection_config.restore_concurrency,
                        seed=collection_config.seed,
                        exp_logger=exp_logger,
                        env_factory=ranking_env_factory,
                        restore_fn=ranking_restore_fn,
                        prompt_budget=ranking_prompt_budget,
                        format_reminder=ranking_format_reminder,
                        max_steps=context.max_steps,
                        inject_task_text=context.inject_task_text_in_prompts,
                    )
                finally:
                    env.answer_extractor = original_extractor
                    if collection_config.ranking_backend_names:
                        for ranking_backend in reversed(ranking_backends):
                            try:
                                ranking_backend.close()
                            except Exception as exc:
                                logger.warning("Backend cleanup failed: %s", exc)
            logger.info(
                "Collected %d ranking points (k=%d, mode=%s)",
                len(ranking_points),
                ranking_k,
                ranking_mode,
            )

        # 7. Compute trajectory returns (already loaded when resuming)
        if not config.resume_ranking_from:
            traj_returns = _compute_trajectory_returns(
                all_trajectory_results,
                points,
            )

        logger.info(
            "Total trajectories: %d in %.1fs",
            len(all_trajectory_results),
            time.time() - start,
        )
        _log_trajectory_summary_event(
            exp_logger,
            phase="trajectory_collection",
            event_type="collection_summary",
            results=all_trajectory_results,
            trajectory_returns=traj_returns,
            extra={
                "backend_names": backend_names,
                "num_ranking_points": len(ranking_points),
                "num_attempted_trajectories": attempted_trajectory_count,
                "num_skipped_trajectories": skipped_trajectory_count,
                "drop_rate": (
                    skipped_trajectory_count / attempted_trajectory_count
                    if attempted_trajectory_count > 0
                    else 0.0
                ),
                "completed": True,
            },
        )

        elapsed = time.time() - start

        # Stop resource monitor and log summary
        resource_summary = resource_monitor.stop()
        if resource_summary:
            exp_logger.log_event("resource_summary", resource_summary)

        if not config.resume_ranking_from and not all_trajectory_results:
            raise RuntimeError(
                "No trajectories were collected successfully; aborting dataset save"
            )

        # 8. Save dataset
        dataset = _build_dataset(
            evaluation_points=points,
            trajectory_returns=traj_returns,
            partial=False,
            elapsed_seconds=elapsed,
        )

        save_dataset(dataset, dataset_path)
        logger.info("Saved dataset to %s (%.1fs)", dataset_path, elapsed)

        # Remove the resume checkpoint now that the final dataset is saved
        if resume_ckpt_path is not None and resume_ckpt_path.exists():
            resume_ckpt_path.unlink()
            logger.info("Removed resume checkpoint %s", resume_ckpt_path)

        # Final save of logs (sink already streams incrementally)
        exp_logger.save_logs(log_path)
        logger.info("Logs saved to %s (%d records)", log_path, len(exp_logger.records))
    except BaseException:
        import sys

        # Emit a collection_summary so the JSONL always has one, even on crash
        try:
            try:
                partial_ranking_count = len(ranking_points)
            except UnboundLocalError:
                partial_ranking_count = 0
            if config.resume_ranking_from:
                partial_traj_returns = traj_returns
            else:
                partial_traj_returns = (
                    _compute_trajectory_returns(
                        all_trajectory_results,
                        points,
                    )
                    if all_trajectory_results
                    else []
                )
            _log_trajectory_summary_event(
                exp_logger,
                phase="trajectory_collection",
                event_type="collection_summary",
                results=all_trajectory_results,
                trajectory_returns=partial_traj_returns,
                extra={
                    "backend_names": backend_names,
                    "num_ranking_points": partial_ranking_count,
                    "num_attempted_trajectories": attempted_trajectory_count,
                    "num_skipped_trajectories": skipped_trajectory_count,
                    "drop_rate": (
                        skipped_trajectory_count / attempted_trajectory_count
                        if attempted_trajectory_count > 0
                        else 0.0
                    ),
                    "completed": False,
                    "abort_reason": str(sys.exc_info()[1]),
                },
            )
        except Exception:
            logger.warning("Failed to emit collection_summary on crash", exc_info=True)

        if attempted_trajectory_count > 0 or (
            config.resume_ranking_from and existing is not None
        ):
            logger.warning(
                "Collection interrupted with %d retained trajectories from %d attempts — saving partial dataset",
                len(all_trajectory_results),
                attempted_trajectory_count,
            )
            try:
                if config.resume_ranking_from:
                    partial_points = points
                    partial_returns = traj_returns
                else:
                    partial_points = collect_evaluation_points(
                        all_trajectory_results,
                        config.points_config,
                    )
                    partial_returns = _compute_trajectory_returns(
                        all_trajectory_results,
                        partial_points,
                    )
                partial_dataset = _build_dataset(
                    evaluation_points=partial_points,
                    trajectory_returns=partial_returns,
                    partial=True,
                    partial_reason=str(sys.exc_info()[1]),
                    ranking_points_value=ranking_points,
                )
                save_dataset(partial_dataset, dataset_path)
                logger.info("Partial dataset saved to %s", dataset_path)
            except Exception:
                logger.exception("Failed to save partial dataset")
        # Save logs unconditionally on crash (sink already has incremental
        # records, but do a final save to ensure nothing is lost)
        try:
            exp_logger.save_logs(log_path)
            logger.info(
                "Logs saved to %s (%d records)", log_path, len(exp_logger.records)
            )
        except Exception:
            logger.exception("Failed to save logs")
        raise
    finally:
        for created_backend in reversed(created_backends):
            try:
                created_backend.close()
            except Exception as exc:
                logger.warning("Backend cleanup failed: %s", exc)


def _timestamped_path(path: str | None, label: str, suffix: str | None = None) -> str:
    if not path:
        raise ValueError(f"{label}_path is required (set in config or CLI)")
    base = Path(path)
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix_value = suffix if suffix is not None else base.suffix
    stem = base.stem
    return str(base.with_name(f"{stem}_{ts}{suffix_value}"))


def _log_trajectory_summary_event(
    exp_logger: ExperimentLogger,
    *,
    phase: str,
    event_type: str,
    results: list[Any],
    trajectory_returns: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a trajectory summary event for collection logs."""
    exp_logger.set_phase(phase)
    payload = summarize_trajectory_results(
        results, trajectory_returns=trajectory_returns
    )
    if extra:
        payload.update(extra)
    exp_logger.log_event(event_type, payload)


def _log_skipped_trajectory_event(
    exp_logger: ExperimentLogger,
    *,
    phase: str,
    backend_name: str,
    skipped: dict[str, Any],
) -> None:
    """Emit a structured event for a skipped errored trajectory."""
    exp_logger.set_phase(phase)
    payload = dict(skipped)
    payload["backend_name"] = backend_name
    exp_logger.log_event("skipped_trajectory", payload)


if __name__ == "__main__":
    main()

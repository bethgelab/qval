"""Benchmark utilities: trajectory collection, display point preparation, backend routing."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable

logger = logging.getLogger(__name__)

from llenvs.core.environment import Environment
from llenvs.evaluation.history import (
    HistoryEntry,
    HistoryFn,
    full_history,
    last_n_history,
)
from llenvs.evaluation.runner import TrajectoryResult, TrajectoryRunner
from llenvs.inference.protocol import ChatMessage, ModelBackend, SamplingParams

from qval.config import BenchmarkConfig, EstimationConfig
from qval.error_handling import retry_on_transient
from qval.methods.serialization import action_text_for_display, serialize_action
from qval.types import EvaluationPoint


def resolve_backend(
    config: EstimationConfig,
    default_backend: ModelBackend,
    backends_dict: dict[str, ModelBackend] | None,
) -> ModelBackend:
    """Resolve the backend for an estimation config.

    Args:
        config: Estimation config that may reference a named backend.
        default_backend: Fallback backend when no name is specified.
        backends_dict: Optional dict mapping backend names to instances.

    Returns:
        The resolved ModelBackend.

    Raises:
        ValueError: If the named backend is not found.
    """
    if config.backend_name and backends_dict:
        if config.backend_name not in backends_dict:
            raise ValueError(
                f"Unknown backend_name: {config.backend_name!r}. "
                f"Available: {sorted(backends_dict)}"
            )
        return backends_dict[config.backend_name]
    if config.backend_name and not backends_dict:
        raise ValueError(
            f"Unknown backend_name: {config.backend_name!r}. "
            "No backends provided"
        )
    return default_backend


# Keep underscore alias for scripts that import the private name
_resolve_backend = resolve_backend


def prepare_display_points(
    points: list[EvaluationPoint],
    include_actor_thinking: bool = False,
) -> list[EvaluationPoint]:
    """Create display copies of evaluation points with cleaned action text.

    For tool-calling actions, ``serialize_action`` includes formatted tool
    calls. For text-only actions, uses ``resolved_action`` from the llenvs
    adapter when available, falling back to ``serialize_action(action)``
    with thinking tokens stripped.

    With ``include_actor_thinking=True``, always uses raw action text
    (preserving thinking traces and tags).
    """
    if include_actor_thinking:
        return [replace(p, action=serialize_action(p.action).strip()) for p in points]

    return [
        replace(
            p,
            action=action_text_for_display(
                p.action, p.resolved_action, p.extracted_action,
            ),
        )
        for p in points
    ]


def _middle_truncate(text: str, max_chars: int) -> str:
    """Truncate the middle of *text*, keeping the beginning and end."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n[... {omitted} characters omitted ...]\n\n"
        + text[-half:]
    )


def content_truncated_history(
    min_action_chars: int | None = None,
    min_observation_chars: int | None = None,
    inner: HistoryFn | None = None,
) -> HistoryFn:
    """Truncate individual action/observation text while preserving all turns.

    Middle-truncation keeps the beginning and end of each text, replacing the
    middle with a notice.  The ``min_*_chars`` parameters set the floor on how
    short truncation can make an entry — each entry is truncated to at most
    that many characters.

    Composable: *inner* defaults to ``full_history`` but can be any
    ``HistoryFn`` (e.g. ``truncated_last_n_history``).
    """
    if inner is None:
        inner = full_history

    def _fn(entries: list[HistoryEntry]) -> list[ChatMessage]:
        truncated: list[HistoryEntry] = []
        for entry in entries:
            action = entry.action_text
            obs = entry.observation_text
            if min_action_chars is not None:
                action = _middle_truncate(action, min_action_chars)
            if min_observation_chars is not None and obs:
                obs = _middle_truncate(obs, min_observation_chars)
            truncated.append(
                HistoryEntry(
                    action_text=action,
                    observation_text=obs,
                    observation_images=entry.observation_images,
                    step=entry.step,
                )
            )
        return inner(truncated)

    return _fn


def truncated_last_n_history(
    n: int,
    inner: HistoryFn | None = None,
) -> HistoryFn:
    """Sliding window history that notifies the model when turns are dropped.

    *inner* transforms the kept entries into messages (default:
    ``full_history``).  Pass a ``content_truncated_history`` to combine
    turn-dropping with per-turn content truncation.
    """
    keep_last = last_n_history(n)
    if inner is None:
        inner = full_history

    def _fn(entries: list[HistoryEntry]) -> list[ChatMessage]:
        # First apply inner (e.g. content truncation) to ALL entries,
        # then slice to keep last N and convert to messages.
        kept = entries[-n:] if n > 0 else []
        messages = inner(kept)
        if len(entries) > n > 0:
            notice = (
                f"[Note: Only the last {n} of {len(entries)} "
                f"turns are shown. {len(entries) - n} earlier "
                f"turns have been omitted.]"
            )
            messages.insert(0, ChatMessage(role="user", content=notice))
        return messages

    return _fn


def collect_trajectories(
    env: Environment[Any],
    backend: ModelBackend,
    config: BenchmarkConfig,
    sampling_params: SamplingParams | None,
    system_prompt: str | None,
    sampling_extra: dict | None = None,
    turn_info: bool = False,
    env_factory: Callable[[], Environment[Any]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    history_fn: HistoryFn | None = None,
    prompt_budget: Any | None = None,
    on_skipped_trajectory: Callable[[dict[str, Any]], None] | None = None,
    format_reminder: str | None = None,
) -> list[TrajectoryResult]:
    """Collect trajectories by running the environment."""
    if sampling_params is None:
        sampling_params = SamplingParams(extra=sampling_extra or {})

    runner_kwargs: dict = {
        "environment": env,
        "backend": backend,
        "sampling_params": sampling_params,
        "system_prompt": system_prompt,
        "format_reminder": format_reminder,
    }
    if env_factory is not None:
        runner_kwargs["env_factory"] = env_factory
    if prompt_budget is not None:
        runner_kwargs["prompt_budget"] = prompt_budget
    elif history_fn is not None:
        runner_kwargs["history_fn"] = history_fn
    if turn_info:
        from llenvs.evaluation.runner import TurnInfoConfig

        max_steps = env.spec.max_steps
        if system_prompt is not None and max_steps is not None:
            system_prompt = system_prompt + f"\n\nYou have a maximum of {max_steps} turns."
        runner_kwargs["turn_info"] = TurnInfoConfig(
            task_suffix="", task_suffix_no_max="",
        )

    runner = TrajectoryRunner(**runner_kwargs)

    task_indices: list[int]
    if config.points_config.task_indices is not None:
        task_indices = list(config.points_config.task_indices)
    else:
        num_tasks = config.points_config.num_trajectories
        task_indices = list(range(num_tasks))

    batch_result = retry_on_transient(
        lambda: runner.run_batch(
            task_indices,
            batch_size=config.batch_size,
            progress_callback=progress_callback,
        ),
        max_retries=6,
        base_delay=4.0,
        description="collect_trajectories run_batch",
    )

    collected_results = []
    for result in batch_result.trajectory_results:
        skipped = _skipped_trajectory_metadata(result)
        if skipped["error"] is None:
            collected_results.append(result)
            continue
        if on_skipped_trajectory is not None:
            on_skipped_trajectory(skipped)
        if skipped["error_kind"] == "reset_error":
            logger.warning(
                "Skipping task %s: reset error: %s",
                skipped["task_index"],
                skipped["error"],
            )
        elif skipped["error_kind"] == "generation_error":
            logger.warning(
                "Skipping task %s: generation error: %s",
                skipped["task_index"],
                skipped["error"],
            )
        else:
            logger.warning(
                "Skipping task %s: step error: %s",
                skipped["task_index"],
                skipped["error"],
            )

    skipped_count = len(batch_result.trajectory_results) - len(collected_results)

    if collected_results:
        logger.info(
            "Collected %d/%d trajectories (%d errors skipped)",
            len(collected_results),
            len(batch_result.trajectory_results),
            skipped_count,
        )
    elif batch_result.trajectory_results:
        logger.error("All %d trajectories errored", len(batch_result.trajectory_results))

    return collected_results


def _skipped_trajectory_metadata(result: TrajectoryResult) -> dict[str, Any]:
    """Build structured metadata for a skipped errored trajectory."""
    error = result.metadata.get("error")
    task_name = result.metadata.get("task_name")
    if task_name is None:
        try:
            task_name = getattr(result.trajectory.initial_state.hidden, "task_name", None)
        except Exception:
            task_name = None

    if isinstance(error, str) and error.startswith("Generation error:"):
        return {
            "task_index": result.metadata.get("task_index"),
            "task_name": task_name,
            "error": error,
            "error_kind": "generation_error",
        }

    is_reset_error = False
    try:
        info = result.trajectory.initial_state.metadata.info
        is_reset_error = bool(info.get("error"))
    except Exception:
        is_reset_error = False

    return {
        "task_index": result.metadata.get("task_index"),
        "task_name": task_name,
        "error": error,
        "error_kind": "reset_error" if is_reset_error else "step_error",
    }

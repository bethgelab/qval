"""Ranking point collection: generate alternative actions and build RankingPoints.

Supports two modes:
- **manual**: uses a registered action sampler to enumerate alternatives.
- **llm**: generates additional actions via the actor backend, with
  deduplication and batched resampling.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import random
import time
from typing import TYPE_CHECKING, Any, Callable

from llenvs.core.state import Action
from llenvs.evaluation.history import HistoryEntry, PromptBudget
from llenvs.inference.protocol import ChatMessage, GenerationResult

from qval.error_handling import (
    EnvError,
    classify_rollout_error,
    generate_batch_with_transient_retry,
)
from qval.methods.serialization import (
    action_text_for_display,
    extract_thought,
    serialize_observation,
)
from qval.prompts import (
    _history_turn_label,
    _task_text_for_injection,
    _turn_label_prefix,
)
from qval.types import (
    ActionCandidate,
    EvaluationPoint,
    RankingCandidateSource,
    RankingPoint,
)

if TYPE_CHECKING:
    from llenvs.core.environment import Environment
    from llenvs.inference.protocol import ModelBackend, SamplingParams

    from qval.experiment_logging import ExperimentLogger

_log = logging.getLogger(__name__)


def build_ranking_sampling_user_content(
    state_text: str,
    seen_actions: list[str],
    strategy: str = "avoid_seen",
    format_reminder: str | None = None,
) -> str:
    """Build the user-turn content for a ranking-sampling prompt.

    Extracted so that preview scripts and the actual collection pipeline
    share the same prompt text.

    Args:
        format_reminder: Optional response-format instruction appended at
            the very end so it is closest to the model's generation point.
            Typically the same text as the scheme instruction in the system
            prompt.
    """
    parts: list[str] = [state_text]

    if strategy != "independent":
        non_empty = [action for action in seen_actions if action]
        any_multiline = any("\n" in action for action in non_empty)
        if any_multiline:
            avoid_lines = "\n".join(
                f"<forbidden_action>\n{action}\n</forbidden_action>"
                for action in non_empty
            )
        else:
            avoid_lines = "\n".join(f"- {action}" for action in non_empty)
        parts.append(
            "Generate exactly one action for the current state. "
            "Your chosen action should be a valid and sensible action to "
            "attempt in the current state. "
            "However, do NOT generate any of the following "
            "actions; they are forbidden at this moment:\n"
            f"{avoid_lines}"
        )

    if format_reminder:
        parts.append(format_reminder)

    return "\n\n".join(parts)


# Hardcoded thresholds
_MAX_RESAMPLE_ROUNDS = 5
_MAX_DROP_FRACTION = 0.20


class _RestoreSetupError(RuntimeError):
    """Raised when a non-pure env cannot be created/restored for candidate stepping."""


def _step_with_restore(
    env_factory: Callable[[], Any],
    restore_fn: Callable[[Any, Any], Any],
    state: Any,
    action: Action,
    *,
    answer_extractor: Any | None = None,
) -> Any:
    """Step a candidate action in a fresh restored environment."""
    try:
        step_env = env_factory()
    except Exception as exc:
        raise _RestoreSetupError(
            "Failed to initialize environment for ranking candidate"
        ) from exc
    try:
        if answer_extractor is not None and hasattr(step_env, "answer_extractor"):
            step_env.answer_extractor = answer_extractor
        try:
            restored_state = restore_fn(step_env, state)
        except Exception as exc:
            raise _RestoreSetupError(
                "Failed to restore environment for ranking candidate"
            ) from exc
        return step_env.step(restored_state, action)
    finally:
        try:
            step_env.close()
        except Exception:
            pass


def _validate_candidate_step(
    state: Any,
    action: Action,
    *,
    env: Any | None = None,
    env_factory: Callable[[], Any] | None = None,
    restore_fn: Callable[[Any, Any], Any] | None = None,
    answer_extractor: Any | None = None,
    debug_context: tuple[int, int, int, int, int, bool] | None = None,
) -> tuple[Any | None, Exception | None]:
    """Execute one candidate step and capture any raised exception."""
    debug_enabled = (
        debug_context is not None and _log.isEnabledFor(logging.DEBUG)
    )
    started_at = time.monotonic() if debug_enabled else 0.0
    if debug_enabled:
        (
            slot_idx,
            resample_round,
            point_idx,
            trajectory_index,
            step_index,
            needs_restore,
        ) = debug_context
        _log.debug(
            "Ranking slot %d round %d point=%d traj=%d step=%d: "
            "candidate validation start restore=%s action_chars=%d",
            slot_idx + 1,
            resample_round + 1,
            point_idx,
            trajectory_index,
            step_index,
            needs_restore,
            len(action.text or ""),
        )

    try:
        if env_factory is not None and restore_fn is not None:
            step_result = _step_with_restore(
                env_factory,
                restore_fn,
                state,
                action,
                answer_extractor=answer_extractor,
            )
        else:
            if env is None:
                raise ValueError(
                    "env or env_factory/restore_fn is required for candidate validation"
                )
            step_result = env.step(state, action)
    except Exception as exc:
        if debug_enabled:
            _log.debug(
                "Ranking slot %d round %d point=%d traj=%d step=%d: "
                "candidate validation raised after %.2fs: %s: %s",
                slot_idx + 1,
                resample_round + 1,
                point_idx,
                trajectory_index,
                step_index,
                max(0.0, time.monotonic() - started_at),
                type(exc).__name__,
                exc,
            )
        return None, exc

    if debug_enabled:
        _log.debug(
            "Ranking slot %d round %d point=%d traj=%d step=%d: "
            "candidate validation end duration=%.2fs terminated=%s "
            "truncated=%s extracted=%s resolved=%s",
            slot_idx + 1,
            resample_round + 1,
            point_idx,
            trajectory_index,
            step_index,
            max(0.0, time.monotonic() - started_at),
            step_result.terminated,
            step_result.truncated,
            step_result.extracted_action is not None,
            step_result.resolved_action is not None,
        )
    return step_result, None


def _run_candidate_validation_jobs(
    jobs: list[dict[str, Any]],
    *,
    restore_concurrency: int,
) -> list[tuple[Any | None, Exception | None]]:
    """Run candidate validation jobs, parallelizing restore-heavy work when requested."""
    if not jobs:
        return []
    max_workers = min(restore_concurrency, len(jobs))
    if max_workers <= 1:
        return [_validate_candidate_step(**job) for job in jobs]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_validate_candidate_step, **job) for job in jobs]
        return [future.result() for future in futures]


def collect_ranking_points(
    points: list[EvaluationPoint],
    k: int,
    mode: str,
    *,
    env: Environment[Any] | None = None,
    sampler_name: str | None = None,
    backend: ModelBackend | None = None,
    backends: list[ModelBackend] | None = None,
    system_prompt: str | None = None,
    ranking_sampling_params: SamplingParams | None = None,
    ranking_sampling_strategy: str = "avoid_seen",
    batch_size: int = 256,
    restore_concurrency: int = 16,
    seed: int = 42,
    exp_logger: ExperimentLogger | None = None,
    env_factory: Callable[[], Any] | None = None,
    restore_fn: Callable[[Any, Any], Any] | None = None,
    prompt_budget: PromptBudget | None = None,
    format_reminder: str | None = None,
    max_steps: int | None = None,
    inject_task_text: bool = False,
) -> list[RankingPoint]:
    """Build ranking points from evaluation points.

    For each evaluation point, produces a ``RankingPoint`` with up to *k*
    candidates: the primary trajectory action plus as many unique alternatives
    as collection can obtain.

    Extraction of raw LLM text is handled by the environment's own
    ``answer_extractor``.  Callers should set ``env.answer_extractor``
    to the appropriate ranking extractor before calling this function.

    Args:
        points: Evaluation points to build ranking points from.
        k: Target maximum candidates per ranking point (including the primary
            action).
        mode: ``"manual"`` or ``"llm"``.
        env: Environment for stepping/validating LLM-sampled actions.
        sampler_name: Name of registered manual sampler (required for manual mode).
        backend: Single backend for LLM sampling mode.
        backends: Optional backend list for LLM sampling mode. When multiple
            backends are provided, auxiliary candidate slots are assigned in
            round-robin order.
        system_prompt: System prompt for LLM sampling messages.
        ranking_sampling_params: Sampling params for the extra actions.
        ranking_sampling_strategy: ``"avoid_seen"`` (default) or ``"independent"``.
        batch_size: LLM request chunk size for batched generation.
        restore_concurrency: Maximum concurrent restore-based candidate
            validations for non-pure environments.
        seed: Random seed for shuffling and manual sampling.
        exp_logger: Optional experiment logger.
        env_factory: Factory for creating fresh env instances (required for
            non-pure envs like Harbor).
        restore_fn: Function to restore env to a given state (required for
            non-pure envs like Harbor).
        prompt_budget: Token budget for truncating prompts to fit the ranking
            backend's context window. ``None`` disables truncation.
        format_reminder: Response-format instruction appended at the end of
            every user message (e.g. the scheme formatting instruction).
        max_steps: Episode step limit, used for turn labels in history
            messages (e.g. ``[Turn 2/50]``).

    Returns:
        List of RankingPoint, one per successfully sampled evaluation point.
    """
    if restore_concurrency < 1:
        raise ValueError("restore_concurrency must be a positive integer")

    if mode == "manual":
        return _collect_manual(
            points,
            k,
            sampler_name=sampler_name,
            seed=seed,
            env=env,
            env_factory=env_factory,
            restore_fn=restore_fn,
            restore_concurrency=restore_concurrency,
        )
    elif mode == "llm":
        return _collect_llm(
            points,
            k,
            env=env,
            backend=backend,
            backends=backends,
            system_prompt=system_prompt,
            ranking_sampling_params=ranking_sampling_params,
            ranking_sampling_strategy=ranking_sampling_strategy,
            batch_size=batch_size,
            restore_concurrency=restore_concurrency,
            seed=seed,
            exp_logger=exp_logger,
            env_factory=env_factory,
            restore_fn=restore_fn,
            prompt_budget=prompt_budget,
            format_reminder=format_reminder,
            max_steps=max_steps,
            inject_task_text=inject_task_text,
        )
    else:
        raise ValueError(f"Unknown ranking sampling mode: {mode!r}")


def _primary_candidate(point: EvaluationPoint) -> ActionCandidate:
    """Build an ActionCandidate from an evaluation point's primary action."""
    return ActionCandidate(
        action=point.action,
        source=RankingCandidateSource.ACTOR_PRIMARY,
        extracted_action=point.extracted_action,
        resolved_action=point.resolved_action,
        next_state=point.next_state,
        thought=point.current_thought,
    )


def _candidate_key(c: ActionCandidate) -> str:
    """Deduplication key for a candidate: resolved_action or extracted_action."""
    return (c.resolved_action or c.extracted_action or "").strip().lower()


def _candidate_prompt_text(c: ActionCandidate) -> str:
    """Best-effort display text for ranking prompt constraints."""
    return action_text_for_display(c.action, c.resolved_action, c.extracted_action)


def _reorder_budget_history_for_ranking(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Convert assistant/action→user/observation history into ranking order.

    ``PromptBudget`` history builders operate on trajectory-style entries where
    each turn is represented as assistant(action) followed by user(observation).
    Ranking points instead store prior history as user(state) followed by
    assistant(action). This helper preserves any leading non-assistant notice
    message and swaps only well-formed assistant/user pairs.
    """
    if not messages:
        return []

    reordered: list[ChatMessage] = []
    start_idx = 0
    if messages[0].role != "assistant":
        reordered.append(messages[0])
        start_idx = 1

    i = start_idx
    while i < len(messages):
        if i + 1 >= len(messages):
            reordered.extend(messages[i:])
            break
        assistant_msg = messages[i]
        user_msg = messages[i + 1]
        if assistant_msg.role == "assistant" and user_msg.role == "user":
            reordered.append(
                ChatMessage(
                    role="user",
                    content=user_msg.content,
                    images=user_msg.images,
                )
            )
            reordered.append(
                ChatMessage(role="assistant", content=assistant_msg.content)
            )
        else:
            reordered.extend([assistant_msg, user_msg])
        i += 2

    return reordered


def _shuffle_candidates(
    candidates: list[ActionCandidate],
    trajectory_index: int,
    step_index: int,
) -> tuple[ActionCandidate, ...]:
    """Deterministically shuffle candidates to avoid position bias."""
    rng = random.Random(trajectory_index * 1000 + step_index)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return tuple(shuffled)


def _collect_manual(
    points: list[EvaluationPoint],
    k: int,
    *,
    sampler_name: str | None,
    seed: int,
    restore_concurrency: int,
    env: Any | None = None,
    env_factory: Callable[[], Any] | None = None,
    restore_fn: Callable[[Any, Any], Any] | None = None,
) -> list[RankingPoint]:
    """Collect ranking points using a manual action sampler."""
    from qval.action_samplers import get_sampler

    if sampler_name is None:
        raise ValueError("sampler_name is required for manual ranking mode")

    _needs_restore = env is not None and not env.spec.pure_step
    if _needs_restore and (env_factory is None or restore_fn is None):
        raise ValueError(
            "Ranking collection on non-pure environments (e.g., Harbor) "
            "requires env_factory and restore_fn for state restoration"
        )

    sampler_fn = get_sampler(sampler_name)
    rng = random.Random(seed)
    result: list[RankingPoint] = []

    for point in points:
        primary = _primary_candidate(point)
        primary_key = _candidate_key(primary)

        # Ask for up to k alternatives so one duplicate-primary collision
        # does not unnecessarily reduce the retained set.
        alternatives = sampler_fn(point.state, k, rng)

        # Populate next_state for alternatives by stepping the env
        if env is not None:
            import dataclasses as _dc

            validation_jobs = []
            for alt in alternatives:
                job: dict[str, Any] = {
                    "state": point.state,
                    "action": alt.action,
                }
                if _needs_restore:
                    job["env_factory"] = env_factory
                    job["restore_fn"] = restore_fn
                else:
                    job["env"] = env
                validation_jobs.append(job)

            validation_outcomes = _run_candidate_validation_jobs(
                validation_jobs,
                restore_concurrency=restore_concurrency if _needs_restore else 1,
            )
            populated: list[ActionCandidate] = []
            _manual_restore_failures = 0
            point_aborted = False
            fatal_exc: EnvError | None = None
            for alt, (sr, error) in zip(alternatives, validation_outcomes, strict=True):
                try:
                    if error is not None:
                        raise error
                    if point_aborted:
                        continue
                    alt = _dc.replace(
                        alt,
                        extracted_action=sr.extracted_action,
                        resolved_action=sr.resolved_action,
                        next_state=sr.next_state,
                    )
                except Exception as exc:
                    error_kind = classify_rollout_error(exc)
                    if error_kind is not None:
                        point_aborted = True
                        _log.warning(
                            "Point (traj=%d, step=%d): aborting manual ranking point "
                            "due to recoverable env error (%s): %s",
                            point.trajectory_index,
                            point.step_index,
                            error_kind,
                            exc,
                        )
                        continue
                    if isinstance(exc, EnvError):
                        fatal_exc = exc
                        break
                    if _needs_restore:
                        _manual_restore_failures += 1
                        if _manual_restore_failures <= 3:
                            _log.warning(
                                "Manual ranking restore/step failed for candidate",
                                exc_info=True,
                            )
                        continue  # skip candidate — next_state=None defeats unbiased prompts
                    # pure envs: keep candidate with next_state=None (existing behavior)
                populated.append(alt)
            if fatal_exc is not None:
                raise fatal_exc
            if point_aborted:
                continue
            alternatives = populated

        # Deduplicate against the primary action
        unique: list[ActionCandidate] = [primary]
        seen = {primary_key}
        for alt in alternatives:
            alt_key = _candidate_key(alt)
            if alt_key not in seen:
                unique.append(alt)
                seen.add(alt_key)

        # Keep the primary action and trim the remaining actions
        # deterministically so collection is stable across runs.
        if len(unique) > k:
            primary_candidate = unique[0]
            shuffled_alts = list(
                _shuffle_candidates(
                    unique[1:],
                    point.trajectory_index,
                    point.step_index,
                )
            )
            unique = [primary_candidate, *shuffled_alts[: k - 1]]

        # If we still have enough unique candidates (may be < k if action space exhausted)
        if len(unique) < 2:
            _log.warning(
                "Point (traj=%d, step=%d): only %d unique candidates, skipping",
                point.trajectory_index,
                point.step_index,
                len(unique),
            )
            continue

        shuffled = _shuffle_candidates(unique, point.trajectory_index, point.step_index)
        result.append(
            RankingPoint(
                state=point.state,
                candidates=shuffled,
                trajectory_index=point.trajectory_index,
                step_index=point.step_index,
                history=point.history,
            )
        )

    return result


def build_ranking_sampling_messages(
    point: EvaluationPoint,
    *,
    seen_actions: list[str],
    strategy: str = "avoid_seen",
    system_prompt: str | None = None,
    format_reminder: str | None = None,
    max_steps: int | None = None,
    prompt_budget: PromptBudget | None = None,
    task_text_to_inject: str | None = None,
) -> list[ChatMessage]:
    """Build the full message list for a ranking-sampling LLM call.

    This is the single source of truth for ranking-collection prompt
    construction.  Both the collection pipeline and preview scripts
    should call this function.
    """
    state_text = serialize_observation(point.state)

    user_content = build_ranking_sampling_user_content(
        state_text,
        seen_actions,
        strategy,
        format_reminder=format_reminder,
    )

    # Build history messages — with or without budget-aware truncation
    if prompt_budget is not None:
        non_history_tokens = 0
        if system_prompt:
            non_history_tokens += prompt_budget.estimate_tokens(system_prompt)
        non_history_tokens += prompt_budget.estimate_tokens(user_content)
        if task_text_to_inject:
            non_history_tokens += prompt_budget.estimate_tokens(task_text_to_inject) + 1
        non_history_tokens += 20  # per-message overhead

        available = max(0, prompt_budget.max_prompt_tokens - non_history_tokens)

        entries = []
        for i, turn in enumerate(point.history):
            label = (
                _history_turn_label(
                    i,
                    len(point.history),
                    point.step_index,
                    max_steps,
                )
                + "]"
            )
            entries.append(
                HistoryEntry(
                    observation_text=f"{label}\n{turn.state_text}",
                    action_text=turn.action_text,
                    step=i,
                )
            )
        history_msgs = _reorder_budget_history_for_ranking(
            prompt_budget.build_history(entries, available)
        )

        if prompt_budget.min_current_observation_chars is not None:
            total_tokens = non_history_tokens + sum(
                prompt_budget.estimate_tokens(m.content) for m in history_msgs
            )
            excess = total_tokens - prompt_budget.max_prompt_tokens
            if excess > 0:
                from qval.token_budget import truncate_text_to_token_savings

                truncated = truncate_text_to_token_savings(
                    state_text,
                    excess,
                    prompt_budget.min_current_observation_chars,
                    prompt_budget.estimate_tokens,
                )
                if truncated != state_text:
                    user_content = build_ranking_sampling_user_content(
                        truncated,
                        seen_actions,
                        strategy,
                        format_reminder=format_reminder,
                    )
    else:
        history_msgs = []
        for i, turn in enumerate(point.history):
            label = (
                _history_turn_label(
                    i,
                    len(point.history),
                    point.step_index,
                    max_steps,
                )
                + "]"
            )
            history_msgs.append(
                ChatMessage(role="user", content=f"{label}\n{turn.state_text}")
            )
            history_msgs.append(ChatMessage(role="assistant", content=turn.action_text))

    cur_label = _turn_label_prefix(point.step_index + 1, max_steps) + "]"
    msgs: list[ChatMessage] = []
    if system_prompt:
        msgs.append(ChatMessage(role="system", content=system_prompt))
    msgs.extend(history_msgs)
    msgs.append(ChatMessage(role="user", content=f"{cur_label}\n{user_content}"))

    if task_text_to_inject:
        for i, m in enumerate(msgs):
            if m.role == "user":
                msgs[i] = ChatMessage(
                    role="user",
                    content=f"{task_text_to_inject}\n\n{m.content}",
                )
                break

    return msgs


def _collect_llm(
    points: list[EvaluationPoint],
    k: int,
    *,
    env: Any,
    backend: Any,
    backends: list[Any] | None,
    system_prompt: str | None,
    ranking_sampling_params: Any | None,
    ranking_sampling_strategy: str,
    batch_size: int,
    restore_concurrency: int,
    seed: int,
    exp_logger: Any | None,
    env_factory: Callable[[], Any] | None = None,
    restore_fn: Callable[[Any, Any], Any] | None = None,
    prompt_budget: PromptBudget | None = None,
    format_reminder: str | None = None,
    max_steps: int | None = None,
    inject_task_text: bool = False,
) -> list[RankingPoint]:
    """Collect ranking points by LLM-sampling alternative actions."""
    from llenvs.inference.protocol import SamplingParams

    sampling_backends = list(backends) if backends is not None else []
    if backend is not None:
        sampling_backends = [backend] if not sampling_backends else sampling_backends
    if not sampling_backends:
        raise ValueError("backend or backends is required for LLM ranking sampling")
    if env is None:
        raise ValueError("env is required for LLM ranking sampling")
    if ranking_sampling_strategy not in {"avoid_seen", "independent"}:
        raise ValueError(
            f"Unknown ranking_sampling_strategy: {ranking_sampling_strategy!r}"
        )

    _needs_restore = not env.spec.pure_step
    if _needs_restore and (env_factory is None or restore_fn is None):
        raise ValueError(
            "Ranking collection on non-pure environments (e.g., Harbor) "
            "requires env_factory and restore_fn for state restoration"
        )

    sampling_params = ranking_sampling_params or SamplingParams(
        temperature=1.5,
        top_p=1.0,
        top_k=0,
        max_tokens=1024,
    )

    if exp_logger is not None:
        exp_logger.set_phase("ranking_sampling")

    # Track per-point state: accepted candidates and dedup keys.
    per_point: list[dict[str, Any]] = []
    for point in points:
        primary = _primary_candidate(point)
        primary_key = _candidate_key(primary)
        per_point.append(
            {
                "point": point,
                "candidates": [primary],
                "seen": {primary_key},
                "aborted_error_kind": None,
                "aborted_error": None,
            }
        )

    max_drop = max(1, int(len(points) * _MAX_DROP_FRACTION))
    aux_slots = max(0, k - 1)
    aborted_points: set[int] = set()
    for slot_idx in range(aux_slots):
        slot_backend = sampling_backends[slot_idx % len(sampling_backends)]
        accepted_points: set[int] = set()

        try:
            for resample_round in range(_MAX_RESAMPLE_ROUNDS):
                requests: list[tuple[int, list[ChatMessage]]] = []
                for point_idx, info in enumerate(per_point):
                    if point_idx in accepted_points or point_idx in aborted_points:
                        continue
                    seen_actions = [
                        _candidate_prompt_text(candidate)
                        for candidate in info["candidates"]
                    ]
                    point_task_text = None
                    if inject_task_text:
                        point_task_text = _task_text_for_injection(info["point"])
                    requests.append(
                        (
                            point_idx,
                            build_ranking_sampling_messages(
                                info["point"],
                                seen_actions=seen_actions,
                                strategy=ranking_sampling_strategy,
                                system_prompt=system_prompt,
                                format_reminder=format_reminder,
                                max_steps=max_steps,
                                prompt_budget=prompt_budget,
                                task_text_to_inject=point_task_text,
                            ),
                        )
                    )

                if not requests:
                    break

                _log.info(
                    "Ranking sampling slot %d round %d: %d requests across %d points",
                    slot_idx + 1,
                    resample_round + 1,
                    len(requests),
                    len({idx for idx, _ in requests}),
                )

                all_messages = [msgs for _, msgs in requests]
                all_results: list[GenerationResult | None] = [None] * len(all_messages)
                debug_enabled = _log.isEnabledFor(logging.DEBUG)
                num_chunks = (len(all_messages) + batch_size - 1) // batch_size
                for chunk_start in range(0, len(all_messages), batch_size):
                    chunk = all_messages[chunk_start : chunk_start + batch_size]
                    chunk_index = (chunk_start // batch_size) + 1
                    chunk_end = chunk_start + len(chunk)
                    if debug_enabled:
                        chunk_started_at = time.monotonic()
                        _log.debug(
                            "Ranking slot %d round %d chunk %d/%d: backend batch start "
                            "requests=[%d:%d) size=%d backend=%s",
                            slot_idx + 1,
                            resample_round + 1,
                            chunk_index,
                            num_chunks,
                            chunk_start,
                            chunk_end,
                            len(chunk),
                            type(slot_backend).__name__,
                        )
                    try:
                        chunk_results = generate_batch_with_transient_retry(
                            slot_backend,
                            chunk,
                            sampling_params,
                        )
                    except Exception:
                        # Ranking is auxiliary data — swallow fatal errors
                        _log.error(
                            "Non-recoverable error in ranking chunk, skipping",
                            exc_info=True,
                        )
                        chunk_results = [None] * len(chunk)
                    if debug_enabled:
                        chunk_elapsed_sec = max(0.0, time.monotonic() - chunk_started_at)
                        _log.debug(
                            "Ranking slot %d round %d chunk %d/%d: backend batch end "
                            "duration=%.2fs results=%d/%d missing=%d",
                            slot_idx + 1,
                            resample_round + 1,
                            chunk_index,
                            num_chunks,
                            chunk_elapsed_sec,
                            sum(result is not None for result in chunk_results),
                            len(chunk_results),
                            sum(result is None for result in chunk_results),
                        )
                    for i, result in enumerate(chunk_results):
                        all_results[chunk_start + i] = result

                round_empty = 0
                round_backend_fail = 0
                round_step_fail = 0
                round_dedup = 0
                round_accepted = 0
                round_unparsed = 0
                round_aborted = 0
                pending_validations: list[tuple[int, dict[str, Any], str, Action]] = []
                validation_jobs: list[dict[str, Any]] = []

                for (point_idx, _), gen_result in zip(requests, all_results, strict=True):
                    if gen_result is None:
                        round_backend_fail += 1
                        continue
                    info = per_point[point_idx]
                    raw_text = gen_result.text or ""
                    if not raw_text.strip():
                        round_empty += 1
                        continue

                    step_action = Action(text=raw_text)
                    pending_validations.append((point_idx, info, raw_text, step_action))
                    job = {
                        "state": info["point"].state,
                        "action": step_action,
                        "debug_context": (
                            slot_idx,
                            resample_round,
                            point_idx,
                            info["point"].trajectory_index,
                            info["point"].step_index,
                            _needs_restore,
                        ) if debug_enabled else None,
                    }
                    if _needs_restore:
                        job["env_factory"] = env_factory
                        job["restore_fn"] = restore_fn
                        job["answer_extractor"] = env.answer_extractor
                    else:
                        job["env"] = env
                    validation_jobs.append(job)

                validation_outcomes = _run_candidate_validation_jobs(
                    validation_jobs,
                    restore_concurrency=restore_concurrency if _needs_restore else 1,
                )

                for (
                    point_idx,
                    info,
                    raw_text,
                    step_action,
                ), (step_result, error) in zip(
                    pending_validations,
                    validation_outcomes,
                    strict=True,
                ):
                    if error is not None:
                        exc = error
                        error_kind = classify_rollout_error(exc)
                        if error_kind is not None:
                            aborted_points.add(point_idx)
                            info["aborted_error_kind"] = error_kind
                            info["aborted_error"] = str(exc)
                            round_aborted += 1
                            _log.warning(
                                "Point (traj=%d, step=%d): aborting ranking point "
                                "due to recoverable env error (%s): %s",
                                info["point"].trajectory_index,
                                info["point"].step_index,
                                error_kind,
                                exc,
                            )
                            continue
                        if isinstance(exc, EnvError):
                            raise
                        round_step_fail += 1
                        if round_step_fail <= 3:
                            _log.warning(
                                "Point %d: env.step() failed for action %r",
                                point_idx,
                                raw_text[:80],
                                exc_info=True,
                            )
                        continue

                    extracted = step_result.extracted_action
                    resolved = step_result.resolved_action
                    if extracted is None:
                        round_unparsed += 1
                        continue

                    candidate = ActionCandidate(
                        action=step_action,
                        source=RankingCandidateSource.RANKING_LLM,
                        extracted_action=extracted,
                        resolved_action=resolved,
                        next_state=step_result.next_state,
                        thought=extract_thought(raw_text),
                    )
                    cand_key = _candidate_key(candidate)
                    if cand_key and cand_key not in info["seen"]:
                        info["candidates"].append(candidate)
                        info["seen"].add(cand_key)
                        accepted_points.add(point_idx)
                        round_accepted += 1
                    else:
                        round_dedup += 1
                        _log.debug(
                            "Point %d: duplicate key %r (seen: %s)",
                            point_idx,
                            cand_key[:80] if cand_key else "",
                            ", ".join(repr(s[:40]) for s in sorted(info["seen"])),
                        )

                _log.info(
                    "Ranking slot %d round %d results: %d accepted, %d duplicate, "
                    "%d step-failed, %d aborted, %d empty, %d backend-failed, %d unparsed",
                    slot_idx + 1,
                    resample_round + 1,
                    round_accepted,
                    round_dedup,
                    round_step_fail,
                    round_aborted,
                    round_empty,
                    round_backend_fail,
                    round_unparsed,
                )
        except Exception as exc:
            if isinstance(exc, EnvError):
                raise  # fatal — let caller save and exit
            _log.error(
                "Ranking collection interrupted during slot %d — "
                "building results from accumulated candidates",
                slot_idx + 1,
                exc_info=True,
            )
            break  # exit slot loop, fall through to result building

    # Build RankingPoints from successful collections
    result: list[RankingPoint] = []
    dropped_count = 0
    for info in per_point:
        if info["aborted_error_kind"] is not None:
            dropped_count += 1
            _log.warning(
                "Point (traj=%d, step=%d): dropped after recoverable env error (%s): %s",
                info["point"].trajectory_index,
                info["point"].step_index,
                info["aborted_error_kind"],
                info["aborted_error"],
            )
            continue
        if len(info["candidates"]) < 2:
            dropped_count += 1
            seen_keys = sorted(info["seen"])
            _log.warning(
                "Point (traj=%d, step=%d): dropped with only %d candidates. Seen keys: %s",
                info["point"].trajectory_index,
                info["point"].step_index,
                len(info["candidates"]),
                [k[:80] for k in seen_keys],
            )
            continue
        shuffled = _shuffle_candidates(
            info["candidates"],
            info["point"].trajectory_index,
            info["point"].step_index,
        )
        result.append(
            RankingPoint(
                state=info["point"].state,
                candidates=shuffled,
                trajectory_index=info["point"].trajectory_index,
                step_index=info["point"].step_index,
                history=info["point"].history,
            )
        )

    if dropped_count > max_drop:
        _log.error(
            "Too many ranking points dropped (%d > %d max). Saving partial results.",
            dropped_count,
            max_drop,
        )

    _log.info(
        "LLM ranking sampling complete: %d/%d points collected, %d dropped",
        len(result),
        len(points),
        dropped_count,
    )
    return result

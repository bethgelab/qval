"""Ground truth estimation utilities.

Provides ``create_gt_method()`` factory that dispatches to the appropriate
estimation method (MCMethod or MCTSMethod), a standalone
``aggregate_returns()`` function for aggregating raw rollout returns, and
``collect_grouped_gt()`` for collecting GT with rollout reuse and V(s)
cross-group sharing.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from llenvs.core.environment import Environment

from qval.config import EstimationConfig
from qval.dense_signal import DenseSignalMethod
from qval.error_handling import (
    MC_ROLLOUT_MAX_RETRIES,
    MC_ROLLOUT_RETRY_BASE_DELAY,
    classify_rollout_error,
    is_recoverable_backend_error,
    retry_on_transient,
)
from qval.types import (
    AggregationMethod,
    EvaluationPoint,
    MethodContext,
    Policy,
    RawPointReturns,
    SignalEstimate,
    SignalType,
)

if TYPE_CHECKING:
    from qval.experiment_logging import ExperimentLogger

_log = logging.getLogger(__name__)


def _on_generation_error(exc: Exception) -> str:
    """Error callback for runner: skip recoverable errors, raise others."""
    if is_recoverable_backend_error(exc):
        return "skip"
    return "raise"


def aggregate_returns(
    raw_returns: list[RawPointReturns],
    aggregation: AggregationMethod,
    *,
    is_advantage: bool = False,
) -> list[SignalEstimate]:
    """Aggregate raw rollout returns into signal estimates.

    For non-advantage signal types: ``agg(returns)``.
    For ADVANTAGE (``is_advantage=True``):
    ``agg(next_state_returns) - agg(returns)``.

    Args:
        raw_returns: Raw per-point rollout returns from ``collect_returns()``.
        aggregation: Aggregation method (MEAN or MAX).
        is_advantage: When True, compute advantage as ``V(s') - V(s)``
            using ``next_state_returns`` for V(s') and ``returns`` for V(s).

    Returns:
        List of SignalEstimate, one per point.
    """
    results: list[SignalEstimate] = []
    for raw in raw_returns:
        returns = raw.returns
        next_returns = raw.next_state_returns

        if raw.aborted:
            value = math.nan
            all_returns = returns + next_returns
            num_rollouts = len(all_returns)
        elif is_advantage:
            # ADVANTAGE: both sides must be present for a valid estimate.
            if returns and next_returns:
                v_s = _agg(returns, aggregation)
                v_next = _agg(next_returns, aggregation)
                value = v_next - v_s
                all_returns = returns + next_returns
                num_rollouts = len(returns) + len(next_returns)
            else:
                # One or both sides empty (aborted point) → NaN
                value = math.nan
                all_returns = returns + (next_returns or ())
                num_rollouts = len(returns) + len(next_returns or ())
        else:
            value = _agg(returns, aggregation)
            all_returns = returns
            num_rollouts = len(returns)

        results.append(
            SignalEstimate(
                value=value,
                point=raw.point,
                num_rollouts=num_rollouts,
                raw_returns=all_returns,
            )
        )
    return results


def _agg(returns: tuple[float, ...], aggregation: AggregationMethod) -> float:
    """Aggregate a tuple of returns."""
    if not returns:
        return math.nan
    if aggregation == AggregationMethod.MEAN:
        return sum(returns) / len(returns)
    elif aggregation == AggregationMethod.MAX:
        return max(returns)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def create_gt_method(
    env: Environment[Any],
    policy: Policy,
    config: EstimationConfig,
    signal_type: SignalType,
    *,
    reward_signal_name: str | None = "correctness",
    batch_size: int | None = None,
    turn_info: bool = False,
    discount_factor: float = 1.0,
    env_factory: Any | None = None,
    restore_fn: Any | None = None,
    history_fn: Any | None = None,
    prompt_budget: Any | None = None,
    format_reminder: str | None = None,
    step_penalty: float | None = None,
    generation_descriptor: dict[str, Any] | None = None,
) -> DenseSignalMethod:
    """Create a ground truth estimation method from an EstimationConfig.

    Returns a ``DenseSignalMethod`` (MCMethod or MCTSMethod) that uses the
    provided ``Policy`` for rollouts.

    Args:
        env: The environment for rollouts.
        policy: The rollout policy (backend + sampling_params + system_prompt).
        config: Estimation configuration.
        signal_type: Type of signal to estimate.
        reward_signal_name: Which reward signal to use (None for total).
        batch_size: Maximum rollouts per lockstep batch.
        turn_info: When True, prepend turn info to user messages during rollouts.
        env_factory: Optional factory for creating per-rollout env instances
            (non-pure environments).
        restore_fn: Optional function to restore a fresh env to a saved state.
        history_fn: Optional history function for the TrajectoryRunner.
        prompt_budget: Optional PromptBudget for budget-aware history
            truncation. Takes precedence over ``history_fn``.

    Returns:
        A DenseSignalMethod instance (MCMethod or MCTSMethod).

    Raises:
        ValueError: If the estimation method is not supported.
    """
    if signal_type == SignalType.SHAPED_REWARD:
        raise ValueError("SHAPED_REWARD does not use rollout-based GT estimation")

    # POTENTIAL uses V(s) rollouts — create context with STATE_VALUE
    gt_signal_type = (
        SignalType.STATE_VALUE if signal_type == SignalType.POTENTIAL else signal_type
    )
    context = MethodContext(signal_type=gt_signal_type)

    if config.method == "mc":
        from qval.mc_method import MCMethod

        return MCMethod(
            context=context,
            env=env,
            policy=policy,
            num_rollouts=config.num_rollouts,
            aggregation=config.aggregation,
            reward_signal_name=reward_signal_name,
            batch_size=batch_size,
            turn_info=turn_info,
            discount_factor=discount_factor,
            env_factory=env_factory,
            restore_fn=restore_fn,
            history_fn=history_fn,
            prompt_budget=prompt_budget,
            format_reminder=format_reminder,
            step_penalty=step_penalty,
            generation_descriptor=generation_descriptor,
        )
    elif config.method == "mcts":
        from qval.mcts_method import MCTSMethod

        return MCTSMethod(
            context=context,
            env=env,
            policy=policy,
            mcts_iterations=config.mcts_iterations,
            mcts_expansion_width=config.mcts_expansion_width,
            mcts_exploration_constant=config.mcts_exploration_constant,
            aggregation=config.aggregation,
            reward_signal_name=reward_signal_name,
            batch_size=batch_size,
            turn_info=turn_info,
            discount_factor=discount_factor,
            history_fn=history_fn,
            prompt_budget=prompt_budget,
            format_reminder=format_reminder,
        )
    else:
        raise ValueError(f"Unknown estimation method: {config.method}")


def _v_s_rollout_key(identity: tuple) -> tuple:
    """Strip rollout_type from identity for cross-group V(s) matching.

    The identity key format is ``(backend_id, method_type, rollout_type, ...)``.
    Removes index 2 (rollout_type) so that V(s) and ADVANTAGE groups with the
    same backend and rollout params can match.
    """
    return identity[:2] + identity[3:]


def _slice_raw_point_returns(raw: list[RawPointReturns], num_rollouts: int) -> list[RawPointReturns]:
    """Take the first ``num_rollouts`` samples from each raw-return item."""
    return [
        RawPointReturns(
            point=item.point,
            returns=item.returns[:num_rollouts],
            step_counts=item.step_counts[:num_rollouts],
            next_state_returns=item.next_state_returns[:num_rollouts],
            next_state_step_counts=item.next_state_step_counts[:num_rollouts],
            aborted=item.aborted,
        )
        for item in raw
    ]


def _default_point_keys(points: list[EvaluationPoint]) -> list[tuple[int, ...]]:
    return [
        (point.trajectory_index, point.step_index)
        for point in points
    ]


def _chunked[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _task_name_for_point(point: EvaluationPoint) -> str | None:
    replay_spec = getattr(point, "replay_spec", None)
    if replay_spec is not None:
        return replay_spec.task_name
    hidden_task_name = getattr(point.state.hidden, "task_name", None)
    return hidden_task_name if isinstance(hidden_task_name, str) else None


def _log_rollout_abort_events(
    logger: ExperimentLogger | None,
    *,
    method_name: str,
    points: list[EvaluationPoint],
    point_keys: list[tuple[int, ...]],
    aborted_indices: set[int],
    raw: list[RawPointReturns] | None,
    failure_details: dict[tuple[int, ...], tuple[dict[str, Any], ...]],
) -> None:
    if logger is None or not aborted_indices:
        return
    logger.set_phase(f"gt:{method_name}")
    for idx in sorted(aborted_indices):
        point = points[idx]
        point_key = point_keys[idx]
        details = failure_details.get(point_key, ())
        primary = details[0] if details else {}
        error_text = primary.get("error")
        error_kind = (
            classify_rollout_error(RuntimeError(error_text))
            if isinstance(error_text, str)
            else None
        ) or "step_failure"
        retained_rollouts = 0
        if raw is not None:
            retained_rollouts = len(raw[idx].returns)
        logger.log_event(
            "rollout_abort",
            {
                "method_name": method_name,
                "task_index": point.state.metadata.info.get(
                    "task_index",
                    getattr(point.state.hidden, "task_index", None),
                ),
                "task_name": _task_name_for_point(point),
                "point_key": list(point_key),
                "error_kind": error_kind,
                "error": error_text,
                "phase": primary.get("phase"),
                "retained_rollouts": retained_rollouts,
            },
        )


def _method_return_spec(method: Any) -> Any:
    from qval.mc_rollout_store import ReturnSpec

    return ReturnSpec(
        discount_factor=method.discount_factor,
        reward_signal_name=method.reward_signal_name,
        step_penalty=getattr(method, "step_penalty", None),
    )


def _raw_from_state_trajectories(
    method: Any,
    points: list[EvaluationPoint],
    trajectories_by_point: dict[tuple[int, ...], tuple[Any, ...]],
    point_keys: list[tuple[int, ...]],
    *,
    num_rollouts: int,
) -> list[RawPointReturns]:
    from qval.mc_rollout_store import stored_trajectory_return

    return [
        RawPointReturns(
            point=point,
            returns=tuple(
                stored_trajectory_return(trajectory, _method_return_spec(method))
                for trajectory in trajectories_by_point[key][:num_rollouts]
            ),
            step_counts=tuple(
                len(trajectory.transitions)
                for trajectory in trajectories_by_point[key][:num_rollouts]
            ),
            aborted=len(trajectories_by_point[key][:num_rollouts]) < num_rollouts,
        )
        for point, key in zip(points, point_keys, strict=True)
    ]


def _raw_from_state_action_trajectories(
    method: Any,
    points: list[EvaluationPoint],
    trajectories_by_point: dict[tuple[int, ...], tuple[Any, ...]],
    point_keys: list[tuple[int, ...]],
    *,
    num_rollouts: int,
) -> list[RawPointReturns]:
    from qval.mc_rollout_store import stored_trajectory_return

    return [
        RawPointReturns(
            point=point,
            returns=tuple(
                stored_trajectory_return(trajectory, _method_return_spec(method))
                for trajectory in trajectories_by_point[key][:num_rollouts]
            ),
            step_counts=tuple(
                len(trajectory.transitions)
                for trajectory in trajectories_by_point[key][:num_rollouts]
            ),
            next_state_returns=tuple(
                stored_trajectory_return(
                    trajectory,
                    _method_return_spec(method),
                    start_index=1,
                )
                for trajectory in trajectories_by_point[key][:num_rollouts]
            ),
            next_state_step_counts=tuple(
                max(0, len(trajectory.transitions) - 1)
                for trajectory in trajectories_by_point[key][:num_rollouts]
            ),
            aborted=len(trajectories_by_point[key][:num_rollouts]) < num_rollouts,
        )
        for point, key in zip(points, point_keys, strict=True)
    ]


def _raw_from_advantage_trajectories(
    method: Any,
    points: list[EvaluationPoint],
    state_trajectories: dict[tuple[int, ...], tuple[Any, ...]],
    next_state_trajectories: dict[tuple[int, ...], tuple[Any, ...]],
    point_keys: list[tuple[int, ...]],
    *,
    num_rollouts: int,
) -> list[RawPointReturns]:
    from qval.mc_rollout_store import stored_trajectory_return

    results: list[RawPointReturns] = []
    for point, key in zip(points, point_keys, strict=True):
        state_rollouts = state_trajectories[key][:num_rollouts]
        next_rollouts = next_state_trajectories[key][:num_rollouts]
        results.append(
            RawPointReturns(
                point=point,
                returns=tuple(
                    stored_trajectory_return(trajectory, _method_return_spec(method))
                    for trajectory in state_rollouts
                ),
                step_counts=tuple(len(trajectory.transitions) for trajectory in state_rollouts),
                next_state_returns=tuple(
                    stored_trajectory_return(trajectory, _method_return_spec(method))
                    for trajectory in next_rollouts
                ),
                next_state_step_counts=tuple(
                    len(trajectory.transitions) for trajectory in next_rollouts
                ),
                aborted=(
                    len(state_rollouts) < num_rollouts
                    or len(next_rollouts) < num_rollouts
                ),
            )
        )
    return results


def _store_root_for_spec(root_dir: Path, generation_spec: Any) -> Path:
    return (
        root_dir
        / generation_spec.namespace
        / generation_spec.primitive
        / generation_spec.dataset_fingerprint.sha256
        / generation_spec.generation_hash
    )


def _resolve_sibling_store_root(expected: Path) -> Path | None:
    """Return the unique non-empty sibling rollout dir under ``expected.parent``.

    Used to recover from additive ``BackendConfig`` changes that altered
    the generation_hash but left the underlying rollouts semantically
    valid. A candidate is a sibling directory whose ``manifest.json``
    both lists at least one shard *and* has the corresponding
    ``shards/*.pkl`` files on disk — empty-init stores and
    manifest-without-data stores are skipped because adopting them would
    be a no-op disguised as a recovery (logging a misleading "adopted"
    warning). Returns ``None`` when the parent doesn't exist or no
    sibling has recoverable data. Raises ``ValueError`` when more than
    one sibling has data — adoption would be ambiguous, and the user
    must remove all but one before retrying. JSON parse errors and
    deleted manifests are treated as "not a candidate"; OS-level errors
    (PermissionError, etc.) propagate so real disk problems aren't
    silently masked as "no sibling found".
    """
    import json as _json

    parent = expected.parent
    if not parent.exists():
        return None

    def _has_recoverable_data(candidate: Path) -> bool:
        manifest_path = candidate / "manifest.json"
        try:
            data = _json.loads(manifest_path.read_text())
        except (FileNotFoundError, _json.JSONDecodeError):
            return False
        if not data.get("shards"):
            return False
        # The manifest claims shards; verify the listed .pkl files actually
        # exist on disk. ``MCRolloutStore.open`` reconciles against actual
        # files, but the trust path would have already fired an "Adopting ..."
        # warning by then — misleading the user about what happened.
        shards_dir = candidate / "shards"
        if not shards_dir.is_dir():
            return False
        for shard in data["shards"]:
            if not isinstance(shard, dict):
                return False
            filename = shard.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                return False
            if not (shards_dir / filename).is_file():
                return False
        return True

    candidates = sorted(
        path for path in parent.iterdir()
        if path.is_dir()
        and path != expected
        and (path / "manifest.json").exists()
        and _has_recoverable_data(path)
    )
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "trust_sibling_hash: refusing to adopt; multiple sibling rollout "
            f"stores with data found under {parent}. Remove all but one and retry. "
            f"Candidates (under that parent): {[p.name for p in candidates]}"
        )
    return candidates[0]


def _is_empty_rollout_store(root: Path) -> bool:
    """Return True for a placeholder store with no recoverable data."""
    import json as _json

    if not root.exists():
        return False
    try:
        children = list(root.iterdir())
    except FileNotFoundError:
        return False
    if any(child.name not in {"manifest.json", "shards"} for child in children):
        return False

    shards_dir = root / "shards"
    if shards_dir.exists():
        if not shards_dir.is_dir():
            return False
        if any(shards_dir.iterdir()):
            return False

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            data = _json.loads(manifest_path.read_text())
        except _json.JSONDecodeError:
            return False
        if data.get("shards") or data.get("point_counts"):
            return False

    return True


def _remove_empty_rollout_store(root: Path) -> None:
    """Remove a placeholder store after _is_empty_rollout_store() approved it."""
    shards_dir = root / "shards"
    if shards_dir.exists():
        shards_dir.rmdir()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    root.rmdir()


def _ensure_primitive_trajectories(
    method: Any,
    primitive: str,
    points: list[EvaluationPoint],
    point_keys: list[tuple[int, ...]],
    *,
    namespace: str,
    required_rollouts: int,
    rollout_store_root: Path,
    dataset_fingerprint: Any,
    resume: bool,
    trajectory_cache: dict[str, dict[tuple[int, ...], tuple[Any, ...]]],
    store_cache: dict[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
    required_rollouts_by_point: dict[tuple[int, ...], int] | None = None,
    collect_missing: bool = True,
    failure_recorder: dict[tuple[int, ...], tuple[dict[str, Any], ...]] | None = None,
    trust_sibling_hash: bool = False,
    adopted_siblings: dict[str, str] | None = None,
) -> tuple[dict[tuple[int, ...], tuple[Any, ...]], set[tuple[int, ...]]]:
    # ``trust_sibling_hash``: when the expected hash dir is missing,
    # adopt a sibling under the same dataset-fingerprint parent. See
    # ``MCRolloutPersistenceConfig.trust_sibling_hash`` for the policy.
    # ``adopted_siblings`` maps per-run physical paths to the requested
    # generation_hash that claimed them. The same path may only be reused
    # for the same requested spec; a different spec would silently share
    # rollouts via ``store_cache``/``trajectory_cache``.
    from qval.mc_rollout_store import MCRolloutStore, stored_trajectory_from_live

    target_counts = (
        dict(required_rollouts_by_point)
        if required_rollouts_by_point is not None
        else {point_key: required_rollouts for point_key in point_keys}
    )

    aborted_keys: set[tuple[int, ...]] = set()

    if max(target_counts.values(), default=0) <= 0:
        return {point_key: () for point_key in point_keys}, aborted_keys

    generation_spec = method.build_generation_spec(
        namespace=namespace,
        primitive=primitive,
        dataset_fingerprint=dataset_fingerprint,
    )
    store_root = _store_root_for_spec(rollout_store_root, generation_spec)
    # `adopted_from_sibling` propagates to MCRolloutStore.open as
    # `trust_existing_hash`, which is where the WARNING-level diagnostic
    # (with field-level diff) is emitted. We log only an INFO breadcrumb
    # here so the user can correlate the chosen path with the missing
    # expected one without duplicating the safety warning.
    #
    # On adoption we *rename* the sibling into the expected (requested-
    # hash) directory before opening, restoring the dirname == manifest
    # generation_hash invariant. This lets a follow-up run for the same
    # spec find the rollouts at the expected path without needing the
    # recovery flag again.
    adopted_from_sibling = False
    expected_empty_store = _is_empty_rollout_store(store_root)
    if (
        trust_sibling_hash
        and (not (store_root / "manifest.json").exists() or expected_empty_store)
    ):
        sibling_root = _resolve_sibling_store_root(store_root)
        if sibling_root is not None:
            sibling_key = str(sibling_root)
            requested_hash = generation_spec.generation_hash
            previous_hash = (
                adopted_siblings.get(sibling_key)
                if adopted_siblings is not None
                else None
            )
            if previous_hash is not None and previous_hash != requested_hash:
                raise ValueError(
                    "trust_sibling_hash: refusing to adopt sibling "
                    f"{sibling_root} for spec {requested_hash} — "
                    "this physical sibling was already adopted earlier in this "
                    f"run for different spec {previous_hash}. The recovery flag does not "
                    "support sharing one set of rollouts across multiple "
                    "semantically different specs. If the rollouts really are "
                    "valid for both, manually duplicate the sibling directory "
                    "under each expected hash path and rerun without the flag."
                )
            if expected_empty_store:
                _remove_empty_rollout_store(store_root)
            elif store_root.exists():
                # Empty placeholder dir from a prior partial init: rmdir is
                # safe. Anything else means a concurrent writer or stale
                # state we shouldn't clobber.
                try:
                    store_root.rmdir()
                except OSError as exc:
                    raise ValueError(
                        "trust_sibling_hash: cannot rename sibling "
                        f"{sibling_root} into expected path {store_root} "
                        "because that path exists and is non-empty. Manual "
                        "cleanup required before retrying."
                    ) from exc
            sibling_root.rename(store_root)
            _log.info(
                "trust_sibling_hash: adopted sibling rollout store %s by "
                "renaming it into expected path %s",
                sibling_root,
                store_root,
            )
            adopted_from_sibling = True
            if adopted_siblings is not None:
                # After rename, the data lives at ``store_root``. A second
                # group whose sibling search resolves to the same physical
                # location (now ``store_root``) must trip the collision
                # check above, so track the post-rename path.
                adopted_siblings[sibling_key] = requested_hash
                adopted_siblings[str(store_root)] = requested_hash
    store_key = str(store_root)

    store = store_cache.get(store_key)
    if store is None:
        store = MCRolloutStore.open(
            store_root,
            generation_spec,
            trust_existing_hash=adopted_from_sibling,
        )
        store_cache[store_key] = store
    cached = trajectory_cache.setdefault(store_key, {})

    loaded_total = 0
    if resume:
        load_counts: dict[tuple[int, ...], int] = {}
        for point_key in point_keys:
            cached_count = len(cached.get(point_key, ()))
            available_count = store.count_for(point_key)
            target_count = min(target_counts.get(point_key, 0), available_count)
            if target_count > cached_count:
                load_counts[point_key] = target_count - cached_count
        if load_counts:
            loaded = store.load_point_trajectories(load_counts)
            for point_key, trajectories in loaded.items():
                cached[point_key] = cached.get(point_key, ()) + trajectories
                loaded_total += len(trajectories)

    missing_items: list[tuple[EvaluationPoint, tuple[int, ...], int]] = []
    for point, point_key in zip(points, point_keys, strict=True):
        missing = target_counts.get(point_key, 0) - len(cached.get(point_key, ()))
        if missing > 0:
            missing_items.append((point, point_key, missing))

    total_requested = sum(target_counts.values())
    total_to_collect = sum(m for _, _, m in missing_items)
    if loaded_total or total_to_collect < total_requested:
        _log.info(
            "Rollout store %s/%s: %d points, %d rollouts requested, "
            "%d loaded from store, %d to collect",
            namespace, primitive, len(point_keys), total_requested,
            loaded_total, total_to_collect,
        )

    if missing_items and collect_missing:
        grouped: dict[int, list[tuple[EvaluationPoint, tuple[int, ...]]]] = {}
        for point, point_key, missing in missing_items:
            grouped.setdefault(missing, []).append((point, point_key))

        completed = 0
        total = len(missing_items)
        method_batch_size = getattr(method, "batch_size", None)
        for missing_count, grouped_items in sorted(grouped.items()):
            # Derive shard size from the natural parallel lockstep unit:
            # one shard per batch of rollouts. With batch_size=4 and
            # missing_count=4, that is 1 point/shard — durability matches
            # parallelism. When batch_size is None ("no chunking"), flush
            # per point for crash safety on slow rollouts.
            shard_points = max(
                1, (method_batch_size or 1) // max(1, missing_count),
            )
            for chunk in _chunked(grouped_items, shard_points):
                chunk_points = [point for point, _ in chunk]
                if primitive == "state":
                    collector = method.collect_state_trajectories
                elif primitive == "state_action":
                    collector = method.collect_state_action_trajectories
                elif primitive == "next_state":
                    collector = method.collect_next_state_trajectories
                else:
                    raise ValueError(f"Unknown primitive: {primitive}")

                def _progress(done: int, _total: int) -> None:
                    if progress_callback is None:
                        return
                    progress_callback(completed + done, total)

                live, chunk_aborted = retry_on_transient(
                    lambda: collector(
                        chunk_points,
                        num_rollouts=missing_count,
                        progress_callback=_progress,
                        on_generation_error=_on_generation_error,
                    ),
                    max_retries=MC_ROLLOUT_MAX_RETRIES,
                    base_delay=MC_ROLLOUT_RETRY_BASE_DELAY,
                    description=f"MC shard {namespace}/{primitive}",
                )
                # Map local aborted indices to point keys
                for local_idx in chunk_aborted:
                    _, aborted_point_key = chunk[local_idx]
                    aborted_keys.add(aborted_point_key)
                    if (
                        failure_recorder is not None
                        and hasattr(method, "last_rollout_failures")
                        and local_idx in method.last_rollout_failures
                    ):
                        failure_recorder[aborted_point_key] = (
                            failure_recorder.get(aborted_point_key, ())
                            + tuple(method.last_rollout_failures[local_idx])
                        )
                stored_chunk = {
                    point_key: tuple(
                        stored_trajectory_from_live(trajectory)
                        for trajectory in point_trajectories
                    )
                    for (_, point_key), point_trajectories in zip(
                        chunk,
                        live,
                        strict=True,
                    )
                    if point_trajectories  # skip aborted points (empty tuples)
                }
                store.append(stored_chunk)
                for point_key, trajectories in stored_chunk.items():
                    cached[point_key] = cached.get(point_key, ()) + trajectories
                completed += len(chunk)

    return {
        point_key: cached.get(point_key, ())[:target_counts.get(point_key, 0)]
        for point_key in point_keys
    }, aborted_keys


def collect_grouped_gt(
    gt_methods: dict[str, DenseSignalMethod],
    points: list[EvaluationPoint],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    logger: ExperimentLogger | None = None,
    rollout_store_root: str | Path | None = None,
    dataset_fingerprint: Any | None = None,
    point_keys: list[tuple[int, ...]] | None = None,
    point_namespace: str = "evaluation_points",
    resume: bool = True,
    trust_sibling_hash: bool = False,
    on_group_complete: Callable[[str, list[float], list[RawPointReturns], set[int]], None] | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[RawPointReturns]], dict[str, set[int]]]:
    """Collect ground truth values with rollout reuse and V(s) cross-group sharing.

    Groups GT methods by ``rollout_identity()``, collects rollouts once per
    group, and reuses V(s) returns from "v_s" groups for "advantage" groups
    when the non-rollout-type parameters match.

    Shard-flush granularity for persistent MC storage is derived internally
    as ``method.batch_size // missing_count`` (clamped to at least 1) — i.e.
    one shard per natural parallel lockstep unit. This keeps durability
    aligned with the batch size without a separate config knob.

    Args:
        gt_methods: Dict mapping estimation names to DenseSignalMethod instances.
        points: Evaluation points to estimate at.
        progress_callback: Optional callback(completed, total) for the current
            collection unit. For MC-based methods this may correspond to rollout
            progress; for other methods it may correspond to evaluation points.
        logger: Optional ExperimentLogger for phase tagging.
        rollout_store_root: Optional root directory for persistent MC rollout
            storage. When provided together with ``dataset_fingerprint``,
            MC groups load/save append-only trajectory shards and derive raw
            returns from stored trajectories.
        dataset_fingerprint: Required when ``rollout_store_root`` is set.
        point_keys: Optional logical point ids aligned with ``points``.
        point_namespace: Persistent point namespace (evaluation vs ranking).
        resume: When True, load existing compatible trajectory shards before
            collecting missing rollouts.

    Returns:
        A tuple of ``(gt_values, raw_returns, aborted_points)`` where:
        - ``gt_values[name]`` is a list of aggregated GT values for each point
        - ``raw_returns[name]`` is a list of RawPointReturns (shared across
          methods in the same rollout group)
        - ``aborted_points[name]`` is a set of point indices that were aborted
          due to backend errors
    """
    from qval.mc_method import MCMethod
    from qval.mcts_method import MCTSMethod

    persistent_mc = rollout_store_root is not None and dataset_fingerprint is not None
    point_keys = _default_point_keys(points) if point_keys is None else point_keys
    store_cache: dict[str, Any] = {}
    trajectory_cache: dict[str, dict[tuple[int, ...], tuple[Any, ...]]] = {}
    # Track sibling-store paths already adopted via trust_sibling_hash in
    # this call, along with the requested generation hash that claimed
    # them. Two different specs claiming the same sibling would otherwise
    # silently share rollouts through the caches above.
    adopted_siblings: dict[str, str] = {}

    # Group methods by rollout identity.
    # Include backend identity so methods using different backends never share.
    groups: dict[tuple, list[tuple[str, DenseSignalMethod]]] = {}
    order: list[tuple] = []
    for name, method in gt_methods.items():
        if isinstance(method, (MCMethod, MCTSMethod)):
            key = (id(method.policy.backend),) + method.rollout_identity()
        else:
            key = (id(method),)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((name, method))

    # Classify groups by rollout type
    v_s_groups: list[tuple] = []
    q_sa_groups: list[tuple] = []
    advantage_groups: list[tuple] = []
    other_groups: list[tuple] = []

    for key in order:
        representative = groups[key][0][1]
        if isinstance(representative, (MCMethod, MCTSMethod)):
            # Key format: (backend_id, method_type, rollout_type, ...)
            rollout_type = key[2] if len(key) > 2 else None
            if rollout_type == "v_s":
                v_s_groups.append(key)
            elif rollout_type == "q_sa":
                q_sa_groups.append(key)
            elif rollout_type == "advantage":
                advantage_groups.append(key)
            else:
                other_groups.append(key)
        else:
            other_groups.append(key)

    gt_values: dict[str, list[float]] = {}
    raw_returns: dict[str, list[RawPointReturns]] = {}
    aborted_points: dict[str, set[int]] = {}

    # Build V(s) cache keyed by non-rollout-type params
    v_s_cache_map: dict[tuple, list[RawPointReturns]] = {}
    # Build Q continuation cache: V(s') from Q_VALUE rollouts
    q_continuation_cache_map: dict[tuple, list[RawPointReturns]] = {}

    def _collect_group(
        key: tuple,
        v_s_cache: list[RawPointReturns] | None = None,
        v_next_cache: list[RawPointReturns] | None = None,
    ) -> None:
        group = groups[key]
        rep_name, rep_method = group[0]
        group_progress = progress_callback

        if (
            logger
            and group_progress is None
            and isinstance(rep_method, (MCMethod, MCTSMethod))
        ):
            unit_name = "rollouts" if isinstance(rep_method, MCMethod) else "points"
            group_progress = logger.make_progress_reporter(
                name=f"GT {rep_name} {unit_name}",
                logger_name=__name__,
            )

        group_aborted: set[int] = set()
        failure_details: dict[tuple[int, ...], tuple[dict[str, Any], ...]] = {}

        if isinstance(rep_method, (MCMethod, MCTSMethod)):
            if logger:
                logger.set_phase(f"gt:{rep_name}")
            requested_rollouts: int | None = None
            if isinstance(rep_method, MCMethod) and persistent_mc:
                mc_group = [
                    method for _, method in group
                    if isinstance(method, MCMethod)
                ]
                requested_rollouts = max(method.num_rollouts for method in mc_group)
                rollout_root_path = Path(rollout_store_root)

                # Helper: map aborted point keys back to indices
                key_to_idx = {pk: i for i, pk in enumerate(point_keys)}

                def _keys_to_indices(aborted_keys: set[tuple[int, ...]]) -> set[int]:
                    return {key_to_idx[k] for k in aborted_keys if k in key_to_idx}

                if rep_method.context.signal_type in (
                    SignalType.STATE_VALUE,
                    SignalType.POTENTIAL,
                ):
                    state_trajectories, aborted_keys = _ensure_primitive_trajectories(
                        rep_method,
                        "state",
                        points,
                        point_keys,
                        namespace=point_namespace,
                        required_rollouts=requested_rollouts,
                        rollout_store_root=rollout_root_path,
                        dataset_fingerprint=dataset_fingerprint,
                        resume=resume,
                        trajectory_cache=trajectory_cache,
                        store_cache=store_cache,
                        progress_callback=group_progress,
                        failure_recorder=failure_details,
                        trust_sibling_hash=trust_sibling_hash,
                        adopted_siblings=adopted_siblings,
                    )
                    group_aborted |= _keys_to_indices(aborted_keys)
                    raw = _raw_from_state_trajectories(
                        rep_method,
                        points,
                        state_trajectories,
                        point_keys,
                        num_rollouts=requested_rollouts,
                    )
                elif rep_method.context.signal_type == SignalType.Q_VALUE:
                    state_action_trajectories, aborted_keys = _ensure_primitive_trajectories(
                        rep_method,
                        "state_action",
                        points,
                        point_keys,
                        namespace=point_namespace,
                        required_rollouts=requested_rollouts,
                        rollout_store_root=rollout_root_path,
                        dataset_fingerprint=dataset_fingerprint,
                        resume=resume,
                        trajectory_cache=trajectory_cache,
                        store_cache=store_cache,
                        progress_callback=group_progress,
                        failure_recorder=failure_details,
                        trust_sibling_hash=trust_sibling_hash,
                        adopted_siblings=adopted_siblings,
                    )
                    group_aborted |= _keys_to_indices(aborted_keys)
                    raw = _raw_from_state_action_trajectories(
                        rep_method,
                        points,
                        state_action_trajectories,
                        point_keys,
                        num_rollouts=requested_rollouts,
                    )
                elif rep_method.context.signal_type == SignalType.ADVANTAGE:
                    state_trajectories, aborted_keys_s = _ensure_primitive_trajectories(
                        rep_method,
                        "state",
                        points,
                        point_keys,
                        namespace=point_namespace,
                        required_rollouts=requested_rollouts,
                        rollout_store_root=rollout_root_path,
                        dataset_fingerprint=dataset_fingerprint,
                        resume=resume,
                        trajectory_cache=trajectory_cache,
                        store_cache=store_cache,
                        failure_recorder=failure_details,
                        trust_sibling_hash=trust_sibling_hash,
                        adopted_siblings=adopted_siblings,
                    )
                    group_aborted |= _keys_to_indices(aborted_keys_s)
                    state_action_needs = {
                        point_key: 0 if point_key in aborted_keys_s else requested_rollouts
                        for point_key in point_keys
                    }
                    q_state_action_trajectories, aborted_keys_q = _ensure_primitive_trajectories(
                        rep_method,
                        "state_action",
                        points,
                        point_keys,
                        namespace=point_namespace,
                        required_rollouts=0,
                        required_rollouts_by_point=state_action_needs,
                        rollout_store_root=rollout_root_path,
                        dataset_fingerprint=dataset_fingerprint,
                        resume=resume,
                        trajectory_cache=trajectory_cache,
                        store_cache=store_cache,
                        collect_missing=False,
                        failure_recorder=failure_details,
                        trust_sibling_hash=trust_sibling_hash,
                        adopted_siblings=adopted_siblings,
                    )
                    group_aborted |= _keys_to_indices(aborted_keys_q)
                    next_state_needs = {
                        point_key: 0
                        if point_key in aborted_keys_s or point_key in aborted_keys_q
                        else max(
                            0,
                            requested_rollouts - len(q_state_action_trajectories.get(point_key, ())),
                        )
                        for point_key in point_keys
                    }
                    next_state_trajectories, aborted_keys_ns = _ensure_primitive_trajectories(
                        rep_method,
                        "next_state",
                        points,
                        point_keys,
                        namespace=point_namespace,
                        required_rollouts=0,
                        required_rollouts_by_point=next_state_needs,
                        rollout_store_root=rollout_root_path,
                        dataset_fingerprint=dataset_fingerprint,
                        resume=resume,
                        trajectory_cache=trajectory_cache,
                        store_cache=store_cache,
                        progress_callback=group_progress,
                        failure_recorder=failure_details,
                        trust_sibling_hash=trust_sibling_hash,
                        adopted_siblings=adopted_siblings,
                    )
                    group_aborted |= _keys_to_indices(aborted_keys_ns)
                    raw = _raw_from_state_trajectories(
                        rep_method,
                        points,
                        state_trajectories,
                        point_keys,
                        num_rollouts=requested_rollouts,
                    )
                    q_raw = _raw_from_state_action_trajectories(
                        rep_method,
                        points,
                        q_state_action_trajectories,
                        point_keys,
                        num_rollouts=requested_rollouts,
                    )
                    next_raw = _raw_from_state_trajectories(
                        rep_method,
                        points,
                        next_state_trajectories,
                        point_keys,
                        num_rollouts=requested_rollouts,
                    )
                    raw = [
                        RawPointReturns(
                            point=state_item.point,
                            returns=state_item.returns,
                            step_counts=state_item.step_counts,
                            next_state_returns=(
                                q_item.next_state_returns + next_item.returns
                            )[:requested_rollouts],
                            next_state_step_counts=(
                                q_item.next_state_step_counts + next_item.step_counts
                            )[:requested_rollouts],
                            aborted=(
                                state_item.aborted or q_item.aborted or next_item.aborted
                            ),
                        )
                        for state_item, q_item, next_item in zip(
                            raw,
                            q_raw,
                            next_raw,
                            strict=True,
                        )
                    ]
                else:
                    raise ValueError(f"Unsupported MC signal type: {rep_method.context.signal_type}")

            elif isinstance(rep_method, MCMethod):
                mc_group = [
                    method for _, method in group
                    if isinstance(method, MCMethod)
                ]
                requested_rollouts = max(method.num_rollouts for method in mc_group)
                raw, group_aborted = retry_on_transient(
                    lambda: rep_method.collect_returns(
                        points,
                        progress_callback=group_progress,
                        v_s_cache=v_s_cache,
                        v_next_cache=v_next_cache,
                        num_rollouts=requested_rollouts,
                    ),
                    max_retries=MC_ROLLOUT_MAX_RETRIES,
                    base_delay=MC_ROLLOUT_RETRY_BASE_DELAY,
                    description=f"MC collect_returns {rep_name}",
                )
                failure_details = {
                    point_keys[idx]: tuple(rep_method.last_rollout_failures.get(idx, ()))
                    for idx in group_aborted
                }
            else:
                raw, group_aborted = rep_method.collect_returns(
                    points,
                    progress_callback=group_progress,
                    v_s_cache=v_s_cache,
                    v_next_cache=v_next_cache,
                )
        else:
            if logger:
                logger.set_phase(f"gt:{rep_name}")
            raw = None

        _log_rollout_abort_events(
            logger,
            method_name=rep_name,
            points=points,
            point_keys=point_keys,
            aborted_indices=group_aborted,
            raw=raw,
            failure_details=failure_details,
        )

        sliced_cache: dict[int, list[RawPointReturns]] = {}
        for name, method in group:
            if raw is not None and isinstance(method, (MCMethod, MCTSMethod)):
                if isinstance(method, MCMethod):
                    requested = method.num_rollouts
                    if requested == (requested_rollouts or requested):
                        method_raw = raw
                    else:
                        method_raw = sliced_cache.get(requested)
                        if method_raw is None:
                            method_raw = _slice_raw_point_returns(raw, requested)
                            sliced_cache[requested] = method_raw
                else:
                    method_raw = raw
                raw_returns[name] = method_raw
                is_adv = method.context.signal_type == SignalType.ADVANTAGE
                estimates = aggregate_returns(
                    method_raw, method.aggregation, is_advantage=is_adv,
                )
                gt_values[name] = [e.value for e in estimates]
            else:
                if logger:
                    logger.set_phase(f"gt:{name}")
                gt_values[name] = method.evaluate_batch(points)
                raw_returns[name] = []
            aborted_points[name] = group_aborted

        if on_group_complete is not None:
            for name, _ in group:
                on_group_complete(
                    name, gt_values[name], raw_returns[name], aborted_points[name],
                )

    # 1. Process V(s) groups first
    for key in v_s_groups:
        _collect_group(key)
        cache_key = _v_s_rollout_key(key)
        first_name = groups[key][0][0]
        v_s_cache_map[cache_key] = raw_returns[first_name]

    # 2. Process Q(s,a) groups — cache continuation returns for ADVANTAGE
    for key in q_sa_groups:
        _collect_group(key)
        cache_key = _v_s_rollout_key(key)
        first_name = groups[key][0][0]
        q_continuation_cache_map[cache_key] = raw_returns[first_name]

    # 3. Process advantage groups with V(s) and Q continuation caches
    for key in advantage_groups:
        cache_key = _v_s_rollout_key(key)
        v_s_cached = v_s_cache_map.get(cache_key)
        q_cont_cached = q_continuation_cache_map.get(cache_key)
        _collect_group(key, v_s_cache=v_s_cached, v_next_cache=q_cont_cached)

    # 4. Process remaining groups normally
    for key in other_groups:
        _collect_group(key)

    return gt_values, raw_returns, aborted_points

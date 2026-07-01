"""Fingerprint-keyed rolling checkpoints for resumable pipeline runs.

A run (test-time scaling strategy, trajectory collection) periodically saves
its completed trajectories to a rolling ``Dataset`` pickle whose filename
embeds a fingerprint of every content-affecting config field. On restart with
the same config, the exact path is recomputed, the checkpoint is loaded, and
only the missing work runs. A config change yields a different fingerprint —
and therefore a different file — so stale checkpoints are never reused; they
are simply left in place as orphans (safe to delete manually).

Only successful trajectories are checkpointed: failed, skipped, or discarded
ones are retried on resume. Checkpoints are deleted after the run's final
artifacts are saved.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qval.data_cache import (
    Dataset,
    _task_index_for_result,
    load_dataset,
    save_dataset,
)
from qval.ranking_identity import _normalize_descriptor, stable_hash

logger = logging.getLogger(__name__)

RESUME_FINGERPRINT_SCHEMA_VERSION = 1

# Stripped recursively from descriptors before hashing so credentials never
# influence (or leak into) fingerprints.
_CREDENTIAL_KEYS = frozenset(
    {"api_key", "api_key_env", "token", "authorization", "secret"}
)

# BackendConfig fields that do not affect generated content: transport,
# placement, parallelism, container plumbing, and retry/rate-limit policy.
_BACKEND_INFRA_FIELDS = frozenset(
    {
        "backend_url",
        "device",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "vllm_kwargs",
        "singularity_sif",
        "singularity_binds",
        "singularity_extra_vllm_args",
        "singularity_startup_timeout",
        "singularity_cuda_visible_devices",
        "max_concurrency",
        "rate_limit_wait",
        "rate_limit_max_retries",
        "connect_timeout",
        "api_max_retries",
        "quota_retry_policy",
    }
)


def backend_content_descriptor(backend_config: Any) -> dict[str, Any]:
    """Reduce a ``BackendConfig`` to its content-affecting fields.

    Keeps model identity and generation behavior (type, model, dtype,
    max_model_len, enable_thinking, sampling, provider preferences, CLI
    overrides); drops transport/parallelism/retry plumbing.
    """
    descriptor = dataclasses.asdict(backend_config)
    return {
        key: value
        for key, value in descriptor.items()
        if key not in _BACKEND_INFRA_FIELDS
    }


# EnvironmentContext fields that shape prompts/episodes. Only plain scalar
# fields are fingerprinted: extractor objects can hold callables whose repr
# embeds memory addresses, which would make the fingerprint differ on every
# process and silently defeat resume.
_CONTEXT_CONTENT_FIELDS = (
    "env_name",
    "adapter",
    "max_steps",
    "make_kwargs",
    "step_penalty",
    "reward_signal_name",
    "prompting_scheme",
    "system_prompt_file",
    "turn_info",
    "min_action_chars",
    "min_observation_chars",
    "min_current_observation_chars",
    "early_turns_to_discard",
    "late_turns_to_discard",
    "inject_task_text_in_prompts",
    "invalid_action_text",
    "invalid_action_observation",
    "advance_on_invalid",
)

# make_kwargs keys that select runtime/transport rather than content (mirrors
# predict.py's rollout-persistence normalization for harbor environments).
_INFRA_MAKE_KWARGS = frozenset(
    {
        "environment_type",
        "podman_command",
        "apptainer_command",
        "trials_dir",
        "sif_cache_dir",
        "browsergym_call_timeout",
    }
)


def environment_context_descriptor(context: Any) -> dict[str, Any]:
    """Content-affecting scalar fields of a loaded environment context."""
    return {name: getattr(context, name, None) for name in _CONTEXT_CONTENT_FIELDS}


def content_make_kwargs(
    make_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Drop runtime/transport make_kwargs keys before fingerprinting."""
    if make_kwargs is None:
        return None
    return {
        key: value
        for key, value in make_kwargs.items()
        if key not in _INFRA_MAKE_KWARGS
    }


def _strip_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_credentials(inner)
            for key, inner in value.items()
            if key not in _CREDENTIAL_KEYS
        }
    if isinstance(value, list):
        return [_strip_credentials(item) for item in value]
    return value


def build_resume_fingerprint(descriptor: Mapping[str, Any]) -> str:
    """Hash a descriptor of content-affecting run parameters (16-char sha256).

    Normalization makes the hash independent of dict ordering, tuple-vs-list
    distinctions, and dataclass-vs-dict representation; credential-like keys
    are removed at any nesting depth.
    """
    normalized = _normalize_descriptor(dict(descriptor))
    return stable_hash(_strip_credentials(normalized))


def checkpoint_path(directory: Path | str, stem: str, fingerprint: str) -> Path:
    """Stable (non-timestamped) checkpoint location for a run fingerprint."""
    return Path(directory) / f"{stem}_{fingerprint}.checkpoint.pkl"


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A validated rolling checkpoint loaded from disk."""

    path: Path
    trajectory_results: list[Any]
    metadata: dict[str, Any]
    config: dict[str, Any]


def save_rolling_checkpoint(
    path: Path | str,
    *,
    trajectory_results: Sequence[Any],
    config_dict: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    """Atomically persist the trajectories completed so far.

    Never raises into the run: checkpointing is best-effort and a failed
    flush only costs recovery granularity. Returns ``False`` on failure.
    """
    try:
        checkpoint = Dataset(
            evaluation_points=[],
            trajectory_results=list(trajectory_results),
            config=config_dict,
            metadata={
                "checkpoint": True,
                "checkpoint_timestamp": datetime.now().astimezone().isoformat(),
                **metadata,
            },
        )
        save_dataset(checkpoint, path)
        return True
    except Exception:
        logger.warning("Failed to save resume checkpoint to %s", path, exc_info=True)
        return False


def load_rolling_checkpoint(
    path: Path | str, *, expected_fingerprint: str
) -> LoadedCheckpoint | None:
    """Load and validate a rolling checkpoint; ``None`` when unusable.

    Missing file is the normal fresh-start case. Unreadable pickles, datasets
    that are not checkpoints, and fingerprint mismatches (belt-and-braces —
    the fingerprint is already part of the filename) warn and return ``None``
    so the run falls back to collecting from scratch.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        dataset = load_dataset(path)
    except Exception:
        logger.warning("Ignoring unreadable resume checkpoint %s", path, exc_info=True)
        return None
    metadata = dict(getattr(dataset, "metadata", None) or {})
    if metadata.get("checkpoint") is not True:
        logger.warning("Ignoring %s: not a resume checkpoint dataset", path)
        return None
    stored = metadata.get("resume_fingerprint")
    if stored != expected_fingerprint:
        logger.warning(
            "Ignoring resume checkpoint %s: fingerprint mismatch "
            "(stored %s, expected %s)",
            path,
            stored,
            expected_fingerprint,
        )
        return None
    results = getattr(dataset, "trajectory_results", None)
    if results is None:
        logger.warning("Ignoring resume checkpoint %s: no trajectory results", path)
        return None
    return LoadedCheckpoint(
        path=path,
        trajectory_results=list(results),
        metadata=metadata,
        config=dict(getattr(dataset, "config", None) or {}),
    )


class ResumeState:
    """Completed-trajectory tracker keyed by ``trajectory_index``.

    Single source of truth for a resumable per-trajectory rollout loop:
    loaded checkpoint results and fresh completions land in the same map, so
    ``ordered_results()`` reproduces the never-interrupted result sequence.
    """

    def __init__(self) -> None:
        self._results: dict[int, Any] = {}
        self._resumed_count = 0

    @classmethod
    def from_checkpoint(cls, loaded: LoadedCheckpoint | None) -> ResumeState:
        """Key checkpointed results by their ``trajectory_index`` metadata.

        Any result missing the key makes the whole checkpoint unusable
        (positions could not be trusted) — warn and start fresh.
        """
        state = cls()
        if loaded is None:
            return state
        keyed: dict[int, Any] = {}
        for result in loaded.trajectory_results:
            metadata = getattr(result, "metadata", None) or {}
            index = metadata.get("trajectory_index")
            if not isinstance(index, int):
                logger.warning(
                    "Resume checkpoint %s has results without trajectory_index "
                    "metadata; starting fresh",
                    loaded.path,
                )
                return cls()
            keyed[index] = result
        state._results = keyed
        state._resumed_count = len(keyed)
        return state

    @property
    def resumed_count(self) -> int:
        return self._resumed_count

    def record(self, trajectory_index: int, result: Any) -> None:
        self._results[trajectory_index] = result

    def is_completed(self, trajectory_index: int) -> bool:
        return trajectory_index in self._results

    def remaining(self, task_indices: Sequence[int]) -> list[tuple[int, int]]:
        """``(trajectory_index, task_index)`` pairs still to run, in order."""
        return [
            (position, task_index)
            for position, task_index in enumerate(task_indices)
            if position not in self._results
        ]

    def ordered_results(self) -> list[Any]:
        return [self._results[index] for index in sorted(self._results)]

    def completed_trajectory_indices(self) -> list[int]:
        return sorted(self._results)


def collection_completed_counts(
    trajectory_results: Sequence[Any],
) -> dict[int, Counter[int]] | None:
    """Per-policy multiset of completed task indices for collection resume.

    Task indices can repeat within a policy (``resolve_task_indices`` cycles
    when more trajectories than tasks are requested), so counts — not sets —
    are required. Returns ``None`` when any result lacks provenance (legacy
    checkpoints): the caller must then start fresh.
    """
    counts: dict[int, Counter[int]] = {}
    for result in trajectory_results:
        metadata = getattr(result, "metadata", None) or {}
        policy_index = metadata.get("collection_policy_index")
        task_index = _task_index_for_result(result)
        if not isinstance(policy_index, int) or task_index is None:
            logger.warning(
                "Resume checkpoint results lack collection provenance "
                "(collection_policy_index/task_index); cannot resume"
            )
            return None
        counts.setdefault(policy_index, Counter())[task_index] += 1
    return counts


def subtract_completed_tasks(
    requested: Sequence[int], completed: Counter[int]
) -> list[int]:
    """Order-preserving removal of one occurrence per completed count."""
    budget = Counter(completed)
    remaining: list[int] = []
    for task_index in requested:
        if budget[task_index] > 0:
            budget[task_index] -= 1
        else:
            remaining.append(task_index)
    return remaining

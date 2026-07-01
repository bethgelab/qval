"""Shared helpers for CLI scripts (collect_dataset, predict)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import logging
import os
import random
import shutil
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llenvs.inference import DEFAULT_EARLY_STOPPING_SUFFIX
from llenvs.inference.protocol import SamplingParams
from qval.config import (
    BackendConfig,
    BackendSamplingConfig,
    OpenRouterProviderPreferences,
)
from qval.types import EnvironmentContext

logger = logging.getLogger(__name__)

_HARBOR_PREFLIGHT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_HARBOR_PREFLIGHT_CACHE_LOCK = threading.Lock()

BACKEND_CHOICES = [
    "openai",
    "huggingface",
    "vllm",
    "vllm_singularity",
    "anthropic",
    "openrouter",
    "litellm",
    "codex",
    "claude_code",
]

_CLI_BACKEND_TYPES = frozenset({"codex", "claude_code"})

# Adapters whose reset() accepts task_index values beyond len(env) —
# Jericho wraps via modulo, Craftax derives a unique seed from any index.
# Adapters NOT listed here (e.g., ALFWorld, Harbor) bounds-check and raise
# on out-of-range indices, so scripts must cycle within [0, len(env)).
ADAPTERS_WITH_UNBOUNDED_TASK_INDEX: frozenset[str] = frozenset({
    "jericho",
    "craftax",
})


def resolve_task_indices(
    env: Any,
    *,
    adapter: str,
    num_tasks: int | None,
    shuffle: bool,
    seed: int,
    overflow: str = "cap",
    logger: logging.Logger | None = None,
) -> list[int]:
    """Resolve the ordered list of environment task indices to run.

    Owns the shared "derive task indices from the environment" logic used by the
    collection, Pass@k, and test-time-scaling pipelines. Callers handle their own
    explicit task-index lists and pre-filtered pools (e.g. Harbor snapshot /
    runtime filter reports) before falling back to this helper.

    Behavior:

    * ``num_tasks is None`` — every available task (``range(len(env))``).
    * ``num_tasks <= len(env)`` — the first ``num_tasks`` tasks.
    * ``num_tasks > len(env)`` — overflow handling:
        - Unbounded-task-index adapters (Jericho/Craftax) use ``range(num_tasks)``
          directly, since their ``reset()`` wraps or derives a seed per index.
        - Bounded adapters either ``cap`` at ``len(env)`` (keeps per-task sample
          counts even — correct for Pass@k / test-time scaling) or ``cycle``
          through the pool to reach ``num_tasks`` indices (correct for collection,
          which wants exactly ``num_tasks`` trajectories).

    When ``shuffle`` is true the task pool is shuffled with ``seed`` before
    truncation/cycling. Environments that do not support ``len()`` require
    ``num_tasks`` (otherwise ``ValueError``) and fall back to sequential
    ``range(num_tasks)`` indices.

    Args:
        env: The environment (probed via ``len(env)`` for its task count).
        adapter: Adapter name, used to detect unbounded-task-index adapters.
        num_tasks: Requested number of task slots (``None`` = all available).
        shuffle: Whether to shuffle the task pool before selection.
        seed: Seed for the shuffle.
        overflow: ``"cap"`` or ``"cycle"`` — how to handle ``num_tasks`` exceeding
            the available task count for bounded adapters.
        logger: Logger for capping/no-len warnings (defaults to this module's).

    Returns:
        The ordered list of task indices to run.
    """
    if overflow not in ("cap", "cycle"):
        raise ValueError(f"overflow must be 'cap' or 'cycle', got {overflow!r}")
    log = logger if logger is not None else logging.getLogger(__name__)

    try:
        total_tasks = len(env)
    except TypeError:
        total_tasks = None

    if total_tasks is None:
        if num_tasks is None:
            raise ValueError(
                "num_tasks must be set when the environment does not support len()"
            )
        log.warning(
            "Environment does not support len(); using sequential task indices "
            "[0, %d)",
            num_tasks,
        )
        indices = list(range(num_tasks))
        if shuffle:
            random.Random(seed).shuffle(indices)
        return indices

    indices = list(range(total_tasks))
    if shuffle:
        random.Random(seed).shuffle(indices)

    if num_tasks is None:
        return indices
    if num_tasks <= total_tasks:
        return indices[:num_tasks]

    # num_tasks > total_tasks
    if adapter in ADAPTERS_WITH_UNBOUNDED_TASK_INDEX:
        extended = list(range(num_tasks))
        if shuffle:
            random.Random(seed).shuffle(extended)
        return extended

    if overflow == "cycle":
        full_cycles = num_tasks // total_tasks
        remainder = num_tasks % total_tasks
        return indices * full_cycles + indices[:remainder]

    log.warning(
        "Requested %d tasks but environment only has %d available; capping at %d",
        num_tasks,
        total_tasks,
        total_tasks,
    )
    return indices


def resolve_effective_max_steps(
    max_steps: int | None,
    make_kwargs: Mapping[str, Any] | None,
) -> int | None:
    """Return the effective runtime step limit.

    When both top-level ``max_steps`` and ``make_kwargs["max_steps"]`` are
    present, they must agree. This avoids silently choosing between two
    conflicting representations of the same setting.
    """
    normalized_max_steps = None if max_steps is None else int(max_steps)
    if isinstance(make_kwargs, Mapping):
        make_kwargs_max_steps = make_kwargs.get("max_steps")
        if make_kwargs_max_steps is not None:
            normalized_make_kwargs_max_steps = int(make_kwargs_max_steps)
            if (
                normalized_max_steps is not None
                and normalized_make_kwargs_max_steps != normalized_max_steps
            ):
                raise ValueError(
                    "Conflicting max_steps values: "
                    f"max_steps={normalized_max_steps}, "
                    f"make_kwargs['max_steps']={normalized_make_kwargs_max_steps}"
                )
            return normalized_make_kwargs_max_steps
    return normalized_max_steps


def resolve_system_prompt(
    *,
    override_system_prompt: str | None,
    adapter: str | None = None,
    prompting_scheme: str | None = None,
    context_system_prompt_file: str | None = None,
    adapter_system_prompt: str | None = None,
    fallback_system_prompt: str | None = None,
    make_kwargs: dict[str, Any] | None = None,
    env_name: str | None = None,
) -> str | None:
    """Resolve actor system prompt: override > context file > composed > adapter > fallback."""
    if override_system_prompt is not None:
        return override_system_prompt
    if context_system_prompt_file is not None:
        return Path(context_system_prompt_file).read_text().strip()
    from qval.system_prompts import ENV_PROMPTS, build_actor_system_prompt

    if adapter is not None and adapter in ENV_PROMPTS:
        built = build_actor_system_prompt(
            adapter,
            prompting_scheme or "answer_tags",
            env_name=env_name,
            make_kwargs=make_kwargs,
        )
        if adapter_system_prompt:
            return built + "\n\n" + adapter_system_prompt
        return built
    if adapter_system_prompt is not None:
        return adapter_system_prompt
    return fallback_system_prompt


def resolve_ranking_system_prompt(
    *,
    override_system_prompt: str | None,
    adapter: str | None = None,
    prompting_scheme: str | None = None,
    context_system_prompt_file: str | None = None,
    adapter_system_prompt: str | None = None,
    fallback_system_prompt: str | None = None,
    make_kwargs: dict[str, Any] | None = None,
    env_name: str | None = None,
) -> str | None:
    """Resolve ranking-sampling system prompt.

    This mirrors :func:`resolve_system_prompt`, but the built-in composed prompt
    uses action-only ranking instructions instead of the rollout actor prompt.
    Explicit overrides still take precedence.
    """
    if override_system_prompt is not None:
        return override_system_prompt
    if context_system_prompt_file is not None:
        return Path(context_system_prompt_file).read_text().strip()
    from qval.system_prompts import (
        ENV_PROMPTS,
        build_ranking_sampling_system_prompt,
    )

    if adapter is not None and adapter in ENV_PROMPTS:
        built = build_ranking_sampling_system_prompt(
            adapter,
            prompting_scheme or "answer_tags",
            env_name=env_name,
            make_kwargs=make_kwargs,
        )
        if adapter_system_prompt:
            return built + "\n\n" + adapter_system_prompt
        return built
    if adapter_system_prompt is not None:
        return adapter_system_prompt
    return fallback_system_prompt


def apply_cli_backend_tmux_prompt_hardening(
    system_prompt: str | None,
    *,
    enabled: bool,
    adapter: str | None,
    make_kwargs: dict[str, Any] | None,
    backend_config: Any | None,
) -> str | None:
    """Append CLI-backend tmux execution guidance when explicitly enabled.

    Activates for both ``codex`` and ``claude_code`` backends running on
    Harbor's ``tmux_session`` text-exec mode, where the model emits one
    shell command per turn that an external runner replays into the task's
    persistent tmux shell.
    """
    from qval.system_prompts import CLI_BACKEND_TMUX_PROMPT_HARDENING

    if (
        system_prompt is None
        or not enabled
        or adapter != "harbor"
        or (make_kwargs or {}).get("text_exec_mode") != "tmux_session"
        or getattr(backend_config, "type", None) not in _CLI_BACKEND_TYPES
    ):
        return system_prompt

    if CLI_BACKEND_TMUX_PROMPT_HARDENING in system_prompt:
        return system_prompt
    return system_prompt + "\n\n" + CLI_BACKEND_TMUX_PROMPT_HARDENING


@dataclass(frozen=True)
class HarborSnapshotFilterReport:
    """Resolved snapshot-eligible Harbor task selection."""

    total_tasks: int
    eligible_task_indices: tuple[int, ...]
    filtered_task_indices: tuple[int, ...]
    filtered_tasks: tuple[dict[str, Any], ...]
    reason_counts: dict[str, int]
    explicit_task_selection: bool
    requested_num_trajectories: int | None
    requested_task_indices: tuple[int, ...] | None
    selected_task_indices: tuple[int, ...]


def _freeze_cache_value(value: Any) -> Any:
    """Convert nested values into a stable, hashable cache representation."""
    if dataclasses.is_dataclass(value):
        return (
            "__dataclass__",
            f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            tuple(
                (field.name, _freeze_cache_value(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_cache_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return (
            "__callable__",
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", repr(value)),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__") and vars(value):
        return (
            "__object__",
            f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            _freeze_cache_value(vars(value)),
        )
    try:
        hash(value)
    except TypeError:
        return ("__repr__", repr(value))
    return value


def create_adapter_env(
    adapter: str,
    env_name: str,
    *,
    env_size: int,
    seed: int,
    max_steps: int | None,
    answer_extractor: Any = None,
    make_kwargs: dict[str, Any] | None = None,
    extra_rewards: tuple = (),
    invalid_action_text: str | None = None,
    invalid_action_observation: str | None = None,
    advance_on_invalid: str | None = None,
):
    """Create an environment using the specified adapter.

    Returns:
        A tuple of ``(env, system_prompt)`` where *system_prompt* may be
        ``None`` for adapters that manage prompts internally.
    """
    if adapter == "reasoning_gym":
        from llenvs.adapters.reasoning_gym import ReasoningGymAdapter

        env = ReasoningGymAdapter().get_environment(
            env_name,
            size=env_size,
            seed=seed,
            answer_extractor=answer_extractor,
        )
        system_prompt = None
    elif adapter == "gem":
        from llenvs.adapters.gem import GemAdapter

        kwargs: dict = {"seed": seed}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            kwargs["answer_extractor"] = answer_extractor
        env = GemAdapter().get_environment(env_name, **kwargs)
        # GEM manages observations internally; no system prompt needed
        system_prompt = None
    elif adapter == "agentgym":
        from llenvs.adapters.agentgym import AgentGymAdapter

        kwargs = {"data_len": env_size}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        # AgentGym does not support answer_extractor
        env = AgentGymAdapter().get_environment(env_name, **kwargs)
        system_prompt = env.prompts.get("system_prompt")
    elif adapter == "gymnasium":
        from llenvs.adapters.gymnasium import GymnasiumAdapter

        gym_kwargs: dict[str, Any] = {}
        if max_steps is not None:
            gym_kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            gym_kwargs["answer_extractor"] = answer_extractor
        if make_kwargs:
            gym_kwargs.update(make_kwargs)
        if extra_rewards:
            gym_kwargs["extra_rewards"] = extra_rewards
        env = GymnasiumAdapter().get_environment(
            env_name,
            seeds=list(range(seed, seed + env_size)),
            pure_step=True,
            **gym_kwargs,
        )
        # Gymnasium builds prompts internally; no system prompt needed
        system_prompt = None
    elif adapter == "harbor":
        from llenvs.adapters.harbor import HarborAdapter

        kwargs = {}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            kwargs["answer_extractor"] = answer_extractor
        if invalid_action_text is not None:
            kwargs["invalid_action_text"] = invalid_action_text
        if invalid_action_observation is not None:
            kwargs["invalid_action_observation"] = invalid_action_observation
        if make_kwargs:
            kwargs.update(make_kwargs)
        if extra_rewards:
            kwargs["extra_rewards"] = extra_rewards
        kwargs = _apply_harbor_env_overrides(kwargs)
        env = HarborAdapter().get_environment(env_name, **kwargs)
        system_prompt = None
    elif adapter == "alfworld":
        from llenvs.adapters.alfworld import AlfWorldAdapter

        alfworld_kwargs: dict[str, Any] = {}
        if max_steps is not None:
            alfworld_kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            alfworld_kwargs["answer_extractor"] = answer_extractor
        if invalid_action_text is not None:
            alfworld_kwargs["invalid_action_text"] = invalid_action_text
        if invalid_action_observation is not None:
            alfworld_kwargs["invalid_action_observation"] = invalid_action_observation
        if advance_on_invalid is not None:
            alfworld_kwargs["advance_on_invalid"] = advance_on_invalid
        if make_kwargs:
            alfworld_kwargs.update(make_kwargs)
        if extra_rewards:
            alfworld_kwargs["extra_rewards"] = extra_rewards
        env = AlfWorldAdapter().get_environment(env_name, **alfworld_kwargs)
        system_prompt = None
    elif adapter == "jericho":
        from llenvs.adapters.jericho import JerichoAdapter

        jericho_kwargs: dict[str, Any] = {}
        if max_steps is not None:
            jericho_kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            jericho_kwargs["answer_extractor"] = answer_extractor
        if invalid_action_text is not None:
            jericho_kwargs["invalid_action_text"] = invalid_action_text
        if invalid_action_observation is not None:
            jericho_kwargs["invalid_action_observation"] = invalid_action_observation
        if advance_on_invalid is not None:
            jericho_kwargs["advance_on_invalid"] = advance_on_invalid
        if make_kwargs:
            jericho_kwargs.update(make_kwargs)
        if extra_rewards:
            jericho_kwargs["extra_rewards"] = extra_rewards
        env = JerichoAdapter().get_environment(
            env_name,
            pure_step=True,
            **jericho_kwargs,
        )
        system_prompt = None
    elif adapter == "webshop":
        from llenvs.adapters.webshop import WebShopAdapter

        webshop_kwargs: dict[str, Any] = {"num_tasks": env_size}
        if max_steps is not None:
            webshop_kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            webshop_kwargs["answer_extractor"] = answer_extractor
        if invalid_action_text is not None:
            webshop_kwargs["invalid_action_text"] = invalid_action_text
        if invalid_action_observation is not None:
            webshop_kwargs["invalid_action_observation"] = invalid_action_observation
        if advance_on_invalid is not None:
            webshop_kwargs["advance_on_invalid"] = advance_on_invalid
        if make_kwargs:
            webshop_kwargs.update(make_kwargs)
        if extra_rewards:
            webshop_kwargs["extra_rewards"] = extra_rewards
        env = WebShopAdapter().get_environment(
            env_name,
            pure_step=True,
            **webshop_kwargs,
        )
        system_prompt = None
    elif adapter == "craftax":
        from llenvs.adapters.craftax import CraftaxAdapter

        craftax_kwargs: dict[str, Any] = {"num_tasks": env_size}
        if max_steps is not None:
            craftax_kwargs["max_steps"] = max_steps
        if answer_extractor is not None:
            craftax_kwargs["answer_extractor"] = answer_extractor
        if make_kwargs:
            craftax_kwargs.update(make_kwargs)
        if extra_rewards:
            craftax_kwargs["extra_rewards"] = extra_rewards
        env = CraftaxAdapter().get_environment(
            env_name,
            **craftax_kwargs,
        )
        # Static game description + action space → merged into system prompt.
        system_prompt = getattr(env, "task_description", None)
    elif adapter == "open_apps":
        from llenvs.adapters.open_apps import OpenAppsAdapter

        oa_kwargs: dict[str, Any] = {}
        if max_steps is not None:
            oa_kwargs["max_steps"] = max_steps
        if make_kwargs:
            oa_kwargs.update(make_kwargs)
        if extra_rewards:
            oa_kwargs["extra_rewards"] = extra_rewards
        # The task name is passed as env_name (e.g. "add_call_mom_to_my_todo")
        env = OpenAppsAdapter().get_environment(env_name, **oa_kwargs)
        system_prompt = None  # resolved by resolve_system_prompt() via ENV_PROMPTS
    else:
        raise ValueError(
            f"Unknown adapter: {adapter!r}. "
            "Valid values: reasoning_gym, gem, agentgym, gymnasium, alfworld, "
            "harbor, jericho, webshop, craftax, open_apps"
        )

    return env, system_prompt


def _apply_harbor_env_overrides(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply local-NVMe and VB_* env var overrides to Harbor adapter kwargs.

    When ``$TMPDIR`` is set (typical SLURM job-local NVMe) and the config
    specifies a ``sif_cache_dir``, this copies SIF images to local storage
    and redirects both ``sif_cache_dir`` and ``trials_dir`` to avoid
    hammering shared filesystems during parallel container starts.

    Explicit ``VB_SIF_CACHE_DIR`` / ``VB_TRIALS_DIR`` env vars take
    precedence over the automatic ``$TMPDIR``-based redirection.
    """
    result = dict(kwargs)
    tmpdir = os.environ.get("TMPDIR")

    # Auto-redirect to local NVMe when $TMPDIR is available
    if tmpdir and result.get("sif_cache_dir"):
        src_sif = Path(result["sif_cache_dir"])
        local_sif = Path(tmpdir) / "sif_cache"
        if src_sif.exists() and not local_sif.exists():
            local_sif.mkdir(parents=True, exist_ok=True)
            copied = 0
            for f in src_sif.iterdir():
                if f.suffix == ".sif" or f.name == "manifest.json":
                    shutil.copy2(f, local_sif / f.name)
                    copied += 1
            if copied:
                logger.info(
                    "Copied %d SIF files to local NVMe: %s → %s",
                    copied, src_sif, local_sif,
                )
                result["sif_cache_dir"] = str(local_sif)
        elif local_sif.exists():
            # Already copied (e.g., by a prior call in the same job)
            result["sif_cache_dir"] = str(local_sif)

    if tmpdir and "trials_dir" not in result and result.get("sif_cache_dir"):
        # Only auto-redirect trials when we also redirected SIF (Harbor context)
        local_trials = Path(tmpdir) / "trials"
        local_trials.mkdir(parents=True, exist_ok=True)
        result["trials_dir"] = str(local_trials)

    # Explicit env vars take final precedence
    if "VB_SIF_CACHE_DIR" in os.environ:
        result["sif_cache_dir"] = os.environ["VB_SIF_CACHE_DIR"]
    if "VB_TRIALS_DIR" in os.environ:
        result["trials_dir"] = os.environ["VB_TRIALS_DIR"]
    return result


def create_harbor_env_factory(
    env_name: str = "terminal-bench@2.0",
    *,
    restore_mode: str = "replay",
    artifact_root: str | None = None,
    **adapter_kwargs: Any,
) -> tuple:
    """Create env_factory and restore_fn for Harbor multi-instance rollouts.

    Returns:
        Tuple of ``(env_factory, restore_fn)`` suitable for passing to
        ``TrajectoryRunner``.
    """
    from llenvs.adapters.harbor import (
        HarborAdapter,
        harbor_restore,
        harbor_snapshot_restore,
    )

    adapter_kwargs = _apply_harbor_env_overrides(adapter_kwargs)
    adapter = HarborAdapter()
    # Use pre-loaded tasks from caller if provided (e.g. snapshot_exact
    # path in collect_dataset.py), otherwise load from adapter.
    tasks = adapter_kwargs.pop("tasks", None)
    if tasks is None:
        dataset_path = adapter_kwargs.get("dataset_path")
        tasks = adapter.load_tasks(env_name, dataset_path=dataset_path)

    def env_factory():
        t0 = time.monotonic()
        try:
            env = adapter.get_environment(
                env_name,
                tasks=tasks,
                **adapter_kwargs,
            )
            logger.debug(
                "Harbor env created in %.1fs", time.monotonic() - t0
            )
            return env
        except FileNotFoundError as e:
            logger.error("Harbor env creation failed (SIF image missing?): %s", e)
            raise
        except RuntimeError as e:
            msg = str(e).lower()
            if "fakeroot" in msg:
                logger.error(
                    "Harbor env creation failed (fakeroot not supported): %s", e
                )
            elif "overlay" in msg:
                logger.error(
                    "Harbor env creation failed (overlay creation error): %s", e
                )
            elif "fuse" in msg:
                logger.error(
                    "Harbor env creation failed (FUSE mount error): %s", e
                )
            else:
                logger.error("Harbor env creation failed: %s", e)
            raise

    if restore_mode == "replay":
        return env_factory, harbor_restore
    if restore_mode == "snapshot_exact":
        if artifact_root is None:
            raise ValueError(
                "artifact_root is required when restore_mode='snapshot_exact'"
            )

        def restore_fn(env, state):
            return harbor_snapshot_restore(
                env,
                state,
                artifact_root=artifact_root,
            )

        return env_factory, restore_fn

    raise ValueError(
        f"Unknown Harbor restore_mode: {restore_mode!r}. "
        "Valid values: replay, snapshot_exact"
    )


def create_harbor_ranking_env_factory(
    env_name: str = "terminal-bench@2.0",
    *,
    restore_mode: str = "replay",
    artifact_root: str | None = None,
    **adapter_kwargs: Any,
) -> tuple:
    """Create Harbor env_factory/restore_fn for per-candidate ranking stepping.

    Ranking candidate validation should use the same restore mode as collection,
    but it should not inherit collection-only live capture features that add
    unnecessary overhead to short-lived candidate-validation containers.
    """
    ranking_adapter_kwargs = {
        key: value for key, value in adapter_kwargs.items()
        if key not in {"runtime_probing", "state_capture_mode", "snapshot_artifact_root"}
    }
    return create_harbor_env_factory(
        env_name,
        restore_mode=restore_mode,
        artifact_root=artifact_root,
        **ranking_adapter_kwargs,
    )


def create_open_apps_env_factory(
    env_name: str,
    **adapter_kwargs: Any,
) -> tuple:
    """Create env_factory and restore_fn for OpenApps replay-based rollouts.

    Each env_factory call instantiates its own ``OpenAppsAdapter`` — this
    is *intentional*. OpenApps' FastHTML apps keep state in module-global
    SQLite databases (e.g. ``global events`` in calendar_app); sharing one
    adapter/server across parallel envs would corrupt each trajectory's
    state with writes from the others.

    The previously-observed "step timed out after 300s" issue was NOT a
    parallelism problem — it was an *accumulation* problem: the adapter
    had no ``__del__``/``close`` that stopped its subprocess when the env
    was discarded, so every batch's 4 servers leaked, port-starving new
    spawns across months of runs (496 orphans found on the machine at
    one point). The fix lives in llenvs: ``OpenAppsAdapter.__del__`` now
    calls ``stop_server()`` best-effort.

    Returns:
        Tuple of ``(env_factory, restore_fn)`` suitable for passing to
        ``TrajectoryRunner``.
    """
    from llenvs.adapters.open_apps import OpenAppsAdapter, open_apps_restore

    def env_factory():
        return OpenAppsAdapter().get_environment(env_name, **adapter_kwargs)

    return env_factory, open_apps_restore


def preflight_harbor_tmux_sessions(
    env_name: str = "terminal-bench@2.0",
    *,
    task_indices: tuple[int, ...],
    **adapter_kwargs: Any,
) -> dict[str, Any]:
    """Sequentially verify tmux-session startup for selected Harbor tasks."""
    from llenvs.adapters.harbor import HarborAdapter

    if not task_indices:
        return {
            "text_exec_mode": adapter_kwargs.get("text_exec_mode", "independent_exec"),
            "task_indices": [],
            "tasks": [],
            "bootstrapped_task_indices": [],
        }

    adapter_kwargs = _apply_harbor_env_overrides(adapter_kwargs)
    dataset_path = adapter_kwargs.get("dataset_path")
    sanitized_kwargs = {
        key: value for key, value in adapter_kwargs.items()
        if key not in {"snapshot_artifact_root"}
    }
    sanitized_kwargs["runtime_probing"] = False
    sanitized_kwargs["state_capture_mode"] = "replay"

    unique_task_indices = tuple(dict.fromkeys(task_indices))

    logger.info(
        "Harbor tmux preflight: checking %d unique task(s) from %d requested trajectory slot(s)",
        len(unique_task_indices),
        len(task_indices),
    )

    adapter = HarborAdapter()
    tasks = adapter.load_tasks(env_name, dataset_path=dataset_path)
    report_tasks: list[dict[str, Any]] = []
    bootstrapped_task_indices: list[int] = []

    for idx, task_index in enumerate(unique_task_indices, start=1):
        logger.info(
            "Harbor tmux preflight: task %d/%d (task_index=%d)",
            idx,
            len(unique_task_indices),
            task_index,
        )
        env = adapter.get_environment(
            env_name,
            tasks=tasks,
            **sanitized_kwargs,
        )
        try:
            _state, info = env.reset(options={"task_index": task_index})
        except Exception as exc:
            try:
                env.close()
            finally:
                raise RuntimeError(
                    f"Harbor tmux preflight failed for task {task_index}: {exc}"
                ) from exc
        try:
            entry = {
                "task_index": task_index,
                "task_name": info.get("task_name"),
                "tmux_bootstrapped": bool(info.get("tmux_bootstrapped", False)),
                "tmux_start_method": info.get("tmux_start_method"),
            }
            report_tasks.append(entry)
            if entry["tmux_bootstrapped"]:
                bootstrapped_task_indices.append(task_index)
        finally:
            env.close()

    return {
        "text_exec_mode": sanitized_kwargs.get("text_exec_mode", "independent_exec"),
        "task_indices": list(unique_task_indices),
        "tasks": report_tasks,
        "bootstrapped_task_indices": bootstrapped_task_indices,
    }


# ── Replay cache for filesystem checkpoint/restore ─────────────


@dataclass
class _CacheEntry:
    """Per-key cache state."""

    tar_path: Path | None = None
    event: threading.Event = field(default_factory=threading.Event)
    remaining: int = 0
    filled: bool = False


class ReplayCache:
    """Thread-safe replay cache with per-entry cleanup.

    First restore for a given state replays + exports a tar.
    Subsequent restores untar from cache. When the last consumer
    finishes, the tar is deleted automatically.
    """

    def __init__(self, cache_dir: Path | str, *, env_name: str = ""):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._env_name = env_name
        self._lock = threading.Lock()
        self._entries: dict[tuple, _CacheEntry] = {}
        self._fills = 0
        self._waiter_restores = 0
        self._late_hits = 0
        self._fallbacks = 0

    def _key(self, state) -> tuple:
        hidden = state.hidden
        return (self._env_name, getattr(hidden, "task_name", ""), getattr(hidden, "trajectory", ()))

    def wrap(self, base_restore_fn):
        """Return a restore_fn with caching."""

        def cached_restore(env, state):
            key = self._key(state)
            role = self._acquire(key)
            if role == "filler":
                return self._on_fill(env, state, key, base_restore_fn)
            if role == "waiter":
                return self._on_wait(env, state, key, base_restore_fn)
            if role == "hit":
                try:
                    with self._lock:
                        tar_path = self._entries[key].tar_path
                    return _restore_from_fs_cache(env, state, tar_path)
                finally:
                    self._maybe_cleanup(key)
            # role == "replay_fallback"
            return base_restore_fn(env, state)

        return cached_restore

    def _acquire(self, key) -> str:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = _CacheEntry(remaining=1)
                self._fills += 1
                return "filler"
            if not entry.filled:
                entry.remaining += 1
                return "waiter"
            if entry.tar_path is not None and entry.tar_path.exists():
                entry.remaining += 1
                self._late_hits += 1
                return "hit"
            self._fallbacks += 1
            return "replay_fallback"

    def _on_fill(self, env, state, key, base_restore_fn):
        entry = self._entries[key]
        tar_path_attempted: Path | None = None
        try:
            result = base_restore_fn(env, state)
            harbor_env = getattr(env, "_harbor_env", None)
            export_fn = getattr(harbor_env, "export_checkpoint", None) if harbor_env else None
            if callable(export_fn):
                digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
                tar_path_attempted = self._cache_dir / f"{digest}.tar"
                from llenvs.core.async_utils import run_async

                try:
                    run_async(export_fn(tar_path_attempted))
                except Exception as exc:
                    logger.warning(
                        "Replay cache export failed for %s: %s. "
                        "Falling back to normal replay for this state.",
                        tar_path_attempted,
                        exc,
                    )
                    with self._lock:
                        self._fallbacks += 1
                    if tar_path_attempted.exists():
                        try:
                            tar_path_attempted.unlink()
                        except OSError:
                            pass
                    tar_path_attempted = None
                else:
                    with self._lock:
                        entry.tar_path = tar_path_attempted
                    tar_path_attempted = None  # committed successfully, don't clean up
            return result
        except Exception:
            # Clean up partially written tar file
            if tar_path_attempted is not None and tar_path_attempted.exists():
                try:
                    tar_path_attempted.unlink()
                except OSError:
                    pass
            raise
        finally:
            with self._lock:
                entry.filled = True
            entry.event.set()
            self._maybe_cleanup(key)

    def _on_wait(self, env, state, key, base_restore_fn):
        entry = self._entries[key]
        entry.event.wait()
        with self._lock:
            tar_path = entry.tar_path
        if tar_path is not None:
            with self._lock:
                self._waiter_restores += 1
            try:
                return _restore_from_fs_cache(env, state, tar_path)
            finally:
                self._maybe_cleanup(key)
        with self._lock:
            self._fallbacks += 1
        self._maybe_cleanup(key)
        return base_restore_fn(env, state)

    def _maybe_cleanup(self, key):
        tar_to_delete: Path | None = None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.remaining -= 1
            if entry.remaining <= 0 and entry.filled:
                tar_to_delete = entry.tar_path
                del self._entries[key]
        if tar_to_delete is not None and tar_to_delete.exists():
            try:
                tar_to_delete.unlink()
            except OSError:
                pass

    def cleanup(self):
        """Safety net: delete all remaining tars."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            if entry.tar_path and entry.tar_path.exists():
                try:
                    entry.tar_path.unlink()
                except OSError:
                    pass

    def stats(self) -> dict[str, int]:
        with self._lock:
            active = sum(1 for e in self._entries.values() if e.tar_path is not None)
        return {
            "fills": self._fills,
            "waiter_restores": self._waiter_restores,
            "late_hits": self._late_hits,
            "fallbacks": self._fallbacks,
            "active_entries": active,
        }


def _restore_from_fs_cache(env, state, tar_path: Path):
    """Restore env from cached filesystem snapshot.

    Bypasses the full ``env.reset()`` to avoid a wasted seed-copy +
    instance-start cycle.  Instead we:

    1. Stop the old container (if any).
    2. Create a new ``_harbor_env`` via the factory (constructor only, no start).
    3. Call ``restore_checkpoint()`` which untars the cached rootfs and starts
       the instance — one start instead of two.
    """
    from llenvs.core.async_utils import run_async

    hidden = state.hidden
    task_index = hidden.task_index

    # 1. Stop previous container
    if env._harbor_env is not None:
        try:
            run_async(env._harbor_env.stop(delete=True))
        except Exception:
            pass

    # 2. Create new harbor_env without starting
    task = env._tasks[task_index]
    env._current_task = task
    env._harbor_env = env._harbor_env_factory(task)

    # Task name validation
    actual_name = getattr(task, "name", str(task_index))
    if hidden.task_name and actual_name and hidden.task_name != actual_name:
        raise ValueError(
            f"Task name mismatch: expected {hidden.task_name!r}, "
            f"got {actual_name!r} at index {task_index}. "
            f"Dataset version may have changed."
        )

    # 3. Restore from tar (not started yet → no wasted stop/delete)
    run_async(env._harbor_env.restore_checkpoint(tar_path))
    if hasattr(env, "_trajectory_started_at_monotonic"):
        env._trajectory_started_at_monotonic = time.monotonic()

    env._state_tracker.track(state)
    return state


def resolve_harbor_snapshot_filter_report(
    eligibility: tuple[Any, ...],
    *,
    task_indices: tuple[int, ...] | None,
    num_trajectories: int,
) -> HarborSnapshotFilterReport:
    """Resolve Harbor task indices for exact snapshot collection."""
    total_tasks = len(eligibility)
    eligible_entries = tuple(entry for entry in eligibility if entry.eligible)
    filtered_entries = tuple(entry for entry in eligibility if not entry.eligible)
    eligible_task_indices = tuple(entry.task_index for entry in eligible_entries)
    filtered_task_indices = tuple(entry.task_index for entry in filtered_entries)
    reason_counts = dict(
        sorted(
            Counter(
                entry.reason_code or "unknown"
                for entry in filtered_entries
            ).items()
        )
    )
    filtered_tasks = tuple(
        {
            "task_index": entry.task_index,
            "task_name": entry.task_name,
            "reason_code": entry.reason_code,
            "reason_detail": entry.reason_detail,
        }
        for entry in filtered_entries
    )

    explicit_task_selection = task_indices is not None
    if explicit_task_selection:
        requested_task_indices = tuple(task_indices or ())
        invalid_indices = tuple(
            idx for idx in requested_task_indices if idx < 0 or idx >= total_tasks
        )
        if invalid_indices:
            raise ValueError(
                f"Harbor task_indices out of bounds for dataset size {total_tasks}: "
                f"{list(invalid_indices)}"
            )
        unsupported = tuple(
            entry for entry in filtered_entries if entry.task_index in requested_task_indices
        )
        if unsupported:
            details = ", ".join(
                f"{entry.task_index}:{entry.task_name} ({entry.reason_code})"
                for entry in unsupported
            )
            raise ValueError(
                "Snapshot exact Harbor collection does not support the explicitly "
                f"requested task_indices: {details}"
            )
        selected_task_indices = requested_task_indices
        requested_num_trajectories: int | None = len(requested_task_indices)
    else:
        requested_task_indices = None
        requested_num_trajectories = num_trajectories
        selected_task_indices = eligible_task_indices[:num_trajectories]
        if len(selected_task_indices) < num_trajectories:
            raise ValueError(
                "Snapshot exact Harbor collection requested "
                f"{num_trajectories} tasks, but only {len(eligible_task_indices)} "
                "snapshot-eligible tasks are available."
            )

    return HarborSnapshotFilterReport(
        total_tasks=total_tasks,
        eligible_task_indices=eligible_task_indices,
        filtered_task_indices=filtered_task_indices,
        filtered_tasks=filtered_tasks,
        reason_counts=reason_counts,
        explicit_task_selection=explicit_task_selection,
        requested_num_trajectories=requested_num_trajectories,
        requested_task_indices=requested_task_indices,
        selected_task_indices=selected_task_indices,
    )


def inspect_harbor_snapshot_filter(
    env_name: str,
    *,
    dataset_path: str | None = None,
    environment_type: str = "docker",
    task_indices: tuple[int, ...] | None,
    num_trajectories: int,
    difficulties: set[str] | list[str] | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], HarborSnapshotFilterReport]:
    """Load Harbor tasks and resolve snapshot-eligible collection indices.

    When *difficulties* is set, tasks are filtered by difficulty before
    checking snapshot eligibility, so the returned indices are into the
    filtered task set (matching what the env will use).
    """
    from llenvs.adapters.harbor import HarborAdapter

    adapter = HarborAdapter()
    tasks = adapter.load_tasks(env_name, dataset_path=dataset_path)
    tasks = adapter.filter_tasks(tasks, difficulties=difficulties)
    eligibility = adapter.inspect_snapshot_eligibility(
        name=env_name,
        tasks=tasks,
        environment_type=environment_type,
    )
    report = resolve_harbor_snapshot_filter_report(
        eligibility,
        task_indices=task_indices,
        num_trajectories=num_trajectories,
    )
    return tasks, eligibility, report


def serialize_harbor_snapshot_filter_report(
    report: HarborSnapshotFilterReport,
) -> dict[str, Any]:
    """Convert a Harbor snapshot filter report into dataset/report metadata."""
    return {
        "total_tasks": report.total_tasks,
        "eligible_task_count": len(report.eligible_task_indices),
        "filtered_task_count": len(report.filtered_task_indices),
        "eligible_task_indices": list(report.eligible_task_indices),
        "filtered_task_indices": list(report.filtered_task_indices),
        "filtered_tasks": [dict(item) for item in report.filtered_tasks],
        "reason_counts": dict(report.reason_counts),
        "explicit_task_selection": report.explicit_task_selection,
        "requested_num_trajectories": report.requested_num_trajectories,
        "requested_task_indices": (
            None
            if report.requested_task_indices is None
            else list(report.requested_task_indices)
        ),
        "selected_task_indices": list(report.selected_task_indices),
    }


def inspect_harbor_runtime_eligibility(
    env_name: str,
    *,
    dataset_path: str | None = None,
    environment_type: str = "docker",
    sif_cache_dir: str | None = None,
    task_indices: tuple[int, ...] | None,
    num_trajectories: int | None = None,
    difficulties: set[str] | list[str] | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Filter Harbor tasks by runtime eligibility.

    When *difficulties* is set, tasks are filtered by difficulty before
    checking runtime eligibility, so the returned indices are into the
    filtered task set (matching what the env will use).

    Returns ``(selected_task_indices, report_dict)`` where
    ``selected_task_indices`` are the indices of eligible tasks and
    ``report_dict`` contains filtering metadata for logging.
    """
    from llenvs.adapters.harbor import (
        HarborAdapter,
        inspect_harbor_runtime_eligibility as _inspect,
    )

    adapter = HarborAdapter()
    tasks = adapter.load_tasks(env_name, dataset_path=dataset_path)
    tasks = adapter.filter_tasks(tasks, difficulties=difficulties)
    eligibility = _inspect(
        tasks, environment_type, sif_cache_dir=sif_cache_dir
    )

    eligible_indices = tuple(e.task_index for e in eligibility if e.eligible)
    filtered_entries = [e for e in eligibility if not e.eligible]
    reason_counts: dict[str, int] = {}
    for entry in filtered_entries:
        code = entry.reason_code or "unknown"
        reason_counts[code] = reason_counts.get(code, 0) + 1

    if task_indices is not None:
        # Explicit selection: fail fast if any requested task is ineligible
        ineligible = [
            e for e in filtered_entries if e.task_index in task_indices
        ]
        if ineligible:
            details = ", ".join(
                f"{e.task_index}:{e.task_name} ({e.reason_code})"
                for e in ineligible
            )
            raise ValueError(
                f"Harbor runtime '{environment_type}' does not support the "
                f"explicitly requested task_indices: {details}"
            )
        selected = task_indices
    else:
        if num_trajectories is not None:
            selected = eligible_indices[:num_trajectories]
            if len(selected) < num_trajectories:
                raise ValueError(
                    f"Harbor runtime '{environment_type}' collection requested "
                    f"{num_trajectories} tasks, but only {len(eligible_indices)} "
                    "eligible tasks are available."
                )
        else:
            selected = eligible_indices

    report = {
        "total_tasks": len(eligibility),
        "eligible_count": len(eligible_indices),
        "filtered_count": len(filtered_entries),
        "reason_counts": reason_counts,
        "selected_task_indices": list(selected),
    }
    return selected, report


def _import_vllm_backend():
    """Lazy import for VLLMBackend (requires vllm installed)."""
    from llenvs.inference.backends import VLLMBackend

    return VLLMBackend


_BACKEND_TYPES_WITHOUT_THINKING_CONTROL = frozenset(
    {"openai", "anthropic", "codex", "vllm_singularity"}
)

# API backends that DO honor thinking_budget, via a provider-side knob
# (named here for warning messages). Suffix/soft-ratio stay inert on these.
_API_THINKING_BUDGET_MECHANISM = {
    "openrouter": "OpenRouter's reasoning.max_tokens parameter",
    "litellm": "litellm's thinking budget_tokens parameter",
}


def _warn_if_thinking_controls_are_inert(config: BackendConfig) -> None:
    """Warn when thinking-budget controls are silently no-op on this backend.

    ``thinking_budget``, ``thinking_budget_suffix`` (populated from
    ``early_stopping_suffix=True``), and ``thinking_budget_soft_ratio``
    are implemented inside the llenvs vLLM and HuggingFace backends via
    logits processors. Most API backends (``openai``, ``anthropic``,
    ``codex``, ``vllm_singularity``) have no way to control token
    generation at that level, so these sampling flags silently no-op.
    OpenRouter and litellm are partial exceptions: both expose a
    provider-side budget knob (OpenRouter ``reasoning.max_tokens``,
    litellm ``thinking.budget_tokens``) which llenvs maps onto
    ``thinking_budget``; only the suffix and soft-ratio knobs remain
    inert there.
    """
    sampling = config.sampling
    if sampling.thinking_budget is None:
        return

    if config.type in _BACKEND_TYPES_WITHOUT_THINKING_CONTROL:
        logger.warning(
            "Backend %s (type=%s): thinking_budget=%d is set, but %s backends "
            "do not implement local thinking-budget control. thinking_budget, "
            "thinking_budget_suffix (populated from early_stopping_suffix=True), "
            "and thinking_budget_soft_ratio will be silently ignored. "
            "These controls only take effect for vllm and huggingface backends.",
            config.model,
            config.type,
            sampling.thinking_budget,
            config.type,
        )
        return

    budget_mechanism = _API_THINKING_BUDGET_MECHANISM.get(config.type)
    if budget_mechanism is None:
        return

    if sampling.thinking_budget_soft_ratio is not None:
        logger.warning(
            "Backend %s (type=%s): thinking_budget=%d is honored "
            "via %s, but thinking_budget_soft_ratio=%.2f requires local "
            "logits intervention and will be silently ignored. Soft-ratio "
            "early stopping only takes effect for vllm and huggingface "
            "backends.",
            config.model,
            config.type,
            sampling.thinking_budget,
            budget_mechanism,
            sampling.thinking_budget_soft_ratio,
        )

    if sampling.max_tokens:
        ratio = sampling.thinking_budget / sampling.max_tokens
        if ratio >= 0.75:
            logger.warning(
                "Backend %s (type=%s): thinking_budget=%d consumes "
                "%.0f%% of max_tokens=%d. The API counts reasoning tokens "
                "as completion tokens, returns visible answer text separately "
                "from reasoning text, and cannot use qval's local "
                "thinking_budget_suffix/soft-ratio transition. This setting "
                "can produce empty visible responses with MAX_TOKENS.",
                config.model,
                config.type,
                sampling.thinking_budget,
                ratio * 100,
                sampling.max_tokens,
            )


def create_backend_from_config(
    config: BackendConfig,
    *,
    chat_template_kwargs: dict | None = None,
    tensor_parallel_size_override: int | None = None,
):
    """Create a model backend from a BackendConfig.

    Args:
        config: Backend configuration.
        chat_template_kwargs: Extra kwargs for chat template application.
        tensor_parallel_size_override: When set, overrides the config's
            ``tensor_parallel_size`` for vLLM backends. Used by the
            ``--tensor-parallel-size`` CLI flag to flow the GPU count
            from SLURM to all vLLM backends.

    Callers own the backend lifecycle and should call ``close()`` when done.
    """
    _warn_if_thinking_controls_are_inert(config)
    if config.type == "openai":
        from llenvs.inference.backends import OpenAIBackend

        openai_kwargs: dict[str, Any] = {
            "model": config.model,
            "base_url": config.backend_url,
            "api_key": "dummy",
        }
        if config.connect_timeout is not None:
            openai_kwargs["timeout"] = config.connect_timeout
        if config.api_max_retries is not None:
            openai_kwargs["max_retries"] = config.api_max_retries
        return OpenAIBackend(**openai_kwargs)
    if config.type == "huggingface":
        from llenvs.inference.backends import HuggingFaceBackend

        return HuggingFaceBackend(
            model_path=config.model,
            device=config.device,
            dtype=config.dtype,
            chat_template_kwargs=chat_template_kwargs,
        )
    if config.type == "vllm":
        VLLMBackend = _import_vllm_backend()
        tp_size = tensor_parallel_size_override or config.tensor_parallel_size

        return VLLMBackend(
            model_path=config.model,
            tensor_parallel_size=tp_size,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            chat_template_kwargs=chat_template_kwargs,
            post_shutdown_delay=60,
            **dict(config.vllm_kwargs),
        )
    if config.type == "vllm_singularity":
        from llenvs.inference.backends import SingularityVLLMBackend

        tp_size = tensor_parallel_size_override or config.tensor_parallel_size
        sv_kwargs: dict[str, Any] = {
            "model_path": config.model,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "dtype": config.dtype,
            "max_concurrency": config.max_concurrency,
            "singularity_binds": config.singularity_binds,
            "extra_vllm_args": config.singularity_extra_vllm_args,
            "startup_timeout": config.singularity_startup_timeout,
        }
        # Omit empty/None kwargs so llenvs falls back to env vars
        # (LLENVS_SIF, inherited CUDA_VISIBLE_DEVICES).
        if config.singularity_sif:
            sv_kwargs["sif"] = config.singularity_sif
        if config.singularity_cuda_visible_devices:
            sv_kwargs["cuda_visible_devices"] = (
                config.singularity_cuda_visible_devices
            )
        return SingularityVLLMBackend(**sv_kwargs)
    if config.type == "anthropic":
        from llenvs.inference.backends import AnthropicBackend

        return AnthropicBackend(model=config.model)
    if config.type == "openrouter":
        from llenvs.inference.backends import OpenRouterBackend

        or_kwargs: dict[str, Any] = {
            "model": config.model,
            "max_concurrency": config.max_concurrency,
            "rate_limit_wait": config.rate_limit_wait,
            "rate_limit_max_retries": config.rate_limit_max_retries,
        }
        if config.connect_timeout is not None:
            or_kwargs["timeout"] = config.connect_timeout
        if config.api_max_retries is not None:
            or_kwargs["max_retries"] = config.api_max_retries
        if config.openrouter_provider is not None:
            or_kwargs["provider"] = config.openrouter_provider.to_request_dict()
        return OpenRouterBackend(**or_kwargs)
    if config.type == "litellm":
        from llenvs.inference.backends import LiteLLMBackend

        # No api_key here by design: litellm reads the provider's native
        # env var (LITELLM_PROXY_API_KEY for litellm_proxy/ models).
        ll_kwargs: dict[str, Any] = {
            "model": config.model,
            "max_concurrency": config.max_concurrency,
            "rate_limit_wait": config.rate_limit_wait,
            "rate_limit_max_retries": config.rate_limit_max_retries,
        }
        if config.api_base is not None:
            ll_kwargs["api_base"] = config.api_base
        if config.connect_timeout is not None:
            ll_kwargs["timeout"] = config.connect_timeout
        if config.api_max_retries is not None:
            ll_kwargs["num_retries"] = config.api_max_retries
        return LiteLLMBackend(**ll_kwargs)
    if config.type == "codex":
        from llenvs.inference.backends import CodexCLIBackend

        codex_kwargs: dict[str, Any] = {
            "model": config.model,
            "max_concurrency": config.max_concurrency,
        }
        if config.connect_timeout is not None:
            codex_kwargs["timeout"] = config.connect_timeout
        if config.config_overrides:
            codex_kwargs["config_overrides"] = dict(config.config_overrides)
        return CodexCLIBackend(**codex_kwargs)
    if config.type == "claude_code":
        from llenvs.inference.backends import ClaudeCodeBackend

        cc_kwargs: dict[str, Any] = {
            "model": config.model,
            "max_concurrency": config.max_concurrency,
        }
        if config.connect_timeout is not None:
            cc_kwargs["timeout"] = config.connect_timeout
        for key, value in config.claude_code_kwargs:
            cc_kwargs[key] = list(value) if isinstance(value, tuple) else value
        return ClaudeCodeBackend(**cc_kwargs)
    raise ValueError(
        f"Unknown backend type: {config.type!r}. "
        "Valid values: openai, huggingface, vllm, vllm_singularity, "
        "anthropic, openrouter, litellm, codex, claude_code"
    )


def sampling_params_from_config(
    config: BackendSamplingConfig,
    *,
    backend_type: str | None = None,
    provider_preferences: OpenRouterProviderPreferences | None = None,
) -> SamplingParams:
    """Build SamplingParams from a BackendSamplingConfig.

    When ``backend_type`` is a CLI-style backend (``codex`` or
    ``claude_code``), sampling knobs the underlying CLI rejects
    (temperature, top_p, top_k, thinking/second-elicitation suffixes) are
    omitted so SamplingParams falls back to its CLI-compatible defaults.
    ``max_tokens`` is still forwarded (Codex accepts but ignores it; Claude
    Code's CLI honours it via the ``--max-tokens`` flag).

    When ``provider_preferences`` is set and ``backend_type == "openrouter"``,
    the preferences are forwarded as
    ``extra={"extra_body": {"provider": ...}}`` so the OpenAI SDK sends them
    in the request body. ``reasoning_effort`` and ``reasoning_exclude`` are
    forwarded through the same ``extra_body`` for openrouter; for litellm,
    ``reasoning_effort`` is forwarded as a top-level
    ``extra={"reasoning_effort": ...}`` kwarg instead (litellm's native
    parameter) and ``reasoning_exclude`` has no equivalent. For any other
    ``backend_type`` a warning is logged and those request-body extras are
    dropped — the YAML parser already blocks these combinations, so the
    warning branches only fire on programmatic misuse.
    """
    if backend_type in _CLI_BACKEND_TYPES:
        if provider_preferences is not None:
            logger.warning(
                "provider_preferences supplied for backend_type=%r; ignoring. "
                "Provider routing only applies to openrouter backends.",
                backend_type,
            )
        if (
            config.reasoning_effort is not None
            or config.reasoning_exclude is not None
        ):
            logger.warning(
                "reasoning_effort/reasoning_exclude supplied for backend_type=%r; "
                "ignoring. OpenRouter reasoning controls only apply to "
                "openrouter backends.",
                backend_type,
            )
        return SamplingParams(
            max_tokens=2048 if config.max_tokens is None else config.max_tokens,
        )

    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": 0 if config.top_k is None else config.top_k,
        "max_tokens": 2048 if config.max_tokens is None else config.max_tokens,
    }

    if config.thinking_budget is not None:
        kwargs["thinking_budget"] = config.thinking_budget
        if config.early_stopping_suffix:
            kwargs["thinking_budget_suffix"] = DEFAULT_EARLY_STOPPING_SUFFIX

    if config.thinking_budget_soft_ratio is not None:
        kwargs["thinking_budget_soft_ratio"] = config.thinking_budget_soft_ratio

    if config.second_elicitation:
        kwargs["second_elicitation_suffix"] = DEFAULT_EARLY_STOPPING_SUFFIX
        kwargs["second_elicitation_max_tokens"] = config.second_elicitation_max_tokens

    extra_body: dict[str, Any] = {}
    extra_top_level: dict[str, Any] = {}
    if config.reasoning_effort is not None or config.reasoning_exclude is not None:
        if backend_type == "openrouter":
            reasoning: dict[str, Any] = {}
            if config.reasoning_effort is not None:
                reasoning["effort"] = config.reasoning_effort
            elif (
                config.thinking_budget is not None
                and config.reasoning_exclude is not None
            ):
                # llenvs' OpenRouter backend shallow-merges caller-supplied
                # extra_body over its own reasoning.max_tokens mapping. Keep
                # the budget in the override when we only add exclude.
                reasoning["max_tokens"] = config.thinking_budget
            if config.reasoning_exclude is not None:
                reasoning["exclude"] = config.reasoning_exclude
            extra_body["reasoning"] = reasoning
        elif backend_type == "litellm":
            # litellm takes reasoning_effort as a top-level completion
            # kwarg, which llenvs' LiteLLMBackend forwards from
            # SamplingParams.extra.
            if config.reasoning_effort is not None:
                extra_top_level["reasoning_effort"] = config.reasoning_effort
            if config.reasoning_exclude is not None:
                logger.warning(
                    "reasoning_exclude supplied for backend_type='litellm'; "
                    "ignoring. litellm has no equivalent of OpenRouter's "
                    "reasoning.exclude flag.",
                )
        else:
            logger.warning(
                "reasoning_effort/reasoning_exclude supplied for backend_type=%r; "
                "ignoring. Reasoning controls only apply to openrouter and "
                "litellm backends.",
                backend_type,
            )

    if provider_preferences is not None:
        if backend_type != "openrouter":
            logger.warning(
                "provider_preferences supplied for backend_type=%r; ignoring. "
                "Provider routing only applies to openrouter backends.",
                backend_type,
            )
        else:
            provider_dict = provider_preferences.to_request_dict()
            if provider_dict:
                extra_body["provider"] = provider_dict

    extra: dict[str, Any] = dict(extra_top_level)
    if extra_body:
        extra["extra_body"] = extra_body
    if extra:
        kwargs["extra"] = extra

    return SamplingParams(**kwargs)


def chat_template_kwargs_from_config(config: BackendConfig) -> dict | None:
    """Build chat_template_kwargs from BackendConfig."""
    return {"enable_thinking": config.enable_thinking}


# Backend types that bake ``enable_thinking`` into the constructed instance by
# forwarding it as a chat-template kwarg at construction time (the
# ``huggingface`` and ``vllm`` branches of ``create_backend_from_config``).
# For every other type, thinking is a per-request concern (driven by
# ``sampling.thinking_budget``/``disable_thinking`` via ``extra_body``), so
# ``enable_thinking`` does not affect which backend instance can be shared.
_THINKING_BAKED_AT_CONSTRUCTION = frozenset({"vllm", "huggingface"})


def backend_cache_key(config: BackendConfig) -> BackendConfig:
    """Return the key identifying a shareable backend instance for ``config``.

    Two ``BackendConfig`` values map to the same cached backend iff they agree
    on every construction-time field. Per-request-only fields are neutralized
    so configs that differ only in them reuse a single instance:

    * ``sampling`` is always stripped (applied per call).
    * ``enable_thinking`` is stripped unless the backend type bakes it in at
      construction (``vllm``/``huggingface``); for all other types thinking is
      controlled per request, so two configs differing only in
      ``enable_thinking`` share one instance — e.g. a ``react`` and a
      ``thinking`` ``vllm_singularity`` config for the same model reuse a
      single container instead of spawning two.
    """
    key = dataclasses.replace(config, sampling=BackendSamplingConfig())
    if config.type not in _THINKING_BAKED_AT_CONSTRUCTION:
        key = dataclasses.replace(key, enable_thinking=False)
    return key


def attach_backend_logging_metadata(backend: Any, config: BackendConfig) -> None:
    """Attach backend settings that should appear in experiment logs."""
    try:
        backend.enable_thinking = config.enable_thinking
    except Exception:
        return


def resolve_backend_name(
    name: str | None,
    backends: dict[str, BackendConfig],
    default_name: str | None,
) -> str:
    """Resolve a backend name against the backend registry."""
    resolved = name or default_name
    if not resolved:
        raise ValueError("backend name not provided and no default_backend set")
    if resolved not in backends:
        raise ValueError(
            f"Unknown backend name: {resolved!r}. Available: {sorted(backends)}"
        )
    return resolved


def annotate_codegen_records(
    exp_logger,
    method_name: str,
    method_obj,
) -> None:
    """Add generated_code to the first method record for LLMCodeGen methods.

    Method records already have ``extracted_answer`` annotations set by
    the evaluation loop.  This adds ``generated_code`` to the first record
    in the phase so the full generated source is captured in the logs.
    """
    if not method_obj.generated_code:
        return
    phase_records = exp_logger.get_phase_records(f"method:{method_name}")
    if phase_records:
        ann = phase_records[0].get("annotation", {})
        ann["generated_code"] = method_obj.generated_code
        phase_records[0]["annotation"] = ann


# ---------------------------------------------------------------------------
# Harbor pre-flight diagnostics
# ---------------------------------------------------------------------------


def check_harbor_environment(
    adapter_kwargs: dict[str, Any],
    env_name: str = "terminal-bench@2.0",
) -> dict[str, Any]:
    """Run pre-flight checks for Harbor/Apptainer and log results.

    Checks binary availability, fakeroot support, SIF cache, disk space,
    and directory writability.  All issues are logged but never abort;
    the actual container creation will fail with a clear message.

    Returns a dict of check results for inclusion in experiment logs.
    """
    adapter_kwargs = _apply_harbor_env_overrides(adapter_kwargs)
    cache_key = (env_name, _freeze_cache_value(adapter_kwargs))
    with _HARBOR_PREFLIGHT_CACHE_LOCK:
        cached = _HARBOR_PREFLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    results: dict[str, Any] = {}
    env_type = str(adapter_kwargs.get("environment_type", "docker"))
    is_apptainer = env_type in ("apptainer-hpc", "singularity-hpc")

    # Log config summary
    summary_fields = {
        "environment_type": env_type,
        "fakeroot": adapter_kwargs.get("fakeroot", False),
        "rootfs_mode": adapter_kwargs.get("rootfs_mode", "auto"),
        "overlay_size_mb": adapter_kwargs.get("overlay_size_mb", 512),
        "sif_cache_dir": adapter_kwargs.get("sif_cache_dir"),
        "trials_dir": adapter_kwargs.get("trials_dir"),
        "writable_tmpfs": adapter_kwargs.get("writable_tmpfs", False),
    }
    logger.info("Harbor config: %s", summary_fields)
    results["config"] = summary_fields

    if not is_apptainer:
        logger.info("Harbor environment_type=%s, skipping Apptainer checks", env_type)
        with _HARBOR_PREFLIGHT_CACHE_LOCK:
            _HARBOR_PREFLIGHT_CACHE[cache_key] = copy.deepcopy(results)
        return results

    cmd = str(adapter_kwargs.get("apptainer_command", "apptainer"))

    # 1. Binary availability and version
    binary_path = shutil.which(cmd)
    if binary_path is None:
        logger.error(
            "Harbor pre-flight: %s binary not found in PATH", cmd
        )
        results["binary"] = "not_found"
    else:
        try:
            proc = subprocess.run(
                [cmd, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            version = proc.stdout.strip()
            logger.info("Harbor pre-flight: %s version %s", cmd, version)
            results["binary"] = version
        except Exception as e:
            logger.warning("Harbor pre-flight: could not get %s version: %s", cmd, e)
            results["binary"] = f"error: {e}"

    # 2. SIF cache
    sif_dir = adapter_kwargs.get("sif_cache_dir")
    if sif_dir is not None:
        sif_path = Path(sif_dir)
        if not sif_path.exists():
            logger.error(
                "Harbor pre-flight: SIF cache directory does not exist: %s",
                sif_path,
            )
            results["sif_cache"] = "not_found"
        else:
            sif_files = list(sif_path.glob("*.sif"))
            if not sif_files:
                logger.error(
                    "Harbor pre-flight: no .sif files in %s", sif_path
                )
                results["sif_cache"] = "empty"
            else:
                total_mb = sum(f.stat().st_size for f in sif_files) / 1024 / 1024
                logger.info(
                    "Harbor pre-flight: SIF cache at %s — %d images (%.0f MB total)",
                    sif_path, len(sif_files), total_mb,
                )
                results["sif_cache"] = {
                    "path": str(sif_path),
                    "count": len(sif_files),
                    "total_mb": round(total_mb, 1),
                }

    # 3. Trials directory disk space and writability
    trials_dir = Path(adapter_kwargs.get("trials_dir", "trials"))
    trials_dir.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(trials_dir)
        free_gb = usage.free / 1024 / 1024 / 1024
        if free_gb < 10:
            logger.warning(
                "Harbor pre-flight: low disk space for trials at %s — %.1f GB free",
                trials_dir, free_gb,
            )
        else:
            logger.info(
                "Harbor pre-flight: trials dir %s — %.1f GB free",
                trials_dir, free_gb,
            )
        results["trials_disk_free_gb"] = round(free_gb, 1)
    except OSError as e:
        logger.error(
            "Harbor pre-flight: cannot check disk space for %s: %s",
            trials_dir, e,
        )
        results["trials_disk_free_gb"] = f"error: {e}"

    # 4. Directory writability
    test_file = trials_dir / ".vb_preflight_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        results["trials_writable"] = True
    except OSError as e:
        logger.error(
            "Harbor pre-flight: trials dir %s is not writable: %s",
            trials_dir, e,
        )
        results["trials_writable"] = False

    # 5. Fakeroot support (quick test if fakeroot is enabled)
    if adapter_kwargs.get("fakeroot", False) and sif_dir is not None:
        sif_files = list(Path(sif_dir).glob("*.sif"))
        if sif_files:
            test_sif = str(sif_files[0])
            try:
                proc = subprocess.run(
                    [cmd, "exec", "--fakeroot", test_sif, "true"],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode == 0:
                    logger.info("Harbor pre-flight: fakeroot supported")
                    results["fakeroot"] = "supported"
                else:
                    logger.warning(
                        "Harbor pre-flight: fakeroot test failed (rc=%d): %s",
                        proc.returncode, proc.stderr.strip()[:200],
                    )
                    results["fakeroot"] = f"failed: {proc.stderr.strip()[:200]}"
            except subprocess.TimeoutExpired:
                logger.warning("Harbor pre-flight: fakeroot test timed out")
                results["fakeroot"] = "timeout"
            except Exception as e:
                logger.warning("Harbor pre-flight: fakeroot test error: %s", e)
                results["fakeroot"] = f"error: {e}"

    with _HARBOR_PREFLIGHT_CACHE_LOCK:
        _HARBOR_PREFLIGHT_CACHE[cache_key] = copy.deepcopy(results)
    return results


# ---------------------------------------------------------------------------
# Shared collection helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedEnvParams:
    """Resolved environment parameters from context + config overrides."""

    env_name: str
    adapter: str
    max_steps: int | None
    make_kwargs: dict[str, Any] | None
    env_make_kwargs: dict[str, Any]
    step_penalty: float | None
    extra_rewards: tuple
    reward_signal_name: str | None


def resolve_env_params(
    context: EnvironmentContext,
    *,
    env_name_override: str | None = None,
    adapter_override: str | None = None,
    max_steps_override: int | None = None,
    make_kwargs_override: dict[str, Any] | None = None,
    step_penalty_override: float | None = None,
) -> ResolvedEnvParams:
    """Resolve environment parameters from context with optional config overrides.

    Override semantics match ``collect_dataset.py``:
    - env_name, adapter, max_steps, make_kwargs: truthiness-based ``or``.
    - step_penalty: explicit None check (None = use context value).

    Does NOT log — logging stays at call sites.
    """
    env_name = env_name_override or context.env_name
    if not env_name:
        raise ValueError(
            "Environment name must be specified in config or context"
        )
    adapter = adapter_override or context.adapter or ""
    make_kwargs = make_kwargs_override or context.make_kwargs
    max_steps = resolve_effective_max_steps(
        max_steps_override or context.max_steps,
        make_kwargs,
    )
    env_make_kwargs = dict(make_kwargs or {})

    step_penalty = (
        step_penalty_override
        if step_penalty_override is not None
        else context.step_penalty
    )

    reward_signal_name = context.reward_signal_name
    extra_rewards: tuple = ()
    if step_penalty is not None:
        from llenvs.core.reward import StepPenalty

        extra_rewards = (StepPenalty(penalty=step_penalty),)
        reward_signal_name = None

    return ResolvedEnvParams(
        env_name=env_name,
        adapter=adapter,
        max_steps=max_steps,
        make_kwargs=make_kwargs,
        env_make_kwargs=env_make_kwargs,
        step_penalty=step_penalty,
        extra_rewards=extra_rewards,
        reward_signal_name=reward_signal_name,
    )


@dataclass
class EnvFactoryResult:
    """Result of building an environment factory for multi-instance adapters."""

    env_factory: Any
    harbor_kwargs: dict[str, Any] | None
    harbor_preflight: dict[str, Any] | None
    restore_fn: Any = None


def build_env_factory(
    adapter: str,
    env_name: str,
    env_make_kwargs: dict[str, Any],
    *,
    max_steps: int | None = None,
    extra_rewards: tuple = (),
    answer_extractor: Any | None = None,
    runtime_probing: bool = False,
) -> EnvFactoryResult:
    """Build an environment factory for Harbor or open_apps adapters."""
    if adapter == "harbor":
        harbor_kwargs: dict[str, Any] = (
            {"max_steps": max_steps} if max_steps else {}
        )
        if env_make_kwargs:
            harbor_kwargs.update(env_make_kwargs)
        if extra_rewards:
            harbor_kwargs["extra_rewards"] = extra_rewards
        if answer_extractor is not None:
            harbor_kwargs["answer_extractor"] = answer_extractor
        if runtime_probing:
            harbor_kwargs["runtime_probing"] = True

        harbor_preflight = check_harbor_environment(harbor_kwargs, env_name)
        env_factory, _ = create_harbor_env_factory(env_name, **harbor_kwargs)
        return EnvFactoryResult(env_factory, harbor_kwargs, harbor_preflight)

    if adapter == "open_apps":
        open_apps_kwargs: dict[str, Any] = (
            {"max_steps": max_steps} if max_steps else {}
        )
        if env_make_kwargs:
            open_apps_kwargs.update(env_make_kwargs)
        if extra_rewards:
            open_apps_kwargs["extra_rewards"] = extra_rewards
        env_factory, _ = create_open_apps_env_factory(
            env_name, **open_apps_kwargs
        )
        return EnvFactoryResult(env_factory, None, None)

    if adapter == "gymnasium" and env_make_kwargs and env_make_kwargs.get("use_images"):
        # Image-rendering gym envs (e.g., FrozenLake with pygame) cannot
        # use pure_step (pygame Surface objects aren't picklable). Provide
        # an env_factory + restore_fn so the runner can create fresh envs
        # per task for collection AND restore to arbitrary mid-trajectory
        # states for GT/ranking rollouts.
        from llenvs.adapters.gymnasium import GymnasiumAdapter

        gym_kwargs: dict[str, Any] = dict(env_make_kwargs)
        if max_steps is not None:
            gym_kwargs.setdefault("max_steps", max_steps)
        if extra_rewards:
            gym_kwargs["extra_rewards"] = extra_rewards
        if answer_extractor is not None:
            gym_kwargs["answer_extractor"] = answer_extractor

        def gym_env_factory():
            return GymnasiumAdapter().get_environment(env_name, **gym_kwargs)

        # Only FrozenLake has a restore_fn right now. Other gym envs get
        # None, which is fine for forward-only collection but will raise
        # if the caller tries mid-trajectory restoration (GT/ranking).
        restore_fn = None
        if env_name.startswith("frozen_lake") or gym_kwargs.get("map_name"):
            from llenvs.adapters.gymnasium import frozen_lake_restore

            restore_fn = frozen_lake_restore

        return EnvFactoryResult(
            gym_env_factory, None, None, restore_fn=restore_fn,
        )

    return EnvFactoryResult(None, None, None)


def build_history_fn(
    context: EnvironmentContext,
    *,
    max_history_turns: int | None = None,
) -> Any:
    """Build a history truncation function from environment context."""
    from qval.benchmark import (
        content_truncated_history,
        truncated_last_n_history,
    )

    history_fn = None
    if (
        context.min_action_chars is not None
        or context.min_observation_chars is not None
    ):
        history_fn = content_truncated_history(
            min_action_chars=context.min_action_chars,
            min_observation_chars=context.min_observation_chars,
        )
    if max_history_turns is not None:
        history_fn = truncated_last_n_history(
            max_history_turns, inner=history_fn
        )
    return history_fn


def build_prompt_budget(
    backend: Any,
    backend_config: BackendConfig,
    context: EnvironmentContext,
    *,
    max_history_turns: int | None = None,
) -> Any:
    """Build a budget-aware prompt truncation object, or None."""
    if (
        context.min_action_chars is None
        and context.min_observation_chars is None
        and context.min_current_observation_chars is None
    ):
        return None

    from qval.token_budget import make_prompt_budget

    return make_prompt_budget(
        backend,
        backend_config,
        min_observation_chars=context.min_observation_chars,
        min_action_chars=context.min_action_chars,
        min_current_observation_chars=context.min_current_observation_chars,
        max_history_turns=max_history_turns,
    )

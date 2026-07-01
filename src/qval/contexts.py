"""Context loaders for per-environment YAML files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from qval.types import EnvironmentContext, MethodContext, SignalType

_VALID_KEYS = {
    "task_description", "reward_description", "evaluator_extractor",
    "environment_extractor", "example_trajectories_file", "include_actor_thinking",
    "adapter", "env_name", "max_steps", "make_kwargs", "step_penalty", "turn_info",
    "reward_signal_name", "min_action_chars", "min_observation_chars",
    "min_current_observation_chars", "min_next_observation_chars",
    "min_ranking_candidate_action_chars", "min_ranking_next_observation_chars",
    "ranking_environment_extractor", "prompting_scheme", "system_prompt_file",
    "early_turns_to_discard", "late_turns_to_discard",
    "invalid_action_text", "invalid_action_observation", "advance_on_invalid",
    "inject_task_text_in_prompts", "include_state_text_when_images",
    "include_images",
}

# Default environments directory: <project_root>/shared/configs/environments/
_DEFAULT_CONTEXTS_DIR = (
    Path(__file__).resolve().parents[2] / "shared" / "configs" / "environments"
)


def _expand_make_kwargs_paths(value: Any) -> Any:
    """Expand ``$VARS`` and ``~`` recursively inside ``make_kwargs`` only."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand_make_kwargs_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_make_kwargs_paths(v) for v in value]
    return value


def _parse_extractor(spec: dict[str, Any]) -> Any:
    """Parse an extractor specification into an llenvs AnswerExtractor.

    Supports:
    - ``type``: registry name (e.g., ``"tag_based"``, ``"numeric"``)
    - ``pre_cleaners``: list of cleaner names applied before extraction
    - ``post_cleaners``: list of cleaner names applied after extraction
    - ``type: composite`` with ``extractors: [...]``: recursive parsing

    Args:
        spec: Dictionary with extractor configuration.

    Returns:
        An AnswerExtractor instance.

    Raises:
        ValueError: If the spec is missing ``type`` or contains invalid values.
    """
    from llenvs.core.cleaning import resolve_cleaners
    from llenvs.core.extraction import CleanedExtractor, CompositeExtractor, SingleLineExtractor
    from llenvs.core.registry import answer_extractor_registry

    if "type" not in spec:
        raise ValueError("Extractor spec must include 'type'")

    extractor_type = spec["type"]
    pre_cleaner_names = spec.get("pre_cleaners")
    post_cleaner_names = spec.get("post_cleaners")

    # Build the inner extractor
    if extractor_type == "composite":
        sub_specs = spec.get("extractors", [])
        if not sub_specs:
            raise ValueError("Composite extractor requires a non-empty 'extractors' list")
        inner = CompositeExtractor(
            extractors=[_parse_extractor(s) for s in sub_specs],
        )
    elif extractor_type == "single_line":
        inner_spec = spec.get("inner")
        if inner_spec is None:
            raise ValueError("Single-line extractor requires an 'inner' extractor spec")
        kwargs = {
            k: v for k, v in spec.items()
            if k not in ("type", "inner", "pre_cleaners", "post_cleaners")
        }
        inner = SingleLineExtractor(
            inner=_parse_extractor(inner_spec),
            **kwargs,
        )
    else:
        # Pass remaining kwargs (excluding known meta-keys) to the registry
        kwargs = {
            k: v for k, v in spec.items()
            if k not in ("type", "pre_cleaners", "post_cleaners")
        }
        inner = answer_extractor_registry.create(extractor_type, **kwargs)

    # Wrap with cleaners if specified
    pre_cleaners = resolve_cleaners(pre_cleaner_names, "pre") if pre_cleaner_names is not None else []
    post_cleaners = resolve_cleaners(post_cleaner_names, "post") if post_cleaner_names is not None else []

    if pre_cleaners or post_cleaners:
        return CleanedExtractor(
            inner=inner,
            pre_cleaners=pre_cleaners,
            post_cleaners=post_cleaners,
        )

    return inner


def _load_context_data(
    env_name: str,
    contexts_dir: str | Path | None = None,
) -> dict[str, Any]:
    ctx_dir = Path(contexts_dir) if contexts_dir is not None else _DEFAULT_CONTEXTS_DIR
    ctx_path = ctx_dir / f"{env_name}.yaml"

    if not ctx_path.exists():
        available = sorted(
            p.stem for p in ctx_dir.glob("*.yaml") if p.is_file()
        )
        raise FileNotFoundError(
            f"No context file found for '{env_name}' at {ctx_path}. "
            f"Available contexts: {available}"
        )

    with ctx_path.open() as f:
        data = yaml.safe_load(f) or {}

    unknown = set(data) - _VALID_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in {ctx_path.name}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {sorted(_VALID_KEYS)}"
        )

    evaluator_extractor = None
    if "evaluator_extractor" in data and data["evaluator_extractor"] is not None:
        evaluator_extractor = _parse_extractor(data["evaluator_extractor"])

    environment_extractor = None
    if "environment_extractor" in data and data["environment_extractor"] is not None:
        environment_extractor = _parse_extractor(data["environment_extractor"])

    ranking_environment_extractor = None
    if "ranking_environment_extractor" in data and data["ranking_environment_extractor"] is not None:
        ranking_environment_extractor = _parse_extractor(data["ranking_environment_extractor"])

    example_trajectories = None
    if "example_trajectories_file" in data and data["example_trajectories_file"] is not None:
        from qval.trajectory_store import load_trajectories

        traj_path = ctx_dir / data["example_trajectories_file"]
        example_trajectories = load_trajectories(traj_path)

    # Resolve system_prompt_file relative to the contexts directory
    system_prompt_file = data.get("system_prompt_file")
    if system_prompt_file is not None:
        system_prompt_file = str(ctx_dir / system_prompt_file)

    return {
        "task_description": data.get("task_description"),
        "reward_description": data.get("reward_description"),
        "example_trajectories": example_trajectories,
        "evaluator_extractor": evaluator_extractor,
        "environment_extractor": environment_extractor,
        "include_actor_thinking": data.get("include_actor_thinking", False),
        "adapter": data.get("adapter"),
        "env_name": data.get("env_name"),
        "max_steps": data.get("max_steps"),
        "make_kwargs": _expand_make_kwargs_paths(data.get("make_kwargs")),
        "step_penalty": data.get("step_penalty"),
        "turn_info": data.get("turn_info"),
        "reward_signal_name": data.get("reward_signal_name", "correctness"),
        "min_action_chars": data.get("min_action_chars"),
        "min_observation_chars": data.get("min_observation_chars"),
        "min_current_observation_chars": data.get("min_current_observation_chars"),
        "min_next_observation_chars": data.get("min_next_observation_chars"),
        "min_ranking_candidate_action_chars": data.get("min_ranking_candidate_action_chars"),
        "min_ranking_next_observation_chars": data.get("min_ranking_next_observation_chars"),
        "ranking_environment_extractor": ranking_environment_extractor,
        "prompting_scheme": data.get("prompting_scheme"),
        "system_prompt_file": system_prompt_file,
        "early_turns_to_discard": data.get("early_turns_to_discard"),
        "late_turns_to_discard": data.get("late_turns_to_discard"),
        "invalid_action_text": data.get("invalid_action_text"),
        "invalid_action_observation": data.get("invalid_action_observation"),
        "advance_on_invalid": data.get("advance_on_invalid"),
        "inject_task_text_in_prompts": data.get(
            "inject_task_text_in_prompts", False,
        ),
        "include_state_text_when_images": data.get(
            "include_state_text_when_images", True,
        ),
        "include_images": data.get("include_images", True),
    }


def load_environment_context(
    env_name: str,
    contexts_dir: str | Path | None = None,
) -> EnvironmentContext:
    """Load environment-side context from a per-environment YAML file."""
    return EnvironmentContext(**_load_context_data(env_name, contexts_dir))


def load_context(
    env_name: str,
    signal_type: SignalType,
    contexts_dir: str | Path | None = None,
) -> MethodContext:
    """Load a MethodContext from a per-environment YAML file.

    Looks for ``{contexts_dir}/{env_name}.yaml`` and returns a MethodContext
    with ``signal_type`` set and environment-side fields populated from YAML.
    """
    base = _load_context_data(env_name, contexts_dir)

    return MethodContext(
        signal_type=signal_type,
        task_description=base["task_description"],
        reward_description=base["reward_description"],
        example_trajectories=base["example_trajectories"],
        evaluator_extractor=base["evaluator_extractor"],
        environment_extractor=base["environment_extractor"],
        include_actor_thinking=base["include_actor_thinking"],
        include_state_text_when_images=base["include_state_text_when_images"],
        include_images=base["include_images"],
        adapter=base["adapter"],
        env_name=base["env_name"],
        max_steps=base["max_steps"],
        make_kwargs=base["make_kwargs"],
        step_penalty=base["step_penalty"],
        turn_info=base["turn_info"],
        reward_signal_name=base["reward_signal_name"],
        min_action_chars=base["min_action_chars"],
        min_observation_chars=base["min_observation_chars"],
        min_current_observation_chars=base["min_current_observation_chars"],
        min_next_observation_chars=base["min_next_observation_chars"],
        min_ranking_candidate_action_chars=base["min_ranking_candidate_action_chars"],
        min_ranking_next_observation_chars=base["min_ranking_next_observation_chars"],
        prompting_scheme=base["prompting_scheme"],
        system_prompt_file=base["system_prompt_file"],
        invalid_action_text=base["invalid_action_text"],
        invalid_action_observation=base["invalid_action_observation"],
        advance_on_invalid=base["advance_on_invalid"],
    )

"""Registry and backend adapters for scripted rollout policies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
)

ScriptedPolicyFn = Callable[[str, dict[str, Any]], str]


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    """Registered scripted policy with static metadata."""

    fn: ScriptedPolicyFn
    metadata: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ScriptedPolicyConfig] = {}


def register_policy(
    name: str,
    fn: ScriptedPolicyFn,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a scripted policy under a unique name."""
    if name in _REGISTRY:
        raise ValueError(f"Scripted policy already registered: {name!r}")
    _REGISTRY[name] = ScriptedPolicyConfig(fn=fn, metadata=dict(metadata or {}))


def get_policy(name: str) -> ScriptedPolicyConfig:
    """Look up a scripted policy by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown scripted policy: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def list_policies() -> list[str]:
    """List registered scripted policy names."""
    return sorted(_REGISTRY)


def make_stochastic_policy(
    base_policy_name: str,
    epsilon: float,
    seed: int | None = None,
) -> ScriptedPolicyConfig:
    """Wrap a registered policy with epsilon-uniform-random action mixing.

    With probability ``epsilon`` the wrapper returns a uniformly random
    action sampled from the base policy's action space; with probability
    ``1 - epsilon`` it delegates to the base policy.

    The action space is declared in ``base.metadata["actions"]`` and may
    be either:

    * **A fixed sequence** (tuple/list) of action strings — use for
      environments with a static discrete action space such as
      FrozenLake (e.g. ``("left", "down", "right", "up")``).
    * **A callable** ``(state_text, metadata) -> Sequence[str]`` that
      returns the currently-valid actions given the observation — use
      for environments with a state-dependent action space such as
      OpenApps (where clickable element bids change every step).  When
      the callable returns an empty sequence the wrapper falls through
      to the base policy rather than raising.

    Use for collection (not GT): a perfect scripted policy on a cheap
    env like FrozenLake collapses the GT distribution because every
    trajectory succeeds with the same return; mixing in random actions
    yields a balanced dataset of good and bad trajectories.

    Args:
        base_policy_name: Name of a registered scripted policy. Its
            metadata must contain an ``actions`` key (sequence or
            callable as described above).
        epsilon: Probability of taking a random action (0 ≤ epsilon ≤ 1).
        seed: RNG seed for reproducibility. ``None`` → non-deterministic.

    Returns:
        A new ``ScriptedPolicyConfig`` ready to wrap with
        ``ScriptedBackend``.  Carries through the base policy's metadata.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    base = get_policy(base_policy_name)
    actions = base.metadata.get("actions")
    if actions is None:
        raise ValueError(
            f"Policy {base_policy_name!r} has no 'actions' in metadata; "
            "cannot mix in random actions. Register the policy with an "
            "'actions' tuple or callable to enable stochastic wrapping."
        )

    if callable(actions):
        action_fn: Callable[[str, dict[str, Any]], Any] = actions
    else:
        static_tuple = tuple(actions)
        if not static_tuple:
            raise ValueError(
                f"Policy {base_policy_name!r} has an empty 'actions' tuple; "
                "stochastic wrapping requires at least one action."
            )

        def _static_actions(
            _state: str, _metadata: dict[str, Any]
        ) -> tuple[str, ...]:
            return static_tuple

        action_fn = _static_actions

    rng = random.Random(seed)

    def stochastic_fn(state: str, metadata: dict[str, Any]) -> str:
        if rng.random() < epsilon:
            choices = action_fn(state, metadata)
            if choices:
                return rng.choice(tuple(choices))
            # State-dependent action space happens to be empty this
            # turn — fall back to the base policy rather than crashing.
        return base.fn(state, metadata)

    return ScriptedPolicyConfig(fn=stochastic_fn, metadata=dict(base.metadata))


class ScriptedBackend(ModelBackend):
    """ModelBackend wrapper around a deterministic scripted policy."""

    def __init__(
        self,
        policy_fn: ScriptedPolicyFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._policy_fn = policy_fn
        self._metadata = dict(metadata or {})

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True)

    @property
    def model_name(self) -> str:
        return "scripted-policy"

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        del params
        return [self._result_for_state(prompt) for prompt in prompts]

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        del params
        state_text = self._extract_state_text(messages)
        return self._result_for_state(state_text, messages=messages)

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        return [self.generate_chat(messages, params) for messages in messages_batch]

    def _extract_state_text(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content is not None:
                return message.content
        raise ValueError("ScriptedBackend requires a user message with state text")

    def _result_for_state(
        self,
        state_text: str,
        messages: list[ChatMessage] | None = None,
    ) -> GenerationResult:
        # Inject the conversation history into a transient metadata dict
        # under the ``_messages`` key so policies that need turn-zero
        # context (e.g. the initial "Goal: ..." line) can access it.
        # Existing policies that ignore ``_messages`` are unaffected.
        metadata = dict(self._metadata)
        if messages is not None:
            metadata["_messages"] = messages
        action_text = self._policy_fn(state_text, metadata)
        return GenerationResult(
            text=action_text,
            finish_reason=StopReason.END_OF_TEXT,
        )


from . import alfworld as _alfworld  # noqa: E402,F401
from . import frozen_lake as _frozen_lake  # noqa: E402,F401
from . import open_apps as _open_apps  # noqa: E402,F401


__all__ = [
    "ScriptedBackend",
    "ScriptedPolicyConfig",
    "ScriptedPolicyFn",
    "get_policy",
    "list_policies",
    "make_stochastic_policy",
    "register_policy",
]

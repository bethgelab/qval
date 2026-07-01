"""Serialize environment states and actions to text for LLM prompts."""

from __future__ import annotations

import re
from typing import Any

from llenvs.core.cleaning import strip_thinking_tokens
from llenvs.core.state import Action, Observation, ObservationImages, State
from llenvs.core.tools import format_tool_call, format_tool_result_data

UNPARSED_ACTION_PLACEHOLDER = "[invalid action: could not parse from model response]"

# ── Thought extraction (ReAct-style) ─────────────────────────────

_TAG_THOUGHT = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_CLASSIC_THOUGHT = re.compile(
    r"Thought:\s*(.*?)(?=\nAction:|\Z)", re.DOTALL | re.IGNORECASE
)


def extract_thought(raw_text: str) -> str | None:
    """Extract thought from ReAct-formatted text.

    Supports two formats, tried in order:

    1. **Tag format**: ``<thought>...</thought>``
    2. **Classic format**: ``Thought: ...`` followed by ``\\nAction:`` or end of string

    Returns the content of the **last** match (consistent with llenvs
    extractor conventions).  Returns ``None`` when no thought is found or
    when the matched content is empty/whitespace-only.
    """
    # Try tags first (preferred — unambiguous, multi-line safe)
    matches = list(_TAG_THOUGHT.finditer(raw_text))
    if matches:
        content = matches[-1].group(1).strip()
        return content or None

    # Fall back to classic Thought:/Action: format
    matches = list(_CLASSIC_THOUGHT.finditer(raw_text))
    if matches:
        content = matches[-1].group(1).strip()
        return content or None

    return None


def strip_thought_tags(text: str) -> str:
    """Remove ``<thought>...</thought>`` tags from text, keeping surrounding content."""
    return _TAG_THOUGHT.sub("", text).strip()


def action_text_for_display(action: Any, resolved: str | None, extracted: str | None) -> str:
    """Pick the best action text for display.

    For tool-calling actions, ``serialize_action`` includes formatted tool
    calls which ``resolved_action``/``extracted_action`` cannot carry.
    For text-only actions, the three-tier priority chain applies:
    ``resolved_action`` → ``extracted_action`` → raw text with thinking stripped.
    """
    if isinstance(action, Action) and action.tool_calls:
        return serialize_action(action)
    text = (
        resolved
        or extracted
        or strip_thinking_tokens(serialize_action(action))
    )
    return text.strip() or UNPARSED_ACTION_PLACEHOLDER


def serialize_observation(state_or_obs: Any) -> str:
    """Convert a state or observation to a text representation.

    Handles:
    - Observation: renders prompt and message history.
    - State: extracts and serializes its observation.
    - str: returned as-is.
    - None: returns empty string.
    - Other types: falls back to str().
    """
    if state_or_obs is None:
        return ""

    if isinstance(state_or_obs, str):
        return state_or_obs

    if isinstance(state_or_obs, State):
        return serialize_observation(state_or_obs.observation)

    if isinstance(state_or_obs, Observation):
        return _serialize_text_observation(state_or_obs)

    return str(state_or_obs)


def extract_images(state_or_obs: Any) -> ObservationImages:
    """Extract images from a state or observation, separated by source.

    Mirrors the dispatch logic of :func:`serialize_observation` but returns
    an :class:`ObservationImages` with task and state images kept separate.
    Returns an empty ``ObservationImages`` for text-only inputs.

    Handles:
    - State: extracts images from its observation via ``get_images()``.
    - Observation: calls ``get_images()`` which reads from ``task`` and ``state``.
    - None, str, other types: returns empty ``ObservationImages()``.
    """
    if state_or_obs is None or isinstance(state_or_obs, str):
        return ObservationImages()

    if isinstance(state_or_obs, State):
        return extract_images(state_or_obs.observation)

    if isinstance(state_or_obs, Observation):
        return state_or_obs.get_images()

    return ObservationImages()


def serialize_action(action: Any) -> str:
    """Convert an action to a text representation.

    For tool-calling actions (``action.tool_calls`` non-empty), appends
    formatted tool calls after any reasoning text using
    ``format_tool_call()`` from llenvs.

    Handles:
    - Action: returns text + formatted tool calls.
    - str: returned as-is.
    - None: returns empty string.
    - Other types: falls back to str().
    """
    if action is None:
        return ""

    if isinstance(action, str):
        return action

    if isinstance(action, Action):
        parts: list[str] = []
        if action.text:
            parts.append(action.text)
        if action.tool_calls:
            tool_lines = [f"- {format_tool_call(tc)}" for tc in action.tool_calls]
            parts.append("[Tool Calls]\n" + "\n".join(tool_lines))
        return "\n\n".join(parts) if parts else ""

    return str(action)


def _serialize_text_observation(obs: Observation) -> str:
    """Serialize an Observation for evaluator prompts.

    Prefers structured observation fields when available:
    - ``obs.state`` (dynamic per-step observation) — canonical for multi-turn.
      When ``obs.state.data`` contains ``"tool_results"``, appends formatted
      tool results after the state text.
    - ``obs.task`` (static task description) — used for single-turn envs where
      the task IS the state and the evaluator needs the instance-specific question.

    Falls back to legacy heuristic parsing when both are None:
    - No messages (initial state / single-turn): return obs.prompt as-is.
    - Has messages (multi-turn subsequent states): return only user-role
      message contents joined by newline.
    """
    # Structured multi-turn: obs.state is the canonical dynamic state.
    if obs.state is not None:
        text = obs.state.text
        # Append tool results if present in state data
        if obs.state.data and "tool_results" in obs.state.data:
            tool_results = obs.state.data["tool_results"]
            if tool_results:
                formatted = format_tool_result_data(tool_results)
                text = f"{text}\n\n[Tool Results]\n{formatted}"
        return text

    # Structured single-turn: obs.task IS the state (no separate dynamic state).
    # The evaluator needs the instance-specific question to estimate V(s).
    if obs.task is not None:
        return obs.task.text

    # Legacy: no structured obs, fall back to heuristic
    if not obs.messages:
        return obs.prompt or ""

    # Return only the last user-role message — the latest environment feedback.
    for msg in reversed(obs.messages):
        role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
        if role != "user":
            continue
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content:
            return content
    return ""

"""Registry for manual action samplers used in ranking point construction.

An action sampler generates alternative candidate actions for a given state.
Each registered sampler is a callable with signature::

    (state: Any, k: int, rng: random.Random) -> list[ActionCandidate]

where *k* is the number of alternative candidates to produce. The sampler
should return up to *k* candidates (fewer if the action space is smaller).
"""

from __future__ import annotations

import random
from typing import Any, Callable

from qval.types import ActionCandidate, RankingCandidateSource

ActionSamplerFn = Callable[[Any, int, random.Random], list[ActionCandidate]]
"""Signature: (state, k, rng) -> list[ActionCandidate]."""

_REGISTRY: dict[str, ActionSamplerFn] = {}


def register_sampler(name: str, fn: ActionSamplerFn) -> None:
    """Register a manual action sampler under a unique name."""
    if name in _REGISTRY:
        raise ValueError(f"Action sampler already registered: {name!r}")
    _REGISTRY[name] = fn


def get_sampler(name: str) -> ActionSamplerFn:
    """Look up a registered action sampler by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown action sampler: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def list_samplers() -> list[str]:
    """List registered action sampler names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in: FrozenLake
# ---------------------------------------------------------------------------

_FROZEN_LAKE_ACTIONS = ("left", "down", "right", "up")


def _frozen_lake_sampler(
    state: Any, k: int, rng: random.Random
) -> list[ActionCandidate]:
    """Sample up to *k* FrozenLake actions (without replacement).

    Returns ``ActionCandidate`` objects with the action text set as both
    ``extracted_action`` and ``resolved_action``.
    """
    from llenvs.core.state import Action

    available = list(_FROZEN_LAKE_ACTIONS)
    if k >= len(available):
        selected = available
    else:
        selected = rng.sample(available, k)
    return [
        ActionCandidate(
            action=Action(text=a),
            source=RankingCandidateSource.RANKING_MANUAL,
            extracted_action=a,
            resolved_action=a,
        )
        for a in selected
    ]


register_sampler("frozen_lake", _frozen_lake_sampler)


# ---------------------------------------------------------------------------
# Built-in: OpenApps (BrowserGym)
# ---------------------------------------------------------------------------


def _open_apps_sampler(
    state: Any, k: int, rng: random.Random
) -> list[ActionCandidate]:
    """Sample up to *k* browser actions from the current AXTree.

    Parses the accessibility tree text to find clickable, fillable, and other
    interactive elements, then generates candidate BrowserGym actions.
    """
    import re

    from llenvs.core.state import Action

    axtree = ""
    if hasattr(state, "observation"):
        obs = state.observation
        if hasattr(obs, "state") and obs.state is not None:
            axtree = obs.state.text or ""

    candidates: list[str] = []

    for line in axtree.split("\n"):
        stripped = line.strip()
        bid_match = re.match(r"\[(\d+)\]", stripped)
        if bid_match is None:
            continue
        bid = bid_match.group(1)

        lower = stripped.lower()
        if "clickable" not in lower:
            continue

        if "textbox" in lower or "input" in lower:
            candidates.append(f"fill('{bid}', 'example text')")
        elif "select" in lower:
            candidates.append(f"select_option('{bid}', '0')")
        else:
            candidates.append(f"click('{bid}')")

    if not candidates:
        candidates = ["noop(1000)"]

    if k >= len(candidates):
        selected = candidates
    else:
        selected = rng.sample(candidates, k)

    return [
        ActionCandidate(
            action=Action(text=a),
            source=RankingCandidateSource.RANKING_MANUAL,
            extracted_action=a,
            resolved_action=a,
        )
        for a in selected
    ]


register_sampler("open_apps", _open_apps_sampler)


# ---------------------------------------------------------------------------
# Built-in: ALFWorld
# ---------------------------------------------------------------------------


def _alfworld_sampler(
    state: Any, k: int, rng: random.Random
) -> list[ActionCandidate]:
    """Sample up to *k* ALFWorld actions from the state's admissible commands.

    Reads ``state.hidden.admissible_commands`` (the TextWorld-provided
    list of valid commands at the current step) and picks *k* without
    replacement. Falls back to the single literal ``"look"`` command if
    no admissible list is available — rare, but keeps ranking points
    well-formed.
    """
    from llenvs.core.state import Action

    admissible: tuple[str, ...] = ()
    if hasattr(state, "hidden") and state.hidden is not None:
        admissible = tuple(getattr(state.hidden, "admissible_commands", ()) or ())

    candidates = list(admissible) if admissible else ["look"]
    if k >= len(candidates):
        selected = candidates
    else:
        selected = rng.sample(candidates, k)
    return [
        ActionCandidate(
            action=Action(text=a),
            source=RankingCandidateSource.RANKING_MANUAL,
            extracted_action=a,
            resolved_action=a,
        )
        for a in selected
    ]


register_sampler("alfworld", _alfworld_sampler)


__all__ = [
    "ActionSamplerFn",
    "get_sampler",
    "list_samplers",
    "register_sampler",
]

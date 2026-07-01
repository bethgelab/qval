"""Compose per-game Jericho system prompts from fragments."""

from __future__ import annotations

from qval.jericho_prompts.fragments import (
    ACTION_STRUCTURE,
    GAME_MECHANICS,
    PREPOSITIONS,
    ROLE_AND_GOAL,
)
from qval.jericho_prompts.game_data import get_game_data
from qval.jericho_prompts.verb_extraction import extract_verb_reference

_CACHE: dict[str, str] = {}


def _render_game_description(game_name: str) -> str:
    """Fragment 2: per-game description."""
    data = get_game_data(game_name)
    return (
        f"## About This Game\n\n"
        f"{data.display_name} is a {data.genre} game. {data.premise}"
    )


def _render_scoring(game_name: str) -> str:
    """Fragment 6: per-game scoring system."""
    data = get_game_data(game_name)
    return (
        f"## Scoring\n\n"
        f"Maximum score: {data.max_score} points.\n"
        f"{data.scoring_description}"
    )


def build_jericho_env_prompt(game_name: str) -> str:
    """Build the full Layer-1 environment prompt for a Jericho game.

    Composes 7 fragments (4 universal + 3 per-game) into a single
    string suitable for use as the ``ENV_PROMPTS`` layer in the
    three-layer actor system prompt.
    """
    if game_name in _CACHE:
        return _CACHE[game_name]

    fragments = [
        ROLE_AND_GOAL,                          # 1. Universal
        _render_game_description(game_name),    # 2. Per-game
        ACTION_STRUCTURE,                       # 3. Universal
        extract_verb_reference(game_name),      # 4. Per-game
        PREPOSITIONS,                           # 5. Universal
        _render_scoring(game_name),             # 6. Per-game
        GAME_MECHANICS,                         # 7. Universal
    ]
    result = "\n\n".join(fragments)
    _CACHE[game_name] = result
    return result

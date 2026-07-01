"""Extract and categorize verbs from Jericho grammar templates."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Verb -> category mapping.  Categories are ordered for display.
CATEGORY_ORDER = (
    "Movement",
    "Observation",
    "Manipulation",
    "Combat",
    "Communication",
    "Interaction",
    "Magic",
    "Light & Fire",
    "Body & Consumption",
    "Other",
)

VERB_CATEGORIES: dict[str, set[str]] = {
    "Movement": {
        "north", "south", "east", "west", "up", "down", "enter", "exit",
        "go", "climb", "leave", "cross", "board", "swim", "jump", "dive",
        "crawl", "run", "walk", "step", "proceed",
    },
    "Observation": {
        "look", "examine", "read", "search", "listen", "smell", "find",
        "inventory", "hear", "taste", "touch", "feel", "describe", "gaze",
        "stare", "sniff", "watch", "check", "inspect",
    },
    "Manipulation": {
        "take", "get", "drop", "put", "open", "close", "pick", "insert",
        "remove", "move", "pull", "push", "press", "lift", "turn", "flip",
        "set", "tie", "fill", "empty", "pour", "lock", "unlock", "wear",
        "hold", "wave", "dig", "squeeze", "wind", "lower", "raise", "shake",
        "spin", "slide", "drag", "roll", "peel", "hang", "place", "stuff",
        "lay", "grab", "catch", "toss", "hurl", "chuck",
    },
    "Combat": {
        "attack", "hit", "kill", "fight", "stab", "cut", "break", "throw",
        "swing", "strike", "smash", "slice", "punch", "kick", "bite",
        "murder", "slay",
    },
    "Communication": {
        "say", "ask", "tell", "answer", "shout", "hello", "talk", "yell",
        "scream", "speak", "consult", "call", "reply",
    },
    "Interaction": {
        "give", "feed", "offer", "show", "display", "present", "buy", "pay",
        "trade",
    },
    "Magic": {
        "cast", "chant", "enchant", "wish", "pray", "bless", "exorcise",
        "banish",
    },
    "Light & Fire": {
        "light", "douse", "burn", "extinguish", "ignite", "blow",
    },
    "Body & Consumption": {
        "eat", "drink", "sleep", "wake", "sit", "stand", "wait", "rest",
    },
}

# Reverse lookup: verb string -> category name.
_VERB_TO_CATEGORY: dict[str, str] = {}
for _cat, _verbs in VERB_CATEGORIES.items():
    for _v in _verbs:
        _VERB_TO_CATEGORY[_v] = _cat


def _get_game_grammar(game_name: str) -> str | None:
    """Load the raw grammar string for *game_name* from jericho."""
    try:
        from jericho import game_info  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("jericho not installed; cannot extract verb reference")
        return None

    info = getattr(game_info, game_name, None)
    if info is None:
        # Some games use different attribute names (e.g., "nine05" for "905").
        for attr in dir(game_info):
            candidate = getattr(game_info, attr)
            if isinstance(candidate, dict) and candidate.get("name") == game_name:
                info = candidate
                break
    if info is None or not isinstance(info, dict):
        return None
    return info.get("grammar")


def _categorize_verb(verb: str, synonym_group: list[str]) -> str:
    """Return the category for *verb*, checking synonyms if needed."""
    cat = _VERB_TO_CATEGORY.get(verb)
    if cat is not None:
        return cat
    for syn in synonym_group:
        cat = _VERB_TO_CATEGORY.get(syn)
        if cat is not None:
            return cat
    return "Other"


# Preferred verbs when multiple synonyms are in the category map.
# Ordered by preference (most natural/common first).
_PREFERRED_VERBS = (
    "take", "get", "drop", "put", "open", "close", "look", "examine",
    "go", "walk", "enter", "exit", "climb", "jump", "swim", "dive",
    "attack", "kill", "hit", "throw", "break", "cut",
    "say", "ask", "tell", "answer", "talk", "shout",
    "give", "show", "buy",
    "eat", "drink", "wait", "sit", "stand", "sleep", "wake",
    "light", "burn",
    "push", "pull", "turn", "move", "press", "lift", "fill",
    "lock", "unlock", "wear", "tie", "dig", "pour", "wave",
    "search", "read", "listen", "smell", "feel", "taste",
    "pray", "cast", "wish",
)
_PREFERRED_SET = frozenset(_PREFERRED_VERBS)


def _pick_representative(synonym_group: list[str]) -> str:
    """Pick the best representative verb from a synonym group.

    Prefers common verbs from ``_PREFERRED_VERBS``, then any verb in
    our category mapping, then falls back to the first synonym.
    """
    # First pass: find the highest-priority preferred verb.
    for preferred in _PREFERRED_VERBS:
        if preferred in synonym_group:
            return preferred
    # Second pass: any verb in the category map.
    for v in synonym_group:
        if v in _VERB_TO_CATEGORY:
            return v
    return synonym_group[0]


def _extract_verbs_from_grammar(grammar: str) -> dict[str, list[str]]:
    """Parse grammar string and return ``{category: [verbs]}``."""
    templates = [t.strip() for t in grammar.split(";") if t.strip()]
    seen: set[str] = set()
    categorized: dict[str, list[str]] = {}

    for template in templates:
        # Remove OBJ markers to get verb/prep tokens.
        tokens = template.replace(" OBJ", "").split()
        if not tokens:
            continue
        verb_group_str = tokens[0]
        synonyms = verb_group_str.split("/")

        representative = _pick_representative(synonyms)
        if representative in seen:
            continue
        seen.add(representative)

        cat = _categorize_verb(representative, synonyms)
        categorized.setdefault(cat, []).append(representative)

    return categorized


def _render_verb_reference(categorized: dict[str, list[str]]) -> str:
    """Render categorized verbs as a prompt fragment."""
    lines = ["## Available Verbs"]
    for cat in CATEGORY_ORDER:
        if cat == "Other":
            continue
        verbs = categorized.get(cat)
        if verbs:
            lines.append(f"\n{cat}: {', '.join(sorted(verbs))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_CACHE: dict[str, str] = {}

_GENERIC_FALLBACK = (
    "## Available Verbs\n\n"
    "Movement: north, south, east, west, up, down, go, enter, exit, "
    "climb, jump\n"
    "Observation: look, examine, read, search, listen, inventory\n"
    "Manipulation: take, drop, put, open, close, pick up, move, pull, "
    "push, turn, lock, unlock, fill, wear, tie, dig\n"
    "Combat: attack, hit, kill, throw, break, cut\n"
    "Communication: say, ask, tell, answer\n"
    "Interaction: give, show\n"
    "Light & Fire: light, burn\n"
    "Body & Consumption: eat, drink, wait"
)


def extract_verb_reference(game_name: str) -> str:
    """Build the verb reference fragment for *game_name*.

    Returns a formatted string listing available verbs by category.
    Falls back to a generic reference when jericho is not installed or
    the game is unknown.
    """
    if game_name in _CACHE:
        return _CACHE[game_name]

    grammar = _get_game_grammar(game_name)
    if grammar is None:
        return _GENERIC_FALLBACK

    categorized = _extract_verbs_from_grammar(grammar)
    result = _render_verb_reference(categorized)
    _CACHE[game_name] = result
    return result

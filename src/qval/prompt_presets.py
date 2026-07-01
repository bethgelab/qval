"""Prompt preset library with explicit building blocks.

Every variable text piece in prompt assembly is a named block registered in
``_BLOCK_REGISTRY``.  A ``PromptPreset`` composes blocks by name reference,
making the final prompt content fully transparent from the preset definition
alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from llenvs.inference import PromptFragment


# ---------------------------------------------------------------------------
# Block registry
# ---------------------------------------------------------------------------

_BLOCK_REGISTRY: dict[str, PromptFragment] = {}


def _register_block(name: str, content: str) -> None:
    """Register a named prompt block (internal helper)."""
    if name in _BLOCK_REGISTRY:
        raise ValueError(f"Block {name!r} is already registered")
    _BLOCK_REGISTRY[name] = PromptFragment(
        name=name,
        content=content,
        category="vb_preset",
    )


def get_block(name: str) -> PromptFragment:
    """Retrieve a registered block by name.

    Raises:
        KeyError: If no block with the given name exists.
    """
    try:
        return _BLOCK_REGISTRY[name]
    except KeyError:
        available = sorted(_BLOCK_REGISTRY.keys())
        raise KeyError(
            f"Unknown prompt block: {name!r}. Available: {available}"
        ) from None


def list_blocks() -> list[str]:
    """Return sorted list of registered block names."""
    return sorted(_BLOCK_REGISTRY.keys())


def resolve_block(
    block_name: str | None,
    *,
    signal_name: str | None = None,
) -> str | None:
    """Resolve a block name to its content string, optionally formatting it.

    Args:
        block_name: Name of a registered block, or ``None``.
        signal_name: If provided, ``{signal_name}`` placeholders in the
            block content are replaced with this value.

    Returns:
        The block's content string (possibly formatted), or ``None`` if
        *block_name* is ``None``.
    """
    if block_name is None:
        return None
    content = get_block(block_name).content
    if signal_name is not None:
        content = content.format(signal_name=signal_name)
    return content


# ---------------------------------------------------------------------------
# Built-in blocks
# ---------------------------------------------------------------------------

_register_block(
    "role_opener_direct_default",
    "You are an expert at estimating {signal_name} functions for "
    "reinforcement learning environments.",
)

_register_block(
    "role_opener_codegen_default",
    "You are an expert at writing {signal_name} functions for "
    "reinforcement learning environments.",
)

_register_block(
    "closing_direct_default",
    "After any reasoning, respond with ONLY a single numeric "
    "{signal_name} estimate. Do not include any explanation after "
    "your final number.",
)

_register_block(
    "codegen_closing_default",
    "You may use the `collections`, `itertools`, `json`, `math`, `re`, "
    "`statistics`, and `string` standard library modules. "
    "The function will run in a restricted sandbox that does not expose "
    "the introspection builtins `locals`, `globals`, `vars`, `dir`, `eval`, "
    "or `exec` — calling any of them raises NameError at runtime, so do "
    "not rely on them (in particular, do not use `locals()` to assemble "
    "return values: build any dictionaries with explicit literal keys). "
    "After any reasoning, output ONLY a single Python code block "
    "containing the function definition. No explanations or "
    "commentary after the code block.",
)

_register_block(
    "approximation_guidance_v1",
    "Your goal is to produce an informed approximation — not to compute "
    "the exact value. Enumerating all possible future trajectories is "
    "intractable. Instead, leverage your understanding of the environment "
    "dynamics, the reward structure, and the state's features to reason "
    "about how favorable the situation is. A well-reasoned estimate "
    "grounded in domain knowledge is more valuable than an attempt at "
    "exact computation.",
)

_register_block(
    "codegen_constraints_v1",
    "Important constraints:\n"
    "- Do NOT use recursive search, tree expansion, lookahead, or any "
    "form of simulation/rollout — these will time out.\n"
    "- Base the estimate on direct analysis of the state representation, "
    "using your understanding of what features predict good outcomes.",
)

_register_block(
    "efficiency_guidance_v1",
    "All else being equal, reaching the goal in fewer steps is preferable — "
    "states and actions that lead to shorter successful trajectories should "
    "generally be valued higher, reflecting the importance of efficiency.",
)

_register_block(
    "actor_reasoning_guidance",
    "The provided information includes the acting agent's brief reasoning "
    "at each step, representing their stated rationale for each action. "
    "This provides context about the agent's decision-making process, "
    "which may help you better assess the quality and consequences of the "
    "chosen actions. Your evaluation should focus on the actual quality "
    "of the actions and resulting states.",
)

_register_block(
    "role_opener_ranking_default",
    "You are an expert at comparing and ranking actions by their "
    "{signal_name} in reinforcement learning environments.",
)

_register_block(
    "role_opener_verifier_default",
    "You are an expert verifier for reinforcement learning environments. "
    "Estimate {signal_name} using an ordered discrete score scale.",
)

_register_block(
    "closing_ranking_default",
    "After any reasoning, output ONLY a comma-separated list of "
    "action numbers from best to worst. For example: 2, 1, 3, 4",
)

_register_block(
    "closing_direct_answer_tags",
    "After any reasoning, provide your numeric {signal_name} estimate "
    "as a single number inside <answer> tags, e.g. <answer>NUMBER</answer>.",
)

_register_block(
    "closing_ranking_answer_tags",
    "After any reasoning, output a comma-separated list of action numbers "
    "from best to worst inside <answer> tags, "
    "e.g. <answer>#, #, #, ...</answer>.",
)


# ---------------------------------------------------------------------------
# PromptPreset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptPreset:
    """A named composition of prompt blocks.

    Required block-name fields reference blocks from the registry.  Optional
    slots use ``None`` to mean "no text in this position" — an explicit
    choice, not a hidden default.

    Attributes:
        name: Unique identifier for this preset.
        role_opener_direct: Block name for the direct method system prompt opener.
        role_opener_codegen: Block name for the codegen method system prompt opener.
        closing_direct: Block name for the direct method closing instruction.
        codegen_closing: Block name for the codegen user prompt closing.
        approximation_guidance: Block name or ``None`` (no guidance text).
        codegen_constraints: Block name or ``None`` (no constraints).
        efficiency_guidance: Block name or ``None`` (no efficiency hint).
        role_opener_ranking: Block name for the ranking method system prompt opener.
        closing_ranking: Block name for the ranking method closing instruction.
        role_opener_verifier: Block name for the verifier method system prompt opener.
        uses_answer_tags: When ``True``, methods use ``<answer>`` tag
            extraction and tag-aware user-prompt closings.
    """

    name: str
    role_opener_direct: str
    role_opener_codegen: str
    closing_direct: str
    codegen_closing: str
    approximation_guidance: str | None = None
    codegen_constraints: str | None = None
    efficiency_guidance: str | None = None
    role_opener_ranking: str = "role_opener_ranking_default"
    closing_ranking: str = "closing_ranking_default"
    role_opener_verifier: str = "role_opener_verifier_default"
    uses_answer_tags: bool = False


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, PromptPreset] = {}


def register_preset(preset: PromptPreset) -> None:
    """Register a prompt preset by name.

    Validates that all block-name references point to registered blocks.

    Raises:
        ValueError: If a preset with the same name is already registered,
            or if any block reference is invalid.
    """
    if preset.name in _REGISTRY:
        raise ValueError(f"Preset {preset.name!r} is already registered")

    # Validate all block references
    required_refs = [
        ("role_opener_direct", preset.role_opener_direct),
        ("role_opener_codegen", preset.role_opener_codegen),
        ("closing_direct", preset.closing_direct),
        ("codegen_closing", preset.codegen_closing),
    ]
    optional_refs = [
        ("approximation_guidance", preset.approximation_guidance),
        ("codegen_constraints", preset.codegen_constraints),
        ("efficiency_guidance", preset.efficiency_guidance),
        ("role_opener_ranking", preset.role_opener_ranking),
        ("closing_ranking", preset.closing_ranking),
        ("role_opener_verifier", preset.role_opener_verifier),
    ]
    for field_name, block_name in required_refs + optional_refs:
        if block_name is not None and block_name not in _BLOCK_REGISTRY:
            available = sorted(_BLOCK_REGISTRY.keys())
            raise ValueError(
                f"Preset {preset.name!r} field {field_name!r} references "
                f"unknown block {block_name!r}. Available: {available}"
            )

    _REGISTRY[preset.name] = preset


def get_preset(name: str) -> PromptPreset:
    """Retrieve a registered prompt preset by name.

    Raises:
        KeyError: If no preset with the given name exists.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown prompt preset: {name!r}. Available: {available}"
        ) from None


def list_presets() -> list[str]:
    """Return sorted list of registered preset names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

V0 = PromptPreset(
    name="v0",
    role_opener_direct="role_opener_direct_default",
    role_opener_codegen="role_opener_codegen_default",
    closing_direct="closing_direct_default",
    codegen_closing="codegen_closing_default",
)
"""Original behavior — all required slots use the default blocks, no optional blocks."""

V1 = PromptPreset(
    name="v1",
    role_opener_direct="role_opener_direct_default",
    role_opener_codegen="role_opener_codegen_default",
    closing_direct="closing_direct_default",
    codegen_closing="codegen_closing_default",
    approximation_guidance="approximation_guidance_v1",
    codegen_constraints="codegen_constraints_v1",
    efficiency_guidance="efficiency_guidance_v1",
    role_opener_ranking="role_opener_ranking_default",
    closing_ranking="closing_ranking_default",
)
"""Approximation-oriented — adds guidance, codegen constraints, and efficiency guidance."""

V2 = PromptPreset(
    name="v2",
    role_opener_direct="role_opener_direct_default",
    role_opener_codegen="role_opener_codegen_default",
    closing_direct="closing_direct_answer_tags",
    codegen_closing="codegen_closing_default",
    approximation_guidance="approximation_guidance_v1",
    codegen_constraints="codegen_constraints_v1",
    efficiency_guidance="efficiency_guidance_v1",
    role_opener_ranking="role_opener_ranking_default",
    closing_ranking="closing_ranking_answer_tags",
    uses_answer_tags=True,
)
"""Answer-tag-based — extends V1 with <answer>...</answer> extraction tags."""

register_preset(V0)
register_preset(V1)
register_preset(V2)

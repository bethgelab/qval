"""Token-budget-aware history truncation.

Provides tiered token estimation (exact → reference → heuristic),
budget resolution from backend capabilities, and a selective truncation
algorithm that truncates only what is needed, starting from earliest
entries and observations before actions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from llenvs.core.state import ImageContent, ObservationImages
from llenvs.evaluation.history import HistoryEntry, PromptBudget, full_history

from qval.types import HistoryTurn
from llenvs.inference.protocol import ChatMessage, ModelBackend

from qval.benchmark import _middle_truncate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Estimated tokens per history entry for direct prompt formatting overhead
# ("[Turn N - State]\n...\n[Turn N - Action]\n...\n\n")
DIRECT_PROMPT_ENTRY_OVERHEAD_TOKENS: int = 12

# Estimated tokens per message pair (assistant + user) for chat template overhead
CHAT_MSG_OVERHEAD_TOKENS: int = 10

# Heuristic fallback: characters per token when no tokenizer is available
_CHARS_PER_TOKEN: float = 3.5

# Estimated tokens per image for VLM APIs (conservative high-detail default).
# OpenAI GPT-4V: ~765 tokens per 1024x1024 tile (detail=high).
# Anthropic Claude: similar cost structure.
DEFAULT_TOKENS_PER_IMAGE: int = 765

# Default reference tokenizer (used when model family is unknown)
_DEFAULT_REFERENCE_TOKENIZER: str = "Qwen/Qwen3-0.6B"

# (substring_patterns, HuggingFace_tokenizer_repo_id)
# First match wins.  Matched against lowercased model_name.
_FAMILY_TOKENIZER_MAP: list[tuple[list[str], str]] = [
    (["deepseek"],            "deepseek-ai/DeepSeek-V3"),
    (["qwen"],                "Qwen/Qwen3-0.6B"),
    (["llama", "meta-llama"], "baseten/Meta-Llama-3-tokenizer"),
    (["mistral"],             "mistralai/Mistral-7B-v0.1"),
]

# Cache: tokenizer_name -> estimator (None = load failed, missing key = not attempted)
_reference_estimator_cache: dict[str, Callable[[str], int] | None] = {}

# Safety margin applied when using the default reference tokenizer for a
# non-matching model family.  A larger-vocab tokenizer (Qwen, 151K) will
# undercount tokens relative to a smaller-vocab model (DeepSeek, 129K).
# 15 % overestimate keeps truncation conservative.
_FALLBACK_SAFETY_MARGIN: float = 1.15


def estimate_image_tokens(
    images: tuple[ImageContent, ...] | ObservationImages,
    tokens_per_image: int = DEFAULT_TOKENS_PER_IMAGE,
) -> int:
    """Estimate the token cost of images for VLM budget calculations.

    Args:
        images: Either a flat tuple of ImageContent or an ObservationImages.
        tokens_per_image: Token cost per image (default: 765, conservative
            high-detail estimate for OpenAI/Anthropic VLMs).

    Returns:
        Estimated total token cost of the images.
    """
    if isinstance(images, ObservationImages):
        return len(images.all) * tokens_per_image
    return len(images) * tokens_per_image


@dataclass(frozen=True)
class _EstimatorResolution:
    """Resolved token-estimation strategy."""

    estimate_tokens: Callable[[str], int]
    source: str
    tokenizer_name: str | None = None
    model_name: str = ""
    uses_safety_margin: bool = False
    failed_family_tokenizer: str | None = None


# ---------------------------------------------------------------------------
# Token estimation (tiered)
# ---------------------------------------------------------------------------

def _select_reference_tokenizer_name(model_name: str) -> tuple[str, bool]:
    """Choose a reference tokenizer based on the model family.

    Matches *model_name* against known family patterns and returns
    ``(tokenizer_repo_id, is_family_match)``.  When no pattern matches,
    returns the default reference tokenizer with ``is_family_match=False``.
    """
    lower = model_name.lower()
    for patterns, tokenizer_name in _FAMILY_TOKENIZER_MAP:
        for pattern in patterns:
            if pattern in lower:
                return tokenizer_name, True
    return _DEFAULT_REFERENCE_TOKENIZER, False


def _load_reference_tokenizer(name: str) -> Callable[[str], int] | None:
    """Try loading a named reference tokenizer. Returns None on failure."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        logger.debug("Loaded reference tokenizer: %s", name)
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    except Exception:
        logger.debug(
            "Could not load reference tokenizer %s, will use char heuristic",
            name,
        )
        return None


def _char_heuristic(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN + 0.5))


def _resolve_token_estimator(backend: ModelBackend) -> _EstimatorResolution:
    """Resolve the token estimator and describe the selected strategy."""
    tokenizer = getattr(backend, "tokenizer", None)
    model_name = getattr(backend, "model_name", "") or ""
    if tokenizer is not None:
        try:
            # Verify it works
            tokenizer.encode("test")
            tokenizer_name = getattr(tokenizer, "name_or_path", None)
            return _EstimatorResolution(
                estimate_tokens=lambda text: len(
                    tokenizer.encode(text, add_special_tokens=False)
                ),
                source="backend",
                tokenizer_name=tokenizer_name,
                model_name=model_name,
            )
        except Exception:
            pass

    # Tier 2: family-matched reference tokenizer (cached per name)
    ref_name, is_family_match = _select_reference_tokenizer_name(model_name)

    if ref_name not in _reference_estimator_cache:
        _reference_estimator_cache[ref_name] = _load_reference_tokenizer(ref_name)
    estimator = _reference_estimator_cache[ref_name]
    if estimator is not None:
        if is_family_match:
            return _EstimatorResolution(
                estimate_tokens=estimator,
                source="family_reference",
                tokenizer_name=ref_name,
                model_name=model_name,
            )
        # Unknown family → apply safety margin to compensate for
        # possible vocab-size mismatch with the default tokenizer.
        margin = _FALLBACK_SAFETY_MARGIN
        return _EstimatorResolution(
            estimate_tokens=lambda text, _e=estimator, _m=margin: int(
                _e(text) * _m + 0.5
            ),
            source="default_reference",
            tokenizer_name=ref_name,
            model_name=model_name,
            uses_safety_margin=True,
        )

    # If family-specific tokenizer failed, try default as fallback
    if ref_name != _DEFAULT_REFERENCE_TOKENIZER:
        if _DEFAULT_REFERENCE_TOKENIZER not in _reference_estimator_cache:
            _reference_estimator_cache[_DEFAULT_REFERENCE_TOKENIZER] = (
                _load_reference_tokenizer(_DEFAULT_REFERENCE_TOKENIZER)
            )
        fallback = _reference_estimator_cache[_DEFAULT_REFERENCE_TOKENIZER]
        if fallback is not None:
            # Wrong family → always apply safety margin.
            margin = _FALLBACK_SAFETY_MARGIN
            return _EstimatorResolution(
                estimate_tokens=lambda text, _e=fallback, _m=margin: int(
                    _e(text) * _m + 0.5
                ),
                source="default_reference",
                tokenizer_name=_DEFAULT_REFERENCE_TOKENIZER,
                model_name=model_name,
                uses_safety_margin=True,
                failed_family_tokenizer=ref_name,
            )

    # Tier 3: char heuristic
    return _EstimatorResolution(
        estimate_tokens=_char_heuristic,
        source="char_heuristic",
        model_name=model_name,
        failed_family_tokenizer=ref_name if ref_name != _DEFAULT_REFERENCE_TOKENIZER else None,
    )


def _log_estimator_resolution(resolution: _EstimatorResolution) -> None:
    """Log which tokenizer path will drive prompt-budget estimation."""
    model_label = resolution.model_name or "<unknown model>"
    if resolution.failed_family_tokenizer is not None:
        logger.warning(
            "Could not load family-matched reference tokenizer %s for %s.",
            resolution.failed_family_tokenizer,
            model_label,
        )

    if resolution.source == "backend":
        tokenizer_label = resolution.tokenizer_name or "<unnamed backend tokenizer>"
        logger.info(
            "Prompt-budget token estimation for %s will use the backend tokenizer (%s).",
            model_label,
            tokenizer_label,
        )
        return

    if resolution.source == "family_reference":
        logger.info(
            "Prompt-budget token estimation for %s will use the family-matched "
            "reference tokenizer (%s).",
            model_label,
            resolution.tokenizer_name or "<unknown>",
        )
        return

    if resolution.source == "default_reference":
        margin_suffix = ""
        if resolution.uses_safety_margin:
            margin_pct = int((_FALLBACK_SAFETY_MARGIN - 1.0) * 100)
            margin_suffix = f" with a {margin_pct}% safety margin"
        logger.info(
            "Prompt-budget token estimation for %s will use the "
            "default-reference fallback tokenizer (%s)%s.",
            model_label,
            resolution.tokenizer_name or "<unknown>",
            margin_suffix,
        )
        return

    logger.warning(
        "Prompt-budget token estimation for %s is using the character heuristic fallback.",
        model_label,
    )


def make_token_estimator(backend: ModelBackend) -> Callable[[str], int]:
    """Create a token estimator, preferring exact tokenizer when available.

    Tier 1: Backend's own tokenizer (vLLM, HuggingFace).
    Tier 2: Family-matched reference tokenizer (cached per tokenizer name).
    Tier 3: Character-based heuristic (chars / 3.5).
    """
    return _resolve_token_estimator(backend).estimate_tokens


def resolve_max_prompt_tokens(
    backend: ModelBackend,
    backend_config: Any,
) -> int | None:
    """Compute max prompt tokens as max_context_length - max_tokens.

    Checks backend capabilities first, then config's max_model_len.
    Returns None if context length is unknown.

    Args:
        backend: The ModelBackend instance.
        backend_config: BackendConfig (has ``max_model_len`` and ``sampling``).
    """
    max_context = backend.capabilities.max_context_length

    # Override with config's max_model_len if set (vLLM explicit limit)
    config_max = getattr(backend_config, "max_model_len", None)
    if config_max is not None:
        max_context = config_max

    if max_context is None:
        return None

    # Get max generation tokens from sampling config
    sampling = getattr(backend_config, "sampling", None)
    max_tokens = getattr(sampling, "max_tokens", None) if sampling else None
    if max_tokens is None:
        max_tokens = 2048  # sensible default

    return max(0, max_context - max_tokens)


# ---------------------------------------------------------------------------
# Core truncation algorithm
# ---------------------------------------------------------------------------

def budget_aware_truncation(
    history: Sequence[HistoryTurn],
    available_tokens: int,
    min_observation_chars: int | None,
    min_action_chars: int | None,
    estimate_tokens: Callable[[str], int],
    per_entry_overhead_tokens: int = DIRECT_PROMPT_ENTRY_OVERHEAD_TOKENS,
) -> list[HistoryTurn]:
    """Selectively truncate history entries to fit within a token budget.

    Only truncates what is needed, starting from earliest entries,
    observations before actions. Uses ``_middle_truncate`` for actual
    text truncation.

    Args:
        history: Sequence of HistoryTurn entries, oldest first.
        available_tokens: Token budget for the entire history section.
        min_observation_chars: Floor on observation length after truncation.
            ``None`` means observations cannot be truncated.
        min_action_chars: Floor on action length after truncation.
            ``None`` means actions cannot be truncated.
        estimate_tokens: Token counter function.
        per_entry_overhead_tokens: Formatting overhead per entry.

    Returns:
        List of HistoryTurn entries, possibly with truncated text.
    """
    if not history:
        return []

    n = len(history)
    # Work on mutable copies
    obs_texts = [h.state_text for h in history]
    act_texts = [h.action_text for h in history]

    # Compute per-entry token costs
    obs_tokens = [estimate_tokens(t) if t else 0 for t in obs_texts]
    act_tokens = [estimate_tokens(t) for t in act_texts]
    overhead = per_entry_overhead_tokens * n

    total = sum(obs_tokens) + sum(act_tokens) + overhead

    if total <= available_tokens:
        return list(history)

    excess = total - available_tokens

    # Phase 1: truncate observations from earliest
    if min_observation_chars is not None:
        for i in range(n):
            if excess <= 0:
                break
            obs = obs_texts[i]
            if not obs or len(obs) <= min_observation_chars:
                continue

            truncated_obs = _middle_truncate(obs, min_observation_chars)
            max_saveable = obs_tokens[i] - estimate_tokens(truncated_obs)
            if max_saveable <= 0:
                continue

            if max_saveable <= excess:
                # Full truncation to floor
                obs_texts[i] = truncated_obs
                obs_tokens[i] = estimate_tokens(truncated_obs)
                excess -= max_saveable
            else:
                # Partial truncation: find target_chars that saves exactly enough
                target_chars = _find_partial_target(
                    obs, obs_tokens[i], excess, min_observation_chars,
                    estimate_tokens,
                )
                obs_texts[i] = _middle_truncate(obs, target_chars)
                new_tokens = estimate_tokens(obs_texts[i])
                excess -= (obs_tokens[i] - new_tokens)
                obs_tokens[i] = new_tokens

    # Phase 2: truncate actions from earliest
    if min_action_chars is not None and excess > 0:
        for i in range(n):
            if excess <= 0:
                break
            act = act_texts[i]
            if len(act) <= min_action_chars:
                continue

            truncated_act = _middle_truncate(act, min_action_chars)
            max_saveable = act_tokens[i] - estimate_tokens(truncated_act)
            if max_saveable <= 0:
                continue

            if max_saveable <= excess:
                act_texts[i] = truncated_act
                act_tokens[i] = estimate_tokens(truncated_act)
                excess -= max_saveable
            else:
                target_chars = _find_partial_target(
                    act, act_tokens[i], excess, min_action_chars,
                    estimate_tokens,
                )
                act_texts[i] = _middle_truncate(act, target_chars)
                new_tokens = estimate_tokens(act_texts[i])
                excess -= (act_tokens[i] - new_tokens)
                act_tokens[i] = new_tokens

    return [
        HistoryTurn(
            state_text=obs_texts[i],
            action_text=act_texts[i],
            thought=history[i].thought,
            state_images=history[i].state_images,
        )
        for i in range(n)
    ]


def _find_partial_target(
    text: str,
    current_tokens: int,
    needed_savings: int,
    min_chars: int,
    estimate_tokens: Callable[[str], int],
) -> int:
    """Find target_chars for partial truncation that saves approximately
    ``needed_savings`` tokens. Uses local chars/token ratio as initial
    estimate, then floors at ``min_chars``.
    """
    # Estimate chars/token ratio for this text
    chars_per_token = len(text) / max(current_tokens, 1)
    # Target: reduce by needed_savings tokens worth of chars
    target_reduction_chars = int(needed_savings * chars_per_token)
    target_chars = max(min_chars, len(text) - target_reduction_chars)
    return target_chars


def truncate_text_to_token_savings(
    text: str,
    needed_savings: int,
    min_chars: int,
    estimate_tokens: Callable[[str], int],
) -> str:
    """Middle-truncate *text* to save approximately ``needed_savings`` tokens.

    Returns the minimally truncated text that is estimated to save at least the
    requested number of tokens, while never going below ``min_chars``. If the
    requested savings are not achievable, returns the floor-truncated text.
    """
    if needed_savings <= 0 or len(text) <= min_chars:
        return text

    current_tokens = estimate_tokens(text)
    floor_text = _middle_truncate(text, min_chars)
    floor_tokens = estimate_tokens(floor_text)
    max_saveable = current_tokens - floor_tokens

    if max_saveable <= 0:
        return text
    if max_saveable <= needed_savings:
        return floor_text

    low = min_chars
    high = len(text)
    best = min_chars
    while low <= high:
        mid = (low + high) // 2
        candidate = _middle_truncate(text, mid)
        saved = current_tokens - estimate_tokens(candidate)
        if saved >= needed_savings:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return _middle_truncate(text, best)


# ---------------------------------------------------------------------------
# HistoryEntry-based variant for TrajectoryRunner path
# ---------------------------------------------------------------------------

def _entries_to_messages(entries: list[HistoryEntry]) -> list[ChatMessage]:
    """Convert history entries to alternating assistant/user messages.

    Mirror of llenvs ``_entries_to_messages`` — kept here to avoid coupling
    with the internal helper.  Every entry produces both an assistant and
    a user message so that turn boundaries are always preserved.
    """
    messages: list[ChatMessage] = []
    for entry in entries:
        messages.append(ChatMessage(role="assistant", content=entry.action_text))
        messages.append(
            ChatMessage(
                role="user",
                content=entry.observation_text or "",
                images=entry.observation_images,
            )
        )
    return messages


def build_budget_aware_history(
    min_observation_chars: int | None,
    min_action_chars: int | None,
    estimate_tokens: Callable[[str], int],
    max_history_turns: int | None = None,
) -> Callable[[list[HistoryEntry], int], list[ChatMessage]]:
    """Create a ``build_history`` callback for ``PromptBudget``.

    The returned callable:
    1. Applies ``max_history_turns`` (fixed, not adaptive).
    2. Applies budget-aware content truncation.
    3. Converts to messages.

    Args:
        min_observation_chars: Floor for observation truncation.
        min_action_chars: Floor for action truncation.
        estimate_tokens: Token counter.
        max_history_turns: Fixed turn limit (applied before budget truncation).
    """

    def _build(entries: list[HistoryEntry], available_tokens: int) -> list[ChatMessage]:
        # Apply fixed turn limit
        kept = entries
        notice_msg: ChatMessage | None = None
        if max_history_turns is not None and len(entries) > max_history_turns:
            kept = entries[-max_history_turns:] if max_history_turns > 0 else []
            if max_history_turns > 0:
                dropped = len(entries) - max_history_turns
                notice_msg = ChatMessage(
                    role="user",
                    content=(
                        f"[Note: Only the last {max_history_turns} of {len(entries)} "
                        f"turns are shown. {dropped} earlier turns have been omitted.]"
                    ),
                )

        if not kept:
            return [notice_msg] if notice_msg else []

        # Convert to HistoryTurn for truncation
        turns = [
            HistoryTurn(
                state_text=e.observation_text,
                action_text=e.action_text,
                state_images=ObservationImages(state=e.observation_images),
            )
            for e in kept
        ]

        # Apply budget-aware truncation
        truncated_pairs = budget_aware_truncation(
            turns,
            available_tokens,
            min_observation_chars=min_observation_chars,
            min_action_chars=min_action_chars,
            estimate_tokens=estimate_tokens,
            per_entry_overhead_tokens=CHAT_MSG_OVERHEAD_TOKENS,
        )

        # Rebuild entries with truncated content
        truncated_entries = [
            HistoryEntry(
                action_text=turn.action_text,
                observation_text=turn.state_text,
                observation_images=entry.observation_images,
                step=entry.step,
            )
            for turn, entry in zip(truncated_pairs, kept)
        ]

        messages = _entries_to_messages(truncated_entries)
        if notice_msg:
            messages.insert(0, notice_msg)
        return messages

    return _build


def make_prompt_budget(
    backend: ModelBackend,
    backend_config: Any,
    *,
    min_observation_chars: int | None = None,
    min_action_chars: int | None = None,
    min_current_observation_chars: int | None = None,
    max_history_turns: int | None = None,
) -> PromptBudget | None:
    """Create a PromptBudget from a backend if budget can be determined.

    Returns None if context length is unknown (API backend without
    max_model_len in config).
    """
    max_prompt_tokens = resolve_max_prompt_tokens(backend, backend_config)
    if max_prompt_tokens is None:
        return None

    resolution = _resolve_token_estimator(backend)
    _log_estimator_resolution(resolution)
    estimate_tokens = resolution.estimate_tokens
    build_history = build_budget_aware_history(
        min_observation_chars=min_observation_chars,
        min_action_chars=min_action_chars,
        estimate_tokens=estimate_tokens,
        max_history_turns=max_history_turns,
    )

    return PromptBudget(
        max_prompt_tokens=max_prompt_tokens,
        estimate_tokens=estimate_tokens,
        build_history=build_history,
        min_current_observation_chars=min_current_observation_chars,
    )

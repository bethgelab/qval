"""Shared prompt content and builders for dense signal methods."""

from __future__ import annotations

from dataclasses import dataclass

from llenvs.core.state import ImageContent
from llenvs.inference import PromptFragment, compose_system_prompt

from qval.methods.serialization import (
    action_text_for_display,
    extract_images,
    serialize_observation,
    strip_thought_tags,
)
from qval.prompt_presets import PromptPreset, get_block, resolve_block
from qval.types import (
    EvaluationPoint,
    HistoryTurn,
    MethodContext,
    PolicyAssumption,
    RankingPoint,
    SignalType,
)


ContentBlock = str | ImageContent


@dataclass(frozen=True)
class PromptWithImages:
    """User prompt as interleaved text and image blocks.

    VLM APIs accept content arrays where text and images alternate.  Each
    image sits right after the text section it illustrates so the model
    can associate them naturally.  Text-only environments produce a
    sequence of text blocks with no images.

    The ``text`` property returns a plain-text rendition (for text-only
    backends or logging), while ``blocks`` provides the full interleaved
    sequence for VLM message construction.
    """

    blocks: tuple[ContentBlock, ...] = ()

    @property
    def text(self) -> str:
        """Plain-text rendition (concatenation of text blocks only)."""
        return "\n\n".join(b for b in self.blocks if isinstance(b, str))

    @property
    def images(self) -> tuple[ImageContent, ...]:
        """All images in document order."""
        return tuple(b for b in self.blocks if isinstance(b, ImageContent))

    @property
    def has_images(self) -> bool:
        return any(isinstance(b, ImageContent) for b in self.blocks)


@dataclass(frozen=True)
class GVLContextTransitionText:
    """Rendered surrounding trajectory context for GVL prompts."""

    state_text: str
    action_text: str
    step_index: int
    state_images: tuple[ImageContent, ...] = ()


def _task_text_for_injection(point: "EvaluationPoint | RankingPoint") -> str | None:
    """Return ``obs.task.text`` (stripped) for injection, or None if unset/empty.

    Callers gate on the ``inject_task_text_in_prompts`` env-context flag
    before calling this; the helper itself just reads the point's
    observation. Used by WebShop to surface ``<goal>…</goal>`` at the top
    of evaluator/ranking user prompts.
    """
    obs = getattr(getattr(point, "state", None), "observation", None)
    if obs is None or obs.task is None:
        return None
    text = (obs.task.text or "").strip()
    return text or None


SIGNAL_TYPE_NAMES: dict[SignalType, str] = {
    SignalType.STATE_VALUE: "state-value",
    SignalType.Q_VALUE: "Q-value",
    SignalType.ADVANTAGE: "advantage",
    SignalType.POTENTIAL: "potential",
    SignalType.SHAPED_REWARD: "shaped-reward",
}

_VERIFIER_SCORE_LABELS = tuple(chr(ord("A") + i) for i in range(20))
_VERIFIER_SCORE_SCALE_DESCRIPTION = (
    "Use an ordered 20-point score scale with the single-letter bins A through T, "
    "where A is best and T is worst. Choose exactly one score bin per datapoint.\n"
    "A = clearly and completely favorable with strong evidence of success (best)\n"
    "B-D = strongly favorable with only minor remaining concerns\n"
    "E-G = above average, mostly favorable with some issues\n"
    "H-J = uncertain, leans favorable\n"
    "K-M = uncertain, leans unfavorable\n"
    "N-P = below average, significant issues remain\n"
    "Q-S = poor, with only limited signs of progress\n"
    "T = clearly and completely unfavorable or failed (worst)"
)

_REASONING_PREAMBLE = "After any reasoning, "


def _adapt_closing(text: str, enable_thinking: bool) -> str:
    """Strip reasoning preamble when thinking is disabled.

    When the backend has thinking/reasoning tokens enabled, the model reasons
    inside ``<think>`` blocks and the visible output should start directly with
    the answer.  The "After any reasoning, ..." preamble is appropriate because
    the model *will* reason — just not in its visible output.

    When thinking is **off**, keeping the preamble encourages the model to
    reason visibly, wasting output tokens.  This helper strips it.

    Handles both leading preambles ("After any reasoning, output ...") and
    mid-sentence occurrences ("... modules. After any reasoning, output ...").
    """
    if enable_thinking:
        return text
    # Leading preamble
    if text.startswith(_REASONING_PREAMBLE):
        rest = text[len(_REASONING_PREAMBLE):]
        return rest[0].upper() + rest[1:]
    # Mid-sentence preamble (after ". ")
    mid = ". " + _REASONING_PREAMBLE
    idx = text.find(mid)
    if idx != -1:
        before = text[: idx + 2]  # includes ". "
        after = text[idx + len(mid) :]
        return before + after[0].upper() + after[1:]
    return text


# ---------------------------------------------------------------------------
# Policy-dependent language helpers
# ---------------------------------------------------------------------------

def _policy_phrase(policy: PolicyAssumption) -> str:
    """Return a short phrase describing the policy assumption."""
    if policy == PolicyAssumption.OPTIMAL:
        return "assuming optimal play thereafter"
    return "assuming you are the agent making all future decisions"


def _policy_noun(policy: PolicyAssumption) -> str:
    """Return noun phrase for the policy."""
    if policy == PolicyAssumption.OPTIMAL:
        return "the optimal policy"
    return "your own policy"


def _return_phrase(discount_factor: float, *, disclosed: bool = True) -> str:
    """Describe the return type given the discount factor.

    Args:
        discount_factor: Discount factor γ (0.0–1.0).
        disclosed: Whether the exact discount value should appear in the text.
            When ``False`` and γ<1.0, mentions discounting without revealing
            the exact value.
    """
    if discount_factor == 1.0:
        return "undiscounted cumulative reward"
    if disclosed:
        return f"discounted cumulative reward (discount factor γ={discount_factor})"
    return "discounted cumulative reward"


# ---------------------------------------------------------------------------
# Turn label helpers
# ---------------------------------------------------------------------------


def _turn_label_prefix(
    turn_num: int, max_steps: int | None,
) -> str:
    """Return a turn label prefix like ``[Turn 3/50`` or ``[Turn 3``.

    The caller appends the suffix (e.g. `` - State]``, `` - Action]``, or ``]``).
    """
    if max_steps is not None:
        return f"[Turn {turn_num}/{max_steps}"
    return f"[Turn {turn_num}"


def _history_turn_label(
    index: int,
    history_len: int,
    step_index: int,
    max_steps: int | None,
) -> str:
    """Compute the turn label prefix for a history turn.

    Uses absolute step indices when possible (``step_index`` is large enough
    to explain the history).  Falls back to relative 1-based indexing for
    legacy data where ``step_index=0`` with non-empty history.
    """
    abs_step = step_index - history_len + index
    if abs_step >= 0:
        return _turn_label_prefix(abs_step + 1, max_steps)
    return _turn_label_prefix(index + 1, None)


# ---------------------------------------------------------------------------
# build_signal_type_fragment — the core builder
# ---------------------------------------------------------------------------

def build_signal_type_fragment(
    signal_type: SignalType,
    policy_assumption: PolicyAssumption = PolicyAssumption.OPTIMAL,
    discount_factor: float = 1.0,
    *,
    disclose_discount_factor: bool = True,
) -> PromptFragment:
    """Build a signal-type description fragment with policy and discount context.

    Args:
        signal_type: Which signal type to describe.
        policy_assumption: Whose policy the value functions assume.
        discount_factor: Discount factor γ (0.0–1.0).
        disclose_discount_factor: Whether to reveal the exact γ value in text.

    Returns:
        A PromptFragment with the signal description.
    """
    builders = {
        SignalType.STATE_VALUE: _build_state_value_fragment,
        SignalType.Q_VALUE: _build_q_value_fragment,
        SignalType.ADVANTAGE: _build_advantage_fragment,
        SignalType.POTENTIAL: _build_potential_fragment,
        SignalType.SHAPED_REWARD: _build_shaped_reward_fragment,
    }
    return builders[signal_type](
        policy_assumption, discount_factor, disclosed=disclose_discount_factor,
    )


def _build_state_value_fragment(
    policy: PolicyAssumption,
    discount_factor: float,
    *,
    disclosed: bool = True,
) -> PromptFragment:
    ret = _return_phrase(discount_factor, disclosed=disclosed)
    pol = _policy_phrase(policy)
    content = (
        f"The state-value V(s) represents the expected {ret} starting "
        f"from state s, {pol}. "
        f"It depends only on the current state, not on any specific action."
    )
    return PromptFragment(name="vb_state_value", content=content, category="domain")


def _build_q_value_fragment(
    policy: PolicyAssumption,
    discount_factor: float,
    *,
    disclosed: bool = True,
) -> PromptFragment:
    ret = _return_phrase(discount_factor, disclosed=disclosed)
    pol = _policy_phrase(policy)
    content = (
        f"The Q-value Q(s,a) represents the expected {ret} when "
        f"taking action a in state s and then following "
        f"{_policy_noun(policy)} thereafter. "
        f"In other words, Q(s,a) is the expected return given that action a "
        f"is taken in state s, {pol}. "
        f"It depends on both the state and the specific action taken."
    )
    return PromptFragment(name="vb_q_value", content=content, category="domain")


def _build_advantage_fragment(
    policy: PolicyAssumption,
    discount_factor: float,
    *,
    disclosed: bool = True,
) -> PromptFragment:
    pol_noun = _policy_noun(policy)
    content = (
        f"The advantage A(s,a) = Q(s,a) - V(s) measures how much better or worse "
        f"action a is compared to the average action in state s under {pol_noun}. "
        f"Positive values mean the action is better than average; negative values "
        f"mean worse. A value of 0.0 means no advantage over the average."
    )
    return PromptFragment(name="vb_advantage", content=content, category="domain")


def _build_potential_fragment(
    policy: PolicyAssumption,
    discount_factor: float,
    *,
    disclosed: bool = True,
) -> PromptFragment:
    # Potential is always optimal-policy-based
    ret = _return_phrase(discount_factor, disclosed=disclosed)
    content = (
        "The potential Φ(s) represents the long-term desirability of state s "
        "for reward shaping. The theoretically optimal potential equals V*(s), the "
        f"optimal state-value function — the expected {ret} under optimal play. "
        "It depends only on the current state, not on any specific action."
    )
    return PromptFragment(name="vb_potential", content=content, category="domain")


def _build_shaped_reward_fragment(
    policy: PolicyAssumption,
    discount_factor: float,
    *,
    disclosed: bool = True,
) -> PromptFragment:
    # Shaped reward is per-transition, no policy language needed
    content = (
        "The shaped reward F(s, a, s') provides a dense per-transition reward "
        "signal supplementing the sparse environment reward. A positive value "
        "indicates the transition moved toward the goal; negative indicates "
        "regression. It may depend on the current state, the action taken, and "
        "the resulting next state."
    )
    return PromptFragment(name="vb_shaped_reward", content=content, category="domain")


# ---------------------------------------------------------------------------
# Default dicts (OPTIMAL, γ=1.0) — convenience for backward compatibility
# ---------------------------------------------------------------------------

SIGNAL_TYPE_FRAGMENTS: dict[SignalType, PromptFragment] = {
    st: build_signal_type_fragment(st) for st in SignalType
}

SIGNAL_TYPE_DESCRIPTIONS: dict[SignalType, str] = {
    st: frag.content for st, frag in SIGNAL_TYPE_FRAGMENTS.items()
}

SIGNAL_TYPE_CLOSING_INSTRUCTIONS: dict[SignalType, str] = {
    SignalType.STATE_VALUE: (
        "Now, provide your state-value estimate as a single number."
        " Do not explain — output ONLY the number."
    ),
    SignalType.Q_VALUE: (
        "Now, provide your Q-value estimate as a single number."
        " Do not explain — output ONLY the number."
    ),
    SignalType.ADVANTAGE: (
        "Now, provide your advantage estimate as a single number."
        " Do not explain — output ONLY the number."
    ),
    SignalType.POTENTIAL: (
        "Now, provide your potential value estimate for the current state as a single number."
        " Do not explain — output ONLY the number."
    ),
    SignalType.SHAPED_REWARD: (
        "Now, provide your shaped reward estimate for this transition as a single number."
        " Do not explain — output ONLY the number."
    ),
}

SIGNAL_TYPE_CLOSING_INSTRUCTIONS_ANSWER_TAGS: dict[SignalType, str] = {
    SignalType.STATE_VALUE: (
        "Now, provide your state-value estimate as a single number"
        " inside <answer> tags, e.g. <answer>NUMBER</answer>."
    ),
    SignalType.Q_VALUE: (
        "Now, provide your Q-value estimate as a single number"
        " inside <answer> tags, e.g. <answer>NUMBER</answer>."
    ),
    SignalType.ADVANTAGE: (
        "Now, provide your advantage estimate as a single number"
        " inside <answer> tags, e.g. <answer>NUMBER</answer>."
    ),
    SignalType.POTENTIAL: (
        "Now, provide your potential value estimate for the current state"
        " as a single number inside <answer> tags, e.g. <answer>NUMBER</answer>."
    ),
    SignalType.SHAPED_REWARD: (
        "Now, provide your shaped reward estimate for this transition"
        " as a single number inside <answer> tags, e.g. <answer>NUMBER</answer>."
    ),
}

SIGNAL_TYPE_HEADERS: dict[SignalType, str] = {
    SignalType.STATE_VALUE: (
        "Estimate the state-value of the **Current State** for the following:"
    ),
    SignalType.Q_VALUE: (
        "Estimate the Q-value for the **Current Action** in the "
        "**Current State** for the following:"
    ),
    SignalType.ADVANTAGE: (
        "Estimate the advantage of the **Current Action** in the "
        "**Current State** for the following:"
    ),
    SignalType.POTENTIAL: (
        "Estimate the potential of the **Current State** for the following:"
    ),
    SignalType.SHAPED_REWARD: (
        "Estimate the shaped reward for the transition from "
        "**Current State** via **Current Action** to **Next State** for the following:"
    ),
}

SIGNAL_TYPES_WITH_ACTION: frozenset[SignalType] = frozenset({
    SignalType.Q_VALUE,
    SignalType.ADVANTAGE,
    SignalType.SHAPED_REWARD,
})
"""Signal types whose evaluation depends on action and next state."""


def build_eval_instructions(
    signal_type: SignalType,
    policy_assumption: PolicyAssumption = PolicyAssumption.OPTIMAL,
    discount_factor: float = 1.0,
    *,
    disclose_discount_factor: bool = True,
    include_next_state: bool = True,
    max_history_turns: int | None = None,
) -> str:
    """Build signal-type-specific evaluation instructions.

    Unlike the static descriptions, these instructions tell the LLM what
    data it will receive and what to estimate, adapting to policy and
    discount settings.
    """
    builders = {
        SignalType.STATE_VALUE: _build_sv_eval,
        SignalType.Q_VALUE: _build_qv_eval,
        SignalType.ADVANTAGE: _build_adv_eval,
        SignalType.POTENTIAL: _build_pot_eval,
        SignalType.SHAPED_REWARD: _build_sr_eval,
    }
    return builders[signal_type](
        policy_assumption, discount_factor, disclosed=disclose_discount_factor,
        include_next_state=include_next_state,
        max_history_turns=max_history_turns,
    )


_HISTORY_SUFFIX = (
    "The history provides context about how the agent reached the current state."
)


def _history_preamble(max_history_turns: int | None, *, conjunction: str = ", ") -> str:
    """Return the history data preamble, or empty string when history is disabled."""
    if max_history_turns == 0:
        return ""
    return f"the episode's state-action history (if any){conjunction}"


def _build_sv_eval(
    policy: PolicyAssumption, discount_factor: float, *, disclosed: bool = True,
    include_next_state: bool = True, max_history_turns: int | None = None,
) -> str:
    ret = _return_phrase(discount_factor, disclosed=disclosed)
    pol = _policy_phrase(policy)
    hist = _history_preamble(max_history_turns, conjunction=" and ")
    history_suffix = f" {_HISTORY_SUFFIX}" if max_history_turns != 0 else ""
    return (
        f"You will receive {hist}"
        f"the current state. Estimate V(s) for the **Current State** — the "
        f"expected {ret} from that state, {pol}.{history_suffix}"
    )


def _build_qv_eval(
    policy: PolicyAssumption, discount_factor: float, *, disclosed: bool = True,
    include_next_state: bool = True, max_history_turns: int | None = None,
) -> str:
    ret = _return_phrase(discount_factor, disclosed=disclosed)
    pol = _policy_phrase(policy)
    hist = _history_preamble(max_history_turns)
    if include_next_state:
        data_desc = (
            "the current state, the action taken, and the resulting next state"
        )
    else:
        data_desc = "the current state and the action taken"
    history_suffix = f" {_HISTORY_SUFFIX}" if max_history_turns != 0 else ""
    return (
        f"You will receive {hist}"
        f"{data_desc}. "
        f"Estimate Q(s,a) for the **Current Action** in the **Current State** "
        f"— the expected {ret} when taking this action in this state, {pol}.{history_suffix}"
    )


def _build_adv_eval(
    policy: PolicyAssumption, discount_factor: float, *, disclosed: bool = True,
    include_next_state: bool = True, max_history_turns: int | None = None,
) -> str:
    pol = _policy_phrase(policy)
    hist = _history_preamble(max_history_turns)
    if include_next_state:
        data_desc = (
            "the current state, the action taken, and the resulting next state"
        )
    else:
        data_desc = "the current state and the action taken"
    history_suffix = f" {_HISTORY_SUFFIX}" if max_history_turns != 0 else ""
    return (
        f"You will receive {hist}"
        f"{data_desc}. "
        f"Estimate A(s,a) for the **Current Action** in the **Current State** "
        f"— how much better or worse this action is compared to the average "
        f"action, {pol}.{history_suffix}"
    )


def _build_pot_eval(
    policy: PolicyAssumption, discount_factor: float, *, disclosed: bool = True,
    include_next_state: bool = True, max_history_turns: int | None = None,
) -> str:
    hist = _history_preamble(max_history_turns, conjunction=" and ")
    history_suffix = f" {_HISTORY_SUFFIX}" if max_history_turns != 0 else ""
    return (
        f"You will receive {hist}"
        "the current state. Estimate Φ(s) for the **Current State** — the "
        "potential value representing how desirable this state is for eventual "
        f"success.{history_suffix}"
    )


def _build_sr_eval(
    policy: PolicyAssumption, discount_factor: float, *, disclosed: bool = True,
    include_next_state: bool = True, max_history_turns: int | None = None,
) -> str:
    hist = _history_preamble(max_history_turns)
    if include_next_state:
        data_desc = (
            "the current state, the action taken, and the resulting next state"
        )
    else:
        data_desc = "the current state and the action taken"
    history_suffix = f" {_HISTORY_SUFFIX}" if max_history_turns != 0 else ""
    return (
        f"You will receive {hist}"
        f"{data_desc}. "
        "Estimate F(s, a, s') for this transition — a dense reward signal "
        "indicating whether this transition moved toward or away from the "
        f"goal. Positive means progress, negative means regression.{history_suffix}"
    )


def _format_environment_notes(notes: list[str]) -> str:
    """Format environment notes as a prompt section."""
    if len(notes) == 1:
        return f"## Episode Configuration\n{notes[0]}"
    lines = "\n".join(f"- {note}" for note in notes)
    return f"## Episode Configuration\n{lines}"


def _format_example_trajectories(
    example_trajectories: list[object],
) -> str:
    """Format example trajectories for inclusion in prompts.

    Works with ``ObservableTrajectory`` instances, formatting each
    trajectory's turns with observation, action, and reward.
    """
    from qval.trajectory_store import ObservableTrajectory

    parts = ["## Example Trajectories"]
    for i, traj in enumerate(example_trajectories):
        if isinstance(traj, ObservableTrajectory):
            outcome = "Success" if traj.success else "Failure"
            header = f"### Example {i + 1}\n**Outcome: {outcome}** (reward: {traj.total_reward})"
            turn_parts = [header]
            for j, t in enumerate(traj.transitions, 1):
                action_text = t.resolved_action or t.extracted_action or t.raw_action
                turn_parts.append(
                    f"[Turn {j}]\n"
                    f"Observation: {t.observation}\n"
                    f"Action: {action_text}\n"
                    f"Reward: {t.reward}"
                )
            parts.append("\n\n".join(turn_parts))
        else:
            parts.append(f"### Example {i + 1}\n{str(traj)}")
    return "\n\n".join(parts)


def _build_system_parts(
    context: MethodContext,
    preset: PromptPreset,
    role_slot: str,
) -> list[str | PromptFragment]:
    """Build the common system prompt parts for prompt-driven methods.

    Args:
        context: Method context with signal type, policy, discount, etc.
        preset: Prompt preset with block-name references.
        role_slot: One of ``"direct"``, ``"codegen"``, ``"ranking"``, or
            ``"verifier"`` — selects which role opener block.

    Returns:
        List of prompt parts ready for ``compose_system_prompt()``.
    """
    signal_name = SIGNAL_TYPE_NAMES[context.signal_type]

    # Role opener from preset
    if role_slot == "direct":
        block_name = preset.role_opener_direct
    elif role_slot == "codegen":
        block_name = preset.role_opener_codegen
    elif role_slot == "ranking":
        block_name = preset.role_opener_ranking
    elif role_slot == "verifier":
        block_name = preset.role_opener_verifier
    else:
        raise ValueError(f"Unknown role slot: {role_slot!r}")
    parts: list[str | PromptFragment] = [
        resolve_block(block_name, signal_name=signal_name),
    ]

    # Policy/discount-aware signal type fragment
    fragment = build_signal_type_fragment(
        context.signal_type,
        context.policy_assumption,
        context.discount_factor,
        disclose_discount_factor=context.disclose_discount_factor,
    )
    parts.append(fragment)

    # Approximation guidance from preset (optional)
    guidance = resolve_block(preset.approximation_guidance)
    if guidance is not None:
        parts.append(guidance)

    # Efficiency guidance (runtime-determined, not preset-based)
    if context.efficiency_guidance:
        parts.append(context.efficiency_guidance)

    return parts


def build_direct_system_prompt(
    context: MethodContext,
    preset: PromptPreset,
    *,
    num_points: int = 1,
    joint_trajectory_prompt: bool = False,
    batch_mode: str = "packed",
) -> str:
    """Assemble the system prompt for LLMDirectMethod."""
    if batch_mode not in {"packed", "sequential"}:
        raise ValueError(f"Unknown direct batch_mode: {batch_mode!r}")
    signal_name = SIGNAL_TYPE_NAMES[context.signal_type]
    parts = _build_system_parts(context, preset, "direct")

    if context.include_current_thoughts or context.include_history_thoughts:
        parts.append(get_block("actor_reasoning_guidance").content)

    parts.append(build_eval_instructions(
        context.signal_type, context.policy_assumption, context.discount_factor,
        disclose_discount_factor=context.disclose_discount_factor,
        include_next_state=context.include_next_state,
        max_history_turns=context.max_history_turns,
    ))

    if context.task_description:
        parts.append(f"## Task Description\n{context.task_description.rstrip()}")

    if context.environment_notes:
        parts.append(_format_environment_notes(context.environment_notes))

    if context.reward_description:
        parts.append(f"## Reward Functions\n{context.reward_description.rstrip()}")

    if context.example_trajectories:
        parts.append(_format_example_trajectories(context.example_trajectories))

    if num_points > 1 and batch_mode == "sequential":
        if joint_trajectory_prompt:
            parts.append(
                "You may receive multiple datapoints from the same trajectory "
                "across successive user turns in this conversation. They may "
                "be shown out of chronological order. Estimate them jointly, "
                "keep your earlier estimates fixed and mutually consistent, "
                "and return exactly one estimate for the current datapoint "
                "only on each turn."
            )
        else:
            parts.append(
                "You may receive multiple datapoints across successive user "
                "turns in this conversation. For each turn, evaluate only the "
                "datapoint shown in that turn, keep your earlier estimates "
                "fixed and mutually consistent, and return exactly one "
                "estimate for the current datapoint only."
            )
        closing = resolve_block(preset.closing_direct, signal_name=signal_name)
        parts.append(_adapt_closing(closing, context.enable_thinking))
    elif num_points > 1:
        if joint_trajectory_prompt:
            parts.append(
                "You may receive multiple datapoints from the same trajectory "
                "in one request. They may be shown out of chronological order. "
                "Estimate them jointly, keeping the values mutually consistent, "
                "and return one estimate per datapoint in the same order they "
                "appear."
            )
        else:
            parts.append(
                "You may receive multiple datapoints in one request. Evaluate each "
                "datapoint independently and return one estimate per datapoint in "
                "the same order they appear."
            )
        parts.append(
            _adapt_closing(
                _direct_batch_system_closing(
                    signal_name,
                    use_answer_tags=preset.uses_answer_tags,
                ),
                context.enable_thinking,
            )
        )
    else:
        # Closing instruction from preset
        closing = resolve_block(preset.closing_direct, signal_name=signal_name)
        parts.append(_adapt_closing(closing, context.enable_thinking))

    return compose_system_prompt(*parts)


def build_gvl_system_prompt(
    context: MethodContext,
    preset: PromptPreset,
) -> str:
    """Assemble the system prompt for the dedicated GVL method."""
    signal_name = SIGNAL_TYPE_NAMES[context.signal_type]
    parts = _build_system_parts(context, preset, "direct")
    parts.append(build_eval_instructions(
        context.signal_type,
        context.policy_assumption,
        context.discount_factor,
        disclose_discount_factor=context.disclose_discount_factor,
        include_next_state=context.include_next_state,
        max_history_turns=0,
    ))

    if context.task_description:
        parts.append(f"## Task Description\n{context.task_description.rstrip()}")

    if context.environment_notes:
        parts.append(_format_environment_notes(context.environment_notes))

    if context.reward_description:
        parts.append(f"## Reward Functions\n{context.reward_description.rstrip()}")

    if context.example_trajectories:
        parts.append(_format_example_trajectories(context.example_trajectories))

    parts.append(
        "You will receive a shuffled set of surrounding transitions from the "
        "same trajectory, followed by one final target datapoint. The "
        "surrounding transitions are intentionally out of chronological order "
        "and are provided only as contextual evidence. Estimate only the final "
        "target datapoint."
    )
    if context.include_next_state:
        parts.append(
            "Only the target datapoint includes its resulting next state. "
            "Surrounding context transitions include only state and action."
        )

    closing = resolve_block(preset.closing_direct, signal_name=signal_name)
    parts.append(_adapt_closing(closing, context.enable_thinking))
    return compose_system_prompt(*parts)


def signal_type_header(
    signal_type: SignalType,
    include_next_state: bool = True,
) -> str:
    """Return the user-prompt header for a signal type.

    For SHAPED_REWARD with ``include_next_state=False``, omits the
    "to **Next State**" portion.  All other cases return the standard header.
    """
    if (
        signal_type == SignalType.SHAPED_REWARD
        and not include_next_state
    ):
        return (
            "Estimate the shaped reward for the **Current Action** in the "
            "**Current State** for the following:"
        )
    return SIGNAL_TYPE_HEADERS[signal_type]


def _direct_batch_system_closing(
    signal_name: str,
    *,
    use_answer_tags: bool = False,
) -> str:
    if use_answer_tags:
        return (
            "After any reasoning, provide your numeric "
            f"{signal_name} estimates as a comma-separated list inside "
            "<answer> tags, in datapoint order."
        )
    return (
        "After any reasoning, respond with ONLY a comma-separated list of "
        f"numeric {signal_name} estimates, in datapoint order. Do not include "
        "any explanation after the list."
    )


def _direct_batch_user_header(
    signal_type: SignalType,
    include_next_state: bool = True,
    *,
    joint_trajectory_prompt: bool = False,
) -> str:
    header = signal_type_header(signal_type, include_next_state)
    if header.endswith(" for the following:"):
        header = header.removesuffix(" for the following:")
    if joint_trajectory_prompt:
        return (
            f"{header} for each datapoint below. All datapoints come from the "
            "same trajectory and may be shown out of chronological order. "
            "Estimate them jointly and return one estimate per datapoint in "
            "the same order."
        )
    return (
        f"{header} for each datapoint below. Return one estimate per "
        "datapoint in the same order."
    )


def _direct_batch_user_closing(
    signal_type: SignalType,
    *,
    use_answer_tags: bool = False,
) -> str:
    signal_name = SIGNAL_TYPE_NAMES[signal_type]
    if use_answer_tags:
        return (
            "Now, provide your "
            f"{signal_name} estimates as a comma-separated list inside "
            "<answer> tags, in datapoint order, e.g. "
            "<answer>VALUE_1, VALUE_2, VALUE_3</answer>."
        )
    return (
        "Now, provide your "
        f"{signal_name} estimates as a comma-separated list in datapoint "
        "order. Do not explain — output ONLY the comma-separated list."
    )


def _build_direct_prompt_sections(
    point: EvaluationPoint,
    signal_type: SignalType,
    history: list[HistoryTurn] | None = None,
    history_truncated_from: int | None = None,
    include_next_state: bool = True,
    *,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    state_text_override: str | None = None,
    next_state_text_override: str | None = None,
    max_steps: int | None = None,
    show_turn_labels: bool = True,
    task_text_to_inject: str | None = None,
) -> list[str]:
    state_text = state_text_override or serialize_observation(point.state)
    parts: list[str] = []

    if task_text_to_inject:
        parts.append(task_text_to_inject)

    if history:
        history_lines = ["### State-Action History"]
        if history_truncated_from is not None and len(history) < history_truncated_from:
            history_lines.append(
                f"[Note: Only the last {len(history)} of "
                f"{history_truncated_from} turns are shown.]"
            )
        for i, turn in enumerate(history):
            label = _history_turn_label(i, len(history), point.step_index, max_steps)
            turn_parts = [f"{label} - State]\n{turn.state_text}"]
            if include_history_thoughts and turn.thought is not None:
                turn_parts.append(f"{label} - Reasoning]\n{turn.thought}")
            turn_parts.append(f"{label} - Action]\n{turn.action_text}")
            history_lines.append("\n".join(turn_parts))
        parts.append("\n\n".join(history_lines))

    if show_turn_labels:
        cur_label = _turn_label_prefix(point.step_index + 1, max_steps)
        parts.append(f"### Current State\n{cur_label}]\n{state_text}")
    else:
        parts.append(f"### Current State\n{state_text}")

    if include_current_thoughts and point.current_thought is not None:
        parts.append(f"## Actor's Reasoning\n{point.current_thought}")

    if signal_type in SIGNAL_TYPES_WITH_ACTION:
        action_text = action_text_for_display(
            point.action, point.resolved_action, point.extracted_action,
        )
        if include_current_thoughts and point.current_thought is not None:
            action_text = strip_thought_tags(action_text)
        parts.append(f"### Current Action\n{action_text}")
        if include_next_state:
            next_state_text = (
                next_state_text_override
                or serialize_observation(point.next_state)
            )
            parts.append(f"### Next State\n{next_state_text}")

    return parts


def _build_direct_prompt_blocks(
    point: EvaluationPoint,
    signal_type: SignalType,
    history: list[HistoryTurn] | None = None,
    history_truncated_from: int | None = None,
    include_next_state: bool = True,
    *,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    state_text_override: str | None = None,
    next_state_text_override: str | None = None,
    max_steps: int | None = None,
    show_turn_labels: bool = True,
    task_text_to_inject: str | None = None,
    datapoint_header: str | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> list[ContentBlock]:
    """Interleaved text+image blocks for a single evaluation point.

    When ``include_state_text_when_images=False`` and a state has at least
    one image, that state's serialized text is dropped — only the
    section label (e.g. ``[Turn 1/30 - State]``, ``### Current State``)
    and the image(s) remain.  States with no images always keep their
    text regardless of this flag.

    ``datapoint_header`` (e.g.  ``"## Datapoint 3"``) is prepended to the
    first text block so the caller doesn't have to splice a header in.
    """
    state_text = state_text_override or serialize_observation(point.state)
    blocks: list[ContentBlock] = []

    if task_text_to_inject:
        blocks.append(task_text_to_inject)

    def _state_text_block(label: str, text: str, images: tuple[ImageContent, ...]) -> str:
        if images and not include_state_text_when_images:
            return label
        sep = "\n" if label.endswith("]") or label.endswith("State") else "\n"
        return f"{label}{sep}{text}"

    if history:
        history_header = "### State-Action History"
        if history_truncated_from is not None and len(history) < history_truncated_from:
            history_header += (
                f"\n[Note: Only the last {len(history)} of "
                f"{history_truncated_from} turns are shown.]"
            )
        blocks.append(history_header)
        for i, turn in enumerate(history):
            label = _history_turn_label(i, len(history), point.step_index, max_steps)
            state_label = f"{label} - State]"
            turn_state_imgs = turn.state_images.all if include_images else ()
            if turn_state_imgs and not include_state_text_when_images:
                turn_parts = [state_label]
            else:
                turn_parts = [f"{state_label}\n{turn.state_text}"]
            if include_history_thoughts and turn.thought is not None:
                turn_parts.append(f"{label} - Reasoning]\n{turn.thought}")
            blocks.append("\n".join(turn_parts))
            for img in turn_state_imgs:
                blocks.append(img)
            blocks.append(f"{label} - Action]\n{turn.action_text}")

    cur_images = extract_images(point.state).all if include_images else ()
    if show_turn_labels:
        cur_label = _turn_label_prefix(point.step_index + 1, max_steps)
        cur_header = f"### Current State\n{cur_label}]"
    else:
        cur_header = "### Current State"
    if cur_images and not include_state_text_when_images:
        blocks.append(cur_header)
    else:
        blocks.append(f"{cur_header}\n{state_text}")
    for img in cur_images:
        blocks.append(img)

    if include_current_thoughts and point.current_thought is not None:
        blocks.append(f"## Actor's Reasoning\n{point.current_thought}")

    if signal_type in SIGNAL_TYPES_WITH_ACTION:
        action_text = action_text_for_display(
            point.action, point.resolved_action, point.extracted_action,
        )
        if include_current_thoughts and point.current_thought is not None:
            action_text = strip_thought_tags(action_text)
        blocks.append(f"### Current Action\n{action_text}")
        if include_next_state:
            next_state_text = (
                next_state_text_override
                or serialize_observation(point.next_state)
            )
            next_images = extract_images(point.next_state).all if include_images else ()
            if next_images and not include_state_text_when_images:
                blocks.append("### Next State")
            else:
                blocks.append(f"### Next State\n{next_state_text}")
            for img in next_images:
                blocks.append(img)

    if datapoint_header and blocks:
        first = blocks[0]
        if isinstance(first, str):
            blocks[0] = f"{datapoint_header}\n\n{first}"
        else:
            blocks.insert(0, datapoint_header)
    elif datapoint_header:
        blocks.append(datapoint_header)

    return blocks


def build_direct_user_prompt(
    point: EvaluationPoint,
    signal_type: SignalType,
    history: list[HistoryTurn] | None = None,
    history_truncated_from: int | None = None,
    include_next_state: bool = True,
    *,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    state_text_override: str | None = None,
    next_state_text_override: str | None = None,
    use_answer_tags: bool = False,
    max_steps: int | None = None,
    show_turn_labels: bool = True,
    task_text_to_inject: str | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> PromptWithImages:
    """Assemble the user prompt for LLMDirectMethod.

    Returns a :class:`PromptWithImages` with interleaved text and image
    blocks.  Each state's images sit immediately after that state's text
    block so VLMs can associate them.  Text-only callers can use
    ``prompt.text`` for a plain-text rendition.

    Args:
        include_current_thoughts: When ``True`` and ``point.current_thought``
            is set, add an **Actor's Reasoning:** section between the current
            state and current action.
        include_history_thoughts: When ``True``, include ``[Turn i - Reasoning]``
            sections for history turns that have a ``.thought``.
        state_text_override: Pre-serialized (possibly truncated) current state
            text. When provided, skips ``serialize_observation(point.state)``.
        next_state_text_override: Pre-serialized (possibly truncated) next state
            text. When provided, skips ``serialize_observation(point.next_state)``.
        use_answer_tags: When ``True``, use ``<answer>`` tag closing
            instructions instead of the default "output ONLY" closings.
        include_state_text_when_images: When ``False`` and a state has at
            least one image, that state's serialized text is dropped from
            the prompt (only the section label and the image(s) are kept).
            States without images always keep their text.
    """
    header = signal_type_header(signal_type, include_next_state)
    state_text = state_text_override or serialize_observation(point.state)
    blocks: list[ContentBlock] = [header]

    if task_text_to_inject:
        blocks.append(task_text_to_inject)

    if history:
        history_header = "### State-Action History"
        if history_truncated_from is not None and len(history) < history_truncated_from:
            history_header += (
                f"\n[Note: Only the last {len(history)} of "
                f"{history_truncated_from} turns are shown.]"
            )
        blocks.append(history_header)
        for i, turn in enumerate(history):
            label = _history_turn_label(i, len(history), point.step_index, max_steps)
            state_label = f"{label} - State]"
            turn_state_imgs = turn.state_images.all if include_images else ()
            if turn_state_imgs and not include_state_text_when_images:
                turn_parts = [state_label]
            else:
                turn_parts = [f"{state_label}\n{turn.state_text}"]
            if include_history_thoughts and turn.thought is not None:
                turn_parts.append(f"{label} - Reasoning]\n{turn.thought}")
            blocks.append("\n".join(turn_parts))
            for img in turn_state_imgs:
                blocks.append(img)
            blocks.append(f"{label} - Action]\n{turn.action_text}")

    cur_images = extract_images(point.state).all if include_images else ()
    if show_turn_labels:
        cur_label = _turn_label_prefix(point.step_index + 1, max_steps)
        cur_header = f"### Current State\n{cur_label}]"
    else:
        cur_header = "### Current State"
    if cur_images and not include_state_text_when_images:
        blocks.append(cur_header)
    else:
        blocks.append(f"{cur_header}\n{state_text}")
    for img in cur_images:
        blocks.append(img)

    if include_current_thoughts and point.current_thought is not None:
        blocks.append(f"## Actor's Reasoning\n{point.current_thought}")

    if signal_type in SIGNAL_TYPES_WITH_ACTION:
        action_text = action_text_for_display(
            point.action, point.resolved_action, point.extracted_action,
        )
        if include_current_thoughts and point.current_thought is not None:
            action_text = strip_thought_tags(action_text)
        blocks.append(f"### Current Action\n{action_text}")
        if include_next_state:
            next_state_text = (
                next_state_text_override
                or serialize_observation(point.next_state)
            )
            next_images = extract_images(point.next_state).all if include_images else ()
            if next_images and not include_state_text_when_images:
                blocks.append("### Next State")
            else:
                blocks.append(f"### Next State\n{next_state_text}")
            for img in next_images:
                blocks.append(img)

    closing_dict = (
        SIGNAL_TYPE_CLOSING_INSTRUCTIONS_ANSWER_TAGS
        if use_answer_tags
        else SIGNAL_TYPE_CLOSING_INSTRUCTIONS
    )
    blocks.append(closing_dict[signal_type])

    return PromptWithImages(blocks=tuple(blocks))


def build_direct_user_prompt_sequential_turn(
    point: EvaluationPoint,
    signal_type: SignalType,
    *,
    point_index: int,
    num_points: int,
    history: list[HistoryTurn] | None = None,
    history_truncated_from: int | None = None,
    include_next_state: bool = True,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    state_text_override: str | None = None,
    next_state_text_override: str | None = None,
    use_answer_tags: bool = False,
    max_steps: int | None = None,
    show_turn_labels: bool = True,
    task_text_to_inject: str | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> PromptWithImages:
    """Assemble a per-turn user prompt for sequential direct batching."""
    if num_points < 1:
        raise ValueError("build_direct_user_prompt_sequential_turn requires num_points >= 1")
    if point_index < 0 or point_index >= num_points:
        raise ValueError(
            "build_direct_user_prompt_sequential_turn point_index out of range"
        )

    blocks: list[ContentBlock] = [f"## Datapoint {point_index + 1} of {num_points}"]
    if point_index > 0:
        blocks.append(
            "Keep your earlier answers fixed and answer only for the datapoint below."
        )
    blocks.append(signal_type_header(signal_type, include_next_state))
    blocks.extend(
        _build_direct_prompt_blocks(
            point,
            signal_type,
            history=history,
            history_truncated_from=history_truncated_from,
            include_next_state=include_next_state,
            include_current_thoughts=include_current_thoughts,
            include_history_thoughts=include_history_thoughts,
            state_text_override=state_text_override,
            next_state_text_override=next_state_text_override,
            max_steps=max_steps,
            show_turn_labels=show_turn_labels,
            task_text_to_inject=task_text_to_inject,
            include_state_text_when_images=include_state_text_when_images,
            include_images=include_images,
        )
    )
    closing_dict = (
        SIGNAL_TYPE_CLOSING_INSTRUCTIONS_ANSWER_TAGS
        if use_answer_tags
        else SIGNAL_TYPE_CLOSING_INSTRUCTIONS
    )
    blocks.append(closing_dict[signal_type])
    return PromptWithImages(blocks=tuple(blocks))


def build_direct_user_prompt_batch(
    points: list[EvaluationPoint],
    signal_type: SignalType,
    *,
    histories: list[list[HistoryTurn] | None] | None = None,
    history_truncated_froms: list[int | None] | None = None,
    include_next_state: bool = True,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    state_text_overrides: list[str | None] | None = None,
    next_state_text_overrides: list[str | None] | None = None,
    use_answer_tags: bool = False,
    max_steps: int | None = None,
    joint_trajectory_prompt: bool = False,
    show_turn_labels: bool = True,
    task_text_to_injects: list[str | None] | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> PromptWithImages:
    """Assemble a direct-method user prompt for multiple evaluation points.

    Returns a :class:`PromptWithImages` with per-point state and
    next-state images interleaved directly after their text blocks.
    Text-only datasets produce a prompt with no images.
    """
    if not points:
        raise ValueError("build_direct_user_prompt_batch requires at least one point")

    blocks: list[ContentBlock] = [
        _direct_batch_user_header(
            signal_type,
            include_next_state,
            joint_trajectory_prompt=joint_trajectory_prompt,
        ),
    ]
    for i, point in enumerate(points, start=1):
        point_blocks = _build_direct_prompt_blocks(
            point,
            signal_type,
            history=histories[i - 1] if histories is not None else None,
            history_truncated_from=(
                history_truncated_froms[i - 1]
                if history_truncated_froms is not None
                else None
            ),
            include_next_state=include_next_state,
            include_current_thoughts=include_current_thoughts,
            include_history_thoughts=include_history_thoughts,
            state_text_override=(
                state_text_overrides[i - 1]
                if state_text_overrides is not None
                else None
            ),
            next_state_text_override=(
                next_state_text_overrides[i - 1]
                if next_state_text_overrides is not None
                else None
            ),
            max_steps=max_steps,
            show_turn_labels=show_turn_labels,
            task_text_to_inject=(
                task_text_to_injects[i - 1]
                if task_text_to_injects is not None
                else None
            ),
            datapoint_header=f"## Datapoint {i}",
            include_state_text_when_images=include_state_text_when_images,
            include_images=include_images,
        )
        blocks.extend(point_blocks)

    blocks.append(
        _direct_batch_user_closing(
            signal_type,
            use_answer_tags=use_answer_tags,
        )
    )
    return PromptWithImages(blocks=tuple(blocks))


def build_gvl_user_prompt(
    point: EvaluationPoint,
    signal_type: SignalType,
    *,
    context_transitions: list[GVLContextTransitionText],
    target_state_text: str | None = None,
    target_next_state_text: str | None = None,
    target_state_images: tuple[ImageContent, ...] = (),
    target_next_state_images: tuple[ImageContent, ...] = (),
    include_next_state: bool = True,
    use_answer_tags: bool = False,
    task_text_to_inject: str | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> PromptWithImages:
    """Assemble the user prompt for the dedicated GVL method.

    Returns a :class:`PromptWithImages`.  Each context transition's images
    sit right after that transition's text block; target state (and
    optional target next-state) images follow their text in the same way.
    Text-only datasets produce a prompt with no images.

    When ``include_state_text_when_images=False`` and a state has at
    least one image, the serialized state text is dropped for that
    state — only the heading (e.g. ``#### State``, ``### Target State``)
    and the image(s) remain.
    """
    blocks: list[ContentBlock] = [signal_type_header(signal_type, include_next_state)]

    if task_text_to_inject:
        blocks.append(task_text_to_inject)

    if context_transitions:
        blocks.append(
            "## Shuffled Trajectory Context\n"
            "The surrounding transitions below come from the same trajectory "
            "as the target datapoint, but are intentionally shuffled and "
            "not chronological."
        )
        for idx, transition in enumerate(context_transitions, start=1):
            transition_imgs = transition.state_images if include_images else ()
            if transition_imgs and not include_state_text_when_images:
                blocks.append(
                    f"### Context Transition {idx}\n#### State"
                )
            else:
                blocks.append(
                    f"### Context Transition {idx}\n#### State\n{transition.state_text}"
                )
            for img in transition_imgs:
                blocks.append(img)
            blocks.append(f"#### Action\n{transition.action_text}")

    blocks.append(
        "## Target Transition"
        if signal_type in SIGNAL_TYPES_WITH_ACTION
        else "## Target Datapoint"
    )
    state_text = target_state_text or serialize_observation(point.state)
    target_state_imgs = target_state_images if include_images else ()
    if target_state_imgs and not include_state_text_when_images:
        blocks.append("### Target State")
    else:
        blocks.append(f"### Target State\n{state_text}")
    for img in target_state_imgs:
        blocks.append(img)

    if signal_type in SIGNAL_TYPES_WITH_ACTION:
        action_text = action_text_for_display(
            point.action, point.resolved_action, point.extracted_action,
        )
        blocks.append(f"### Target Action\n{action_text}")
        if include_next_state:
            next_state_text = (
                target_next_state_text
                or serialize_observation(point.next_state)
            )
            target_ns_imgs = target_next_state_images if include_images else ()
            if target_ns_imgs and not include_state_text_when_images:
                blocks.append("### Target Next State")
            else:
                blocks.append(f"### Target Next State\n{next_state_text}")
            for img in target_ns_imgs:
                blocks.append(img)

    closing_dict = (
        SIGNAL_TYPE_CLOSING_INSTRUCTIONS_ANSWER_TAGS
        if use_answer_tags
        else SIGNAL_TYPE_CLOSING_INSTRUCTIONS
    )
    blocks.append(closing_dict[signal_type])
    return PromptWithImages(blocks=tuple(blocks))


def _verifier_batch_header(
    signal_type: SignalType,
    include_next_state: bool = True,
) -> str:
    if signal_type == SignalType.STATE_VALUE:
        return (
            "Estimate a verifier score for the **Current State** of each "
            "datapoint below."
        )
    if signal_type == SignalType.Q_VALUE:
        header = signal_type_header(signal_type, include_next_state)
        if header.endswith(" for the following:"):
            header = header.removesuffix(" for the following:")
        return (
            f"Estimate a verifier score for each datapoint below. {header}. "
            "Judge the chosen action and resulting next state using the "
            "ordered score bins."
        )
    raise ValueError(
        "Verifier prompts currently support only state_value and q_value, "
        f"got {signal_type.name.lower()!r}"
    )


def _verifier_batch_closing(num_points: int) -> str:
    lines = [
        "Now provide your final scores using one uppercase letter from A through T per datapoint.",
        "Output exactly one tag per datapoint in prompt order:",
    ]
    for i in range(1, num_points + 1):
        lines.append(f"<score_{i}>LETTER</score_{i}>")
    return "\n".join(lines)


def build_verifier_system_prompt(
    context: MethodContext,
    preset: PromptPreset,
    *,
    num_points: int = 1,
    criteria: tuple[str, ...] | None = None,
) -> str:
    """Assemble the system prompt for verifier-style direct estimation."""
    parts = _build_system_parts(context, preset, "verifier")
    if context.include_current_thoughts or context.include_history_thoughts:
        parts.append(get_block("actor_reasoning_guidance").content)
    parts.append(build_eval_instructions(
        context.signal_type,
        context.policy_assumption,
        context.discount_factor,
        disclose_discount_factor=context.disclose_discount_factor,
        include_next_state=context.include_next_state,
        max_history_turns=context.max_history_turns,
    ))
    parts.append(_VERIFIER_SCORE_SCALE_DESCRIPTION)
    parts.append(
        "Interpret the score bins as an ordered scale that will be converted "
        "into a scalar value after decoding. Use higher bins for stronger "
        "evidence that the state or chosen action is favorable."
    )

    if context.task_description:
        parts.append(f"## Task Description\n{context.task_description.rstrip()}")

    if context.environment_notes:
        parts.append(_format_environment_notes(context.environment_notes))

    if context.reward_description:
        parts.append(f"## Reward Functions\n{context.reward_description.rstrip()}")

    if context.example_trajectories:
        parts.append(_format_example_trajectories(context.example_trajectories))

    if num_points > 1:
        parts.append(
            "You may receive multiple datapoints in one request. Return one "
            "score tag per datapoint in the same order they appear."
        )
    else:
        parts.append(
            "Return exactly one score tag for the single datapoint."
        )
    return compose_system_prompt(*parts)


def build_verifier_user_prompt_batch(
    points: list[EvaluationPoint],
    signal_type: SignalType,
    *,
    histories: list[list[HistoryTurn] | None] | None = None,
    history_truncated_froms: list[int | None] | None = None,
    include_next_state: bool = True,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    criterion_name: str | None = None,
    criterion_description: str | None = None,
    state_text_overrides: list[str | None] | None = None,
    next_state_text_overrides: list[str | None] | None = None,
    max_steps: int | None = None,
    show_turn_labels: bool = True,
    task_text_to_injects: list[str | None] | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> PromptWithImages:
    """Assemble a verifier-style user prompt for one or more evaluation points.

    Returns a :class:`PromptWithImages` with interleaved text and image
    blocks.  Each point's state/history/next_state images sit immediately
    after their matching text section so VLMs can associate them.
    Text-only callers can use ``prompt.text``.
    """
    if not points:
        raise ValueError("build_verifier_user_prompt_batch requires at least one point")

    blocks: list[ContentBlock] = [_verifier_batch_header(signal_type, include_next_state)]
    if criterion_name is not None and criterion_description is not None:
        blocks.append(
            f"### Evaluation Criterion: {criterion_name}\n{criterion_description}"
        )

    for i, point in enumerate(points, start=1):
        point_blocks = _build_direct_prompt_blocks(
            point,
            signal_type,
            history=histories[i - 1] if histories is not None else None,
            history_truncated_from=(
                history_truncated_froms[i - 1]
                if history_truncated_froms is not None
                else None
            ),
            include_next_state=include_next_state,
            include_current_thoughts=include_current_thoughts,
            include_history_thoughts=include_history_thoughts,
            state_text_override=(
                state_text_overrides[i - 1]
                if state_text_overrides is not None
                else None
            ),
            next_state_text_override=(
                next_state_text_overrides[i - 1]
                if next_state_text_overrides is not None
                else None
            ),
            max_steps=max_steps,
            show_turn_labels=show_turn_labels,
            task_text_to_inject=(
                task_text_to_injects[i - 1]
                if task_text_to_injects is not None
                else None
            ),
            datapoint_header=f"## Datapoint {i}",
            include_state_text_when_images=include_state_text_when_images,
            include_images=include_images,
        )
        blocks.extend(point_blocks)

    blocks.append(_verifier_batch_closing(len(points)))
    return PromptWithImages(blocks=tuple(blocks))


def build_codegen_system_prompt(
    context: MethodContext,
    preset: PromptPreset,
) -> str:
    """Assemble the system prompt for LLMCodeGenMethod."""
    parts = _build_system_parts(context, preset, "codegen")

    if context.task_description:
        parts.append(f"## Task Description\n{context.task_description.rstrip()}")

    if context.environment_notes:
        parts.append(_format_environment_notes(context.environment_notes))

    if context.reward_description:
        parts.append(f"## Reward Functions\n{context.reward_description.rstrip()}")

    if context.example_trajectories:
        parts.append(_format_example_trajectories(context.example_trajectories))

    return compose_system_prompt(*parts)


# -- Codegen: signal-type-aware function signatures --

def codegen_param_names(
    signal_type: SignalType,
    include_next_state: bool = True,
) -> tuple[str, ...]:
    """Parameter names for the generated ``signal_function``.

    Args:
        signal_type: Which signal type the function targets.
        include_next_state: Whether to include ``next_state`` for action-based
            signal types.
    """
    if signal_type in (SignalType.STATE_VALUE, SignalType.POTENTIAL):
        return ("state",)
    if include_next_state:
        return ("state", "action", "next_state")
    return ("state", "action")


# Keep the old dict around for backward-compatible imports (default=True).
CODEGEN_PARAM_NAMES: dict[SignalType, tuple[str, ...]] = {
    st: codegen_param_names(st) for st in SignalType
}
"""Parameter names for the generated ``signal_function`` per signal type (default include_next_state=True)."""


def _codegen_input_description(
    signal_type: SignalType,
    include_next_state: bool = True,
) -> str:
    if signal_type in (SignalType.STATE_VALUE, SignalType.POTENTIAL):
        return "a given state"
    if include_next_state:
        return "a given state, action, and next state"
    return "a given state and action"


def _codegen_receives_description(
    signal_type: SignalType,
    include_next_state: bool = True,
) -> str:
    if signal_type in (SignalType.STATE_VALUE, SignalType.POTENTIAL):
        return "a text representation of the state"
    if include_next_state:
        return (
            "text representations of the state, the action taken, "
            "and the resulting next state"
        )
    return "text representations of the state and the action taken"


def _codegen_signature(
    signal_type: SignalType,
    include_next_state: bool = True,
) -> str:
    """Build the ``def signal_function(...)`` line for the given signal type."""
    params = ", ".join(
        f"{p}: str" for p in codegen_param_names(signal_type, include_next_state)
    )
    return f"def signal_function({params}) -> float:"


def build_codegen_user_prompt(
    signal_type: SignalType,
    preset: PromptPreset,
    include_next_state: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Assemble the user prompt for LLMCodeGenMethod."""
    signal_name = SIGNAL_TYPE_NAMES[signal_type]
    input_desc = _codegen_input_description(signal_type, include_next_state)
    receives_desc = _codegen_receives_description(signal_type, include_next_state)
    signature = _codegen_signature(signal_type, include_next_state)

    parts = [
        f"Write a Python function that estimates the {signal_name} for "
        f"{input_desc}.\n\n"
        f"The function signature must be:\n"
        f"```python\n"
        f"{signature}\n"
        f"```\n\n"
        f"The function receives {receives_desc}. "
        f"It should return a float representing the estimated "
        f"{signal_name}.",
    ]

    constraints = resolve_block(preset.codegen_constraints)
    if constraints is not None:
        parts.append(constraints)

    closing = resolve_block(preset.codegen_closing)
    parts.append(_adapt_closing(closing, enable_thinking))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ranking method prompts
# ---------------------------------------------------------------------------


def _build_ranking_eval(
    discount_factor: float,
    *,
    disclosed: bool = True,
    include_next_state: bool = True,
    max_history_turns: int | None = None,
) -> str:
    """Build ranking-specific evaluation instructions.

    The policy phrase is omitted here because the signal-type fragment
    (Q-value definition) already states it.
    """
    ret = _return_phrase(discount_factor, disclosed=disclosed)
    hist = _history_preamble(max_history_turns)
    if include_next_state:
        data_desc = (
            "the current state, a list of candidate actions, and the "
            "resulting next state for each action"
        )
    else:
        data_desc = "the current state, and a list of candidate actions"
    return (
        f"You will receive {hist}"
        f"{data_desc}. Rank the actions "
        f"by their expected Q-value Q(s,a) — the expected {ret} when taking "
        f"each action. Output the action numbers from best to worst."
    )


def build_ranking_system_prompt(
    context: MethodContext,
    preset: PromptPreset,
) -> str:
    """Assemble the system prompt for LLMRankingMethod."""
    signal_name = SIGNAL_TYPE_NAMES[SignalType.Q_VALUE]
    parts = _build_system_parts(context, preset, "ranking")

    if context.include_current_thoughts or context.include_history_thoughts:
        parts.append(get_block("actor_reasoning_guidance").content)

    parts.append(_build_ranking_eval(
        context.discount_factor,
        disclosed=context.disclose_discount_factor,
        include_next_state=context.include_next_state,
        max_history_turns=context.max_history_turns,
    ))

    if context.task_description:
        parts.append(f"## Task Description\n{context.task_description.rstrip()}")

    if context.environment_notes:
        parts.append(_format_environment_notes(context.environment_notes))

    if context.reward_description:
        parts.append(f"## Reward Functions\n{context.reward_description.rstrip()}")

    if context.example_trajectories:
        parts.append(_format_example_trajectories(context.example_trajectories))

    # Closing instruction from preset
    closing = resolve_block(preset.closing_ranking, signal_name=signal_name)
    parts.append(_adapt_closing(closing, context.enable_thinking))

    return compose_system_prompt(*parts)


def build_ranking_user_prompt(
    point: RankingPoint,
    history: list[HistoryTurn] | None = None,
    history_truncated_from: int | None = None,
    include_next_state: bool = True,
    *,
    include_current_thoughts: bool = False,
    include_history_thoughts: bool = False,
    state_text_override: str | None = None,
    candidate_next_state_overrides: list[str | None] | None = None,
    candidate_action_overrides: list[str | None] | None = None,
    use_answer_tags: bool = False,
    max_steps: int | None = None,
    task_text_to_inject: str | None = None,
    include_state_text_when_images: bool = True,
    include_images: bool = True,
) -> PromptWithImages:
    """Assemble the user prompt for LLMRankingMethod.

    Returns a :class:`PromptWithImages` with interleaved text and image
    blocks, same as :func:`build_direct_user_prompt`.  Each candidate's
    resulting-state images (if any) sit immediately after that
    candidate's ``[**Resulting State N**]`` header.

    Args:
        include_current_thoughts: When ``True`` and **all** candidates have
            a ``.thought``, show ``Reasoning: ...`` before each candidate's
            action line. When any candidate lacks a thought, reasoning is
            omitted for all.
        include_history_thoughts: When ``True``, include ``[Turn i - Reasoning]``
            sections for history turns that have a ``.thought``.
        state_text_override: Pre-serialized (possibly truncated) current state
            text. When provided, skips ``serialize_observation(point.state)``.
        candidate_next_state_overrides: Per-candidate pre-serialized next-state
            texts. When provided, entry *i* replaces
            ``serialize_observation(candidate.next_state)`` for candidate *i*.
            ``None`` entries fall back to fresh serialization.
        candidate_action_overrides: Per-candidate pre-truncated action texts.
            When provided, entry *i* replaces ``action_text_for_display()``
            for candidate *i*. ``None`` entries fall back to fresh display.
        use_answer_tags: When ``True``, use ``<answer>`` tag closing
            instructions instead of the default "output ONLY" closings.
        include_state_text_when_images: When ``False`` and a state has at
            least one image, that state's serialized text is dropped from
            the prompt (only the section label and the image(s) are kept).
            Applied to history, current state, and each candidate's
            resulting state.
    """
    state_text = state_text_override or serialize_observation(point.state)
    k = len(point.candidates)

    blocks: list[ContentBlock] = [
        "Rank the following actions by Q-value (best to worst) for the "
        "Current State:",
    ]

    if task_text_to_inject:
        blocks.append(task_text_to_inject)

    if history:
        history_header = "### State-Action History"
        if history_truncated_from is not None and len(history) < history_truncated_from:
            history_header += (
                f"\n[Note: Only the last {len(history)} of "
                f"{history_truncated_from} turns are shown.]"
            )
        blocks.append(history_header)
        for i, turn in enumerate(history):
            label = _history_turn_label(i, len(history), point.step_index, max_steps)
            state_label = f"{label} - State]"
            turn_state_imgs = turn.state_images.all if include_images else ()
            if turn_state_imgs and not include_state_text_when_images:
                turn_parts = [state_label]
            else:
                turn_parts = [f"{state_label}\n{turn.state_text}"]
            if include_history_thoughts and turn.thought is not None:
                turn_parts.append(f"{label} - Reasoning]\n{turn.thought}")
            blocks.append("\n".join(turn_parts))
            for img in turn_state_imgs:
                blocks.append(img)
            blocks.append(f"{label} - Action]\n{turn.action_text}")

    cur_label = _turn_label_prefix(point.step_index + 1, max_steps)
    cur_header = f"## Current State\n{cur_label}]"
    cur_images = extract_images(point.state).all if include_images else ()
    if cur_images and not include_state_text_when_images:
        blocks.append(cur_header)
    else:
        blocks.append(f"{cur_header}\n{state_text}")
    for img in cur_images:
        blocks.append(img)

    # Show per-candidate reasoning only when ALL candidates have thoughts
    show_candidate_reasoning = (
        include_current_thoughts
        and all(c.thought is not None for c in point.candidates)
    )

    blocks.append(
        "The following candidate actions are being evaluated in the current state:"
    )
    blocks.append("## Actions to rank")

    for i, candidate in enumerate(point.candidates, 1):
        action_override = (
            candidate_action_overrides[i - 1]
            if candidate_action_overrides is not None
            else None
        )
        display = action_override or action_text_for_display(
            candidate.action,
            candidate.resolved_action,
            candidate.extracted_action,
        )
        action_parts: list[str] = []
        if show_candidate_reasoning:
            action_parts.append(f"Reasoning: {candidate.thought}")
        action_parts.append(f"[**Candidate Action {i}**]\n{display}")
        blocks.append("\n\n".join(action_parts))

        if include_next_state and candidate.next_state is not None:
            override = (
                candidate_next_state_overrides[i - 1]
                if candidate_next_state_overrides is not None
                else None
            )
            ns_text = override or serialize_observation(candidate.next_state)
            ns_images = extract_images(candidate.next_state).all if include_images else ()
            ns_header = f"[**Resulting State {i}**]"
            if ns_images and not include_state_text_when_images:
                blocks.append(ns_header)
            else:
                blocks.append(f"{ns_header}\n{ns_text}")
            for img in ns_images:
                blocks.append(img)

    if use_answer_tags:
        blocks.append(
            f"Rank all listed actions ({k} total here) from best (highest Q-value) "
            f"to worst. Output a comma-separated list of action numbers inside "
            f"<answer> tags, e.g. <answer>#, #, #, ...</answer>."
        )
    else:
        example_ranking = ", ".join(str(i) for i in range(1, k + 1))
        blocks.append(
            f"Rank all listed actions ({k} total here) from best (highest Q-value) to worst. "
            f"Output ONLY a comma-separated list of action numbers. "
            f"For example: {example_ranking}"
        )

    return PromptWithImages(blocks=tuple(blocks))

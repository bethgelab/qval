"""Offline SDPO-style ranking using feedback-conditioned self-distillation."""

from __future__ import annotations

import logging
import math
import json
import pickle
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llenvs.inference import compose_system_prompt
from llenvs.inference.protocol import ChatMessage, ModelBackend

from qval.error_handling import is_recoverable_backend_error
from qval.methods.serialization import action_text_for_display, serialize_observation
from qval.prompt_presets import PromptPreset, resolve_block
from qval.prompts import SIGNAL_TYPE_NAMES, build_signal_type_fragment
from qval.types import MethodContext, RankingPoint, SignalType

logger = logging.getLogger(__name__)


def _history_turn_text(turn: Any) -> tuple[str, str]:
    if hasattr(turn, "state_text") and hasattr(turn, "action_text"):
        return str(turn.state_text), str(turn.action_text)
    if isinstance(turn, (tuple, list)) and len(turn) >= 2:
        return str(turn[0]), str(turn[1])
    raise TypeError(f"Unsupported history turn type: {type(turn).__name__}")


def _format_environment_notes(notes: list[str] | None) -> str | None:
    if not notes:
        return None
    if len(notes) == 1:
        return f"## Episode Configuration\n{notes[0]}"
    lines = "\n".join(f"- {note}" for note in notes)
    return f"## Episode Configuration\n{lines}"


def _task_text_for_prompt(point: RankingPoint) -> str | None:
    obs = getattr(getattr(point, "state", None), "observation", None)
    task = getattr(obs, "task", None)
    text = getattr(task, "text", None)
    if not text:
        return None
    stripped = str(text).strip()
    return stripped or None


def _state_text_for_prompt(
    point: RankingPoint,
    *,
    state_text_override: str | None,
    inject_task_text: bool,
) -> str:
    state_text = state_text_override or serialize_observation(point.state)
    if not inject_task_text:
        return state_text

    task_text = _task_text_for_prompt(point)
    if not task_text or task_text in state_text:
        return state_text

    return (
        f"Task Goal:\n{task_text}\n\n"
        f"Accessibility-Tree Observation:\n{state_text}"
    )


def _environment_family(context: MethodContext) -> str:
    adapter = (context.adapter or "").lower()
    env_name = (context.env_name or "").lower()
    if adapter == "open_apps" or adapter == "openapps" or "open_apps" in env_name:
        return "open_apps"
    if adapter == "alfworld" or env_name.startswith("alfworld"):
        return "alfworld"
    if adapter == "harbor" or "terminal-bench" in env_name or "tblite" in env_name:
        return "terminal_bench"
    if "frozen_lake" in env_name or "frozenlake" in env_name:
        return "frozen_lake"
    return "generic"


def _head_tail_truncate_text(text: str, max_chars: int, *, label: str) -> str:
    if len(text) <= max_chars:
        return text
    marker = (
        f"\n\n[{label} truncated: showing head and tail of "
        f"{len(text)} total chars; middle omitted.]\n\n"
    )
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return marker[:max_chars]
    head_chars = remaining // 2
    tail_chars = remaining - head_chars
    if tail_chars == 0:
        return text[:head_chars] + marker
    return text[:head_chars] + marker + text[-tail_chars:]


def _student_opening(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "Choose the next OpenApps browser action for the following decision "
            "point. Use the task goal, the accessibility-tree observation, "
            "visible UI element identifiers, prior browser actions, and the "
            "available BrowserGym primitives such as click('bid'), "
            "fill('bid', 'text'), and scroll(x, y)."
        )
    if family == "alfworld":
        return (
            "Choose the next ALFWorld command for the following decision point. "
            "Use the household objective, current observation, inventory/location "
            "state, and admissible commands when they are shown."
        )
    if family == "terminal_bench":
        return (
            "Choose the next shell command for the following TerminalBench "
            "decision point. The shell session is persistent, so commands should "
            "make concrete progress through inspection, file creation or edits, "
            "validation, or targeted error diagnosis."
        )
    if family == "frozen_lake":
        return (
            "Choose the next FrozenLake move for the following decision point. "
            "Use the current grid, the agent position marked '@', the goal (G), "
            "holes (H), and the available moves: left, down, right, up."
        )
    return "Choose the next action for the following decision point."


def _student_closing(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "Respond with exactly one valid BrowserGym action and no explanation. "
            "Prefer actions that navigate to the correct OpenApps application, "
            "click the intended visible element, enter instruction-matching "
            "field values, submit or confirm at the right time, and avoid "
            "unrelated app state changes. The assistant continuation being "
            "scored is that browser action itself."
        )
    if family == "alfworld":
        return (
            "Respond with exactly one valid ALFWorld command and no explanation. "
            "Prefer commands that make immediate progress toward the stated "
            "household goal while preserving required preconditions. "
            "The assistant continuation being scored is that command itself."
        )
    if family == "terminal_bench":
        return (
            "Respond with exactly one shell command and no explanation. Prefer "
            "commands that produce durable task progress, useful diagnostics, or "
            "validation evidence rather than redundant probing or noisy output. "
            "The assistant continuation being scored is that command itself."
        )
    if family == "frozen_lake":
        return (
            "Respond with exactly one valid FrozenLake move from: left, down, "
            "right, up. Prefer moves that keep the agent on safe frozen tiles, "
            "make progress toward the goal, avoid holes, and avoid wasting the "
            "limited step budget. The assistant continuation being scored is "
            "that move itself."
        )
    return (
        "Respond with exactly one action and no explanation. "
        "The assistant continuation being scored is that action itself."
    )


def _teacher_opening(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "Choose again from the same OpenApps browser state after seeing the "
            "immediate page response produced by one candidate browser action."
        )
    if family == "alfworld":
        return (
            "Choose again from the same ALFWorld decision point after seeing the "
            "immediate environment response produced by one candidate command."
        )
    if family == "terminal_bench":
        return (
            "Choose again from the same TerminalBench shell state after seeing "
            "the immediate shell response produced by one candidate command."
        )
    if family == "frozen_lake":
        return (
            "Choose again from the same FrozenLake grid position after seeing "
            "the immediate environment response produced by one candidate move."
        )
    return (
        "Choose again from the same decision point after seeing the immediate "
        "environment response produced by one candidate action."
    )


def _feedback_heading(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return "**Immediate OpenApps Response After The Candidate Browser Action:**"
    if family == "alfworld":
        return "**Immediate ALFWorld Response After The Candidate Command:**"
    if family == "terminal_bench":
        return "**Immediate Shell Response After The Candidate Command:**"
    if family == "frozen_lake":
        return "**Immediate FrozenLake Response After The Candidate Move:**"
    return "**Immediate Environment Response After The Candidate Action:**"


def _teacher_closing(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "Use this one-turn page response only as evidence about the "
            "candidate browser action's quality in the current OpenApps task: "
            "whether it navigated to the right synthetic app, interacted with "
            "the intended visible element, entered exact instruction values, "
            "created or edited the correct event, todo, message, code, or map "
            "state, resolved or avoided validation errors, and preserved useful "
            "future options. A click or fill with little visible change can "
            "still be good if it sets up the target application state; a "
            "successful-looking page transition is bad if it moves into the "
            "wrong app or commits wrong values. Do not reward a browser action "
            "merely because the response makes the action easy to reconstruct. "
            "The assistant continuation being scored is the original candidate "
            "browser action itself, so do not rewrite or replace it here."
        )
    if family == "alfworld":
        return (
            "Use this one-turn response only as evidence about the candidate "
            "command's quality in the current ALFWorld state: whether it was "
            "valid, respected preconditions, interacted with the right object or "
            "location, reduced distance to the household goal, avoided wrong "
            "objects or wasted navigation, and preserved useful future options. "
            "Do not reward a command merely because the response makes the "
            "command easy to reconstruct. The assistant continuation being "
            "scored is the original candidate command itself, so do not rewrite "
            "or replace it here."
        )
    if family == "terminal_bench":
        return (
            "Use this one-turn shell response only as evidence about the "
            "candidate command's quality in the current TerminalBench task: "
            "whether it made durable file or configuration progress, produced "
            "useful diagnostics, revealed or resolved an error, preserved shell "
            "state, avoided redundant probes, and moved toward eventual verifier "
            "success. A command with no output can still be good if it changed "
            "state usefully; an error can be useful only when it provides needed "
            "diagnostic information. Do not reward a command merely because its "
            "output makes the command easy to reconstruct. The assistant "
            "continuation being scored is the original candidate command itself, "
            "so do not rewrite or replace it here."
        )
    if family == "frozen_lake":
        return (
            "Use this one-turn response only as evidence about the candidate "
            "move's quality in the current FrozenLake state: whether it kept the "
            "agent on safe ice, avoided holes, moved closer to the goal along a "
            "viable path, avoided off-grid no-ops or backtracking, preserved "
            "future safe moves, and used the remaining step budget efficiently. "
            "If the response shows the agent reached the goal, heavily reward "
            "the move; if it shows the agent fell into a hole or entered a "
            "dead-end region, heavily penalize it. Do not reward a move merely "
            "because the response makes the move easy to reconstruct. The "
            "assistant continuation being scored is the original candidate move "
            "itself, so do not rewrite or replace it here."
        )
    return (
        "Use this one-turn response only as evidence about whether the candidate "
        "action improved expected future success from the current state. Do not "
        "reward an action merely because the response makes the action easy to "
        "reconstruct. The assistant continuation being scored is the original "
        "candidate action itself, so do not rewrite or replace it here."
    )


def _short_expert_teacher_opening(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "Choose again from the same OpenApps browser state after seeing the "
            "immediate page response produced by one candidate browser action, "
            "the next expert browser action from the resulting state, and a "
            "structured outcome summary from the stored expert rollout."
        )
    if family == "alfworld":
        return (
            "Choose again from the same ALFWorld decision point after seeing the "
            "immediate environment response produced by one candidate command "
            "and the next expert command from the resulting state, plus a "
            "structured outcome summary from the stored expert rollout."
        )
    if family == "terminal_bench":
        return (
            "Choose again from the same TerminalBench shell state after seeing "
            "the immediate shell response produced by one candidate command, "
            "the next expert command from the resulting state, and a structured "
            "verifier outcome summary from the stored expert rollout."
        )
    if family == "frozen_lake":
        return (
            "Choose again from the same FrozenLake grid position after seeing "
            "the immediate environment response produced by one candidate move, "
            "the next expert move from the resulting grid state, and a structured "
            "outcome summary from the stored expert rollout."
        )
    return _teacher_opening(context)


def _short_expert_heading(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return "**Expert Browser Rollout Evidence After The Candidate Browser Action:**"
    if family == "alfworld":
        return "**Expert Rollout Evidence After The Candidate Command:**"
    if family == "terminal_bench":
        return "**Expert Shell Rollout Evidence After The Candidate Command:**"
    if family == "frozen_lake":
        return "**Expert Rollout Evidence After The Candidate Move:**"
    return "**Short Expert Continuation After The Candidate Action:**"


def _short_expert_teacher_closing(context: MethodContext) -> str:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "Use the immediate page response, next expert browser action, and "
            "structured rollout summary to judge the original candidate browser "
            "action from the current OpenApps state. Prefer actions that navigate "
            "to the correct synthetic app, interact with the intended visible "
            "element, enter exact instruction values, resolve or avoid validation "
            "errors, reduce expert distance to task success, and preserve useful "
            "future browser state. Treat a short successful stored expert rollout "
            "as strong positive evidence; treat no observed success as weaker "
            "evidence, not proof that the state is unsolvable. Do not reward a "
            "browser action merely because the page response or next expert action "
            "makes it easy to reconstruct. The assistant continuation being scored "
            "is the original candidate browser action itself, so do not rewrite or "
            "replace it."
        )
    if family == "alfworld":
        return (
            "Use the immediate response, next expert command, and structured "
            "expert rollout summary to judge the original candidate command from "
            "the current ALFWorld state. Prefer commands that are valid, satisfy "
            "preconditions, move the right object or location toward the household "
            "goal, reduce expert distance to success, and preserve useful future "
            "options. Treat a short successful stored expert rollout as strong "
            "positive evidence; treat no observed success as weaker evidence, not "
            "proof that the state is unsolvable. Do not reward a command merely "
            "because the response or next expert command makes it easy to "
            "reconstruct. The assistant continuation being scored is the original "
            "candidate command itself, so do not rewrite or replace it."
        )
    if family == "terminal_bench":
        return (
            "Use the immediate shell response, next expert command, and structured "
            "verifier outcome summary to judge the original candidate command from "
            "the current TerminalBench shell state. Prefer commands that make "
            "durable file or configuration progress, produce useful diagnostics "
            "or validation evidence, preserve needed shell state, reduce expert "
            "distance to verifier success, and avoid redundant probes or noisy "
            "dead ends. Treat a short successful stored expert rollout as strong "
            "positive evidence; treat no observed success as weaker evidence, not "
            "proof that the state is unsolvable. Do not reward a command merely "
            "because its output or the next expert command makes it easy to "
            "reconstruct. The assistant continuation being scored is the original "
            "candidate command itself, so do not rewrite or replace it."
        )
    if family == "frozen_lake":
        return (
            "Use the immediate grid response, next expert move, and structured "
            "rollout summary to judge the original candidate move from the current "
            "FrozenLake grid state. Prefer moves that keep the agent on safe ice, "
            "avoid holes and off-grid no-ops, reduce expert distance to the goal, "
            "preserve future safe moves, and use the remaining step budget "
            "efficiently. Treat a short successful stored expert rollout as strong "
            "positive evidence; treat no observed success as weaker evidence, not "
            "proof that the state is unsolvable. Do not reward a move merely "
            "because the response or next expert move makes it easy to reconstruct. "
            "The assistant continuation being scored is the original candidate "
            "move itself, so do not rewrite or replace it."
        )
    return _teacher_closing(context)


@dataclass(frozen=True)
class ExpertContinuation:
    """Compact expert rollout evidence for one ranking candidate."""

    best_next_command: str | None
    commands: tuple[str, ...]
    reached_goal: bool | None = None
    steps_to_success: int | None = None
    observed_expert_steps: int | None = None
    final_reward: float | None = None
    ranking_gt_value: float | None = None
    ranking_gt_source: str | None = None

    def format_block(self, *, family: str = "alfworld") -> str:
        has_outcome = (
            self.reached_goal is not None
            or self.steps_to_success is not None
            or self.observed_expert_steps is not None
            or self.final_reward is not None
            or self.ranking_gt_value is not None
        )
        if not self.best_next_command and not self.commands and not has_outcome:
            return "[No expert continuation was available for this candidate.]"
        best_next = self.best_next_command or "unknown"
        if family == "open_apps":
            state_label = "OpenApps browser state"
            expert_unit = "expert browser action"
            candidate_label = "candidate browser action"
            next_label = "Best next expert browser action after candidate result"
            reached_label = "Stored expert rollout reached task success"
            steps_label = "Expert browser steps from candidate result to success"
            observed_steps_label = (
                "Expert browser steps from candidate result observed"
            )
            remaining_success = "expert browser step(s) to task success"
            remaining_missing = (
                "task success was not observed in the stored expert rollout"
            )
            reward_label = "Final observed task reward"
        elif family == "terminal_bench":
            state_label = "persistent TerminalBench shell session"
            expert_unit = "expert command"
            candidate_label = "candidate command"
            next_label = "Best next expert command after candidate result"
            reached_label = "Stored expert rollout reached verifier success"
            steps_label = "Expert shell steps from candidate result to success"
            observed_steps_label = "Expert shell steps from candidate result observed"
            remaining_success = "expert shell step(s) to verifier success"
            remaining_missing = (
                "verifier success was not observed in the stored expert rollout"
            )
            reward_label = "Final observed verifier reward"
        elif family == "frozen_lake":
            state_label = "FrozenLake grid state"
            expert_unit = "expert move"
            candidate_label = "candidate move"
            next_label = "Best next expert move after candidate result"
            reached_label = "Stored expert rollout reached goal"
            steps_label = "Expert moves from candidate result to success"
            observed_steps_label = "Expert moves from candidate result observed"
            remaining_success = "expert move(s) to goal"
            remaining_missing = (
                "goal was not observed in the stored expert rollout"
            )
            reward_label = "Final observed environment reward"
        else:
            state_label = "ALFWorld state"
            expert_unit = "expert command"
            candidate_label = "candidate command"
            next_label = "Best next expert command after candidate result"
            reached_label = "Stored expert rollout reached goal"
            steps_label = "Expert steps from candidate result to success"
            observed_steps_label = "Expert steps from candidate result observed"
            remaining_success = "expert step(s) to task completion"
            remaining_missing = (
                "task completion was not observed in the stored expert rollout"
            )
            reward_label = "Final observed task reward"
        lines = [
            f"The {expert_unit} below occurs after the {candidate_label} has "
            f"already been executed in the {state_label}.",
            f"Use it only as evidence about the original {candidate_label}; it is "
            "not a replacement command to output.",
            f"{next_label}: {best_next}",
        ]
        if has_outcome:
            if self.reached_goal is True:
                reached_goal = "yes"
                rollout_outcome = "success"
            elif self.reached_goal is False:
                reached_goal = "no success observed"
                rollout_outcome = "no success observed"
            else:
                reached_goal = "unknown"
                rollout_outcome = "unknown"

            lines.append("Structured stored expert rollout summary:")
            lines.append(f"{reached_label}: {reached_goal}")
            if self.steps_to_success is not None:
                lines.append(f"{steps_label}: {self.steps_to_success}")
                lines.append(
                    "Remaining distance proxy: "
                    f"{self.steps_to_success} {remaining_success}"
                )
            elif self.observed_expert_steps is not None:
                lines.append(f"{observed_steps_label}: {self.observed_expert_steps}")
                lines.append(
                    f"Remaining distance proxy: {remaining_missing}"
                )
            if self.final_reward is not None:
                lines.append(f"{reward_label}: {self.final_reward:g}")
            if self.ranking_gt_value is not None:
                if self.ranking_gt_source:
                    lines.append(
                        f"Stored ranking GT value ({self.ranking_gt_source}): "
                        f"{self.ranking_gt_value:g}"
                    )
                else:
                    lines.append(
                        "Stored ranking GT value for this candidate: "
                        f"{self.ranking_gt_value:g}"
                    )
            lines.append(f"Stored expert rollout outcome: {rollout_outcome}")
        return "\n".join(lines)


_EXPERT_PLAN_NEXT_RE = re.compile(r"^\[expert_plan_next:\s*(.*?)\]\s*$", re.MULTILINE)


def _state_text(state: Any) -> str | None:
    observation = getattr(state, "observation", None)
    content = getattr(observation, "state", None)
    text = getattr(content, "text", None)
    return str(text) if text is not None else None


def _expert_plan_next_from_state(state: Any) -> str | None:
    text = _state_text(state)
    if not text:
        return None
    match = _EXPERT_PLAN_NEXT_RE.search(text)
    if not match:
        return None
    command = match.group(1).strip()
    return command or None


def _transition_action_text(transition: Any) -> str | None:
    text = action_text_for_display(
        getattr(transition, "action", None),
        getattr(transition, "resolved_action", None),
        getattr(transition, "extracted_action", None),
    )
    text = text.strip()
    return text or None


def _reward_value(rewards: Any) -> float | None:
    if rewards is None:
        return None
    by_name = getattr(rewards, "by_name", None)
    if callable(by_name):
        for name in ("task_completion", "correctness", "gym_reward"):
            try:
                signal = by_name(name, required=False)
            except TypeError:
                try:
                    signal = by_name(name)
                except Exception:
                    signal = None
            except Exception:
                signal = None
            reward = getattr(signal, "reward", None)
            if reward is not None:
                return float(reward)

    signals = getattr(rewards, "signals", None)
    if signals is not None:
        for signal in signals:
            if getattr(signal, "name", None) in {
                "task_completion",
                "correctness",
                "gym_reward",
            }:
                reward = getattr(signal, "reward", None)
                if reward is not None:
                    return float(reward)

    total = getattr(rewards, "total", None)
    if total is not None:
        return float(total)
    return None


def _expert_outcome_from_stored_trajectory(
    trajectory: Any,
) -> tuple[bool | None, int | None, int | None, float | None]:
    transitions = tuple(getattr(trajectory, "transitions", ()) or ())
    if not transitions:
        return None, None, None, None

    observed_expert_steps = max(0, len(transitions) - 1)
    final_reward: float | None = None
    success_index: int | None = None
    for idx, transition in enumerate(transitions):
        reward = _reward_value(getattr(transition, "rewards", None))
        if reward is None:
            continue
        final_reward = reward
        if reward > 0.0 and success_index is None:
            success_index = idx

    if success_index is None:
        if final_reward is None:
            return None, None, observed_expert_steps, None
        return False, None, observed_expert_steps, final_reward
    return True, success_index, observed_expert_steps, final_reward


def _expert_continuation_from_stored_trajectory(
    trajectory: Any,
    *,
    max_steps: int,
) -> ExpertContinuation:
    transitions = tuple(getattr(trajectory, "transitions", ()) or ())
    reached_goal, steps_to_success, observed_expert_steps, final_reward = (
        _expert_outcome_from_stored_trajectory(trajectory)
    )
    best_next = None
    if len(transitions) > 1:
        best_next = _transition_action_text(transitions[1])
    if best_next is None:
        best_next = _expert_plan_next_from_state(
            getattr(transitions[0], "next_state", None)
            if transitions else None
        )
    return ExpertContinuation(
        best_next_command=best_next,
        commands=(),
        reached_goal=reached_goal,
        steps_to_success=steps_to_success,
        observed_expert_steps=observed_expert_steps,
        final_reward=final_reward,
    )


def _steps_from_discounted_value(value: float, discount_factor: float | None) -> int | None:
    if value <= 0.0:
        return None
    if value >= 1.0:
        return 0
    if discount_factor is None or discount_factor <= 0.0 or discount_factor >= 1.0:
        return None
    steps = round(math.log(value) / math.log(discount_factor))
    if steps < 0:
        return None
    reconstructed = discount_factor ** steps
    if math.isclose(reconstructed, value, rel_tol=1e-6, abs_tol=1e-9):
        return int(steps)
    return None


def _expert_continuation_from_ranking_gt_value(
    value: float,
    *,
    discount_factor: float | None,
    source: str | None,
) -> ExpertContinuation:
    steps_to_success = _steps_from_discounted_value(value, discount_factor)
    reached_goal = value > 0.0
    return ExpertContinuation(
        best_next_command=None,
        commands=(),
        reached_goal=reached_goal,
        steps_to_success=steps_to_success,
        observed_expert_steps=steps_to_success,
        final_reward=1.0 if reached_goal else 0.0,
        ranking_gt_value=value,
        ranking_gt_source=source,
    )


@dataclass(frozen=True)
class FlatRankingGTContinuationLookup:
    """Flat ranking-GT values that are keyed once ranking points are available."""

    continuations: tuple[ExpertContinuation, ...]
    candidate_counts: tuple[int, ...]
    source_path: str

    def __len__(self) -> int:
        return len(self.continuations)

    def get(self, key: tuple[int, int, int]) -> ExpertContinuation | None:
        return None

    def materialize(
        self,
        points: list[RankingPoint],
    ) -> dict[tuple[int, int, int], ExpertContinuation]:
        if len(points) != len(self.candidate_counts):
            raise ValueError(
                f"Ranking GT file {self.source_path} has "
                f"{len(self.candidate_counts)} ranking points, but the dataset "
                f"has {len(points)}"
            )

        by_key: dict[tuple[int, int, int], ExpertContinuation] = {}
        offset = 0
        for point_index, point in enumerate(points):
            expected = self.candidate_counts[point_index]
            actual = len(point.candidates)
            if actual != expected:
                raise ValueError(
                    f"Ranking GT file {self.source_path} candidate count mismatch "
                    f"at ranking point {point_index}: expected {expected}, got "
                    f"{actual}"
                )
            for candidate_index in range(actual):
                by_key[(point.trajectory_index, point.step_index, candidate_index)] = (
                    self.continuations[offset]
                )
                offset += 1

        if offset != len(self.continuations):
            raise ValueError(
                f"Ranking GT file {self.source_path} has "
                f"{len(self.continuations)} values, but materialized {offset}"
            )
        return by_key


def _select_expert_trajectory(trajectories: Any) -> Any:
    """Choose the stored rollout with the strongest observed outcome."""

    best_trajectory = None
    best_key: tuple[int, float, int] | None = None
    for trajectory in trajectories:
        reached_goal, steps_to_success, _observed_steps, final_reward = (
            _expert_outcome_from_stored_trajectory(trajectory)
        )
        if reached_goal is True:
            success_rank = 1
        elif reached_goal is False:
            success_rank = 0
        else:
            success_rank = -1
        reward_key = final_reward if final_reward is not None else float("-inf")
        steps_key = -steps_to_success if steps_to_success is not None else 0
        key = (success_rank, reward_key, steps_key)
        if best_key is None or key > best_key:
            best_key = key
            best_trajectory = trajectory
    return best_trajectory


def load_expert_continuation_lookup(
    rollout_store: str | Path,
    *,
    max_steps: int,
) -> dict[tuple[int, int, int], ExpertContinuation] | FlatRankingGTContinuationLookup:
    """Load next-step expert evidence from a rollout store or ranking GT JSON."""

    root = Path(rollout_store)
    if root.is_file() and root.suffix == ".json":
        data = json.loads(root.read_text())
        config = data.get("config", {})
        candidate_counts = tuple(
            int(count) for count in config.get("candidate_counts", ())
        )
        values = tuple(float(value) for value in data.get("values", ()))
        if not candidate_counts:
            raise ValueError(f"ranking GT JSON has no candidate_counts: {root}")
        if sum(candidate_counts) != len(values):
            raise ValueError(
                f"ranking GT JSON candidate_counts sum to {sum(candidate_counts)}, "
                f"but values has length {len(values)}: {root}"
            )
        source = str(data.get("method_name") or root.name)
        discount_factor = config.get("discount_factor")
        if discount_factor is not None:
            discount_factor = float(discount_factor)
        continuations = tuple(
            _expert_continuation_from_ranking_gt_value(
                value,
                discount_factor=discount_factor,
                source=source,
            )
            for value in values
        )
        return FlatRankingGTContinuationLookup(
            continuations=continuations,
            candidate_counts=candidate_counts,
            source_path=str(root),
        )

    manifest_path = root / "manifest.json"
    shards_dir = root / "shards"
    if not manifest_path.exists():
        raise FileNotFoundError(f"expert rollout store manifest not found: {manifest_path}")
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"expert rollout store shards dir not found: {shards_dir}")

    manifest = json.loads(manifest_path.read_text())
    shard_names = [
        str(shard_data["filename"])
        for shard_data in manifest.get("shards", [])
        if "filename" in shard_data
    ]
    if not shard_names:
        shard_names = [path.name for path in sorted(shards_dir.glob("shard_*.pkl"))]

    lookup: dict[tuple[int, int, int], ExpertContinuation] = {}
    for shard_name in shard_names:
        shard_path = shards_dir / shard_name
        with shard_path.open("rb") as f:
            payload = pickle.load(f)
        for key_text, trajectories in payload.get("trajectories", {}).items():
            key_parts = tuple(int(part) for part in str(key_text).split(":"))
            if len(key_parts) != 3:
                logger.debug(
                    "Skipping expert continuation key with unexpected shape: %s",
                    key_text,
                )
                continue
            if not trajectories:
                lookup[key_parts] = ExpertContinuation(
                    best_next_command=None,
                    commands=(),
                )
                continue
            trajectory = _select_expert_trajectory(trajectories)
            if trajectory is None:
                lookup[key_parts] = ExpertContinuation(
                    best_next_command=None,
                    commands=(),
                )
                continue
            lookup[key_parts] = _expert_continuation_from_stored_trajectory(
                trajectory,
                max_steps=max_steps,
            )
    return lookup


def _build_environment_system_guidance(context: MethodContext) -> str | None:
    family = _environment_family(context)
    if family == "open_apps":
        return (
            "For OpenApps, this SDPO signal is per-turn: the teacher may use only "
            "the immediate browser/page response after the candidate action. "
            "Treat that response as evidence about sparse task-completion "
            "progress in a synthetic web app, including correct app navigation, "
            "valid BrowserGym element interactions, exact input fidelity, UI "
            "validation or dead ends, and whether the resulting application "
            "state is closer to the target state described by the instruction. "
            "This SDPO prompt is text-only: use the task goal and serialized "
            "accessibility-tree observations, not screenshots."
        )
    if family == "alfworld":
        return (
            "For ALFWorld, this SDPO signal is per-turn: the teacher may use only "
            "the immediate response after the candidate command. Treat that "
            "response as evidence about sparse task-completion progress, command "
            "validity, precondition satisfaction, target-object handling, and "
            "future options from the resulting household state."
        )
    if family == "terminal_bench":
        return (
            "For TerminalBench, this SDPO signal is per-turn: the teacher may use "
            "only the immediate shell response after the candidate command. Treat "
            "that response as evidence about concrete progress toward the task "
            "and eventual verifier success, including useful file changes, "
            "targeted diagnostics, validation signals, and unresolved errors."
        )
    if family == "frozen_lake":
        return (
            "For FrozenLake, this SDPO signal is per-turn: the teacher may use "
            "only the immediate grid response after the candidate move. Treat "
            "that response as evidence about sparse goal-reaching progress, "
            "hole avoidance, shortest safe path progress, off-grid no-ops, "
            "backtracking, terminal success or failure, and the remaining step "
            "budget from the resulting grid state."
        )
    return None


def _build_history_block(
    history: list[Any] | None,
    *,
    history_truncated_from: int | None,
) -> str | None:
    if not history:
        return None

    history_lines = ["**State-Action History:**"]
    if history_truncated_from is not None and len(history) < history_truncated_from:
        history_lines.append(
            f"[Note: Only the last {len(history)} of "
            f"{history_truncated_from} turns are shown.]"
        )
    for i, turn in enumerate(history, 1):
        state_text, action_text = _history_turn_text(turn)
        history_lines.append(
            f"[Turn {i} - State]\n{state_text}\n[Turn {i} - Action]\n{action_text}"
        )
    return "\n\n".join(history_lines)


def _build_student_user_prompt(
    *,
    point: RankingPoint,
    context: MethodContext,
    history: list[Any] | None,
    history_truncated_from: int | None,
    state_text_override: str | None = None,
    inject_task_text: bool = False,
) -> str:
    parts = [
        _student_opening(context),
    ]
    history_block = _build_history_block(
        history, history_truncated_from=history_truncated_from,
    )
    if history_block:
        parts.append(history_block)

    state_text = _state_text_for_prompt(
        point,
        state_text_override=state_text_override,
        inject_task_text=inject_task_text,
    )
    parts.append(f"**Current State:**\n{state_text}")
    parts.append(_student_closing(context))
    return "\n\n".join(parts)


def _build_teacher_user_prompt(
    *,
    point: RankingPoint,
    context: MethodContext,
    history: list[Any] | None,
    history_truncated_from: int | None,
    feedback_text: str | None,
    state_text_override: str | None = None,
    inject_task_text: bool = False,
) -> str:
    parts = [
        _teacher_opening(context),
    ]
    history_block = _build_history_block(
        history, history_truncated_from=history_truncated_from,
    )
    if history_block:
        parts.append(history_block)

    state_text = _state_text_for_prompt(
        point,
        state_text_override=state_text_override,
        inject_task_text=inject_task_text,
    )
    parts.append(f"**Current State:**\n{state_text}")

    if feedback_text:
        parts.append(
            f"{_feedback_heading(context)}\n{feedback_text}"
        )
    else:
        parts.append(
            f"{_feedback_heading(context)}\n"
            "[No explicit feedback was available for this action.]"
        )

    parts.append(_teacher_closing(context))
    return "\n\n".join(parts)


def _build_short_expert_teacher_user_prompt(
    *,
    point: RankingPoint,
    context: MethodContext,
    history: list[Any] | None,
    history_truncated_from: int | None,
    feedback_text: str | None,
    expert_continuation: ExpertContinuation | None,
    state_text_override: str | None = None,
    inject_task_text: bool = False,
) -> str:
    family = _environment_family(context)
    if family not in {"open_apps", "alfworld", "terminal_bench", "frozen_lake"}:
        return _build_teacher_user_prompt(
            point=point,
            context=context,
            history=history,
            history_truncated_from=history_truncated_from,
            feedback_text=feedback_text,
            state_text_override=state_text_override,
            inject_task_text=inject_task_text,
        )

    parts = [
        _short_expert_teacher_opening(context),
    ]
    history_block = _build_history_block(
        history, history_truncated_from=history_truncated_from,
    )
    if history_block:
        parts.append(history_block)

    state_text = _state_text_for_prompt(
        point,
        state_text_override=state_text_override,
        inject_task_text=inject_task_text,
    )
    parts.append(f"**Current State:**\n{state_text}")

    if feedback_text:
        parts.append(f"{_feedback_heading(context)}\n{feedback_text}")
    else:
        parts.append(
            f"{_feedback_heading(context)}\n"
            "[No explicit feedback was available for this action.]"
        )

    if expert_continuation is not None:
        expert_block = expert_continuation.format_block(family=family)
    else:
        expert_block = "[No expert continuation was available for this candidate.]"
    parts.append(f"{_short_expert_heading(context)}\n{expert_block}")

    parts.append(_short_expert_teacher_closing(context))
    return "\n\n".join(parts)


class SDPORankingMethod:
    """Score ranking candidates with an offline SDPO-style teacher-student delta.

    For each candidate action, the student scores the original action under the
    current state/history. The self-teacher scores the same action after being
    conditioned on candidate-specific feedback, currently represented by the
    candidate's resulting ``next_state`` text. The final scalar is the mean
    per-token log-probability delta:

    ``mean_t [log p_teacher(a_t) - log p_student(a_t)]``.
    """

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        *,
        prompt_preset: PromptPreset,
        batch_size: int | None = None,
        max_history_turns: int | None = None,
        estimate_tokens: Callable[[str], int] | None = None,
        max_prompt_tokens: int | None = None,
        min_observation_chars: int | None = None,
        min_current_observation_chars: int | None = None,
        min_next_observation_chars: int | None = None,
        max_feedback_chars: int | None = None,
        inject_task_text: bool | None = None,
    ) -> None:
        if max_feedback_chars is not None and max_feedback_chars <= 0:
            raise ValueError("max_feedback_chars must be > 0")
        self._backend = backend
        self._scoring_backend = getattr(backend, "_backend", backend)
        self._context = context
        self._prompt_preset = prompt_preset
        self._batch_size = batch_size
        self._max_history_turns = max_history_turns
        self._estimate_tokens = estimate_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._min_observation_chars = min_observation_chars
        self._min_current_observation_chars = min_current_observation_chars
        self._min_next_observation_chars = min_next_observation_chars
        self._max_feedback_chars = max_feedback_chars
        self._inject_task_text = (
            _environment_family(context) == "open_apps"
            if inject_task_text is None
            else inject_task_text
        )
        self._system_content_tokens: int | None = None
        self.last_aborted_indices: set[int] = set()

        if not self._scoring_backend.capabilities.supports_full_scoring:
            raise ValueError(
                "SDPORankingMethod requires a backend with full scoring support "
                "(score_chat / score_chat_batch)."
            )

    @property
    def context(self) -> MethodContext:
        return self._context

    def _prepare_feedback_text(self, text: str) -> str:
        if (
            self._max_feedback_chars is None
            or _environment_family(self._context) != "terminal_bench"
        ):
            return text
        return _head_tail_truncate_text(
            text,
            self._max_feedback_chars,
            label="Shell output",
        )

    def score_batch(
        self,
        points: list[RankingPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not points:
            self.last_aborted_indices = set()
            return []

        total = len(points)
        point_scores: list[list[float]] = []
        aborted_points: set[int] = set()

        for point_idx, point in enumerate(points):
            try:
                scores, had_abort = self._score_point(point)
            except Exception:
                logger.exception(
                    "SDPORanking: scoring failed for point (traj=%d, step=%d)",
                    point.trajectory_index,
                    point.step_index,
                )
                scores = [math.nan] * len(point.candidates)
                had_abort = True

            if had_abort:
                aborted_points.add(point_idx)
            point_scores.append(scores)

            if progress_callback:
                progress_callback(point_idx + 1, total)

        self.last_aborted_indices = aborted_points
        if aborted_points:
            logger.warning(
                "SDPORanking: %d/%d points had one or more aborted candidates",
                len(aborted_points),
                total,
            )
        return point_scores

    def _score_point(self, point: RankingPoint) -> tuple[list[float], bool]:
        student_messages, teacher_messages = self._build_messages(point)

        candidate_texts = [
            action_text_for_display(
                candidate.action, candidate.resolved_action, candidate.extracted_action,
            )
            for candidate in point.candidates
        ]
        scores = [math.nan] * len(candidate_texts)
        valid_indices = [
            i for i, text in enumerate(candidate_texts)
            if text.strip()
        ]
        if not valid_indices:
            return scores, False

        continuations = [candidate_texts[i] for i in valid_indices]
        repeated_student = [student_messages for _ in valid_indices]
        point_teacher_messages = [teacher_messages[i] for i in valid_indices]

        student_results, student_abort = self._score_batch_messages(
            repeated_student, continuations,
        )
        teacher_results, teacher_abort = self._score_batch_messages(
            point_teacher_messages, continuations,
        )

        had_abort = student_abort or teacher_abort
        for local_idx, candidate_idx in enumerate(valid_indices):
            student_result = student_results[local_idx]
            teacher_result = teacher_results[local_idx]
            if student_result is None or teacher_result is None:
                continue
            scores[candidate_idx] = self._compute_sdpo_score(
                teacher_result, student_result,
            )

        return scores, had_abort

    def _score_batch_messages(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> tuple[list[Any | None], bool]:
        if not messages_batch:
            return [], False

        total = len(messages_batch)
        chunk_size = self._batch_size or total
        results: list[Any | None] = [None] * total
        had_abort = False

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_indices = list(range(start, end))
            chunk_messages = [messages_batch[i] for i in chunk_indices]
            chunk_conts = [continuations[i] for i in chunk_indices]

            try:
                chunk_results = self._scoring_backend.score_chat_batch(
                    chunk_messages, chunk_conts,
                )
            except Exception as exc:
                if not is_recoverable_backend_error(exc):
                    raise
                had_abort = True
                offending = getattr(exc, "offending_indices", None)
                if offending and len(offending) < len(chunk_indices):
                    offending_set = set(offending)
                    retry_local = [
                        i for i in range(len(chunk_indices))
                        if i not in offending_set
                    ]
                    retry_messages = [chunk_messages[i] for i in retry_local]
                    retry_conts = [chunk_conts[i] for i in retry_local]
                    try:
                        retry_results = self._scoring_backend.score_chat_batch(
                            retry_messages, retry_conts,
                        )
                    except Exception as retry_exc:
                        if not is_recoverable_backend_error(retry_exc):
                            raise
                        continue
                    for local_idx, result in zip(retry_local, retry_results, strict=True):
                        abs_idx = chunk_indices[local_idx]
                        results[abs_idx] = result
                    continue
                continue

            for local_idx, result in enumerate(chunk_results):
                abs_idx = chunk_indices[local_idx]
                results[abs_idx] = result

        return results, had_abort

    def _compute_sdpo_score(self, teacher_result: Any, student_result: Any) -> float:
        teacher_scores = teacher_result.token_scores
        student_scores = student_result.token_scores
        if not teacher_scores or not student_scores:
            return math.nan
        if len(teacher_scores) != len(student_scores):
            logger.debug(
                "SDPORanking: token-length mismatch teacher=%d student=%d",
                len(teacher_scores),
                len(student_scores),
            )
            return math.nan

        deltas = [
            float(t.logprob) - float(s.logprob)
            for t, s in zip(teacher_scores, student_scores, strict=True)
        ]
        return sum(deltas) / len(deltas)

    def _build_messages(
        self,
        point: RankingPoint,
    ) -> tuple[list[ChatMessage], list[list[ChatMessage]]]:
        system_content = self._build_system_prompt()
        history = list(point.history)
        history_truncated_from: int | None = None
        if self._max_history_turns is not None and len(history) > self._max_history_turns:
            history_truncated_from = len(history)
            history = (
                history[-self._max_history_turns:]
                if self._max_history_turns > 0 else []
            )

        state_text_override: str | None = None
        feedback_overrides: list[str | None] | None = None
        if (
            self._estimate_tokens is not None
            and self._max_prompt_tokens is not None
        ):
            history, state_text_override, feedback_overrides = (
                self._apply_budget_truncation(
                    history,
                    system_content,
                    point,
                    history_truncated_from,
                )
            )

        student_messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(
                role="user",
                content=_build_student_user_prompt(
                    point=point,
                    context=self._context,
                    history=history,
                    history_truncated_from=history_truncated_from,
                    state_text_override=state_text_override,
                    inject_task_text=self._inject_task_text,
                ),
            ),
        ]

        teacher_messages: list[list[ChatMessage]] = []
        for idx, candidate in enumerate(point.candidates):
            feedback_override = (
                feedback_overrides[idx] if feedback_overrides is not None else None
            )
            feedback_text = None
            if self._context.include_next_state and candidate.next_state is not None:
                if feedback_override is not None:
                    feedback_text = feedback_override
                else:
                    feedback_text = self._prepare_feedback_text(
                        serialize_observation(candidate.next_state),
                    )
            teacher_messages.append([
                ChatMessage(role="system", content=system_content),
                ChatMessage(
                    role="user",
                    content=_build_teacher_user_prompt(
                        point=point,
                        context=self._context,
                        history=history,
                        history_truncated_from=history_truncated_from,
                        feedback_text=feedback_text,
                        state_text_override=state_text_override,
                        inject_task_text=self._inject_task_text,
                    ),
                ),
            ])

        return student_messages, teacher_messages

    def _build_system_prompt(self) -> str:
        role_opener = resolve_block(
            self._prompt_preset.role_opener_ranking,
            signal_name=SIGNAL_TYPE_NAMES[SignalType.Q_VALUE],
        )
        parts: list[object] = [
            role_opener,
            build_signal_type_fragment(
                SignalType.Q_VALUE,
                self._context.policy_assumption,
                self._context.discount_factor,
                disclose_discount_factor=self._context.disclose_discount_factor,
            ),
            (
                "You will evaluate one fixed candidate action at a time. "
                "The base policy scores the candidate from the original state. "
                "A self-teacher may additionally see environment feedback produced "
                "after that same action was attempted, and should use that feedback "
                "as per-turn evidence about whether the original action improved "
                "expected future success from the current state."
            ),
        ]

        env_guidance = _build_environment_system_guidance(self._context)
        if env_guidance is not None:
            parts.append(env_guidance)

        guidance = resolve_block(self._prompt_preset.approximation_guidance)
        if guidance is not None:
            parts.append(guidance)
        if self._context.efficiency_guidance:
            parts.append(self._context.efficiency_guidance)
        if self._context.task_description:
            parts.append(f"## Task Description\n{self._context.task_description.rstrip()}")
        env_notes = _format_environment_notes(self._context.environment_notes)
        if env_notes:
            parts.append(env_notes)
        if self._context.reward_description:
            parts.append(
                f"## Reward Functions\n{self._context.reward_description.rstrip()}"
            )
        return compose_system_prompt(*parts)

    def _apply_budget_truncation(
        self,
        history: list[Any],
        system_content: str,
        point: RankingPoint,
        history_truncated_from: int | None,
    ) -> tuple[list[Any], str | None, list[str | None] | None]:
        from qval.token_budget import (
            budget_aware_truncation,
            truncate_text_to_token_savings,
        )

        assert self._estimate_tokens is not None
        assert self._max_prompt_tokens is not None

        if self._system_content_tokens is None:
            self._system_content_tokens = self._estimate_tokens(system_content)

        state_text = serialize_observation(point.state)
        feedback_texts: list[str | None] = []
        for candidate in point.candidates:
            if self._context.include_next_state and candidate.next_state is not None:
                feedback_texts.append(
                    self._prepare_feedback_text(
                        serialize_observation(candidate.next_state)
                    )
                )
            else:
                feedback_texts.append(None)

        def _estimate_teacher_total(
            history_value: list[Any],
            *,
            state_override: str | None = None,
            feedback_override: str | None = None,
        ) -> int:
            user_content = _build_teacher_user_prompt(
                point=point,
                context=self._context,
                history=history_value,
                history_truncated_from=history_truncated_from,
                feedback_text=feedback_override,
                state_text_override=state_override,
                inject_task_text=self._inject_task_text,
            )
            return self._system_content_tokens + self._estimate_tokens(user_content)

        longest_feedback = max(
            (text for text in feedback_texts if text),
            key=len,
            default=None,
        )
        available = max(
            0,
            self._max_prompt_tokens - _estimate_teacher_total(
                [],
                state_override=state_text,
                feedback_override=longest_feedback,
            ),
        )
        if history:
            history = budget_aware_truncation(
                history,
                available,
                min_observation_chars=self._min_observation_chars,
                min_action_chars=None,
                estimate_tokens=self._estimate_tokens,
            )

        feedback_overrides: list[str | None] | None = None
        current_feedbacks = list(feedback_texts)
        point_totals = [
            _estimate_teacher_total(
                history,
                state_override=state_text,
                feedback_override=feedback_text,
            )
            for feedback_text in current_feedbacks
        ]

        if (
            point_totals
            and max(point_totals) > self._max_prompt_tokens
            and self._min_next_observation_chars is not None
        ):
            for idx, feedback_text in enumerate(current_feedbacks):
                if feedback_text is None:
                    continue
                excess = point_totals[idx] - self._max_prompt_tokens
                if excess <= 0:
                    continue
                truncated = truncate_text_to_token_savings(
                    feedback_text,
                    excess,
                    self._min_next_observation_chars,
                    self._estimate_tokens,
                )
                current_feedbacks[idx] = truncated
            feedback_overrides = current_feedbacks
            point_totals = [
                _estimate_teacher_total(
                    history,
                    state_override=state_text,
                    feedback_override=feedback_text,
                )
                for feedback_text in current_feedbacks
            ]

        state_text_override: str | None = None
        if (
            point_totals
            and max(point_totals) > self._max_prompt_tokens
            and self._min_current_observation_chars is not None
        ):
            max_excess = max(point_totals) - self._max_prompt_tokens
            candidate_state_text = truncate_text_to_token_savings(
                state_text,
                max_excess,
                self._min_current_observation_chars,
                self._estimate_tokens,
            )
            if candidate_state_text != state_text:
                state_text_override = candidate_state_text

        return history, state_text_override, feedback_overrides


class SDPOShortExpertContinuationRankingMethod(SDPORankingMethod):
    """SDPO ranking with next-step expert rollout summary teacher context."""

    def __init__(
        self,
        backend: ModelBackend,
        context: MethodContext,
        *,
        prompt_preset: PromptPreset,
        expert_rollout_store: str | Path,
        max_expert_continuation_steps: int = 5,
        batch_size: int | None = None,
        max_history_turns: int | None = None,
        estimate_tokens: Callable[[str], int] | None = None,
        max_prompt_tokens: int | None = None,
        min_observation_chars: int | None = None,
        min_current_observation_chars: int | None = None,
        min_next_observation_chars: int | None = None,
        max_feedback_chars: int | None = None,
    ) -> None:
        if max_expert_continuation_steps < 0:
            raise ValueError("max_expert_continuation_steps must be >= 0")
        family = _environment_family(context)
        if family not in {
            "open_apps",
            "alfworld",
            "terminal_bench",
            "frozen_lake",
        }:
            raise ValueError(
                "SDPOShortExpertContinuationRankingMethod currently supports "
                "OpenApps, ALFWorld, TerminalBench, and FrozenLake only"
            )
        super().__init__(
            backend=backend,
            context=context,
            prompt_preset=prompt_preset,
            batch_size=batch_size,
            max_history_turns=max_history_turns,
            estimate_tokens=estimate_tokens,
            max_prompt_tokens=max_prompt_tokens,
            min_observation_chars=min_observation_chars,
            min_current_observation_chars=min_current_observation_chars,
            min_next_observation_chars=min_next_observation_chars,
            max_feedback_chars=max_feedback_chars,
        )
        self.expert_rollout_store = str(expert_rollout_store)
        self.max_expert_continuation_steps = max_expert_continuation_steps
        self._expert_lookup = load_expert_continuation_lookup(
            expert_rollout_store,
            max_steps=max_expert_continuation_steps,
        )
        logger.info(
            "Loaded %d expert continuations from %s",
            len(self._expert_lookup),
            self.expert_rollout_store,
        )

    def score_batch(
        self,
        points: list[RankingPoint],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if isinstance(self._expert_lookup, FlatRankingGTContinuationLookup):
            self._expert_lookup = self._expert_lookup.materialize(points)
            logger.info(
                "Materialized %d flat ranking GT continuations from %s",
                len(self._expert_lookup),
                self.expert_rollout_store,
            )
        return super().score_batch(points, progress_callback=progress_callback)

    def _build_system_prompt(self) -> str:
        role_opener = resolve_block(
            self._prompt_preset.role_opener_ranking,
            signal_name=SIGNAL_TYPE_NAMES[SignalType.Q_VALUE],
        )
        parts: list[object] = [
            role_opener,
            build_signal_type_fragment(
                SignalType.Q_VALUE,
                self._context.policy_assumption,
                self._context.discount_factor,
                disclose_discount_factor=self._context.disclose_discount_factor,
            ),
            (
                "You will evaluate one fixed candidate action at a time. "
                "The base policy scores the candidate from the original state. "
                "A self-teacher may additionally see environment feedback produced "
                "after that same action was attempted, the next expert action "
                "from the resulting state, and a compact stored expert rollout "
                "summary. It should use that privileged evidence to judge whether "
                "the original action improved expected future success from the "
                "current state."
            ),
        ]
        family = _environment_family(self._context)
        if family == "open_apps":
            parts.append(
                "For OpenApps, this SDPO variant is an oracle-teacher ablation: "
                "the teacher may use the immediate page response after the "
                "candidate browser action, the next expert browser action from "
                "the resulting browser state, and a structured stored rollout "
                "summary. Treat that evidence as information about target-app "
                "navigation, valid BrowserGym element interaction, exact input "
                "fidelity, UI validation or dead ends, remaining distance to task "
                "success, and useful future browser state."
            )
        elif family == "alfworld":
            parts.append(
                "For ALFWorld, this SDPO variant is an oracle-teacher ablation: "
                "the teacher may use the immediate response after the candidate "
                "command, the next scripted expert command from the resulting "
                "household state, and a structured stored rollout summary. Treat "
                "that evidence as information about sparse task-completion "
                "progress, command validity, precondition satisfaction, "
                "target-object handling, remaining distance to the goal, and "
                "future options."
            )
        elif family == "terminal_bench":
            parts.append(
                "For TerminalBench, this SDPO variant is an oracle-teacher "
                "ablation: the teacher may use the immediate shell response after "
                "the candidate command, the next expert shell command from the "
                "resulting persistent shell state, and a structured stored "
                "verifier outcome summary. Treat that evidence as information "
                "about durable task progress, useful diagnostics, validation "
                "signals, shell-state preservation, remaining distance to "
                "verifier success, and unresolved errors."
            )
        elif family == "frozen_lake":
            parts.append(
                "For FrozenLake, this SDPO variant is an oracle-teacher ablation: "
                "the teacher may use the immediate grid response after the "
                "candidate move, the next expert move from the resulting grid "
                "state, and a structured stored rollout summary. Treat that "
                "evidence as information about goal-reaching progress, hole "
                "avoidance, off-grid no-ops, backtracking, remaining safe path "
                "distance to the goal, and terminal success or failure."
            )

        guidance = resolve_block(self._prompt_preset.approximation_guidance)
        if guidance is not None:
            parts.append(guidance)
        if self._context.efficiency_guidance:
            parts.append(self._context.efficiency_guidance)
        if self._context.task_description:
            parts.append(f"## Task Description\n{self._context.task_description.rstrip()}")
        env_notes = _format_environment_notes(self._context.environment_notes)
        if env_notes:
            parts.append(env_notes)
        if self._context.reward_description:
            parts.append(
                f"## Reward Functions\n{self._context.reward_description.rstrip()}"
            )
        return compose_system_prompt(*parts)

    def _fit_expert_continuation_to_budget(
        self,
        *,
        system_content: str,
        point: RankingPoint,
        history: list[Any] | None,
        history_truncated_from: int | None,
        feedback_text: str | None,
        expert_continuation: ExpertContinuation | None,
        state_text_override: str | None,
    ) -> ExpertContinuation | None:
        if (
            expert_continuation is None
            or self._estimate_tokens is None
            or self._max_prompt_tokens is None
        ):
            return expert_continuation

        if self._system_content_tokens is None:
            self._system_content_tokens = self._estimate_tokens(system_content)

        current = expert_continuation
        while True:
            user_content = _build_short_expert_teacher_user_prompt(
                point=point,
                context=self._context,
                history=history,
                history_truncated_from=history_truncated_from,
                feedback_text=feedback_text,
                expert_continuation=current,
                state_text_override=state_text_override,
                inject_task_text=self._inject_task_text,
            )
            total_tokens = self._system_content_tokens + self._estimate_tokens(
                user_content,
            )
            if total_tokens <= self._max_prompt_tokens or not current.commands:
                return current
            current = ExpertContinuation(
                best_next_command=current.best_next_command,
                commands=current.commands[:-1],
                reached_goal=current.reached_goal,
                steps_to_success=current.steps_to_success,
                observed_expert_steps=current.observed_expert_steps,
                final_reward=current.final_reward,
                ranking_gt_value=current.ranking_gt_value,
                ranking_gt_source=current.ranking_gt_source,
            )

    def _build_messages(
        self,
        point: RankingPoint,
    ) -> tuple[list[ChatMessage], list[list[ChatMessage]]]:
        system_content = self._build_system_prompt()
        history = list(point.history)
        history_truncated_from: int | None = None
        if self._max_history_turns is not None and len(history) > self._max_history_turns:
            history_truncated_from = len(history)
            history = (
                history[-self._max_history_turns:]
                if self._max_history_turns > 0 else []
            )

        state_text_override: str | None = None
        feedback_overrides: list[str | None] | None = None
        if (
            self._estimate_tokens is not None
            and self._max_prompt_tokens is not None
        ):
            history, state_text_override, feedback_overrides = (
                self._apply_budget_truncation(
                    history,
                    system_content,
                    point,
                    history_truncated_from,
                )
            )

        student_messages = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(
                role="user",
                content=_build_student_user_prompt(
                    point=point,
                    context=self._context,
                    history=history,
                    history_truncated_from=history_truncated_from,
                    state_text_override=state_text_override,
                    inject_task_text=self._inject_task_text,
                ),
            ),
        ]

        teacher_messages: list[list[ChatMessage]] = []
        for idx, candidate in enumerate(point.candidates):
            feedback_override = (
                feedback_overrides[idx] if feedback_overrides is not None else None
            )
            feedback_text = None
            if self._context.include_next_state and candidate.next_state is not None:
                feedback_text = feedback_override or serialize_observation(
                    candidate.next_state,
                )
            expert_continuation = self._expert_lookup.get(
                (point.trajectory_index, point.step_index, idx)
            )
            expert_continuation = self._fit_expert_continuation_to_budget(
                system_content=system_content,
                point=point,
                history=history,
                history_truncated_from=history_truncated_from,
                feedback_text=feedback_text,
                expert_continuation=expert_continuation,
                state_text_override=state_text_override,
            )
            teacher_messages.append([
                ChatMessage(role="system", content=system_content),
                ChatMessage(
                    role="user",
                    content=_build_short_expert_teacher_user_prompt(
                        point=point,
                        context=self._context,
                        history=history,
                        history_truncated_from=history_truncated_from,
                        feedback_text=feedback_text,
                        expert_continuation=expert_continuation,
                        state_text_override=state_text_override,
                        inject_task_text=self._inject_task_text,
                    ),
                ),
            ])

        return student_messages, teacher_messages

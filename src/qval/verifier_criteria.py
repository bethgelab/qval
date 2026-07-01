"""Built-in criterion registry for verifier-style prompting."""

from __future__ import annotations

from dataclasses import dataclass

from qval.types import SignalType


@dataclass(frozen=True)
class VerifierCriterion:
    """Built-in verifier criterion definition."""

    id: str
    name: str
    state_value_description: str
    q_value_description: str

    def description_for_signal(self, signal_type: SignalType) -> str:
        if signal_type == SignalType.STATE_VALUE:
            return self.state_value_description
        if signal_type == SignalType.Q_VALUE:
            return self.q_value_description
        raise ValueError(
            "Verifier criteria currently support only state_value and q_value, "
            f"got {signal_type.name.lower()!r}"
        )


_REGISTRY: dict[str, VerifierCriterion] = {
    "terminal_bench_correctness": VerifierCriterion(
        id="terminal_bench_correctness",
        name="Correctness",
        state_value_description=(
            "Judge how likely the current state of the trajectory is to lead "
            "to a correct final solution that satisfies the task requirements. "
            "Reward states that already show strong evidence of correct setup, "
            "correct intermediate results, or a clear path to a valid final output."
        ),
        q_value_description=(
            "Judge how much the chosen action and resulting next state improve "
            "the chances of reaching a correct final solution that satisfies the "
            "task requirements. Reward actions that move the agent toward the "
            "right files, commands, outputs, or validation steps."
        ),
    ),
    "terminal_bench_error_detection": VerifierCriterion(
        id="terminal_bench_error_detection",
        name="Error Detection",
        state_value_description=(
            "Judge whether the current state reveals unresolved failure signals "
            "or remains free of them. Penalize states that already contain clear "
            "errors, contradictions, or evidence that the trajectory is heading "
            "toward a broken solution."
        ),
        q_value_description=(
            "Judge whether the chosen action helps detect, avoid, or resolve "
            "errors rather than introducing or ignoring them. Penalize actions "
            "that are likely to produce unresolved failures, misleading output, "
            "or brittle workarounds."
        ),
    ),
    "terminal_bench_efficiency": VerifierCriterion(
        id="terminal_bench_efficiency",
        name="Efficiency",
        state_value_description=(
            "Judge how much useful progress the current state represents toward "
            "solving the task with minimal wasted steps. Reward states that are "
            "close to a validated solution and penalize states that suggest the "
            "agent is stalled, repeating work, or exploring unproductive paths."
        ),
        q_value_description=(
            "Judge how much the chosen action and resulting next state advance "
            "the solution efficiently. Reward actions that make concrete forward "
            "progress and penalize actions that are redundant, low-yield, or likely "
            "to waste turns."
        ),
    ),
    "terminal_bench_resource_usage": VerifierCriterion(
        id="terminal_bench_resource_usage",
        name="Resource Usage",
        state_value_description=(
            "Judge whether the current state reflects disciplined resource use. "
            "Penalize states that suggest the trajectory is relying on excessively "
            "expensive commands, noisy output, or other wasteful behavior that makes "
            "successful completion less reliable."
        ),
        q_value_description=(
            "Judge whether the chosen action uses compute, memory, runtime, and "
            "terminal output responsibly. Reward actions that are targeted and "
            "economical, and penalize actions that are likely to be expensive, "
            "spammy, or unnecessary."
        ),
    ),
    "frozen_lake_correctness": VerifierCriterion(
        id="frozen_lake_correctness",
        name="Correctness",
        state_value_description=(
            "Judge how likely the current position is to lead to the goal (G) "
            "within the remaining step budget while staying on safe ice. Reward "
            "positions that still have a clear, unobstructed route to the goal; "
            "penalize positions that are off-path, cornered, or already terminal "
            "without having reached the goal."
        ),
        q_value_description=(
            "Judge how much the chosen action and resulting next position improve "
            "the chances of reaching the goal (G). Reward actions that step along "
            "a valid path from the current position toward the goal; penalize "
            "actions that step away from the goal or trap the agent in a "
            "dead-end region of the grid."
        ),
    ),
    "frozen_lake_hazard_avoidance": VerifierCriterion(
        id="frozen_lake_hazard_avoidance",
        name="Hazard Avoidance",
        state_value_description=(
            "Judge whether the current position is safely on ice rather than "
            "adjacent to or surrounded by holes (H). Penalize positions whose "
            "neighboring cells make an accidental fall likely and heavily "
            "penalize terminal positions that were reached by falling into a "
            "hole; reward positions with a safety margin around them."
        ),
        q_value_description=(
            "Judge whether the chosen action avoids stepping into a hole (H) or "
            "maneuvering into a cell tightly surrounded by holes. Reward actions "
            "that keep the agent on ice and move it toward hole-free regions; "
            "heavily penalize actions whose resulting cell is a hole."
        ),
    ),
    "frozen_lake_path_optimality": VerifierCriterion(
        id="frozen_lake_path_optimality",
        name="Path Optimality",
        state_value_description=(
            "Judge how close the current position is to the goal (G) along an "
            "optimal path, relative to the remaining steps. Reward positions that "
            "are few moves from the goal with ample step budget left; penalize "
            "positions that are far from the goal, have already consumed many "
            "steps, or sit on a detour relative to the shortest path."
        ),
        q_value_description=(
            "Judge whether the chosen action shortens the remaining distance to "
            "the goal along a valid path rather than backtracking, detouring, or "
            "oscillating. Reward actions that make monotonic progress toward the "
            "goal; penalize actions that move away, bounce off the grid edge, or "
            "return to a previously visited cell."
        ),
    ),
    "webshop_correctness": VerifierCriterion(
        id="webshop_correctness",
        name="Correctness",
        state_value_description=(
            "Judge how likely the current state of the shopping trajectory is to "
            "end in a purchase that matches the instruction's product category, "
            "attributes, options, and price. Reward states that are viewing or "
            "selecting products aligned with all stated criteria; penalize "
            "states that have committed to off-target listings or options."
        ),
        q_value_description=(
            "Judge how much the chosen action and resulting next state improve "
            "the chances of purchasing a product that matches the instruction. "
            "Reward actions that surface better-aligned listings, select matching "
            "options, or advance to checkout on a matching product; penalize "
            "actions that commit to mismatched products, options, or prices."
        ),
    ),
    "webshop_error_detection": VerifierCriterion(
        id="webshop_error_detection",
        name="Error Detection",
        state_value_description=(
            "Judge whether the current state reveals an alignment failure with "
            "the instruction — wrong category, missing required attributes, "
            "options that violate the request, or a price outside the stated "
            "constraints. Penalize states that already show such mismatches or "
            "have dismissed constraints; reward states that remain consistent "
            "with the full instruction."
        ),
        q_value_description=(
            "Judge whether the chosen action helps avoid or correct alignment "
            "failures rather than introducing them. Penalize clicks that select "
            "mismatched options, navigate toward wrong-category listings, or "
            "accept an out-of-budget product; reward actions that filter, "
            "adjust, or back out to preserve alignment with the instruction."
        ),
    ),
    "webshop_efficiency": VerifierCriterion(
        id="webshop_efficiency",
        name="Efficiency",
        state_value_description=(
            "Judge how much useful progress the current state represents toward "
            "a completed matching purchase. Reward states that are close to a "
            "'Buy Now' on a matching product with all required options selected; "
            "penalize states that remain on generic search pages, repeat already "
            "seen results, or stall without narrowing the candidate set."
        ),
        q_value_description=(
            "Judge whether the chosen action advances the purchase efficiently. "
            "Reward actions that take the agent closer to 'Buy Now' on a matching "
            "product — narrowing search, selecting options, or confirming "
            "checkout; penalize actions that re-run broad searches, revisit "
            "rejected listings, or idle on unrelated pages."
        ),
    ),
    "webshop_search_discipline": VerifierCriterion(
        id="webshop_search_discipline",
        name="Search Discipline",
        state_value_description=(
            "Judge whether the current state reflects disciplined use of search "
            "and navigation — a focused query history, targeted option selection, "
            "and minimal noisy exploration. Penalize states that show repeated "
            "vague queries, large unread product lists, or navigation churn; "
            "reward states whose trajectory so far has been purposeful."
        ),
        q_value_description=(
            "Judge whether the chosen action uses the limited shopping step "
            "budget wisely. Reward specific, instruction-grounded search queries, "
            "option clicks on matching listings, and decisive filtering; penalize "
            "broad or duplicate queries, random scrolling, and unnecessary "
            "back-and-forth between unrelated pages."
        ),
    ),
    "open_apps_correctness": VerifierCriterion(
        id="open_apps_correctness",
        name="Correctness",
        state_value_description=(
            "Judge how likely the current state is to reach the app-specific "
            "target state specified by the task (e.g., the right calendar event "
            "added, the right todo item created, the right message sent). Reward "
            "states that are in the correct application, on the correct page, "
            "with in-progress changes aligned with the instruction; penalize "
            "states in the wrong app or with off-target changes."
        ),
        q_value_description=(
            "Judge how much the chosen action and resulting next state advance "
            "the target application state. Reward actions that create, edit, or "
            "confirm the correct entity (event, todo, message, etc.) using values "
            "from the instruction; penalize actions that create wrong entities, "
            "stray into unrelated apps, or discard in-progress work."
        ),
    ),
    "open_apps_error_detection": VerifierCriterion(
        id="open_apps_error_detection",
        name="Error Detection",
        state_value_description=(
            "Judge whether the current state reveals UI or interaction failures "
            "— form-validation errors, failed clicks, wrong-app navigation, "
            "dismissed modals, or broken flows. Penalize states that show error "
            "messages, empty required fields on submit, or navigation dead-ends; "
            "reward clean states consistent with the task goal."
        ),
        q_value_description=(
            "Judge whether the chosen action avoids or resolves UI interaction "
            "failures. Penalize actions that submit incomplete forms, click "
            "disabled widgets, or are likely to trigger validation errors; "
            "reward actions that address error states, correct mistyped fields, "
            "or back out of wrong-app navigation."
        ),
    ),
    "open_apps_efficiency": VerifierCriterion(
        id="open_apps_efficiency",
        name="Efficiency",
        state_value_description=(
            "Judge how much useful progress the current state represents toward "
            "goal completion, relative to the step budget. Reward states that "
            "have completed most setup and are close to a final confirming "
            "action; penalize states that linger on landing pages, scroll "
            "without interaction, or cycle through repeated navigation loops."
        ),
        q_value_description=(
            "Judge whether the chosen action makes concrete progress toward the "
            "goal. Reward clicks, fills, or submissions that directly change "
            "application state in the right direction; penalize redundant "
            "scrolling, reopening already-visited pages, or re-clicking widgets "
            "that produced no effect previously."
        ),
    ),
    "open_apps_input_fidelity": VerifierCriterion(
        id="open_apps_input_fidelity",
        name="Input Fidelity",
        state_value_description=(
            "Judge whether the values entered so far — form fields, selected "
            "widgets, clicked entities — faithfully reflect the instruction's "
            "names, dates, times, recipients, and text. Penalize states where "
            "fields hold approximate, placeholder, or partially-correct values; "
            "reward states where all committed inputs match the instruction "
            "exactly."
        ),
        q_value_description=(
            "Judge whether the chosen action enters values that exactly match "
            "the instruction — correct recipient, date, time, name, or text "
            "string, character-for-character where applicable. Reward precise "
            "fills and clicks on exactly-matching elements; penalize typos, "
            "approximations, wrong-case strings, or selections of "
            "adjacent-but-wrong options."
        ),
    ),
    "alfworld_correctness": VerifierCriterion(
        id="alfworld_correctness",
        name="Correctness",
        state_value_description=(
            "Judge how likely the current state is to reach the household goal "
            "— the target object at the target receptacle, in the target state "
            "(e.g., clean, heated, cooled). Reward states where the agent is "
            "holding or has placed the correct object near the correct "
            "receptacle; penalize states focused on wrong objects, wrong rooms, "
            "or wrong target locations."
        ),
        q_value_description=(
            "Judge how much the chosen action and resulting next state advance "
            "the household goal. Reward actions that pick up the target object, "
            "move toward the target receptacle, or apply the required state "
            "change (clean, heat, cool); penalize actions that target wrong "
            "objects, travel to irrelevant locations, or undo prior correct "
            "progress."
        ),
    ),
    "alfworld_error_detection": VerifierCriterion(
        id="alfworld_error_detection",
        name="Error Detection",
        state_value_description=(
            "Judge whether the current state reveals invalid commands, failed "
            "manipulations, or contradictions with the environment (e.g., trying "
            "to place an object the agent is not holding, or interacting with a "
            "closed receptacle without opening it). Penalize states whose last "
            "transition was a 'Nothing happens' or equivalent no-op; reward "
            "states with clean, effectful transitions."
        ),
        q_value_description=(
            "Judge whether the chosen action is a valid, effectful command given "
            "the current state. Penalize actions likely to fail — manipulating "
            "unheld or out-of-reach objects, using wrong receptacles, or issuing "
            "malformed commands; reward actions whose preconditions are "
            "satisfied and that will produce a meaningful environment change."
        ),
    ),
    "alfworld_efficiency": VerifierCriterion(
        id="alfworld_efficiency",
        name="Efficiency",
        state_value_description=(
            "Judge how much useful progress the current state represents toward "
            "goal completion within the step budget. Reward states that have "
            "already located the target object and are close to the target "
            "receptacle with any required state changes done; penalize states "
            "that still explore irrelevant rooms, revisit already-searched "
            "receptacles, or cycle between locations."
        ),
        q_value_description=(
            "Judge whether the chosen action advances the task efficiently. "
            "Reward actions that directly search likely containers, move between "
            "task-relevant locations, or execute the final placement; penalize "
            "actions that revisit empty containers, re-enter rooms unnecessarily, "
            "or detour into irrelevant inspections."
        ),
    ),
    "alfworld_precondition_awareness": VerifierCriterion(
        id="alfworld_precondition_awareness",
        name="Precondition Awareness",
        state_value_description=(
            "Judge whether the current state reflects good object- and "
            "receptacle-handling discipline — opening containers before looking "
            "inside, picking up before placing, and applying state changes at "
            "the right appliance with the right object in hand. Penalize states "
            "that just attempted an action without meeting its precondition; "
            "reward states that sequence operations correctly."
        ),
        q_value_description=(
            "Judge whether the chosen action respects ALFWorld's object-"
            "manipulation preconditions. Reward actions that open containers "
            "before inspecting, pick up before placing, bring the right object "
            "to the right appliance, and sequence unlock/open/close correctly; "
            "penalize actions that skip prerequisites (e.g., place without "
            "pickup, clean without a sink, cool without a fridge)."
        ),
    ),
}


def list_verifier_criteria() -> list[str]:
    """Return the sorted built-in criterion IDs."""
    return sorted(_REGISTRY)


def get_verifier_criterion(criterion_id: str) -> VerifierCriterion:
    """Return a built-in verifier criterion by ID."""
    try:
        return _REGISTRY[criterion_id]
    except KeyError:
        raise KeyError(
            f"Unknown verifier criterion {criterion_id!r}. "
            f"Available: {list_verifier_criteria()}"
        ) from None

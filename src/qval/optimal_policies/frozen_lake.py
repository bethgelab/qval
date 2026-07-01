"""Exact scripted policies for FrozenLake environments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qval.optimal_policies import register_policy

_FROZEN_LAKE_MAPS: dict[str, tuple[str, ...]] = {
    "4x4": (
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ),
    "8x8": (
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
}

_ACTION_ORDER: tuple[str, ...] = ("left", "down", "right", "up")
_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "left": (0, -1),
    "down": (1, 0),
    "right": (0, 1),
    "up": (-1, 0),
}
_SLIP_ACTIONS: dict[str, tuple[str, str, str]] = {
    "left": ("up", "left", "down"),
    "down": ("left", "down", "right"),
    "right": ("down", "right", "up"),
    "up": ("left", "up", "right"),
}
_TIE_BREAK_PRIORITY: dict[str, int] = {
    "right": 0,
    "down": 1,
    "left": 2,
    "up": 3,
}
_TOL = 1e-12
_MAX_ITERS = 10_000


@dataclass(frozen=True)
class FrozenLakeSpec:
    desc: tuple[str, ...]
    is_slippery: bool
    success_rate: float
    reward_schedule: tuple[float, float, float]
    step_penalty: float
    agent_pos: tuple[int, int]


def frozen_lake_optimal_action(state: str, metadata: dict[str, Any]) -> str:
    """Return the action that is optimal for the resolved FrozenLake MDP.

    The planner maximizes expected return under the underlying stationary MDP
    defined by the current map and transition parameters. When multiple actions
    tie on value, it prefers the one with the smallest expected number of steps
    to termination; remaining ties are broken deterministically.
    """
    spec = _resolve_spec(state, metadata)
    row, col = spec.agent_pos
    current_tile = spec.desc[row][col]
    if current_tile in {"G", "H"}:
        raise ValueError(f"FrozenLake policy called on terminal tile {current_tile!r}")

    action = _solve_optimal_action(
        spec.desc,
        spec.is_slippery,
        spec.success_rate,
        spec.reward_schedule,
        spec.step_penalty,
        spec.agent_pos,
    )
    return action


@lru_cache(maxsize=128)
def _solve_optimal_action(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    agent_pos: tuple[int, int],
) -> str:
    values = _compute_optimal_values(
        desc,
        is_slippery,
        success_rate,
        reward_schedule,
        step_penalty,
    )
    optimal_actions = _compute_optimal_actions(
        desc,
        is_slippery,
        success_rate,
        reward_schedule,
        step_penalty,
        values,
    )
    steps = _compute_min_steps(
        desc,
        is_slippery,
        success_rate,
        reward_schedule,
        step_penalty,
        optimal_actions,
    )

    state_index = _to_index(agent_pos, len(desc[0]))
    return _select_action(
        desc,
        is_slippery,
        success_rate,
        reward_schedule,
        step_penalty,
        state_index,
        values,
        optimal_actions,
        steps,
    )


def _resolve_spec(state: str, metadata: dict[str, Any]) -> FrozenLakeSpec:
    metadata_desc = _resolve_desc(metadata)
    expected_shape = None
    if metadata_desc is not None:
        expected_shape = (len(metadata_desc), len(metadata_desc[0]))
    grid = _parse_grid(state, expected_shape)
    agent_pos = _find_agent(grid)
    desc = _resolve_board(grid, metadata_desc, agent_pos)

    is_slippery = bool(metadata.get("is_slippery", True))
    success_rate = float(metadata.get("success_rate", 1.0 / 3.0))
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("FrozenLake success_rate must be in [0, 1]")

    reward_schedule_raw = metadata.get("reward_schedule", (1.0, 0.0, 0.0))
    reward_schedule = tuple(float(x) for x in reward_schedule_raw)
    if len(reward_schedule) != 3:
        raise ValueError("FrozenLake reward_schedule must have 3 entries")

    step_penalty = float(metadata.get("step_penalty", 0.0))
    if reward_schedule[2] + step_penalty > 0.0:
        raise ValueError(
            "Positive non-terminal FrozenLake rewards are unsupported without turn-aware planning"
        )

    return FrozenLakeSpec(
        desc=desc,
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=reward_schedule,
        step_penalty=step_penalty,
        agent_pos=agent_pos,
    )


def _resolve_desc(metadata: dict[str, Any]) -> tuple[str, ...] | None:
    desc = metadata.get("desc")
    if desc is not None:
        resolved = tuple(str(row) for row in desc)
        _validate_desc(resolved)
        return resolved
    map_name = metadata.get("map_name")
    if map_name is None:
        return None
    if map_name not in _FROZEN_LAKE_MAPS:
        raise ValueError(f"Unknown FrozenLake map_name: {map_name!r}")
    return _FROZEN_LAKE_MAPS[map_name]


def _validate_desc(desc: tuple[str, ...]) -> None:
    if not desc:
        raise ValueError("FrozenLake desc must not be empty")
    width = len(desc[0])
    if width == 0 or any(len(row) != width for row in desc):
        raise ValueError("FrozenLake desc must be rectangular")
    allowed = {"S", "F", "H", "G"}
    if any(set(row) - allowed for row in desc):
        raise ValueError("FrozenLake desc contains invalid tiles")


def _parse_grid(
    state: str,
    expected_shape: tuple[int, int] | None,
) -> tuple[str, ...]:
    blocks: list[tuple[str, ...]] = []
    current_rows: list[str] = []
    allowed = {"S", "F", "H", "G", "@"}

    def flush_current() -> None:
        if current_rows:
            blocks.append(tuple(current_rows))
            current_rows.clear()

    for line in state.splitlines():
        compact = line.replace(" ", "").strip()
        if not compact or set(compact) - allowed:
            flush_current()
            continue

        if expected_shape is not None:
            _, expected_width = expected_shape
            if len(compact) != expected_width:
                flush_current()
                continue
        elif current_rows and len(compact) != len(current_rows[0]):
            flush_current()

        current_rows.append(compact)

    flush_current()

    if not blocks:
        raise ValueError("Could not parse FrozenLake grid from state text")

    if expected_shape is not None:
        height, width = expected_shape
        matching_blocks = [
            block
            for block in blocks
            if len(block) == height and all(len(row) == width for row in block)
        ]
        if not matching_blocks:
            raise ValueError("FrozenLake state grid does not match expected shape")
        return matching_blocks[-1]

    return blocks[-1]


def _find_agent(grid: tuple[str, ...]) -> tuple[int, int]:
    positions = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(grid)
        for col_idx, tile in enumerate(row)
        if tile == "@"
    ]
    if len(positions) != 1:
        raise ValueError("FrozenLake state text must contain exactly one '@'")
    return positions[0]


def _resolve_board(
    grid: tuple[str, ...],
    metadata_desc: tuple[str, ...] | None,
    agent_pos: tuple[int, int],
) -> tuple[str, ...]:
    if metadata_desc is not None and (
        len(metadata_desc) != len(grid) or len(metadata_desc[0]) != len(grid[0])
    ):
        raise ValueError("FrozenLake metadata desc shape does not match state grid")

    rows = [list(row) for row in grid]
    row, col = agent_pos

    if metadata_desc is not None:
        for r, row_text in enumerate(grid):
            for c, tile in enumerate(row_text):
                if (r, c) == agent_pos or tile == "@":
                    continue
                merged_tile = _merge_visible_tile(tile, metadata_desc[r][c])
                if merged_tile is None:
                    raise ValueError(
                        "FrozenLake state grid does not match metadata desc"
                    )
                rows[r][c] = merged_tile
        rows[row][col] = metadata_desc[row][col]
    else:
        rows[row][col] = _infer_hidden_tile(grid)

    resolved = tuple("".join(chars) for chars in rows)
    _validate_desc(resolved)
    if sum(r.count("S") for r in resolved) != 1:
        raise ValueError("FrozenLake board must contain exactly one start tile")
    if sum(r.count("G") for r in resolved) != 1:
        raise ValueError("FrozenLake board must contain exactly one goal tile")
    return resolved


def _merge_visible_tile(state_tile: str, metadata_tile: str) -> str | None:
    if state_tile == metadata_tile:
        return state_tile
    if state_tile == "G" and metadata_tile == "F":
        return "G"
    if state_tile == "S" and metadata_tile == "F":
        return "S"
    return None


def _infer_hidden_tile(grid: tuple[str, ...]) -> str:
    start_count = sum(row.count("S") for row in grid)
    goal_count = sum(row.count("G") for row in grid)
    if start_count == 0 and goal_count == 1:
        return "S"
    if goal_count == 0 and start_count == 1:
        return "G"
    if start_count == 1 and goal_count == 1:
        return "F"
    raise ValueError(
        "Cannot infer hidden tile under '@' without metadata desc or map_name"
    )


def _compute_optimal_values(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
) -> list[float]:
    num_states = len(desc) * len(desc[0])
    values = [0.0] * num_states

    for _ in range(_MAX_ITERS):
        delta = 0.0
        new_values = values.copy()
        for state in range(num_states):
            row, col = _from_index(state, len(desc[0]))
            if desc[row][col] in {"G", "H"}:
                new_values[state] = 0.0
                continue
            q_values = [
                _expected_return(
                    desc,
                    is_slippery,
                    success_rate,
                    reward_schedule,
                    step_penalty,
                    state,
                    action,
                    values,
                )
                for action in _ACTION_ORDER
            ]
            best = max(q_values)
            delta = max(delta, abs(best - values[state]))
            new_values[state] = best
        values = new_values
        if delta < _TOL:
            return values

    raise RuntimeError("FrozenLake value iteration did not converge")


def _compute_optimal_actions(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    values: list[float],
) -> list[tuple[str, ...]]:
    optimal_actions: list[tuple[str, ...]] = []
    for state in range(len(values)):
        row, col = _from_index(state, len(desc[0]))
        if desc[row][col] in {"G", "H"}:
            optimal_actions.append(())
            continue
        q_values = {
            action: _expected_return(
                desc,
                is_slippery,
                success_rate,
                reward_schedule,
                step_penalty,
                state,
                action,
                values,
            )
            for action in _ACTION_ORDER
        }
        best = max(q_values.values())
        optimal_actions.append(
            tuple(action for action, q_value in q_values.items() if abs(q_value - best) < 1e-9)
        )
    return optimal_actions


def _compute_min_steps(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    optimal_actions: list[tuple[str, ...]],
) -> list[float]:
    num_states = len(desc) * len(desc[0])
    steps = [0.0] * num_states

    for _ in range(_MAX_ITERS):
        delta = 0.0
        new_steps = steps.copy()
        for state in range(num_states):
            row, col = _from_index(state, len(desc[0]))
            if desc[row][col] in {"G", "H"}:
                new_steps[state] = 0.0
                continue
            actions = optimal_actions[state]
            if not actions:
                raise RuntimeError("Non-terminal FrozenLake state has no optimal actions")
            best_steps = min(
                _expected_steps(
                    desc,
                    is_slippery,
                    success_rate,
                    reward_schedule,
                    step_penalty,
                    state,
                    action,
                    steps,
                )
                for action in actions
            )
            delta = max(delta, abs(best_steps - steps[state]))
            new_steps[state] = best_steps
        steps = new_steps
        if delta < 1e-9:
            return steps

    raise RuntimeError("FrozenLake step-planning iteration did not converge")


def _select_action(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    state: int,
    values: list[float],
    optimal_actions: list[tuple[str, ...]],
    steps: list[float],
) -> str:
    del values
    candidates = []
    for action in optimal_actions[state]:
        candidates.append(
            (
                _expected_steps(
                    desc,
                    is_slippery,
                    success_rate,
                    reward_schedule,
                    step_penalty,
                    state,
                    action,
                    steps,
                ),
                _TIE_BREAK_PRIORITY[action],
                action,
            )
        )
    candidates.sort()
    return candidates[0][2]


def _expected_return(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    state: int,
    action: str,
    values: list[float],
) -> float:
    total = 0.0
    for prob, next_state, reward, terminated in _transitions(
        desc,
        is_slippery,
        success_rate,
        reward_schedule,
        step_penalty,
        state,
        action,
    ):
        total += prob * (reward + (0.0 if terminated else values[next_state]))
    return total


def _expected_steps(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    state: int,
    action: str,
    steps: list[float],
) -> float:
    total = 1.0
    for prob, next_state, _, terminated in _transitions(
        desc,
        is_slippery,
        success_rate,
        reward_schedule,
        step_penalty,
        state,
        action,
    ):
        if not terminated:
            total += prob * steps[next_state]
    return total


def _transitions(
    desc: tuple[str, ...],
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
    step_penalty: float,
    state: int,
    action: str,
) -> tuple[tuple[float, int, float, bool], ...]:
    row, col = _from_index(state, len(desc[0]))
    if desc[row][col] in {"G", "H"}:
        return ((1.0, state, 0.0, True),)

    if is_slippery:
        fail_rate = (1.0 - success_rate) / 2.0
        action_probs = (
            (fail_rate, _SLIP_ACTIONS[action][0]),
            (success_rate, _SLIP_ACTIONS[action][1]),
            (fail_rate, _SLIP_ACTIONS[action][2]),
        )
    else:
        action_probs = ((1.0, action),)

    outcomes: list[tuple[float, int, float, bool]] = []
    for prob, actual_action in action_probs:
        next_row, next_col = _move(desc, row, col, actual_action)
        next_state = _to_index((next_row, next_col), len(desc[0]))
        next_tile = desc[next_row][next_col]
        terminated = next_tile in {"G", "H"}
        reward = step_penalty + _tile_reward(next_tile, reward_schedule)
        outcomes.append((prob, next_state, reward, terminated))
    return tuple(outcomes)


def _move(desc: tuple[str, ...], row: int, col: int, action: str) -> tuple[int, int]:
    dr, dc = _ACTION_DELTAS[action]
    next_row = min(max(row + dr, 0), len(desc) - 1)
    next_col = min(max(col + dc, 0), len(desc[0]) - 1)
    return next_row, next_col


def _tile_reward(tile: str, reward_schedule: tuple[float, float, float]) -> float:
    if tile == "G":
        return reward_schedule[0]
    if tile == "H":
        return reward_schedule[1]
    return reward_schedule[2]


def _to_index(pos: tuple[int, int], width: int) -> int:
    row, col = pos
    return row * width + col


def _from_index(index: int, width: int) -> tuple[int, int]:
    return (index // width, index % width)


register_policy(
    "frozen_lake_4x4",
    frozen_lake_optimal_action,
    metadata={
        "map_name": "4x4",
        "desc": _FROZEN_LAKE_MAPS["4x4"],
        "is_slippery": False,
        "actions": _ACTION_ORDER,
    },
)
register_policy(
    "frozen_lake_8x8",
    frozen_lake_optimal_action,
    metadata={
        "map_name": "8x8",
        "desc": _FROZEN_LAKE_MAPS["8x8"],
        "is_slippery": False,
        "actions": _ACTION_ORDER,
    },
)
# State-text-parsing variants. Use these when the map varies per trajectory
# (e.g. with FrozenLakeMapCycler). They omit `desc`/`map_name` so the policy
# resolves the board from the rendered state text on every call.
register_policy(
    "frozen_lake_random_4x4",
    frozen_lake_optimal_action,
    metadata={
        "is_slippery": False,
        "actions": _ACTION_ORDER,
    },
)
register_policy(
    "frozen_lake_random_8x8",
    frozen_lake_optimal_action,
    metadata={
        "is_slippery": False,
        "actions": _ACTION_ORDER,
    },
)

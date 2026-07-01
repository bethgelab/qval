def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The Q-value is approximated based on the Manhattan distance to the goal,
    penalizing distance and treating holes as terminal states with 0 value.
    """
    def get_coords(grid: str, char: str):
        lines = grid.splitlines()
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch == char:
                    return (r, c)
        return None

    # 1. Locate the agent and the goal in the next state
    agent_next = get_coords(next_state, '@')
    goal_next = get_coords(next_state, 'G')

    # 2. Handle the case where the agent '@' is not found in the next state
    if agent_next is None:
        # Agent might have vanished because it reached G or H
        agent_prev = get_coords(state, '@')
        if agent_prev:
            r, c = agent_prev
            lines_next = next_state.splitlines()
            if r < len(lines_next) and c < len(lines_next[r]):
                cell = lines_next[r][c]
                if cell == 'G':
                    return 1.0
                if cell == 'H':
                    return 0.0
        return 0.0

    # 3. Check if the agent fell into a hole or reached the goal using the previous state grid
    # (Since the agent '@' might replace the cell value 'H' or 'G' in next_state)
    r_next, c_next = agent_next
    lines_prev = state.splitlines()
    if r_next < len(lines_prev) and c_next < len(lines_prev[r_next]):
        cell_prev = lines_prev[r_next][c_next]
        if cell_prev == 'H':
            return 0.0
        if cell_prev == 'G':
            return 1.0

    # 4. Resolve the goal position if it's not visible in next_state (agent might be on it)
    if goal_next is None:
        goal_next = get_coords(state, 'G')
        if goal_next is None:
            # Goal not found in either; fallback to 0
            return 0.0

    # 5. If the agent is currently on the goal position, value is 1.0
    if agent_next == goal_next:
        return 1.0

    # 6. Compute Manhattan distance to the goal
    dist = abs(agent_next[0] - goal_next[0]) + abs(agent_next[1] - goal_next[1])

    # 7. Approximate Q-value using a discount factor gamma^d.
    # Since the agent just completed a transition to a state that is not the goal,
    # the Q-value is approximately gamma * (gamma^dist), which is gamma^(dist + 1).
    gamma = 0.9
    return gamma ** (dist + 1)
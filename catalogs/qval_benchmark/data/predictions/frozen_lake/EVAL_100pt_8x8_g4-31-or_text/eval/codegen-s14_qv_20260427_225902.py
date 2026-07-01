def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an 8x8 Frozen Lake environment.
    The estimate is based on the Manhattan distance to the goal and whether the action leads 
    the agent into a hole or to the goal.
    """
    # Normalize the grid representations by removing spaces and splitting into lines
    state_lines = [line.replace(' ', '') for line in state.splitlines()]
    next_state_lines = [line.replace(' ', '') for line in next_state.splitlines()]

    def get_coords(grid, char):
        """Finds the (row, col) coordinates of the first occurrence of char in the grid."""
        for r, line in enumerate(grid):
            for c, val in enumerate(line):
                if val == char:
                    return (r, c)
        return None

    # Identify key positions from the grids
    goal_pos = get_coords(state_lines, 'G')
    agent_pos_curr = get_coords(state_lines, '@')
    agent_pos_next = get_coords(next_state_lines, '@')

    # If the agent '@' is not present in next_state, infer the position from state and action.
    # This handles cases where the agent might be removed upon reaching a terminal state.
    if agent_pos_next is None and agent_pos_curr is not None:
        r, c = agent_pos_curr
        if action == 'up':
            r -= 1
        elif action == 'down':
            r += 1
        elif action == 'left':
            c -= 1
        elif action == 'right':
            c += 1
        # Keep the agent within the 8x8 boundaries
        r = max(0, min(7, r))
        c = max(0, min(7, c))
        agent_pos_next = (r, c)

    # If we still cannot find the agent's next position, return a baseline value
    if agent_pos_next is None:
        return 0.0

    # Check for terminal outcomes in the resulting state
    if goal_pos and agent_pos_next == goal_pos:
        # Agent reached the goal
        return 1.0

    rn, cn = agent_pos_next
    # Check if the agent fell into a hole based on the original grid layout
    if rn < len(state_lines) and cn < len(state_lines[rn]):
        if state_lines[rn][cn] == 'H':
            return 0.0

    # For non-terminal states, estimate the Q-value using an exponential decay based 
    # on the Manhattan distance to the goal.
    if goal_pos:
        rg, cg = goal_pos
        dist = abs(rn - rg) + abs(cn - cg)
        # A distance of 0 should be handled by the goal check above, 
        # but 0.9^dist is a smooth approximation of the expected return.
        return 0.9 ** dist

    # Fallback if no goal is found in the map
    return 0.0
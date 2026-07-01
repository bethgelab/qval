def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    Q(s, a) is approximated based on the Manhattan distance to the goal and the outcome of the action.
    """
    def find_pos(grid_str, char):
        lines = grid_str.splitlines()
        for r, line in enumerate(lines):
            for c, cell in enumerate(line):
                if cell == char:
                    return r, c
        return None

    # Identify key positions in the next state
    pos_agent_next = find_pos(next_state, '@')
    pos_goal_next = find_pos(next_state, 'G')
    pos_goal_prev = find_pos(state, 'G')

    # Case 1: Agent is still on the grid
    if pos_agent_next is not None:
        # Use the goal position from the next state, falling back to the previous state if necessary
        goal_pos = pos_goal_next if pos_goal_next else pos_goal_prev
        if goal_pos:
            # Manhattan distance to the goal
            dist = abs(pos_agent_next[0] - goal_pos[0]) + abs(pos_agent_next[1] - goal_pos[1])
            # Q-value decreases as distance increases (exponential decay as a proxy for reachability)
            # 0.9^dist provides a smooth gradient toward the goal.
            return float(0.9 ** dist)
        return 0.0

    # Case 2: Agent disappeared from the grid
    # This typically happens when the agent reaches the goal (G) or falls into a hole (H).
    pos_agent_prev = find_pos(state, '@')
    if pos_agent_prev is None:
        return 0.0
    
    r, c = pos_agent_prev
    if action == 'up':
        r -= 1
    elif action == 'down':
        r += 1
    elif action == 'left':
        c -= 1
    elif action == 'right':
        c += 1
    
    # Determine what the agent transitioned into by checking the next state grid
    lines_next = next_state.splitlines()
    if 0 <= r < len(lines_next) and 0 <= c < len(lines_next[0]):
        cell = lines_next[r][c]
        if cell == 'G':
            return 1.0  # Reached the goal
        if cell == 'H':
            return 0.0  # Fell into a hole
            
    return 0.0
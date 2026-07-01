def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an 8x8 Frozen Lake grid.
    The Q-value is approximated based on the Manhattan distance to the goal and 
    whether the action results in reaching the goal or falling into a hole.
    """
    def get_pos(grid, char):
        """Finds the row and column of a specific character in the grid ASCII representation."""
        rows = grid.strip().split('\n')
        for r, row in enumerate(rows):
            col = 0
            for c_val in row:
                if c_val == ' ':
                    continue
                if c_val == char:
                    return (r, col)
                col += 1
        return None

    # Find critical positions in the current and next states
    pos_goal = get_pos(state, 'G')
    pos_agent_state = get_pos(state, '@')
    pos_agent_next = get_pos(next_state, '@')

    # If the agent is not found in the next state, it has likely reached a terminal state (Goal or Hole)
    if pos_agent_next is None:
        if pos_agent_state and pos_goal:
            r, c = pos_agent_state
            if action == 'up':
                r -= 1
            elif action == 'down':
                r += 1
            elif action == 'left':
                c -= 1
            elif action == 'right':
                c += 1
            
            # If the action results in moving to the goal's position, the reward is 1.0
            if (r, c) == pos_goal:
                return 1.0
        # Otherwise, the agent likely fell into a hole or moved off-grid (if off-grid removes '@', though unusual)
        return 0.0

    # If the agent is still on the grid, estimate Q based on distance to the goal
    if pos_agent_next and pos_goal:
        # If the agent is exactly on the goal position
        if pos_agent_next == pos_goal:
            return 1.0
        
        # Use Manhattan distance to approximate the expected discounted reward
        # Q(s, a) approx gamma^dist, where gamma is a discount factor (e.g., 0.95)
        dist = abs(pos_agent_next[0] - pos_goal[0]) + abs(pos_agent_next[1] - pos_goal[1])
        
        # The maximum distance in an 8x8 grid is 14.
        # We use 0.95 as a heuristic discount factor to value closer states more highly.
        return 0.95 ** dist

    # Default value if goal is not found or coordinates are missing
    return 0.0
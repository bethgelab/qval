def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an 8x8 Frozen Lake environment.
    
    The Q-value is estimated based on the proximity to the goal, the presence of holes, 
    and the outcome of the action taken.
    """
    def get_pos(grid_str, char):
        # Flatten the grid to easily find coordinates regardless of newline representation
        flat = "".join(grid_str.splitlines())
        idx = flat.find(char)
        if idx == -1:
            return None
        # Assume 8x8 grid as per task description
        return (idx // 8, idx % 8)

    # Find the positions of the agent and the goal in the current state
    agent_pos = get_pos(state, '@')
    goal_pos = get_pos(state, 'G')
    
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # Find the agent's position in the resulting next_state
    next_agent_pos = get_pos(next_state, '@')
    
    # Determine the coordinates the agent moved to (nr, nc)
    if next_agent_pos is not None:
        nr, nc = next_agent_pos
    else:
        # If the agent is no longer in the grid, they likely hit a goal or a hole
        r, c = agent_pos
        if action == 'up':
            nr, nc = max(0, r - 1), c
        elif action == 'down':
            nr, nc = min(7, r + 1), c
        elif action == 'left':
            nr, nc = r, max(0, c - 1)
        elif action == 'right':
            nr, nc = r, min(7, c + 1)
        else:
            nr, nc = r, c
            
    # Check if the destination coordinates in the original state are the Goal or a Hole
    flat_state = "".join(state.splitlines())
    if 0 <= nr < 8 and 0 <= nc < 8:
        idx = nr * 8 + nc
        if idx < len(flat_state):
            cell = flat_state[idx]
            if cell == 'G':
                return 1.0  # Immediate success
            if cell == 'H':
                return 0.0  # Immediate failure
    
    # If the agent is still safe, the Q-value is approximated by a decay function 
    # of the Manhattan distance to the goal, reflecting the expected discounted reward.
    dist = abs(nr - goal_pos[0]) + abs(nc - goal_pos[1])
    
    # Using a discount factor of 0.9 as a heuristic for the expected return.
    # Closer positions to the goal are valued higher.
    return 0.9 ** dist
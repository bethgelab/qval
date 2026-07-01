def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the Frozen Lake environment.
    The estimate is based on the proximity of the agent to the goal and avoidance of holes.
    """
    def find_pos(grid: str, char: str):
        # Remove potential leading/trailing whitespace and split into lines
        lines = grid.strip().splitlines()
        for r, line in enumerate(lines):
            for c, val in enumerate(line):
                if val == char:
                    return r, c
        return None

    # Find current and next agent positions
    pos_curr = find_pos(state, '@')
    pos_next = find_pos(next_state, '@')
    
    if pos_curr is None or pos_next is None:
        return 0.0

    # Find the goal position
    # The goal 'G' should be visible in the state unless the agent is standing on it.
    pos_goal = find_pos(state, 'G')
    if pos_goal is None:
        # Fallback: search in the next state if it was obscured by the agent in the current state.
        pos_goal = find_pos(next_state, 'G')
        
    if pos_goal is None:
        # If no goal is found in either, we cannot estimate distance.
        return 0.0

    # Parse the current state grid to see what cell the agent moved into
    state_lines = state.strip().splitlines()
    r_next, c_next = pos_next
    
    # Ensure coordinates are within the grid boundaries to avoid IndexError
    if r_next < len(state_lines) and c_next < len(state_lines[r_next]):
        char_at_next = state_lines[r_next][c_next]
        
        # Reward of 1.0 if the agent reaches the goal
        if char_at_next == 'G':
            return 1.0
        # Reward of 0.0 if the agent falls into a hole
        if char_at_next == 'H':
            return 0.0

    # Otherwise, the Q-value is estimated based on the discounted reward.
    # Q(s, a) ≈ gamma^dist * Reward_goal, where gamma is the discount factor.
    # We use gamma = 0.95 as a standard reinforcement learning discount.
    dist = abs(r_next - pos_goal[0]) + abs(c_next - pos_goal[1])
    return 0.95 ** dist
def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the Frozen Lake environment.
    Q-value represents the expected discounted cumulative reward.
    """
    def get_pos(grid_str, char):
        for r, line in enumerate(grid_str.split('\n')):
            for c, val in enumerate(line):
                if val == char:
                    return (r, c)
        return None

    # Find current agent, next agent, and goal positions
    pos_agent_curr = get_pos(state, '@')
    pos_agent_next = get_pos(next_state, '@')
    pos_goal = get_pos(state, 'G')

    # If agent position cannot be determined, we assume a fail state
    if pos_agent_next is None:
        return 0.0
        
    # If goal cannot be determined (shouldn't happen), we cannot estimate progress
    if pos_goal is None:
        return 0.0

    # If the agent is at the goal in the next state, the reward is 1.0
    if pos_agent_next == pos_goal:
        return 1.0

    # Check if the agent has fallen into a hole ('H') in the original state grid
    state_lines = state.split('\n')
    r_next, c_next = pos_agent_next
    if r_next < len(state_lines) and c_next < len(state_lines[r_next]):
        if state_lines[r_next][c_next] == 'H':
            return 0.0
    
    # Calculate Manhattan distance to the goal from the next state
    dist_next = abs(r_next - pos_goal[0]) + abs(c_next - pos_goal[1])
    
    # We estimate Q(s, a) using a decay function based on the distance to the goal.
    # 0.9 is used as a proxy for the discount factor gamma.
    # As distance decreases, the value increases towards 1.0.
    # This prioritizes efficiency and proximity to the objective.
    q_estimate = 0.9 ** dist_next
    
    return float(q_estimate)
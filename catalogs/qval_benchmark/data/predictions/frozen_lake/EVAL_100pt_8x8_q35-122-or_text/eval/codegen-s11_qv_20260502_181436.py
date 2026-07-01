def signal_function(state: str, action: str, next_state: str) -> float:
    """Estimate Q-value based on state analysis for frozen lake environment."""
    
    # Parse next_state to find agent and goal positions
    lines = next_state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                hole_positions.append((row_idx, col_idx))
    
    # If agent position not found, return low value
    if agent_pos is None:
        return 0.1
    
    # If goal not found, return low value
    if goal_pos is None:
        return 0.1
    
    # Check if agent fell into a hole
    if agent_pos in hole_positions:
        return 0.0
    
    # Check if reached goal
    if agent_pos == goal_pos:
        return 1.0
    
    # Calculate Manhattan distance to goal
    distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Estimate Q-value based on distance and step limit
    # With 30 step limit, closer distance means higher probability of success
    # Closer to goal = higher Q-value
    max_distance = 30
    q_value = 1.0 - (distance / max_distance)
    
    # Ensure non-negative and reasonable
    q_value = max(0.1, min(1.0, q_value))
    
    return q_value
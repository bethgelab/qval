def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the state grid to find positions
    lines = state.strip().split('\n')
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_positions.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Parse next state to find new agent position
    next_lines = next_state.strip().split('\n')
    next_agent_pos = None
    for r, line in enumerate(next_lines):
        for c, char in enumerate(line):
            if char == '@':
                next_agent_pos = (r, c)
                break
        if next_agent_pos:
            break
    
    if next_agent_pos is None:
        return 0.0
    
    # Calculate Manhattan distances to goal
    current_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    next_dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    
    # Check if we reached the goal
    if next_agent_pos == goal_pos:
        return 1.0
    
    # Check if we fell into a hole
    if next_agent_pos in hole_positions:
        return -0.5
    
    # Calculate progress (positive if moving closer to goal)
    progress = current_dist - next_dist
    
    # Base Q-value: inversely proportional to distance to goal
    # Closer to goal = higher value
    base_value = 0.5 / (current_dist + 1)
    
    # Progress bonus: reward actions that move toward goal
    progress_bonus = progress * 0.15
    
    # Hole proximity penalty: penalize being near holes
    hole_penalty = 0.0
    for hole in hole_positions:
        dist_to_hole = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
        if dist_to_hole <= 2:
            hole_penalty += 0.3 / (dist_to_hole + 1)
    
    # Combine features into Q-value estimate
    q_value = base_value + progress_bonus - hole_penalty
    
    # Clip to reasonable range [0, 1]
    return max(0.0, min(1.0, q_value))
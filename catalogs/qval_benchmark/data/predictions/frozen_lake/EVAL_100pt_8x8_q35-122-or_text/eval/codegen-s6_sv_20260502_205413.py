def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                holes.append((row_idx, col_idx))
    
    # Check if already at goal
    if agent_pos == goal_pos:
        return 1.0
    
    # Check if on a hole (immediate failure)
    if agent_pos in holes:
        return 0.0
    
    # If missing agent or goal, can't reach
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Count holes in the bounding box between agent and goal
    row_start = min(agent_pos[0], goal_pos[0])
    row_end = max(agent_pos[0], goal_pos[0])
    col_start = min(agent_pos[1], goal_pos[1])
    col_end = max(agent_pos[1], goal_pos[1])
    
    holes_in_region = 0
    for hole in holes:
        if row_start <= hole[0] <= row_end and col_start <= hole[1] <= col_end:
            holes_in_region += 1
    
    # Base value: closer to goal = higher value
    # Using inverse distance scaling with diminishing returns
    base_value = 1.0 / (1.0 + distance * 0.15)
    
    # Penalty for holes in the path region
    hole_penalty = 0.03 * holes_in_region
    
    # Small bonus if very close (distance <= 2)
    proximity_bonus = 0.1 if distance <= 2 else 0.0
    
    # Final estimate bounded between 0 and 1
    estimated_value = max(0.0, min(1.0, base_value - hole_penalty + proximity_bonus))
    
    return estimated_value
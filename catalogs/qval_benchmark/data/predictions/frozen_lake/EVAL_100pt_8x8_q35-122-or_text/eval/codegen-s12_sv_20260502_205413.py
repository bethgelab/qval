def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    
    # Find positions
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
    
    # If no goal found, return 0.0
    if goal_pos is None:
        return 0.0
    
    # If agent at goal, return 1.0
    if agent_pos == goal_pos:
        return 1.0
    
    # If no agent found, return 0.0
    if agent_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    agent_row, agent_col = agent_pos
    goal_row, goal_col = goal_pos
    dist_to_goal = abs(agent_row - goal_row) + abs(agent_col - goal_col)
    
    # Calculate minimum distance to any hole
    min_dist_to_hole = float('inf')
    for hole_pos in hole_positions:
        hole_row, hole_col = hole_pos
        dist = abs(agent_row - hole_row) + abs(agent_col - hole_col)
        if dist < min_dist_to_hole:
            min_dist_to_hole = dist
    
    # Base value from distance to goal (closer = higher)
    # With 30 step limit, normalize distance appropriately
    max_distance = 30
    goal_factor = max(0.0, 1.0 - dist_to_goal / max_distance)
    
    # Danger factor from proximity to holes
    # If very close to hole, reduce value significantly
    danger_factor = 1.0
    if min_dist_to_hole <= 2:
        danger_factor = 0.3
    elif min_dist_to_hole <= 4:
        danger_factor = 0.6
    elif min_dist_to_hole <= 6:
        danger_factor = 0.8
    elif min_dist_to_hole <= 8:
        danger_factor = 0.9
    
    # Combine factors
    value = goal_factor * danger_factor
    
    return value
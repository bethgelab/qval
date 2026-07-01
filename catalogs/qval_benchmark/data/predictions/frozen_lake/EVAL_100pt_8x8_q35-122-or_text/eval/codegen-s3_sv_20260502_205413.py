def signal_function(state: str) -> float:
    # Parse the grid from state string
    lines = state.strip().split('\n')
    
    # Find agent, goal, and hole positions
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
    
    # Check if agent is at goal
    if agent_pos is not None and goal_pos is not None and agent_pos == goal_pos:
        return 1.0
    
    # Check if agent is missing or goal is missing
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Check if agent is on a hole
    if agent_pos in hole_positions:
        return 0.0
    
    # Calculate Manhattan distance to goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Normalize distance (max Manhattan distance on 8x8 is 14)
    max_dist = 14
    dist_penalty = dist / max_dist
    
    # Check proximity to holes
    min_hole_dist = float('inf')
    for hole in hole_positions:
        hole_dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        min_hole_dist = min(min_hole_dist, hole_dist)
    
    # If agent is very close to a hole, add extra penalty
    hole_penalty = 0.0
    if min_hole_dist <= 1:
        hole_penalty = 0.35
    elif min_hole_dist <= 2:
        hole_penalty = 0.15
    elif min_hole_dist <= 3:
        hole_penalty = 0.05
    
    # Calculate base value (higher when closer to goal)
    base_value = 1.0 - dist_penalty
    
    # Apply hole penalty
    final_value = base_value - hole_penalty
    
    # Ensure value is in [0, 1]
    return max(0.0, min(1.0, final_value))
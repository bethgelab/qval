def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [line for line in lines if line.strip()]
    
    if not grid:
        return 0.0
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for row_idx, row in enumerate(grid):
        for col_idx, char in enumerate(row):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                hole_positions.append((row_idx, col_idx))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    min_dist_to_hole = float('inf')
    if hole_positions:
        for hole in hole_positions:
            dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
            if dist < min_dist_to_hole:
                min_dist_to_hole = dist
    
    max_dist = 14
    
    distance_factor = max(0.0, 1.0 - dist_to_goal / max_dist)
    
    if min_dist_to_hole == float('inf'):
        hole_factor = 1.0
    else:
        hole_factor = max(0.0, 1.0 - min_dist_to_hole / max_dist * 0.5)
    
    value = distance_factor * hole_factor
    
    if dist_to_goal > 30:
        value = 0.0
    
    return max(0.0, min(1.0, value))
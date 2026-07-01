def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    height = len(lines)
    width = len(lines[0]) if height > 0 else 0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == '@':
                agent_pos = (x, y)
            elif char == 'G':
                goal_pos = (x, y)
            elif char == 'H':
                holes.append((x, y))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Manhattan distance to goal
    dx = abs(goal_pos[0] - agent_pos[0])
    dy = abs(goal_pos[1] - agent_pos[1])
    manhattan_dist = dx + dy
    
    # Maximum useful distance (step limit)
    max_dist = 30
    
    # Distance-based probability (closer = higher value)
    dist_factor = max(0.0, 1.0 - manhattan_dist / max_dist)
    
    # Hole density in the grid
    total_cells = height * width
    hole_density = len(holes) / total_cells if total_cells > 0 else 0
    
    # Estimate holes in the bounding box between agent and goal
    min_x = min(agent_pos[0], goal_pos[0])
    max_x = max(agent_pos[0], goal_pos[0])
    min_y = min(agent_pos[1], goal_pos[1])
    max_y = max(agent_pos[1], goal_pos[1])
    
    path_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    path_holes = sum(1 for hx, hy in holes if min_x <= hx <= max_x and min_y <= hy <= max_y)
    path_hole_density = path_holes / path_area if path_area > 0 else 0
    
    # Combine factors
    # Base value from distance
    base_value = dist_factor
    
    # Penalize for holes in the path
    hole_penalty = path_hole_density * 0.5
    
    # Penalize for overall hole density
    density_penalty = hole_density * 0.3
    
    # Final estimate
    value = base_value * (1.0 - hole_penalty) * (1.0 - density_penalty)
    
    return max(0.0, min(1.0, value))
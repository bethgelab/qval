def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    rows = len(lines)
    cols = len(lines[0]) if rows > 0 else 0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Manhattan distance to goal
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If too far to reach within step limit, value is near zero
    if manhattan_dist > 30:
        return 0.01
    
    # Base value from distance (closer = better, efficiency matters)
    distance_value = 1.0 - (manhattan_dist / 30.0)
    
    # Count holes in the bounding box between agent and goal
    min_r = min(agent_pos[0], goal_pos[0])
    max_r = max(agent_pos[0], goal_pos[0])
    min_c = min(agent_pos[1], goal_pos[1])
    max_c = max(agent_pos[1], goal_pos[1])
    
    holes_in_path = 0
    for hole in holes:
        if min_r <= hole[0] <= max_r and min_c <= hole[1] <= max_c:
            holes_in_path += 1
    
    # Penalize for holes in the direct path area
    hole_penalty = min(0.4, holes_in_path * 0.08)
    
    # Count nearby holes (within 2 cells Manhattan distance) - immediate danger
    nearby_holes = 0
    for hole in holes:
        if abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1]) <= 2:
            nearby_holes += 1
    nearby_penalty = min(0.3, nearby_holes * 0.12)
    
    # Overall hole density as difficulty factor
    total_cells = rows * cols
    hole_density = len(holes) / total_cells if total_cells > 0 else 0
    density_penalty = min(0.2, hole_density * 1.5)
    
    # Combine all factors
    value = distance_value * (1.0 - hole_penalty - nearby_penalty - density_penalty)
    
    # Ensure value is in valid range [0, 1]
    return max(0.0, min(1.0, value))
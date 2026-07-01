def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    holes = []
    grid_height = len(lines)
    grid_width = 0
    
    for i, line in enumerate(lines):
        grid_width = max(grid_width, len(line))
        for j, char in enumerate(line):
            if char == '@':
                agent_pos = (i, j)
            elif char == 'G':
                goal_pos = (i, j)
            elif char == 'H':
                holes.append((i, j))
    
    if agent_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    if agent_pos in holes:
        return 0.0
    
    if goal_pos is None:
        return 0.0
    
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    nearby_holes = 0
    for hole in holes:
        dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if dist <= 2:
            nearby_holes += 1
    
    total_holes = len(holes)
    total_cells = grid_height * grid_width if grid_height > 0 and grid_width > 0 else 64
    hole_density = total_holes / total_cells if total_cells > 0 else 0
    
    distance_score = max(0, 1.0 - dist_to_goal / 16.0)
    
    density_penalty = hole_density * 0.5
    
    nearby_penalty = min(0.4, nearby_holes * 0.1)
    
    value = max(0.0, distance_score - density_penalty - nearby_penalty)
    
    return value
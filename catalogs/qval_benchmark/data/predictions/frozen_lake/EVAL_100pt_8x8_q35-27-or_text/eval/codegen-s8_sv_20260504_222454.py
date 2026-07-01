def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == '@':
                agent_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
            elif cell == 'H':
                holes.append((i, j))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Manhattan distance from agent to goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Count holes in the bounding box between agent and goal
    min_row = min(agent_pos[0], goal_pos[0])
    max_row = max(agent_pos[0], goal_pos[0])
    min_col = min(agent_pos[1], goal_pos[1])
    max_col = max(agent_pos[1], goal_pos[1])
    
    holes_in_path = 0
    for h in holes:
        if min_row <= h[0] <= max_row and min_col <= h[1] <= max_col:
            holes_in_path += 1
    
    # Check if distance is reachable within step limit
    if dist > 30:
        return 0.0
    
    # Distance factor (closer is better, max distance in 8x8 is 14)
    dist_factor = max(0.0, 1.0 - dist / 20.0)
    
    # Hole density factor (fewer holes in path is better)
    path_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    hole_density = holes_in_path / max(path_area, 1)
    hole_factor = max(0.0, 1.0 - hole_density * 3.0)
    
    # Combined value (estimate of success probability)
    value = dist_factor * hole_factor
    
    return value
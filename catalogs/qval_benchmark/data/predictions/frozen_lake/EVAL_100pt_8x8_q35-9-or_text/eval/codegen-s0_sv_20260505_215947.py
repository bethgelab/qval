def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines if line.strip()]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    min_dist_to_hole = float('inf')
    for hole in holes:
        d = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if d < min_dist_to_hole:
            min_dist_to_hole = d
    
    if min_dist_to_hole == float('inf'):
        min_dist_to_hole = 100
    
    rows = len(grid)
    cols = len(grid[0]) if grid else 1
    
    max_dist = (rows - 1) + (cols - 1)
    
    if dist_to_goal > 30:
        base_value = 0.0
    else:
        base_value = 1.0 - (dist_to_goal / 30.0)
    
    hole_penalty = min(0.3, min_dist_to_hole / 100.0)
    
    value = max(0.0, base_value - hole_penalty)
    
    return value
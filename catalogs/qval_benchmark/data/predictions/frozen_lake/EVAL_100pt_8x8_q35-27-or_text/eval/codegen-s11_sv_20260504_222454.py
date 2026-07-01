def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    hole_count = 0
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    for r in range(rows):
        row_len = len(grid[r])
        for c in range(row_len):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                hole_count += 1
    
    if agent_pos is None:
        return 0.0
    
    if goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    max_dist = 14
    dist_factor = max(0.0, 1.0 - (dist / max_dist))
    
    hole_density = hole_count / (rows * cols) if rows * cols > 0 else 0
    hole_factor = max(0.0, 1.0 - (hole_density * 3.0))
    
    steps_remaining_estimate = 30 - dist
    step_factor = max(0.0, min(1.0, steps_remaining_estimate / 30.0))
    
    center_bonus = 0.0
    center_row = rows / 2
    center_col = cols / 2
    if agent_pos:
        center_dist = abs(agent_pos[0] - center_row) + abs(agent_pos[1] - center_col)
        if center_dist < 3:
            center_bonus = 0.1
    
    value = (dist_factor * 0.5 + hole_factor * 0.3 + step_factor * 0.2) + center_bonus
    value = max(0.0, min(1.0, value))
    
    return value
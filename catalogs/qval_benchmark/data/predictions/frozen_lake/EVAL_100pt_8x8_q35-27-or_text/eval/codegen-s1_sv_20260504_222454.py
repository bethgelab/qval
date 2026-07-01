def signal_function(state: str) -> float:
    import math
    
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        if line.strip():
            grid.append(line.strip())
    
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r in range(rows):
        for c in range(len(grid[r])):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    manhattan_dist = abs(ar - gr) + abs(ac - gc)
    
    if manhattan_dist == 0:
        return 0.95
    
    remaining_steps = 30
    min_steps_needed = manhattan_dist
    
    if min_steps_needed > remaining_steps:
        return 0.0
    
    hole_count = len(holes)
    hole_density = hole_count / (rows * cols)
    
    nearby_holes = 0
    for hr, hc in holes:
        dist_to_hole = abs(ar - hr) + abs(ac - hc)
        if dist_to_hole <= 2:
            nearby_holes += 1
    
    path_safety = max(0.0, 1.0 - (nearby_holes * 0.15))
    
    distance_factor = max(0.0, 1.0 - (manhattan_dist / (rows + cols)))
    
    step_factor = max(0.0, (remaining_steps - min_steps_needed) / remaining_steps)
    
    hole_penalty = max(0.0, 1.0 - (hole_density * 2.0))
    
    base_value = distance_factor * step_factor * path_safety * hole_penalty
    
    if manhattan_dist <= 3:
        base_value = max(base_value, 0.6)
    elif manhattan_dist <= 6:
        base_value = max(base_value, 0.4)
    
    return max(0.0, min(1.0, base_value))
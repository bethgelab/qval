def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r in range(rows):
        for c in range(cols):
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
    
    manhattan_dist = abs(ar - gr) + abs(ac - gr)
    manhattan_dist = abs(ar - gr) + abs(ac - gc)
    
    nearby_holes = 0
    for hr, hc in holes:
        dist_to_hole = abs(ar - hr) + abs(ac - hc)
        if dist_to_hole <= 3:
            nearby_holes += 1
    
    if manhattan_dist >= 30:
        return 0.0
    
    base_value = 1.0 / (1.0 + manhattan_dist * 0.15)
    hole_penalty = nearby_holes * 0.08
    step_factor = max(0.0, (30.0 - manhattan_dist) / 30.0)
    
    value = base_value * step_factor - hole_penalty
    return max(0.0, min(1.0, value))
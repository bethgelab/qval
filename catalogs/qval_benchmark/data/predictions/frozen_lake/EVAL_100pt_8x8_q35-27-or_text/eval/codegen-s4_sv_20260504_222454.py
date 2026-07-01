def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [line for line in lines if line.strip()]
    
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
    
    if agent_pos == goal_pos:
        return 1.0
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    manhattan_dist = abs(ar - gr) + abs(ac - gc)
    
    if manhattan_dist > 30:
        return 0.0
    
    total_holes = len(holes)
    holes_nearby = 0
    for hr, hc in holes:
        hole_dist = abs(ar - hr) + abs(ac - hc)
        if hole_dist <= 3:
            holes_nearby += 1
    
    dist_factor = max(0.05, 1.0 - (manhattan_dist / 30.0))
    hole_factor = max(0.2, 1.0 - (total_holes / 32.0))
    
    base_value = dist_factor * hole_factor
    
    if holes_nearby > 2:
        base_value *= 0.6
    
    return max(0.0, min(1.0, base_value))
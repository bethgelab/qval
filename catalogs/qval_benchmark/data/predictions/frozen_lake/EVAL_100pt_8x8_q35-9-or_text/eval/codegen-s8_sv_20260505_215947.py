def signal_function(state: str) -> float:
    lines = state.split('\n')
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
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    manhattan_dist = abs(ar - gr) + abs(ac - gc)
    
    if manhattan_dist == 0:
        return 1.0
    
    max_steps = 30
    base_value = 1.0 - (manhattan_dist / max_steps)
    
    danger = 0.0
    for hr, hc in holes:
        hole_dist = abs(ar - hr) + abs(ac - hc)
        if hole_dist <= 3:
            danger += 1.0 / (hole_dist + 1)
    
    final_value = base_value * (1.0 - danger * 0.05)
    
    return max(0.0, min(1.0, final_value))
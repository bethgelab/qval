def signal_function(state: str) -> float:
    lines = [line for line in state.strip().split('\n') if line]
    
    agent_pos = None
    goal_pos = None
    holes = set()
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    manhattan_dist = abs(ar - gr) + abs(ac - gc)
    max_dist = 14
    
    dist_factor = 1.0 - (manhattan_dist / max_dist)
    
    path_factor = 1.0
    for dr_check, dc_check in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr_check, ac + dc_check
        if (nr, nc) in holes:
            path_factor *= 0.85
    
    value = dist_factor * path_factor
    
    if manhattan_dist <= 3:
        value = max(value, 0.75)
    elif manhattan_dist <= 6:
        value = max(value, 0.5)
    elif manhattan_dist <= 9:
        value = max(value, 0.3)
    
    return min(max(value, 0.0), 1.0)
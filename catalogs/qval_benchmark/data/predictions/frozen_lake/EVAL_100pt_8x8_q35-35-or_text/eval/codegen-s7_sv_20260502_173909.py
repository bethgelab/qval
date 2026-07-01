def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
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
    
    ag_r, ag_c = agent_pos
    gl_r, gl_c = goal_pos
    
    dist_goal = abs(ag_r - gl_r) + abs(ag_c - gl_c)
    
    if dist_goal == 0:
        return 1.0
    
    min_dist_hole = float('inf')
    for hr, hc in holes:
        d = abs(ag_r - hr) + abs(ag_c - hc)
        if d < min_dist_hole:
            min_dist_hole = d
    
    if min_dist_hole == 0:
        return 0.0
    
    # Heuristic: Closer to goal increases value, closer to holes decreases value
    # Base value decays with distance to goal
    base_value = 1.0 / (1.0 + 0.3 * dist_goal)
    
    # Hole penalty
    hole_factor = 1.0
    if min_dist_hole == 1:
        hole_factor = 0.1
    elif min_dist_hole == 2:
        hole_factor = 0.4
    elif min_dist_hole == 3:
        hole_factor = 0.7
    elif min_dist_hole >= 4:
        hole_factor = 1.0
        
    return base_value * hole_factor
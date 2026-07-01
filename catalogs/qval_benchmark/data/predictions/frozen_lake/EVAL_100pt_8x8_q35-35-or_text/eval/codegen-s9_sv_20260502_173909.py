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
    
    # If agent or goal not found, assume invalid or failure state
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # If agent is on the goal, value is 1.0 (success)
    if agent_pos == goal_pos:
        return 1.0
        
    # If agent is on a hole, value is 0.0 (failure)
    if agent_pos in holes:
        return 0.0
        
    # Calculate Manhattan distance to goal
    ag_r, ag_c = agent_pos
    gl_r, gl_c = goal_pos
    dist_goal = abs(ag_r - gl_r) + abs(ag_c - gl_c)
    
    # Calculate minimum Manhattan distance to any hole
    min_dist_hole = 1000.0  # Large number representing safety if no holes
    for h_r, h_c in holes:
        d = abs(ag_r - h_r) + abs(ag_c - h_c)
        if d < min_dist_hole:
            min_dist_hole = d
            
    # Heuristic Value Function
    # 1. Closer to goal -> Higher value (inverse relationship)
    # 2. Closer to holes -> Lower value (safety penalty)
    # 3. Bounded between 0.0 and 1.0
    
    goal_factor = 1.0 / (1.0 + dist_goal)
    safety_factor = min_dist_hole / (1.0 + min_dist_hole)
    
    value = goal_factor * safety_factor
    
    return value
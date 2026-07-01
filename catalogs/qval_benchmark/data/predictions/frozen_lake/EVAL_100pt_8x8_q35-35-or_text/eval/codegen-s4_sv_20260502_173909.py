def signal_function(state: str) -> float:
    rows = state.split('\n')
    # Filter out empty rows that might result from trailing newlines
    rows = [r for r in rows if r]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Parse grid to find positions
    for r_idx, row in enumerate(rows):
        for c_idx, char in enumerate(row):
            if char == '@':
                agent_pos = (r_idx, c_idx)
            elif char == 'G':
                goal_pos = (r_idx, c_idx)
            elif char == 'H':
                holes.append((r_idx, c_idx))
    
    # If agent or goal is missing, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    r_a, c_a = agent_pos
    r_g, c_g = goal_pos
    
    # Calculate Manhattan distance to goal
    dist_goal = abs(r_a - r_g) + abs(c_a - c_g)
    
    # If already at goal, value is max
    if dist_goal == 0:
        return 1.0
    
    # Calculate minimum distance to any hole
    min_dist_hole = float('inf')
    for h_r, h_c in holes:
        d = abs(r_a - h_r) + abs(c_a - h_c)
        if d < min_dist_hole:
            min_dist_hole = d
    
    # Calculate hole penalty factor
    # Closer to holes reduces value
    hole_penalty = 0.0
    if min_dist_hole != float('inf'):
        # If adjacent to hole, significant penalty. If far, negligible.
        # Using inverse distance for penalty contribution
        hole_penalty = 0.5 / min_dist_hole
    
    # Base value inversely proportional to distance to goal
    # Adjusted for hole proximity
    value = 1.0 / (1.0 + dist_goal + hole_penalty)
    
    return value
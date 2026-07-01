def signal_function(state: str) -> float:
    # Parse the state string into a grid structure
    lines = state.split('\n')
    grid_rows = []
    
    for line in lines:
        # Filter out whitespace to handle spaces between characters
        row_chars = [c for c in line if not c.isspace()]
        if row_chars:
            grid_rows.append(row_chars)
    
    if not grid_rows:
        return 0.0
    
    # Identify positions
    agent_pos = None
    goal_pos = None
    holes = []
    
    rows_count = len(grid_rows)
    cols_count = len(grid_rows[0]) if grid_rows else 0
    
    for r, row in enumerate(grid_rows):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    # If essential elements are missing, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    # This is the primary heuristic for value (closer is better)
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Base value decreases with distance
    # Using 1/(1+dist) ensures value is 1.0 at goal and decays smoothly
    base_value = 1.0 / (1.0 + dist)
    
    # Calculate safety penalty based on nearby holes
    # Holes adjacent to the agent significantly reduce safety
    hole_penalty = 0.0
    r_a, c_a = agent_pos
    for hr, hc in holes:
        h_dist = abs(hr - r_a) + abs(hc - c_a)
        if h_dist == 1:
            hole_penalty += 0.25  # High penalty for immediate danger
        elif h_dist == 2:
            hole_penalty += 0.05  # Lower penalty for nearby danger
    
    # Safety factor clamped to [0, 1]
    safety_factor = max(0.0, 1.0 - hole_penalty)
    
    # Final estimated value
    estimated_value = base_value * safety_factor
    
    return float(estimated_value)
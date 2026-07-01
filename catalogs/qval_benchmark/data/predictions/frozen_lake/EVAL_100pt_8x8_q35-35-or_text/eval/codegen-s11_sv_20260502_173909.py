def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.split('\n')
    grid = []
    for line in lines:
        # Filter empty lines that might result from trailing newlines
        if line.strip():
            grid.append(line)
    
    if not grid:
        return 0.0
    
    # Identify positions
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    # Fallback for invalid states
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Check if agent is already at the goal
    if agent_pos == goal_pos:
        return 1.0
    
    # Check if agent is in a hole
    if agent_pos in holes:
        return 0.0
    
    # Calculate Manhattan distance to goal
    dist = abs(ar - gr) + abs(ac - gc)
    # Max possible Manhattan distance in 8x8 grid is 14 (7+7)
    max_dist = 14
    
    # Base value decreases with distance (reflecting discounting/efficiency)
    # Start at 1.0, decrease to ~0.1 at max distance
    base_value = 1.0 - (dist / max_dist)
    base_value = max(0.05, base_value)  # Ensure minimum value for reachable states
    
    # Calculate risk based on nearby holes
    # Holes closer to the agent pose a higher risk of falling in
    hole_risk = 0.0
    for hr, hc in holes:
        h_dist = abs(ar - hr) + abs(ac - hc)
        if h_dist == 1:
            hole_risk += 0.35
        elif h_dist == 2:
            hole_risk += 0.1
        elif h_dist == 3:
            hole_risk += 0.05
            
    # Cap risk to ensure value doesn't become negative
    hole_risk = min(1.0, hole_risk)
    
    # Apply risk to base value
    estimated_value = base_value * (1.0 - hole_risk)
    
    # Clamp result to [0.0, 1.0]
    return max(0.0, min(1.0, estimated_value))
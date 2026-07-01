def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake grid.
    The estimate is based on the Manhattan distance to the goal and 
    the proximity to holes, assuming an optimal policy.
    """
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Remove potential trailing carriage returns or whitespace,
        # but keep characters that represent the grid cells.
        row = [char for char in line.strip('\r\n') if char in ('@', 'G', 'H', 'S', '.')]
        if row:
            grid.append(row)
            
    if not grid:
        return 0.0
        
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Parse the grid for agent, goal, and hole locations
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    if agent_pos is None:
        return 0.0
        
    # If 'G' is not found in the grid, it is highly likely that the agent
    # has reached the goal and the 'G' character was replaced by '@'.
    if goal_pos is None:
        return 1.0
        
    dist_g = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the agent is on the goal, the value is 1.0
    if dist_g == 0:
        return 1.0
        
    # Calculate Manhattan distance to the nearest hole
    dist_h = float('inf')
    for h_pos in holes:
        d = abs(agent_pos[0] - h_pos[0]) + abs(agent_pos[1] - h_pos[1])
        if d < dist_h:
            dist_h = d
            
    # Base estimation of V(s) using exponential decay based on distance to goal.
    # Using a discount factor (gamma) of approximately 0.95.
    val = 0.95 ** dist_g
    
    # Safety factor: proximity to holes reduces the expected reward.
    # If the agent is adjacent to a hole, the probability of success decreases.
    # If the agent is on a hole, the episode is terminal (val should be 0.0),
    # but since '@' replaces 'H', we use dist_h as a proximity indicator.
    if dist_h == 1:
        safety = 0.4
    elif dist_h == 2:
        safety = 0.75
    else:
        safety = 1.0
        
    return float(val * safety)
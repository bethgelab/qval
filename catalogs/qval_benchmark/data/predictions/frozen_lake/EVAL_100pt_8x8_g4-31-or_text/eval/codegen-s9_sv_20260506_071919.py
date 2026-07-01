def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given Frozen Lake grid state.
    The value is an approximation of the expected discounted reward, 
    assuming optimal play (shortest safe path to the goal).
    """
    # Split the state into lines and filter out any empty lines
    lines = state.split('\n')
    grid = [line for line in lines if line.strip()]
    
    if not grid:
        return 0.0
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Locate the agent, the goal, and all holes in the grid
    for r, line in enumerate(grid):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # If the agent or goal cannot be found, return 0.0
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Calculate Manhattan distance as the base for the estimate
    dist = abs(ar - gr) + abs(ac - gc)
    
    # If agent is at the goal, the value is 1.0
    if dist == 0:
        return 1.0
        
    # Use a discount factor (gamma) to approximate the value based on distance.
    # In a sparse reward setting with an 8x8 grid, 0.92 is a reasonable decay.
    val = 0.92 ** dist
    
    # Define a bounding box to estimate how many holes might obstruct 
    # the most direct path between the agent and the goal.
    min_r, max_r = min(ar, gr), max(ar, gr)
    min_c, max_c = min(ac, gc), max(ac, gc)
    
    path_penalty = 1.0
    for hr, hc in holes:
        # If a hole is inside the bounding box, it is likely to obstruct 
        # the shortest path, increasing the steps required (and thus lowering value).
        if min_r <= hr <= max_r and min_c <= hc <= max_c:
            path_penalty *= 0.94
        
        # If a hole is adjacent to the goal, it increases the difficulty 
        # of reaching the goal safely (narrowing the window for optimal movement).
        if abs(hr - gr) + abs(hc - gc) == 1:
            path_penalty *= 0.85
            
    val *= path_penalty
    
    # Ensure the result is within the valid range [0, 1]
    return float(max(0.0, min(1.0, val)))
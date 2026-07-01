def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for an 8x8 Frozen Lake environment.
    V(s) is approximated as the probability of reaching the goal G from the 
    current agent position @, penalized by distance and local risk (holes).
    """
    # Parse the input state string into a grid
    lines = [line.strip() for line in state.split('\n') if line.strip()]
    if not lines:
        return 0.0
        
    # Remove spaces to ensure characters are indexed correctly
    grid = [line.replace(' ', '') for line in lines]
    
    # Handle cases where the grid might be provided as a single flat string
    if len(grid) == 1 and len(grid[0]) == 64:
        flat_grid = grid[0]
        grid = [flat_grid[i:i+8] for i in range(0, 64, 8)]
    
    if not grid or not grid[0]:
        return 0.0
        
    rows = len(grid)
    cols = len(grid[0])
    
    agent_pos = None
    goal_pos = None
    
    # Locate the agent and the goal
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # If agent or goal is missing, the state is invalid orTerminal
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    # Immediate reward if agent is on goal
    if agent_pos == goal_pos:
        return 1.0
        
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # Calculate Manhattan distance to the goal
    # In an 8x8 grid, the maximum distance is 14.
    dist = abs(ar - gr) + abs(ac - gc)
    
    # Start with a base value and apply a penalty based on distance.
    # A linear penalty represents the increasing risk of failure over a longer path.
    # 1.0 - (14 * 0.03) = 0.58 at maximum distance.
    value = 1.0 - (dist * 0.03)
    
    # Apply a penalty for every adjacent hole, as these represent immediate risks.
    adj_holes = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 'H':
                adj_holes += 1
                
    # Each adjacent hole significantly reduces the estimated value.
    value -= (adj_holes * 0.15)
    
    # Clamp the final estimate between 0.0 and 1.0
    return max(0.0, min(1.0, value))
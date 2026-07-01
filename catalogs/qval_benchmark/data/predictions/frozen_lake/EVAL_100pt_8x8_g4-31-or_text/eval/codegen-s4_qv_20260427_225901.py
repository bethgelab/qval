def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in an 8x8 Frozen Lake environment.
    The estimate is based on the Manhattan distance to the goal, with special handling for 
    holes and goal-reaching transitions.
    """
    # Clean the state grid by removing spaces and empty lines to get a strict 8x8 structure
    grid = [line.replace(" ", "") for line in state.splitlines() if line.strip()]
    
    if not grid:
        return 0.0
    
    # Find current agent position '@'
    r, c = -1, -1
    for row_idx, line in enumerate(grid):
        col_idx = line.find('@')
        if col_idx != -1:
            r, c = row_idx, col_idx
            break
    
    if r == -1:
        return 0.0
        
    # Find goal position 'G'
    rg, cg = -1, -1
    for row_idx, line in enumerate(grid):
        col_idx = line.find('G')
        if col_idx != -1:
            rg, cg = row_idx, col_idx
            break
            
    # If goal 'G' is not explicitly visible, the agent '@' must be on top of it
    if rg == -1:
        rg, cg = r, c
        
    # Calculate next position based on action
    nr, nc = r, c
    if action == 'up':
        nr -= 1
    elif action == 'down':
        nr += 1
    elif action == 'left':
        nc -= 1
    elif action == 'right':
        nc += 1
        
    # Handle grid boundaries (moving off-grid keeps agent in place)
    if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
        nr, nc = r, c
        
    # Determine the cell value at the next position
    # Using the current grid to see what the agent is moving into
    try:
        cell = grid[nr][nc]
    except IndexError:
        cell = '@'  # Fallback if boundary check failed
        
    # Outcome-based estimates
    if cell == 'G':
        return 1.0
    if cell == 'H':
        return 0.0
        
    # Heuristic: distance-based value estimation
    # We use a discount factor gamma = 0.9. Q(s, a) approx gamma^dist(s', G)
    dist = abs(nr - rg) + abs(nc - cg)
    return 0.9 ** dist
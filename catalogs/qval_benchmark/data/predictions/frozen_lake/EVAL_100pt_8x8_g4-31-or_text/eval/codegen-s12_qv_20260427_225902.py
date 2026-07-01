def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state and action in a Frozen Lake environment.
    The Q-value is based on the proximity to the goal and the immediate outcome of the action.
    """
    # Parse the state to extract the grid layout
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        parts = line.split()
        if len(parts) == 8:
            grid.append(parts)
        else:
            # Handle cases where cells are not space-separated
            grid.append(list(line.replace(' ', '')))
    
    # Ensure the grid is correctly sized (8x8)
    if len(grid) < 8:
        return 0.0
    
    # Find agent (@) and goal (G) coordinates
    r, c = -1, -1
    gr, gc = -1, -1
    for row_idx in range(min(len(grid), 8)):
        row = grid[row_idx]
        for col_idx in range(min(len(row), 8)):
            cell = row[col_idx]
            if cell == '@':
                r, c = row_idx, col_idx
            elif cell == 'G':
                gr, gc = row_idx, col_idx
                
    # If the agent is already on the goal, the goal coordinate is the agent's coordinate
    if gr == -1:
        gr, gc = r, c
    
    # If agent not found, return a neutral value
    if r == -1:
        return 0.0

    # Calculate the target position based on the action
    nr, nc = r, c
    if action == 'up':
        nr -= 1
    elif action == 'down':
        nr += 1
    elif action == 'left':
        nc -= 1
    elif action == 'right':
        nc += 1
        
    # Boundary check: moving off the grid keeps the agent in place
    if not (0 <= nr < 8 and 0 <= nc < 8):
        nr, nc = r, c
        
    # Determine the outcome of the action by checking the target cell in the state grid
    target_cell = grid[nr][nc]
    
    # Immediate reward/outcome based on the cell transitioned into
    if target_cell == 'G':
        return 1.0
    if target_cell == 'H':
        return 0.0
        
    # Heuristic: Estimate Q-value based on discounted Manhattan distance to the goal
    # We use a decay factor (0.9) to represent the probability of failure and efficiency.
    dist = abs(nr - gr) + abs(nc - gc)
    
    # Q = gamma^distance, where gamma is the discount factor
    return 0.9 ** dist
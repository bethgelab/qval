def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and resulting next state 
    in a Frozen Lake environment.
    """
    # Extract all non-whitespace characters from the state string
    chars = [c for c in state if not c.isspace()]
    
    # Ensure the grid is exactly 64 characters to represent an 8x8 grid
    if len(chars) < 64:
        chars.extend(['F'] * (64 - len(chars)))
    elif len(chars) > 64:
        chars = chars[:64]
        
    # Convert the flat character list into an 8x8 2D grid
    grid = [chars[i:i+8] for i in range(0, 64, 8)]
    
    # Locate the agent's current position '@' in the grid
    ar, ac = -1, -1
    for r in range(8):
        for c in range(8):
            if grid[r][c] == '@':
                ar, ac = r, c
                break
        if ar != -1:
            break
            
    # Locate the goal's position 'G' in the grid
    gr, gc = -1, -1
    for r in range(8):
        for c in range(8):
            if grid[r][c] == 'G':
                gr, gc = r, c
                break
        if gr != -1:
            break
            
    # If the goal is not found in the grid, it means the agent is likely on it
    if gr == -1:
        return 1.0
        
    # Calculate the target position based on the action taken
    nr, nc = ar, ac
    if action == 'up':
        nr -= 1
    elif action == 'down':
        nr += 1
    elif action == 'left':
        nc -= 1
    elif action == 'right':
        nc += 1
    
    # If the action moves the agent off the grid, they remain in place
    if not (0 <= nr < 8 and 0 <= nc < 8):
        nr, nc = ar, ac
        
    # Determine the character at the target position
    # If the move is invalid, the target is the current cell (not G or H)
    if nr == ar and nc == ac:
        target_char = 'F'
    else:
        target_char = grid[nr][nc]
        
    # Immediate rewards or penalties
    if target_char == 'G':
        return 1.0
    if target_char == 'H':
        return 0.0
        
    # For other cells, estimate the value based on Manhattan distance to the goal.
    # A higher discount factor-like decay is used to approximate the expected return.
    dist = abs(nr - gr) + abs(nc - gc)
    return 0.9 ** dist
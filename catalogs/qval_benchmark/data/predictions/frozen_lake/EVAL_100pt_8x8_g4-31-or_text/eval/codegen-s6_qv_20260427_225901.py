def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the Frozen Lake environment.
    The estimate is based on the distance to the goal and the presence of hazards.
    """
    def get_pos(grid_str, char):
        lines = grid_str.strip().split('\n')
        for r, line in enumerate(lines):
            # Only consider relevant cell characters to handle potential spacing in ASCII
            filtered_line = [cell for cell in line if cell in 'SGH@.']
            for c, cell in enumerate(filtered_line):
                if cell == char:
                    return r, c
        return None

    # Parse state into a cleaned 2D grid for coordinate lookup
    state_lines = state.strip().split('\n')
    state_grid = [[cell for cell in line if cell in 'SGH@.'] for line in state_lines]
    
    # Locate key positions
    pos_s = get_pos(state, '@')
    pos_g = get_pos(state, 'G')
    pos_ns = get_pos(next_state, '@')
    
    if pos_g is None:
        return 0.0
    
    # Determine the agent's final position after the action
    if pos_ns is not None:
        final_pos = pos_ns
    else:
        # If the agent symbol disappeared, the episode may have ended.
        # Predict the next position based on the action taken from current state.
        if pos_s is None: 
            return 0.0
        r, c = pos_s
        if action == 'up': r -= 1
        elif action == 'down': r += 1
        elif action == 'left': c -= 1
        elif action == 'right': c += 1
        
        # Clip to 8x8 grid boundaries to simulate "staying in place"
        r = max(0, min(r, len(state_grid) - 1))
        c = max(0, min(c, len(state_grid[0]) - 1))
        final_pos = (r, c)
        
    # Analyze the resulting cell type from the state map
    r_final, c_final = final_pos
    try:
        cell_type = state_grid[r_final][c_final]
    except IndexError:
        return 0.0
        
    # Direct outcome rewards
    if cell_type == 'G':
        # Goal reached
        return 1.0
    elif cell_type == 'H':
        # Fell into a hole
        return 0.0
    else:
        # Use a distance-based heuristic for the expected discounted reward.
        # Q is approximated as gamma^dist, where dist is Manhattan distance to goal.
        # gamma = 0.9 reflects a preference for shorter paths.
        dist = abs(r_final - pos_g[0]) + abs(c_final - pos_g[1])
        return 0.9 ** dist
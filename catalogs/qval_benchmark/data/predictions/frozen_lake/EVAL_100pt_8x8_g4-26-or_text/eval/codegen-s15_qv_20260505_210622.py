def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str: str):
        return [line.strip() for line in grid_str.strip().split('\n') if line.strip()]

    s_grid = parse_grid(state)
    n_grid = parse_grid(next_state)
    
    if not s_grid or not n_grid:
        return 0.0
    
    r_a, c_a = -1, -1
    for r in range(len(n_grid)):
        for c in range(len(n_grid[r])):
            if n_grid[r][c] == '@':
                r_a, c_a = r, c
                break
        if r_a != -1:
            break
            
    if r_a == -1:
        return 0.0

    # The terrain at the agent's new position (r_a, c_a) can be found in the 
    # previous state grid, as the '@' symbol only replaces the terrain at 
    # the agent's CURRENT position in that state.
    if 0 <= r_a < len(s_grid) and 0 <= c_a < len(s_grid[r_a]):
        terrain = s_grid[r_a][c_a]
        if terrain == 'G':
            return 1.0
        if terrain == 'H':
            return 0.0

    # Find the goal position in the next_state grid to calculate distance.
    r_g, c_g = -1, -1
    for r in range(len(n_grid)):
        for c in range(len(n_grid[r])):
            if n_grid[r][c] == 'G':
                r_g, c_g = r, c
                break
        if r_g != -1:
            break
            
    if r_g == -1:
        # If 'G' is not found, the agent must be on the goal, 
        # which is already handled by the terrain check above.
        return 0.0
        
    # Use Manhattan distance as a proxy for the distance to the goal.
    # A discount factor of 0.9 per step is a reasonable estimate for Q-values 
    # in a sparse reward grid-world with a step limit.
    dist = abs(r_a - r_g) + abs(c_a - c_g)
    return float(0.9 ** dist)
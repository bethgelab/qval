def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a 
    Frozen Lake 8x8 environment.
    """
    # Parse the ASCII grid states into 2D lists
    def parse_grid(grid_str):
        lines = [line.strip() for line in grid_str.split('\n') if line.strip()]
        # Remove spaces to ensure a consistent grid structure
        return [line.replace(' ', '') for line in lines]

    grid_s = parse_grid(state)
    grid_ns = parse_grid(next_state)

    if not grid_s or not grid_ns:
        return 0.0

    # Find the agent's position in the next state
    r2, c2 = -1, -1
    for r in range(len(grid_ns)):
        for c in range(len(grid_ns[r])):
            if grid_ns[r][c] == '@':
                r2, c2 = r, c
                break
        if r2 != -1:
            break

    if r2 == -1:
        return 0.0

    # Determine what the agent moved onto by checking the character 
    # at the agent's new position in the original state.
    # We assume the grids are aligned spatially.
    try:
        cell_under_agent = grid_s[r2][c2]
    except IndexError:
        return 0.0

    # Reward outcome analysis
    if cell_under_agent == 'G':
        # Reached the goal
        return 1.0
    if cell_under_agent == 'H':
        # Fell into a hole
        return 0.0

    # Find the goal's position in the current grid to estimate distance
    rg, cg = -1, -1
    for r in range(len(grid_s)):
        for c in range(len(grid_s[r])):
            if grid_s[r][c] == 'G':
                rg, cg = r, c
                break
        if rg != -1:
            break

    if rg == -1:
        # Goal not found in grid; could be that agent is already on it.
        # But since the episode ends at the goal, this is a fallback.
        return 0.0

    # Use Manhattan distance as a heuristic for the probability of success.
    # Shorter distance translates to a higher Q-value.
    dist = abs(r2 - rg) + abs(c2 - cg)
    
    # Decay factor gamma represents the discounted expectation.
    # 0.8^dist provides a reasonable approximation for an 8x8 grid.
    gamma = 0.8
    return gamma ** dist
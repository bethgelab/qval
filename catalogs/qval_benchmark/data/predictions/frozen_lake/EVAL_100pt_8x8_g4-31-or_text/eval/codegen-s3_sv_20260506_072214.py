def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a frozen lake environment.
    V(s) = E[discounted cumulative reward | s].
    The estimate is based on Manhattan distance to the goal, proximity to holes,
    and an estimation of whether a path is likely to exist.
    """
    # Parse the ASCII grid
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Remove whitespace to get the core grid characters
        cleaned_row = [char for char in line if char not in (' ', '\t', '\r')]
        if cleaned_row:
            grid.append(cleaned_row)

    if not grid or len(grid) == 0:
        return 0.0

    # Locate agent '@' and goal 'G'
    ra, ca = -1, -1
    rg, cg = -1, -1
    hole_count = 0
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                ra, ca = r, c
            elif char == 'G':
                rg, cg = r, c
            elif char == 'H':
                hole_count += 1

    # If agent or goal not found, we can't estimate accurately
    if ra == -1 or rg == -1:
        return 0.0

    # Manhattan Distance
    dist = abs(ra - rg) + abs(ca - cg)
    
    # Immediate Trap Check: Is agent or goal completely surrounded by holes or boundaries?
    def is_trapped(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != 'H':
                    return False
        return True

    if is_trapped(ra, ca) or is_trapped(rg, cg):
        return 0.0

    # V(s) approximation:
    # 1. Decay based on distance (representing discount factor gamma^dist)
    # 2. Scale based on hole density (probability that a path is not blocked)
    # Using gamma = 0.95 as a heuristic for the discount factor
    gamma = 0.95
    distance_factor = gamma ** dist
    
    # Hole density: more holes generally increase the chance that optimal play is blocked
    # In an 8x8 grid, 64 cells.
    hole_density = hole_count / (rows * cols)
    reachability_factor = 1.0 - (hole_density * 0.7) # Heuristic weight for density
    
    # Final estimate
    value = distance_factor * reachability_factor
    
    return max(0.0, min(1.0, value))
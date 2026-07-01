import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake environment.
    The value is based on the probability of reaching the goal (G) from the agent's
    current position (@) given the grid layout.
    """
    # Parse the ASCII grid from the state string
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines if line.strip()]
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0])

    # Locate the agent (@) and the goal (G)
    ar, ac = -1, -1
    gr, gc = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                ar, ac = r, c
            elif grid[r][c] == 'G':
                gr, gc = r, c
    
    # If agent or goal is not found, or agent is already at goal
    if ar == -1 or gr == -1:
        return 0.0
    if ar == gr and ac == gc:
        return 1.0
    
    # Manhattan distance is a primary feature for the value estimate
    dist = abs(ar - gr) + abs(ac - gc)
    
    # Check if the goal is completely blocked by holes or edges
    g_neighbors = [(gr-1, gc), (gr+1, gc), (gr, gc-1), (gr, gc+1)]
    g_safe = 0
    for nr, nc in g_neighbors:
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 'H':
                g_safe += 1
    if g_safe == 0:
        return 0.0

    # Check if the agent is completely trapped by holes or edges
    a_neighbors = [(ar-1, ac), (ar+1, ac), (ar, ac-1), (ar, ac+1)]
    a_safe = 0
    for nr, nc in a_neighbors:
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 'H':
                a_safe += 1
    if a_safe == 0:
        return 0.0
    
    # Start with a base value that decays as distance to goal increases
    # Using an exponential decay as a proxy for the probability of a clear path
    value = math.exp(-dist / 10.0)
    
    # Calculate hole density within the bounding box between agent and goal.
    # Higher hole density suggests a higher probability that the direct path is blocked.
    r_min, r_max = min(ar, gr), max(ar, gr)
    c_min, c_max = min(ac, gc), max(ac, gc)
    holes = 0
    total_cells = 0
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            total_cells += 1
            if grid[r][c] == 'H':
                holes += 1
    
    hole_density = holes / total_cells if total_cells > 0 else 0
    
    # Adjust the value based on the density of hazards in the immediate region
    value *= (1.0 - hole_density)
    
    # Ensure the result is within the valid range [0, 1]
    return max(0.0, min(1.0, value))
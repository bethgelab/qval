import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given 8x8 frozen lake grid.
    The estimate is based on the Manhattan distance to the goal and 
    the proximity to holes, assuming a discounted reward structure.
    """
    # Parse the grid from the ASCII representation
    lines = state.strip().split('\n')
    grid = []
    for line in lines:
        # Extract non-whitespace characters to build rows
        row = [c for c in line if not c.isspace()]
        if row:
            grid.append(row)
    
    if not grid:
        return 0.0
        
    # If the grid is provided as a single line of 64 characters, reshape it
    if len(grid) == 1 and len(grid[0]) > 8:
        new_grid = []
        for i in range(0, len(grid[0]), 8):
            new_grid.append(grid[0][i:i+8])
        grid = new_grid
        
    rows = len(grid)
    cols = len(grid[0])
    
    ar, ac = -1, -1
    gr, gc = -1, -1
    holes = set()
    
    # First pass: Find goal (G) and holes (H)
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == 'G':
                gr, gc = r, c
            elif char == 'H':
                holes.add((r, c))
    
    # Second pass: Find agent (@) and handle if agent is occupying the goal
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                ar, ac = r, c
                if gr == -1:
                    gr, gc = r, c
                break
        if ar != -1:
            break
            
    # Safety checks for invalid states
    if ar == -1:
        return 0.0
    if gr == -1:
        return 0.0
        
    # Manhattan distance to the goal
    dist = abs(ar - gr) + abs(ac - gc)
    
    # The episode step limit is 30. If the distance exceeds this, value is 0.
    if dist > 30:
        return 0.0
        
    # Base value estimation: Use an exponential decay based on distance.
    # This approximates the discounted reward E[gamma^t * 1.0].
    # 0.9 is a reasonable discount factor heuristic for this grid size.
    val = 0.9 ** dist
    
    # Proximity to holes (Risk assessment):
    # In Frozen Lake, falling into a hole ends the episode with 0 reward.
    # Even with optimal play, being adjacent to holes in a potentially 
    # stochastic environment reduces the expected value.
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr, ac + dc
        if (nr, nc) in holes:
            val *= 0.5
            
    # Ensure the returned value is within the logical bounds of [0.0, 1.0]
    return float(max(0.0, min(1.0, val)))
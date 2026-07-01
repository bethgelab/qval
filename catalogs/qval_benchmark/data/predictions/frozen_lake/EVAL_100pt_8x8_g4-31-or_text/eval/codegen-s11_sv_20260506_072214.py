def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given Frozen Lake 8x8 grid state.
    The estimate is based on the Manhattan distance to the goal, the 
    global density of holes, and local accessibility of the agent and goal.
    """
    # Parse the grid from the text representation
    lines = state.split('\n')
    grid = [list(line) for line in lines if line.strip()]
    if not grid:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if cols == 0:
        return 0.0

    ar, ac = -1, -1  # Agent position
    gr, gc = -1, -1  # Goal position
    holes = []       # Hole positions
    
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char == '@':
                ar, ac = r, c
            elif char == 'G':
                gr, gc = r, c
            elif char == 'H':
                holes.append((r, c))
    
    # If agent or goal is not found, the state is invalid
    if ar == -1 or gr == -1:
        return 0.0
    
    # Calculate Manhattan distance to the goal
    dist = abs(ar - gr) + abs(ac - gc)
    
    # Immediate success
    if dist == 0:
        return 1.0
    
    # Base value: expected discounted reward assuming a path exists.
    # Using a discount factor gamma = 0.99.
    value = 0.99 ** dist
    
    # Penalty for global hole density. Higher density of holes reduces 
    # the probability that a path to the goal exists.
    hole_density = len(holes) / (rows * cols)
    value *= (1.0 - hole_density)
    
    # Local accessibility check: Is the agent trapped?
    # If the agent cannot move to any adjacent cell that isn't a hole, V(s) = 0.
    agent_possible_moves = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 'H':
                agent_possible_moves += 1
    
    if agent_possible_moves == 0:
        return 0.0
        
    # Local accessibility check: Is the goal trapped?
    # If the goal cannot be reached from any adjacent cell, V(s) = 0.
    goal_possible_moves = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = gr + dr, gc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 'H':
                goal_possible_moves += 1
                
    if goal_possible_moves == 0:
        return 0.0

    # Final clamp to ensure value is within [0, 1]
    return max(0.0, min(1.0, value))
def signal_function(state: str) -> float:
    # Parse the state string into a grid structure
    try:
        lines = state.strip().splitlines()
    except Exception:
        return 0.0
        
    if not lines:
        return 0.0
        
    grid = []
    for line in lines:
        # Filter out empty lines resulting from trailing newlines
        if line.strip():
            grid.append(list(line))
            
    if not grid:
        return 0.0
        
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find positions of Agent (@) and Goal (G)
    agent_pos = None
    goal_pos = None
    
    for r in range(rows):
        for c in range(len(grid[r])):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # If agent or goal is not found, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    # If agent is on the goal, value is maximum (1.0)
    if ar == gr and ac == gc:
        return 1.0
    
    # Calculate Manhattan distance to the goal
    dist = abs(ar - gr) + abs(ac - gc)
    
    # If distance exceeds step limit, value is 0
    if dist > 30:
        return 0.0
        
    # Base value decreases with distance (fewer steps is preferable)
    # Normalized by step limit (30). 1.0 at dist 0, 0.0 at dist 30.
    base_value = max(0.0, 1.0 - (dist / 30.0))
    
    # Calculate penalty for nearby holes (risk of falling)
    adj_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
            if grid[nr][nc] == 'H':
                adj_holes += 1
                
    # Apply hole penalty. Each adjacent hole reduces value.
    # If 4 holes surround agent, value becomes 0 (trapped).
    hole_penalty = 1.0 - (0.25 * adj_holes)
    if hole_penalty < 0.0:
        hole_penalty = 0.0
        
    # Final estimated value
    value = base_value * hole_penalty
    
    return value
def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().splitlines()
    grid = []
    for line in lines:
        if line.strip():
            grid.append(list(line))
    
    if not grid:
        return 0.0

    agent_pos = None
    goal_pos = None
    holes = []
    
    # Scan grid for agent, goal, and holes
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    # If agent or goal is missing, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # If agent is at goal, value is 1.0
    if agent_pos == goal_pos:
        return 1.0
    
    r1, c1 = agent_pos
    r2, c2 = goal_pos
    
    # Calculate Manhattan distance
    dist = abs(r1 - r2) + abs(c1 - c2)
    
    # Calculate max possible Manhattan distance for this grid size
    num_rows = len(grid)
    max_cols = max(len(row) for row in grid) if grid else 0
    max_dist = (num_rows - 1) + (max_cols - 1)
    
    if max_dist <= 0:
        return 1.0 if dist == 0 else 0.0
        
    # Distance-based value component (linear decay)
    v_dist = 1.0 - (dist / max_dist)
    if v_dist < 0.0:
        v_dist = 0.0
        
    # Hole penalty components
    # 1. Adjacent holes (immediate risk)
    adj_holes = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in neighbors:
        nr, nc = r1 + dr, c1 + dc
        if 0 <= nr < len(grid):
            row_len = len(grid[nr])
            if 0 <= nc < row_len:
                if grid[nr][nc] == 'H':
                    adj_holes += 1
                    
    # 2. Holes in the bounding box between agent and goal (path obstruction)
    r_min, r_max = min(r1, r2), max(r1, r2)
    c_min, c_max = min(c1, c2), max(c1, c2)
    box_holes = 0
    for r in range(r_min, r_max + 1):
        if 0 <= r < len(grid):
            row_len = len(grid[r])
            for c in range(c_min, c_max + 1):
                if 0 <= c < row_len:
                    if grid[r][c] == 'H':
                        box_holes += 1
                        
    # Calculate hole penalty
    # Adjacent holes are more dangerous than distant ones
    penalty = (adj_holes * 0.2) + (box_holes * 0.05)
    v_holes = 1.0 - penalty
    if v_holes < 0.0:
        v_holes = 0.0
        
    # Combine values
    value = v_dist * v_holes
    
    # Clamp to [0.0, 1.0]
    if value < 0.0:
        value = 0.0
    if value > 1.0:
        value = 1.0
        
    return value
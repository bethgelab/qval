def signal_function(state: str) -> float:
    # Parse the grid
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    # Find agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == '@':
                agent_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
            elif cell == 'H':
                holes.append((i, j))
    
    # If already at goal
    if agent_pos and goal_pos and agent_pos == goal_pos:
        return 1.0
    
    # If agent is in a hole
    if agent_pos and agent_pos in holes:
        return 0.0
    
    # If no goal found, can't reach it
    if not goal_pos:
        return 0.0
    
    # If no agent found, invalid state
    if not agent_pos:
        return 0.0
    
    # Calculate Manhattan distance to goal
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Count holes near agent (within 2 cells)
    nearby_holes = 0
    for hi, hj in holes:
        if abs(hi - agent_pos[0]) <= 2 and abs(hj - agent_pos[1]) <= 2:
            nearby_holes += 1
    
    # Total holes in grid
    total_holes = len(holes)
    
    # Grid dimensions
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 8
    
    # Estimate value based on multiple factors
    # 1. Distance factor: closer to goal = higher value
    # 2. Safety factor: fewer nearby holes = higher value
    # 3. Overall hole density: fewer total holes = higher value
    
    # Distance factor (normalized by step limit)
    dist_factor = max(0, 1.0 - manhattan_dist / 30.0)
    
    # Safety factor (penalty for nearby holes)
    safety_factor = max(0, 1.0 - nearby_holes * 0.15)
    
    # Hole density factor
    grid_size = rows * cols
    hole_density = total_holes / grid_size if grid_size > 0 else 0
    density_factor = max(0, 1.0 - hole_density * 3.0)
    
    # Combine factors (weighted average)
    value = 0.5 * dist_factor + 0.3 * safety_factor + 0.2 * density_factor
    
    # Ensure value is in valid range
    value = max(0.0, min(1.0, value))
    
    return value
def signal_function(state: str) -> float:
    import math
    
    # Parse the grid to find positions
    lines = state.strip().split('\n')
    grid = []
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for row_idx, line in enumerate(lines):
        row = list(line)
        grid.append(row)
        for col_idx, char in enumerate(row):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                hole_positions.append((row_idx, col_idx))
    
    # If we can't find agent or goal, return 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Step limit is 30, so distance beyond that is unreachable
    if manhattan_dist > 30:
        return 0.0
    
    # Base value from distance (closer = higher value)
    # Normalize: max useful distance is 30, so value decreases with distance
    base_value = max(0.0, 1.0 - (manhattan_dist / 30.0))
    
    # Check if agent is adjacent to a hole (dangerous position)
    agent_row, agent_col = agent_pos
    adjacent_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_row + dr, agent_col + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] == 'H':
                adjacent_holes += 1
    
    # Penalty for being adjacent to holes (higher risk)
    hole_penalty = adjacent_holes * 0.15
    
    # Check if agent is currently on a hole (shouldn't happen in valid state, but handle it)
    if grid[agent_row][agent_col] == 'H':
        return 0.0
    
    # Check if agent is at goal
    if agent_pos == goal_pos:
        return 1.0
    
    # Calculate value considering distance and danger
    value = base_value - hole_penalty
    
    # Additional factor: check if there are holes blocking direct path
    # Count holes in the bounding box between agent and goal
    min_row = min(agent_pos[0], goal_pos[0])
    max_row = max(agent_pos[0], goal_pos[0])
    min_col = min(agent_pos[1], goal_pos[1])
    max_col = max(agent_pos[1], goal_pos[1])
    
    path_holes = 0
    for hole_pos in hole_positions:
        hr, hc = hole_pos
        if min_row <= hr <= max_row and min_col <= hc <= max_col:
            path_holes += 1
    
    # Penalty for holes in the general direction of the goal
    path_penalty = min(0.2, path_holes * 0.05)
    
    # Final value
    value = max(0.0, value - path_penalty)
    
    # Bonus for being very close to goal (1-2 steps)
    if manhattan_dist <= 2:
        value = min(1.0, value + 0.1)
    
    return value
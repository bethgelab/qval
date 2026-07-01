def signal_function(state: str) -> float:
    # Parse the grid to find positions
    lines = state.strip().split('\n')
    grid = [line for line in lines if line.strip()]
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    grid_height = len(grid)
    grid_width = max(len(row) for row in grid) if grid else 0
    
    for row_idx, row in enumerate(grid):
        for col_idx, char in enumerate(row):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                hole_positions.append((row_idx, col_idx))
    
    # If agent is at goal, return max value
    if agent_pos == goal_pos:
        return 1.0
    
    # If no valid positions found, return 0
    if not agent_pos or not goal_pos:
        return 0.0
    
    # Calculate Manhattan distance to goal
    distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Calculate danger score based on nearby holes
    danger_score = 0
    for hole in hole_positions:
        hole_dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if hole_dist <= 2:  # Holes within 2 steps are dangerous
            danger_score += 1
    
    # Normalize danger (max danger would be 8 nearby holes in 5x5 area)
    danger_factor = max(0.0, 1.0 - (danger_score / 8.0))
    
    # Base value from distance (closer = higher value)
    # With 30 step limit, distance up to 30 is feasible
    distance_factor = max(0.0, 1.0 - (distance / 30.0))
    
    # Check if path seems blocked (agent surrounded by holes on all sides)
    adjacent_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if 0 <= nr < grid_height and 0 <= nc < grid_width:
            if nr < len(grid) and nc < len(grid[nr]):
                if grid[nr][nc] == 'H':
                    adjacent_holes += 1
    
    # Penalize heavily if surrounded by holes
    blocked_factor = max(0.0, 1.0 - (adjacent_holes / 4.0))
    
    # Combine factors
    value = distance_factor * danger_factor * blocked_factor
    
    return value
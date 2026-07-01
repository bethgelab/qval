def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r in range(rows):
        for c in range(len(grid[r])):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    # If agent is at goal, return 1.0 (terminal state)
    if agent_pos == goal_pos:
        return 1.0
    
    # If no goal or agent found, return 0.0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    dr = abs(agent_pos[0] - goal_pos[0])
    dc = abs(agent_pos[1] - goal_pos[1])
    distance = dr + dc
    
    # If distance exceeds episode limit (30), cannot reach goal
    if distance > 30:
        return 0.0
    
    # Count holes in adjacent cells (up, down, left, right, and diagonals)
    adjacent_holes = 0
    for hole_r, hole_c in holes:
        if abs(hole_r - agent_pos[0]) <= 1 and abs(hole_c - agent_pos[1]) <= 1:
            adjacent_holes += 1
    
    # Base value from distance (closer is better)
    # With 30 step limit, distance of 0 = 1.0, distance of 30 = 0.0
    base_value = 1.0 - (distance / 30.0)
    
    # Penalty for being near holes (risk of falling)
    hole_penalty = 0.15 * adjacent_holes
    
    # Bonus for being very close to goal (1-2 steps away)
    if distance <= 2:
        base_value = min(1.0, base_value + 0.2)
    
    # Final value (clamped to [0, 1])
    estimated_value = max(0.0, min(1.0, base_value - hole_penalty))
    
    return estimated_value
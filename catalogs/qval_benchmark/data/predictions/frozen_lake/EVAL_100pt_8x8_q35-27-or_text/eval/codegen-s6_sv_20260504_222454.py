def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for i, row in enumerate(grid):
        for j, char in enumerate(row):
            if char == '@':
                agent_pos = (i, j)
            elif char == 'G':
                goal_pos = (i, j)
            elif char == 'H':
                holes.append((i, j))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Count holes in 3x3 area around agent (immediate danger)
    nearby_holes = 0
    for hi, hj in holes:
        if abs(hi - agent_pos[0]) <= 1 and abs(hj - agent_pos[1]) <= 1:
            nearby_holes += 1
    
    # Count total holes in grid (overall difficulty)
    total_holes = len(holes)
    
    # Grid dimensions
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 8
    
    # Max possible Manhattan distance in 8x8 grid
    max_dist = (rows - 1) + (cols - 1)
    
    # Base value from distance to goal (closer = better)
    # Normalize distance to [0, 1] range
    if max_dist > 0:
        distance_factor = 1.0 - (dist / max_dist)
    else:
        distance_factor = 1.0
    
    # Penalize for immediate danger (holes nearby)
    # Each nearby hole reduces confidence significantly
    danger_penalty = nearby_holes * 0.15
    
    # Penalize for overall map difficulty
    difficulty_penalty = total_holes * 0.01
    
    # Bonus for being very close (within 2 steps)
    proximity_bonus = 0.0
    if dist <= 2:
        proximity_bonus = 0.2
    
    # Combine factors
    value = distance_factor - danger_penalty - difficulty_penalty + proximity_bonus
    
    # Clamp to valid range [0.0, 1.0]
    value = max(0.0, min(1.0, value))
    
    return value
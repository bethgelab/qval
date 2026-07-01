def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines if line.strip()]
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for r in range(height):
        for c in range(width):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                hole_positions.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    agent_r, agent_c = agent_pos
    goal_r, goal_c = goal_pos
    
    manhattan_distance = abs(agent_r - goal_r) + abs(agent_c - goal_c)
    
    path_holes = 0
    for hole_r, hole_c in hole_positions:
        if (min(agent_r, goal_r) <= hole_r <= max(agent_r, goal_r) and
            min(agent_c, goal_c) <= hole_c <= max(agent_c, goal_c)):
            path_holes += 1
    
    max_steps = 30
    distance_score = max(0.0, 1.0 - manhattan_distance / max_steps)
    
    hole_penalty = min(1.0, path_holes * 0.15)
    
    neighbor_holes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = agent_r + dr, agent_c + dc
        if (nr, nc) in hole_positions:
            neighbor_holes += 1
    neighbor_penalty = neighbor_holes * 0.1
    
    value = distance_score * (1.0 - hole_penalty) * (1.0 - neighbor_penalty)
    
    return max(0.0, value)
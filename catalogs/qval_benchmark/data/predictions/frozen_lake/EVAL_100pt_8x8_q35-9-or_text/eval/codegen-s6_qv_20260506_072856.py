def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        return [list(line) for line in lines]
    
    def find_position(grid, char):
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == char:
                    return (i, j)
        return None
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        i1, j1 = pos1
        i2, j2 = pos2
        return abs(i1 - i2) + abs(j1 - j2)
    
    next_grid = parse_grid(next_state)
    agent_pos = find_position(next_grid, '@')
    goal_pos = find_position(next_grid, 'G')
    
    distance_to_goal = manhattan_distance(agent_pos, goal_pos)
    
    if distance_to_goal is None or distance_to_goal == float('inf'):
        return 0.0
    
    if distance_to_goal <= 0:
        return 1.0
    
    base_value = 1.0 / (1.0 + distance_to_goal)
    
    if agent_pos is None:
        return 0.0
    
    i, j = agent_pos
    nearby_holes = 0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if 0 <= ni < len(next_grid) and 0 <= nj < len(next_grid[0]):
                if next_grid[ni][nj] == 'H':
                    nearby_holes += 1
    
    hole_penalty = nearby_holes * 0.15
    
    q_value = max(0.0, min(1.0, base_value - hole_penalty))
    
    return q_value
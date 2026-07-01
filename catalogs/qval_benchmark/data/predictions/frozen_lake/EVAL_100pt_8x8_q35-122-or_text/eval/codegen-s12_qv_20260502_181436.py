def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(text):
        lines = [line for line in text.strip().split('\n') if line.strip()]
        return lines
    
    def find_positions(grid):
        agent_pos = None
        goal_pos = None
        
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (row_idx, col_idx)
                elif cell == 'G':
                    goal_pos = (row_idx, col_idx)
        
        return agent_pos, goal_pos
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    grid = parse_grid(state)
    next_grid = parse_grid(next_state)
    
    agent_pos, goal_pos = find_positions(grid)
    next_agent_pos, _ = find_positions(next_grid)
    
    if next_agent_pos == goal_pos:
        return 1.0
    
    if next_agent_pos and next_agent_pos[0] < len(next_grid) and next_agent_pos[1] < len(next_grid[next_agent_pos[0]]):
        cell = next_grid[next_agent_pos[0]][next_agent_pos[1]]
        if cell == 'H':
            return 0.0
    
    distance = manhattan_distance(next_agent_pos, goal_pos)
    
    if goal_pos is None or next_agent_pos is None:
        return 0.0
    
    discount = 0.95
    base_value = discount ** distance
    
    if agent_pos and next_agent_pos:
        old_dist = manhattan_distance(agent_pos, goal_pos)
        if next_agent_pos != agent_pos and old_dist > distance:
            base_value *= 1.1
        elif next_agent_pos == agent_pos:
            base_value *= 0.9
    
    return min(1.0, max(0.0, base_value))
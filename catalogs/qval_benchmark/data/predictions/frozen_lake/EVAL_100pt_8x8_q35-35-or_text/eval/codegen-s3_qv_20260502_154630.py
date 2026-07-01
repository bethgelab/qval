def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        return [list(line) for line in lines]
    
    def find_position(grid, char):
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == char:
                    return (r, c)
        return None
    
    def count_holes(grid):
        count = 0
        for row in grid:
            count += row.count('H')
        return count
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_grid = parse_grid(state)
    next_grid = parse_grid(next_state)
    
    current_agent_pos = find_position(current_grid, '@')
    goal_pos = find_position(next_grid, 'G')
    next_agent_pos = find_position(next_grid, '@')
    
    if next_agent_pos is None:
        return 0.0
    
    if goal_pos is None:
        return 0.0
    
    if next_grid[next_agent_pos[0]][next_agent_pos[1]] == 'G':
        return 1.0
    
    if next_grid[next_agent_pos[0]][next_agent_pos[1]] == 'H':
        return 0.0
    
    current_dist = manhattan_distance(current_agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, goal_pos)
    
    if next_dist == 0:
        return 0.99
    
    distance_factor = 1.0 / (next_dist + 1)
    
    if current_agent_pos:
        dx = next_agent_pos[1] - current_agent_pos[1]
        dy = next_agent_pos[0] - current_agent_pos[0]
        goal_dx = goal_pos[1] - current_agent_pos[1]
        goal_dy = goal_pos[0] - current_agent_pos[0]
        
        toward_goal = False
        if goal_dx != 0:
            toward_goal = (dx * goal_dx) > 0
        if goal_dy != 0:
            if toward_goal:
                toward_goal = (dy * goal_dy) > 0
            else:
                toward_goal = (dy * goal_dy) >= 0
        
        if toward_goal:
            action_factor = 0.3
        else:
            action_factor = -0.15
    else:
        action_factor = 0.0
    
    current_holes = count_holes(current_grid)
    next_holes = count_holes(next_grid)
    
    if next_holes <= current_holes:
        safety_factor = 0.15
    else:
        safety_factor = -0.2
    
    q_value = distance_factor * 0.6 + action_factor + safety_factor
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value
def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        return [list(line) for line in lines]
    
    def find_position(grid, char):
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == char:
                    return i, j
        return None
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    grid = parse_grid(state)
    next_grid = parse_grid(next_state)
    
    agent_pos = find_position(grid, '@')
    goal_pos = find_position(grid, 'G')
    hole_pos = find_position(grid, 'H')
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    dist_to_hole = manhattan_distance(agent_pos, hole_pos) if hole_pos else float('inf')
    
    action_moves = {'left': (0, -1), 'down': (1, 0), 'right': (0, 1), 'up': (-1, 0)}
    move = action_moves.get(action, (0, 0))
    
    new_agent_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
    
    next_agent_pos = find_position(next_grid, '@')
    next_hole_pos = find_position(next_grid, 'H')
    next_goal_pos = find_position(next_grid, 'G')
    
    if next_agent_pos is None:
        return 0.0
    
    dist_after_action = manhattan_distance(next_agent_pos, next_goal_pos)
    dist_to_hole_after = manhattan_distance(next_agent_pos, next_hole_pos) if next_hole_pos else float('inf')
    
    q_value = 0.0
    
    if next_agent_pos == next_goal_pos:
        q_value = 1.0
    elif next_agent_pos == next_hole_pos:
        q_value = 0.0
    elif dist_to_goal > 0 and dist_after_action < dist_to_goal:
        progress_ratio = (dist_to_goal - dist_after_action) / dist_to_goal
        q_value = 0.5 + progress_ratio * 0.3
        if hole_pos and move == (hole_pos[0] - agent_pos[0], hole_pos[1] - agent_pos[1]):
            q_value *= 0.5
    elif dist_after_action == dist_to_goal:
        q_value = 0.2
    else:
        q_value = 0.0
    
    if dist_to_hole < 2 and dist_after_action >= dist_to_goal:
        q_value *= 0.3
    
    return float(q_value)
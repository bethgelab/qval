def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_text):
        lines = grid_text.split('\n')
        agent_pos = None
        goal_pos = None
        hole_positions = []
        
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                elif char == 'H':
                    hole_positions.append((row_idx, col_idx))
        
        return agent_pos, goal_pos, hole_positions
    
    agent_pos, goal_pos, hole_positions = parse_grid(state)
    next_agent_pos, next_goal_pos, next_hole_positions = parse_grid(next_state)
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    min_dist_to_hole = float('inf')
    for hole in hole_positions:
        d = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if d < min_dist_to_hole:
            min_dist_to_hole = d
    
    action_to_delta = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    
    delta = action_to_delta.get(action, (0, 0))
    
    next_row = agent_pos[0] + delta[0]
    next_col = agent_pos[1] + delta[1]
    
    rows = len(lines)
    cols = len(lines[0]) if lines else 0
    
    if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
        expected_next_pos = agent_pos
    else:
        expected_next_pos = (next_row, next_col)
    
    is_goal = next_agent_pos == goal_pos
    is_hole = next_agent_pos in next_hole_positions
    
    score = 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    if is_goal:
        score = 1.0
    elif is_hole:
        score = -0.5
    else:
        next_dist = abs(next_agent_pos[0] - next_goal_pos[0]) + abs(next_agent_pos[1] - next_goal_pos[1])
        
        if next_dist < dist_to_goal:
            score = 0.7 - (next_dist / 15)
        elif next_dist == dist_to_goal:
            score = 0.3
        else:
            score = 0.1
    
    if dist_to_goal > 25:
        score *= 0.5
    
    if min_dist_to_hole <= 2:
        score *= 0.3
    
    return max(0.0, min(1.0, score))
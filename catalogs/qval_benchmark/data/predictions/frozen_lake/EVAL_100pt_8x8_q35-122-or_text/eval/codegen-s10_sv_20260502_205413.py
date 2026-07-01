def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    hole_positions = set()
    
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                hole_positions.add((row_idx, col_idx))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    dr = abs(agent_pos[0] - goal_pos[0])
    dc = abs(agent_pos[1] - goal_pos[1])
    manhattan_dist = dr + dc
    
    dist_value = 1.0 - (manhattan_dist / 14.0)
    
    min_row = min(agent_pos[0], goal_pos[0])
    max_row = max(agent_pos[0], goal_pos[0])
    min_col = min(agent_pos[1], goal_pos[1])
    max_col = max(agent_pos[1], goal_pos[1])
    
    holes_in_path = 0
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if (row, col) in hole_positions:
                holes_in_path += 1
    
    hole_penalty = 0.1 * holes_in_path
    
    value = max(0.0, dist_value - hole_penalty)
    
    return value
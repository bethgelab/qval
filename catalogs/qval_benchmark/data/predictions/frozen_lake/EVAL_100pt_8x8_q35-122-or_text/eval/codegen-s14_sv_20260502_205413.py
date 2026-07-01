def signal_function(state: str) -> float:
    import math
    
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    grid_rows = len(lines)
    
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                hole_positions.append((row_idx, col_idx))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    max_dist = grid_rows * 2
    distance_value = max(0.0, 1.0 - (manhattan_dist / max_dist))
    
    nearby_holes = 0
    for hole in hole_positions:
        hole_dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if hole_dist <= 2:
            nearby_holes += 1
    
    hole_penalty = 0.1 * nearby_holes
    
    value = max(0.0, distance_value - hole_penalty)
    
    return value
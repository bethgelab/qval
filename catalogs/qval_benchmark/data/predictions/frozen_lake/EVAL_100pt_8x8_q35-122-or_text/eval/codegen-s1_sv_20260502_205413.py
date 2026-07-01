def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            if char == '@':
                agent_pos = (row_idx, col_idx)
            elif char == 'G':
                goal_pos = (row_idx, col_idx)
            elif char == 'H':
                holes.append((row_idx, col_idx))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    if agent_pos in holes:
        return 0.0
    
    dr = abs(agent_pos[0] - goal_pos[0])
    dc = abs(agent_pos[1] - goal_pos[1])
    distance = dr + dc
    
    base_value = max(0.0, 1.0 - distance / 30.0)
    
    min_row = min(agent_pos[0], goal_pos[0])
    max_row = max(agent_pos[0], goal_pos[0])
    min_col = min(agent_pos[1], goal_pos[1])
    max_col = max(agent_pos[1], goal_pos[1])
    
    blocking_holes = 0
    for hole in holes:
        if min_row <= hole[0] <= max_row and min_col <= hole[1] <= max_col:
            blocking_holes += 1
    
    path_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    if path_area > 0:
        hole_density = blocking_holes / path_area
        base_value *= (1.0 - hole_density * 0.5)
    
    return max(0.0, min(1.0, base_value))
def signal_function(state: str) -> float:
    lines = [line for line in state.strip().split('\n') if line.strip()]
    
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
    
    if agent_pos in holes:
        return 0.0
    
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    nearby_holes = 0
    for hole in holes:
        dist_to_hole = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if dist_to_hole <= 2:
            nearby_holes += 1
    
    base_value = max(0.0, 1.0 - manhattan_dist / 15.0)
    danger_penalty = min(0.4, nearby_holes * 0.1)
    
    value = base_value - danger_penalty
    value = max(0.0, min(1.0, value))
    
    return value
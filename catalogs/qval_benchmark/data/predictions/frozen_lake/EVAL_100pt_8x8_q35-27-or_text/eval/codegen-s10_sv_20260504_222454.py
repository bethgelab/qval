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
    
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    num_holes = len(holes)
    
    holes_nearby = 0
    for hole in holes:
        dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if dist <= 2:
            holes_nearby += 1
    
    dist_value = max(0.0, 1.0 - manhattan_dist / 12.0)
    
    hole_penalty = min(num_holes * 0.025, 0.3)
    nearby_penalty = min(holes_nearby * 0.08, 0.4)
    
    value = dist_value - hole_penalty - nearby_penalty
    value = max(0.0, min(1.0, value))
    
    return value
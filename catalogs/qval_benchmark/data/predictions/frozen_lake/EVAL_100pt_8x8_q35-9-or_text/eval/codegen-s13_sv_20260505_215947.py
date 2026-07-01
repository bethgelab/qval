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
    
    if goal_pos is None or agent_pos is None:
        return 0.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    dist_factor = max(0.0, 1.0 - dist / 30.0)
    
    hole_penalty = 0.0
    for hole in holes:
        if abs(hole[0] - agent_pos[0]) <= 2 and abs(hole[1] - agent_pos[1]) <= 2:
            hole_penalty += 0.1
    
    return dist_factor * (1.0 - hole_penalty)
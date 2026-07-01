def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_positions.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    for hole_pos in hole_positions:
        if agent_pos == hole_pos:
            return 0.0
    
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    min_dist_to_hole = float('inf')
    for hole_pos in hole_positions:
        dist = abs(agent_pos[0] - hole_pos[0]) + abs(agent_pos[1] - hole_pos[1])
        if dist < min_dist_to_hole:
            min_dist_to_hole = dist
    
    max_steps = 30
    if dist_to_goal > max_steps:
        base_value = 0.0
    else:
        base_value = max(0.0, 1.0 - dist_to_goal / max_steps)
    
    if min_dist_to_hole <= 1:
        base_value *= 0.3
    elif min_dist_to_hole <= 2:
        base_value *= 0.6
    elif min_dist_to_hole <= 3:
        base_value *= 0.8
    
    return base_value
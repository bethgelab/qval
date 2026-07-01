def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = [line for line in grid_str.strip().split('\n') if line.strip()]
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
        
        return agent_pos, goal_pos, holes, len(lines)
    
    agent_pos, goal_pos, holes, rows = parse_grid(state)
    next_agent_pos, _, _, _ = parse_grid(next_state)
    
    if goal_pos is None or agent_pos is None or next_agent_pos is None:
        return 0.0
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, goal_pos)
    
    if next_agent_pos == goal_pos:
        return 1.0
    
    if next_agent_pos in holes:
        return 0.0
    
    max_dist = max(rows, 8) * 2
    
    distance_factor = 1.0 - (next_dist / max_dist)
    distance_factor = max(0.0, min(1.0, distance_factor))
    
    if next_dist < current_dist:
        movement_factor = 1.0
    elif next_dist > current_dist:
        movement_factor = 0.3
    else:
        movement_factor = 0.6
    
    hole_risk = 0.0
    for hole in holes:
        dist_to_hole = manhattan_distance(next_agent_pos, hole)
        if dist_to_hole == 1:
            hole_risk += 0.3
        elif dist_to_hole == 2:
            hole_risk += 0.1
    
    safety_factor = max(0.0, 1.0 - hole_risk)
    
    q_value = distance_factor * movement_factor * safety_factor
    
    gamma = 0.9
    discount_factor = gamma ** next_dist
    q_value = q_value * discount_factor
    
    return q_value
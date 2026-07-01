import math

def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
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
    
    if agent_pos == goal_pos:
        return 1.0
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    max_steps = 30
    
    if dist >= max_steps:
        return 0.0
    
    base_value = 1.0 - (dist / max_steps)
    
    min_hole_distance = float('inf')
    for hole in hole_positions:
        hole_dist = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if hole_dist < min_hole_distance:
            min_hole_distance = hole_dist
    
    if min_hole_distance == 0:
        base_value = 0.0
    elif min_hole_distance == 1:
        base_value *= 0.3
    elif min_hole_distance == 2:
        base_value *= 0.5
    elif min_hole_distance == 3:
        base_value *= 0.7
    elif min_hole_distance <= 5:
        base_value *= 0.85
    
    return max(0.0, min(1.0, base_value))
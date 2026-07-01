import math
from collections import Counter

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
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    nearby_holes = sum(1 for hole in holes if abs(hole[0] - agent_pos[0]) + abs(hole[1] - agent_pos[1]) <= 2)
    
    max_dist = 14
    base_value = 1.0 - (dist / max_dist)
    
    hole_penalty = nearby_holes * 0.15
    
    value = max(0.0, min(1.0, base_value - hole_penalty))
    
    return value
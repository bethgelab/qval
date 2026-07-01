import re
import math

def signal_function(state: str) -> float:
    lines = state.split('\n')
    grid_rows = len(lines)
    if grid_rows == 0:
        return 0.0
    
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
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    max_grid_dist = 14
    distance_ratio = min(1.0, manhattan_dist / max_grid_dist)
    distance_score = 1.0 - distance_ratio
    
    steps_remaining = 30 - manhattan_dist
    step_score = max(0.0, steps_remaining / 30.0)
    
    nearby_holes = 0
    for hole in hole_positions:
        if abs(hole[0] - agent_pos[0]) + abs(hole[1] - agent_pos[1]) <= 2:
            nearby_holes += 1
    
    hole_penalty = nearby_holes * 0.15
    
    base_value = (distance_score * 0.6 + step_score * 0.4) - hole_penalty
    return max(0.0, min(1.0, base_value))
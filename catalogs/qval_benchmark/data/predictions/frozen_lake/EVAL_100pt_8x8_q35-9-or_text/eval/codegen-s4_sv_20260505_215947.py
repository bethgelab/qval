import re
from collections import Counter

def signal_function(state: str) -> float:
    lines = state.split('\n')
    grid = [line.strip() for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    if goal_pos is None or agent_pos is None:
        return 0.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    step_limit = 30
    
    if dist > step_limit:
        return 0.0
    
    base_value = 1.0 - (dist / step_limit)
    
    hole_penalty = 0.0
    for hole in holes:
        hole_dist = abs(hole[0] - agent_pos[0]) + abs(hole[1] - agent_pos[1])
        if hole_dist <= 2:
            hole_penalty += 0.15
    
    value = max(0.0, min(1.0, base_value - hole_penalty))
    
    return value
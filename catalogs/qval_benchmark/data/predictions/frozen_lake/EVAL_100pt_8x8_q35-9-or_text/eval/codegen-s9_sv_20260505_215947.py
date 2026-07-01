import math
from collections import Counter

def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == '@':
                agent_pos = (x, y)
            elif cell == 'G':
                goal_pos = (x, y)
            elif cell == 'H':
                holes.append((x, y))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    agent_x, agent_y = agent_pos
    goal_x, goal_y = goal_pos
    manhattan_dist = abs(agent_x - goal_x) + abs(agent_y - goal_y)
    
    min_hole_dist = float('inf')
    for hole_x, hole_y in holes:
        dist = abs(agent_x - hole_x) + abs(agent_y - hole_y)
        min_hole_dist = min(min_hole_dist, dist)
    
    if min_hole_dist == float('inf'):
        min_hole_dist = max_steps * 2
    
    max_steps = 30
    gamma = 0.95
    
    goal_ratio = 1.0 / (1.0 + manhattan_dist / max_steps)
    hole_ratio = 1.0 / (1.0 + min_hole_dist / max_steps)
    
    value = goal_ratio * (1 - gamma ** min_hole_dist)
    
    return max(0.0, min(1.0, value))
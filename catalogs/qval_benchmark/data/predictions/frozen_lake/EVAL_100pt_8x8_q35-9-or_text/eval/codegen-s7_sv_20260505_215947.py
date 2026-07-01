import re

def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [line for line in lines if line.strip()]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(grid):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    if goal_pos is None or agent_pos is None:
        return 0.0
    
    dr = abs(agent_pos[0] - goal_pos[0])
    dc = abs(agent_pos[1] - goal_pos[1])
    manhattan_dist = dr + dc
    
    max_steps = 30
    distance_value = 1.0 - (manhattan_dist / max_steps)
    
    hole_penalty = 0.0
    for hole in holes:
        hole_dr = abs(agent_pos[0] - hole[0])
        hole_dc = abs(agent_pos[1] - hole[1])
        hole_dist = hole_dr + hole_dc
        if hole_dist <= 5:
            hole_penalty += 1.0 / (hole_dist + 1)
    
    value = distance_value * (1.0 - hole_penalty * 0.3)
    
    return max(0.0, min(1.0, value))
import math

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
    
    if agent_pos == goal_pos:
        return 1.0
    
    if agent_pos in holes:
        return 0.0
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    nearby_holes = sum(1 for h in holes 
                      if abs(h[0] - agent_pos[0]) + abs(h[1] - agent_pos[1]) <= 3)
    
    total_cells = len(lines) * len(lines[0]) if lines else 64
    total_holes = len(holes)
    hole_density = total_holes / total_cells if total_cells > 0 else 0
    
    step_limit = 30
    
    if manhattan_dist >= step_limit:
        return 0.01
    
    distance_factor = max(0.1, 1 - manhattan_dist / (step_limit * 1.5))
    
    hole_penalty = min(0.5, nearby_holes * 0.15)
    density_penalty = min(0.3, hole_density * 2)
    
    value = distance_factor * (1 - hole_penalty) * (1 - density_penalty)
    
    return max(0.0, min(1.0, value))
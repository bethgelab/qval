def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [line for line in lines if line.strip()]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for r, line in enumerate(grid):
        for c, cell in enumerate(line):
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    agent_r, agent_c = agent_pos
    nearby_holes = sum(1 for (h_r, h_c) in holes 
                       if abs(h_r - agent_r) <= 2 and abs(h_c - agent_c) <= 2)
    
    base_value = 1.0 - (dist / 14.0)
    hole_penalty = nearby_holes * 0.05
    value = max(0.0, min(1.0, base_value - hole_penalty))
    
    return value
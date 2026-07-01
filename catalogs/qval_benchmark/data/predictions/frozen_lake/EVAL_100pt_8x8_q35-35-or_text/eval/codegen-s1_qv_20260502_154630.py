def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        grid = [line for line in grid_str.split('\n') if line.strip()]
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
        return agent_pos, goal_pos, holes
    
    agent_pos, goal_pos, holes = parse_grid(state)
    next_agent_pos, next_goal_pos, _ = parse_grid(next_state)
    
    if goal_pos is None or agent_pos is None:
        return 0.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    action_dir = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dr, dc = action_dir.get(action, (0, 0))
    
    moved = (next_agent_pos is not None and 
             next_agent_pos != agent_pos and 
             next_goal_pos is not None and
             next_goal_pos == goal_pos)
    
    dist_decreased = (next_agent_pos is not None and 
                      goal_pos is not None and 
                      abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1]) < dist)
    
    moves_toward_goal = (dr * (goal_pos[0] - agent_pos[0]) >= 0 and 
                         dc * (goal_pos[1] - agent_pos[1]) >= 0)
    
    hole_risk = 0.0
    for hr, hc in holes:
        if abs(hr - agent_pos[0]) <= 2 and abs(hc - agent_pos[1]) <= 2:
            hole_risk += 0.15
    
    base_value = 1.0 / (1.0 + 0.1 * dist)
    
    if moved or dist_decreased:
        base_value *= 1.3
    elif not moves_toward_goal:
        base_value *= 0.7
    
    base_value *= (1.0 - hole_risk)
    
    return max(0.0, min(1.0, base_value))
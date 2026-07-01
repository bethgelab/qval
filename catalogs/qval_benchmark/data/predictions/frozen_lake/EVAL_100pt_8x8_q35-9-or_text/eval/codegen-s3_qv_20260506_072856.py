def signal_function(state: str, action: str, next_state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines if line.strip()]
    
    agent_pos = None
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == '@':
                agent_pos = (i, j)
                break
        if agent_pos:
            break
    
    goal_pos = None
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'G':
                goal_pos = (i, j)
                break
        if goal_pos:
            break
    
    if agent_pos and goal_pos:
        dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    else:
        dist = 100
    
    next_lines = next_state.strip().split('\n')
    next_grid = [list(line) for line in next_lines if line.strip()]
    
    next_agent_pos = None
    for i, row in enumerate(next_grid):
        for j, cell in enumerate(row):
            if cell == '@':
                next_agent_pos = (i, j)
                break
        if next_agent_pos:
            break
    
    if next_agent_pos and next_grid[next_agent_pos[0]][next_agent_pos[1]] == 'G':
        return 0.99
    
    if next_agent_pos and next_grid[next_agent_pos[0]][next_agent_pos[1]] == 'H':
        return 0.0
    
    if next_agent_pos and goal_pos:
        new_dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
        max_dist = 14
        q_value = 0.5 * (1.0 - new_dist / max_dist)
    else:
        q_value = 0.3
    
    return q_value
def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == '@':
                agent_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
            elif grid[r][c] == 'H':
                hole_positions.append((r, c))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    d_goal = manhattan_distance(agent_pos, goal_pos)
    
    if not hole_positions:
        d_hole = float('inf')
    else:
        d_hole = min(manhattan_distance(agent_pos, hole) for hole in hole_positions)
    
    if d_goal >= 30:
        return 0.0
    
    value = max(0.0, 1.0 - (d_goal / 30.0))
    
    if d_hole != float('inf'):
        value *= (1.0 - min(d_hole / 30.0, 1.0))
    
    return value
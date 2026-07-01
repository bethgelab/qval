def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
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
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    goal_dist = manhattan_distance(agent_pos, goal_pos)
    min_hole_dist = min(manhattan_distance(agent_pos, hole) for hole in holes) if holes else 100
    
    goal_score = max(0.0, 1.0 - goal_dist / 14.0)
    hole_score = min_hole_dist / 100.0
    
    value = 0.7 * goal_score + 0.3 * hole_score
    
    if goal_dist > 25:
        value *= 0.5
    
    return max(0.0, min(1.0, value))
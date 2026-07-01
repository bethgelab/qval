def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        goal_pos = None
        holes = []
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                if grid[r][c] == '@':
                    agent_pos = (r, c)
                elif grid[r][c] == 'G':
                    goal_pos = (r, c)
                elif grid[r][c] == 'H':
                    holes.append((r, c))
        return grid, agent_pos, goal_pos, holes
    
    grid, agent_pos, goal_pos, holes = parse_grid(state)
    next_grid, next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    if agent_pos is None or goal_pos is None or next_agent_pos is None:
        return 0.0
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, next_goal_pos)
    
    if next_agent_pos in next_holes:
        return 0.0
    
    if next_dist < current_dist:
        base_value = 0.7 + (current_dist - next_dist) / (max(current_dist, 1))
    elif next_dist == current_dist:
        base_value = 0.3
    else:
        base_value = 0.15
    
    if current_dist <= 3:
        base_value += 0.2
    if current_dist <= 5:
        base_value += 0.1
    
    min_hole_dist = float('inf')
    for hole in next_holes:
        hole_dist = manhattan_distance(next_agent_pos, hole)
        if hole_dist < min_hole_dist:
            min_hole_dist = hole_dist
    
    if min_hole_dist <= 2:
        base_value *= 0.4
    elif min_hole_dist <= 4:
        base_value *= 0.7
    
    if current_dist > 25:
        base_value *= 0.5
    
    if next_agent_pos == agent_pos and action in ['left', 'right', 'up', 'down']:
        if current_dist > next_dist:
            base_value += 0.1
    
    return max(0.0, min(1.0, base_value))
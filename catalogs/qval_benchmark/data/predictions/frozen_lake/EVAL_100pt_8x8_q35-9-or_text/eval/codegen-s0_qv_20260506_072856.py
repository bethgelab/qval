def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        return [list(line) for line in lines if line.strip()]
    
    def find_positions(grid):
        positions = {}
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == '@':
                    positions['agent'] = (i, j)
                elif cell == 'G':
                    positions['goal'] = (i, j)
                elif cell == 'H':
                    positions['hole'] = (i, j)
        return positions
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_grid = parse_grid(state)
    next_grid = parse_grid(next_state)
    current_positions = find_positions(current_grid)
    next_positions = find_positions(next_grid)
    
    if 'agent' not in next_positions or 'goal' not in next_positions:
        return 0.0
    
    agent_next = next_positions.get('agent')
    goal_next = next_positions.get('goal')
    
    if agent_next == goal_next:
        return 1.0
    
    if 'hole' in next_positions and next_positions['hole'] == agent_next:
        return 0.0
    
    if 'agent' not in current_positions or 'goal' not in current_positions:
        return 0.5
    
    current_agent = current_positions['agent']
    current_goal = current_positions['goal']
    current_dist = manhattan_distance(current_agent, current_goal)
    next_dist = manhattan_distance(agent_next, goal_next)
    
    if next_dist == 0:
        return 1.0
    
    if next_dist < current_dist:
        progress = (current_dist - next_dist) / max(current_dist, 1)
        return 0.6 + 0.3 * progress
    elif next_dist > current_dist:
        regression = (next_dist - current_dist) / max(current_dist, 1)
        return 0.3 - 0.2 * regression
    else:
        return 0.5
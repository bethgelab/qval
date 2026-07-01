def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        positions = {'@': None, 'G': None, 'H': []}
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == '@':
                    positions['@'] = (i, j)
                elif char == 'G':
                    positions['G'] = (i, j)
                elif char == 'H':
                    positions['H'].append((i, j))
        return positions
    
    agent_pos = parse_grid(state)['@']
    goal_pos = parse_grid(state)['G']
    holes = parse_grid(state)['H']
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    agent_row, agent_col = agent_pos
    goal_row, goal_col = goal_pos
    
    manhattan_dist = abs(agent_row - goal_row) + abs(agent_col - goal_col)
    
    action_to_delta = {
        'left': (0, -1),
        'down': (1, 0),
        'right': (0, 1),
        'up': (-1, 0)
    }
    
    action_delta = action_to_delta.get(action, (0, 0))
    next_agent_row, next_agent_col = agent_row + action_delta[0], agent_col + action_delta[1]
    
    if next_agent_row < 0 or next_agent_row >= 8 or next_agent_col < 0 or next_agent_col >= 8:
        return 0.0
    
    next_state_grid = parse_grid(next_state)
    next_agent_pos = next_state_grid['@']
    
    next_manhattan_dist = abs(next_agent_pos[0] - goal_row) + abs(next_agent_pos[1] - goal_col)
    
    if next_manhattan_dist == 0:
        return 1.0
    
    progress = 1.0 if next_manhattan_dist < manhattan_dist else 0.0
    
    min_hole_dist = float('inf')
    for hole in holes:
        hole_row, hole_col = hole
        dist = abs(agent_row - hole_row) + abs(agent_col - hole_col)
        min_hole_dist = min(min_hole_dist, dist)
    
    if min_hole_dist == float('inf'):
        safety = 1.0
    else:
        safety = 1.0 - min_hole_dist * 0.1
    
    q_value = progress * safety * 0.8
    
    return max(0.0, min(1.0, q_value))
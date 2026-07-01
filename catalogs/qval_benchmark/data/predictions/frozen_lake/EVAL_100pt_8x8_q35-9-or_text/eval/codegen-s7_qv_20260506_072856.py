import re

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        positions = {'@': None, 'G': None, 'H': set()}
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == '@':
                    positions['@'] = (x, y)
                elif char == 'G':
                    positions['G'] = (x, y)
                elif char == 'H':
                    positions['H'].add((x, y))
        
        return positions
    
    current_positions = parse_grid(state)
    next_positions = parse_grid(next_state)
    
    if current_positions['@'] is None or next_positions['@'] is None:
        return 0.0
    
    current_pos = current_positions['@']
    next_pos = next_positions['@']
    goal_pos = current_positions['G']
    
    if goal_pos is None:
        return 0.0
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_dist = manhattan_distance(current_pos, goal_pos)
    next_dist = manhattan_distance(next_pos, goal_pos)
    distance_change = next_dist - current_dist
    
    if next_pos == goal_pos:
        return 0.95
    elif next_pos in next_positions['H']:
        return -1.0
    
    if distance_change < 0:
        base_value = 0.5 + min(abs(distance_change) * 0.15, 0.4)
    elif distance_change == 0:
        base_value = 0.3
    else:
        base_value = 0.15
    
    dx = 0
    dy = 0
    if action == 'left':
        dx = -1
    elif action == 'right':
        dx = 1
    elif action == 'up':
        dy = -1
    elif action == 'down':
        dy = 1
    
    expected_next = (current_pos[0] + dx, current_pos[1] + dy)
    if expected_next != next_pos:
        if expected_next[0] < 0 or expected_next[0] >= 8 or expected_next[1] < 0 or expected_next[1] >= 8:
            base_value *= 0.5
        else:
            base_value *= 0.7
    
    return base_value
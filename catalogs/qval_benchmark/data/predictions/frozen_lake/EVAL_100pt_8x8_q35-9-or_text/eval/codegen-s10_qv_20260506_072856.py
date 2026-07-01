import re

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_text):
        lines = grid_text.strip().split('\n')
        positions = {'@': None, 'G': None, 'H': None}
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == '@':
                    positions['@'] = (y, x)
                elif char == 'G':
                    positions['G'] = (y, x)
                elif char == 'H':
                    positions['H'] = (y, x)
        return positions
    
    state_positions = parse_grid(state)
    next_positions = parse_grid(next_state)
    
    # Check if we reached the goal in next_state
    if next_positions['@'] == next_positions['G']:
        return 1.0
    
    # Check if we fell into a hole in next_state
    if next_positions['@'] == next_positions['H']:
        return -0.5
    
    # Calculate distance to goal from current state
    if state_positions['@'] and state_positions['G']:
        current_y, current_x = state_positions['@']
        goal_y, goal_x = state_positions['G']
        manhattan_dist = abs(current_y - goal_y) + abs(current_x - goal_x)
    else:
        manhattan_dist = 16  # Default max distance for 8x8 grid
    
    # Check proximity to holes in next_state
    if next_positions['@'] and next_positions['H']:
        agent_y, agent_x = next_positions['@']
        hole_y, hole_x = next_positions['H']
        distance_to_hole = abs(agent_y - hole_y) + abs(agent_x - hole_x)
        if distance_to_hole <= 2:
            hole_penalty = 0.5
        elif distance_to_hole <= 4:
            hole_penalty = 0.3
        else:
            hole_penalty = 0.1
    else:
        hole_penalty = 0.0
    
    # Estimate progress toward goal
    progress_score = max(0, 1 - manhattan_dist / 16)
    
    # Combine progress and safety factors
    q_value = progress_score * (1 - hole_penalty)
    
    # Ensure Q-value is in reasonable range
    return max(-0.5, min(1.0, q_value))
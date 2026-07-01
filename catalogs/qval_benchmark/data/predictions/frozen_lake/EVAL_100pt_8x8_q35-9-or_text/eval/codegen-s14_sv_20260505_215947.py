import re

def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == '@':
                agent_pos = (row_idx, col_idx)
            elif cell == 'G':
                goal_pos = (row_idx, col_idx)
            elif cell == 'H':
                hole_positions.append((row_idx, col_idx))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    agent_row, agent_col = agent_pos
    goal_row, goal_col = goal_pos
    
    manhattan_dist = abs(agent_row - goal_row) + abs(agent_col - goal_col)
    max_dist = 14
    
    distance_factor = max(0.0, 1.0 - (manhattan_dist / max_dist))
    
    danger_count = 0
    for hole_row, hole_col in hole_positions:
        if abs(hole_row - agent_row) <= 2 and abs(hole_col - agent_col) <= 2:
            danger_count += 1
    
    danger_factor = max(0.0, 1.0 - (danger_count * 0.15))
    
    return distance_factor * danger_factor
def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        hole_positions = []
        
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                elif char == 'H':
                    hole_positions.append((row_idx, col_idx))
        
        return agent_pos, goal_pos, hole_positions
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    if next_agent_pos is None or next_goal_pos is None:
        return 0.0
    
    if next_agent_pos in next_holes:
        return -1.0
    
    if next_agent_pos == next_goal_pos:
        return 1.0
    
    distance = manhattan_distance(next_agent_pos, next_goal_pos)
    
    if distance > 30:
        return 0.1
    
    q_value = 1.0 / (distance + 1)
    
    return q_value
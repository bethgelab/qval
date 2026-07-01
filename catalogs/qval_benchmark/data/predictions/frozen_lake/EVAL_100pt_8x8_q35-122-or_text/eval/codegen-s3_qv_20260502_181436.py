def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = set()
        safe_tiles = set()
        
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                    safe_tiles.add((row_idx, col_idx))
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                    safe_tiles.add((row_idx, col_idx))
                elif char == 'H':
                    holes.add((row_idx, col_idx))
                elif char == 'S' or char == ' ' or char == '.':
                    safe_tiles.add((row_idx, col_idx))
        
        return agent_pos, goal_pos, holes, safe_tiles
    
    state_agent, state_goal, state_holes, state_safe = parse_grid(state)
    next_agent, next_goal, next_holes, next_safe = parse_grid(next_state)
    
    if state_goal is None:
        return 0.0
    
    if next_agent is None:
        return 0.0
    
    if next_agent in next_holes:
        return 0.0
    
    if next_agent == state_goal:
        return 1.0
    
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_dist = manhattan_distance(state_agent, state_goal)
    next_dist = manhattan_distance(next_agent, state_goal)
    
    steps_remaining = 30
    
    if current_dist >= steps_remaining:
        base_value = 0.0
    else:
        base_value = 1.0 - (current_dist / steps_remaining)
    
    progress_bonus = 0.0
    if next_dist < current_dist:
        progress_bonus = 0.15
    elif next_dist > current_dist:
        progress_bonus = -0.1
    
    hole_risk = 0.0
    if next_agent in next_holes:
        hole_risk = -1.0
    
    q_value = base_value + progress_bonus + hole_risk
    
    q_value = q_value * math.exp(-0.1 * next_dist)
    
    return max(0.0, min(1.0, q_value))
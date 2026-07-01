def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        rows = len(lines)
        cols = len(lines[0]) if lines else 0
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))
        
        return agent_pos, goal_pos, holes, rows, cols
    
    agent_pos, goal_pos, holes, rows, cols = parse_grid(state)
    next_agent_pos, next_goal_pos, next_holes, next_rows, next_cols = parse_grid(next_state)
    
    # Terminal: reached goal
    if next_agent_pos == next_goal_pos:
        return 1.0
    
    # Terminal: fell into hole
    if next_agent_pos in next_holes:
        return -1.0
    
    # Calculate Manhattan distance to goal from next position
    if goal_pos and next_agent_pos:
        dist_to_goal = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    else:
        dist_to_goal = 100
    
    # Check if action moved toward or away from goal
    moved_toward = False
    moved_away = False
    if agent_pos and next_agent_pos and goal_pos:
        old_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        if next_agent_pos != agent_pos:
            moved_toward = dist_to_goal < old_dist
            moved_away = dist_to_goal > old_dist
    
    # Base score from distance to goal (closer = better)
    max_dist = rows + cols
    dist_score = max(0.0, 1.0 - dist_to_goal / max_dist)
    
    # Direction bonus/penalty
    if moved_toward:
        direction_bonus = 0.15
    elif moved_away:
        direction_bonus = -0.15
    else:
        direction_bonus = 0.0
    
    # Hole proximity penalty
    min_hole_dist = float('inf')
    if next_agent_pos and next_holes:
        for hole in next_holes:
            hole_dist = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
            min_hole_dist = min(min_hole_dist, hole_dist)
    
    hole_penalty = 0.0
    if min_hole_dist <= 1:
        hole_penalty = 0.3
    elif min_hole_dist <= 3:
        hole_penalty = 0.15
    
    # Combine components
    q_value = dist_score + direction_bonus - hole_penalty
    
    # Clamp to reasonable range
    return max(-1.0, min(1.0, q_value))
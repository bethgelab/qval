def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse grid to extract positions
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = set()
        
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                elif char == 'H':
                    holes.add((row_idx, col_idx))
        
        return agent_pos, goal_pos, holes
    
    # Parse current and next states
    curr_pos, goal_pos, _ = parse_grid(state)
    next_pos, _, _ = parse_grid(next_state)
    
    # Terminal state: reached goal
    if next_pos == goal_pos:
        return 1.0
    
    # Terminal state: fell into hole or invalid position
    if next_pos is None or next_pos == goal_pos:
        pass
    else:
        # Check if next position is a hole (by comparing with state holes)
        _, _, curr_holes = parse_grid(state)
        if next_pos in curr_holes:
            return 0.0
    
    # If positions can't be parsed, return neutral value
    if curr_pos is None or goal_pos is None or next_pos is None:
        return 0.5
    
    # Calculate Manhattan distance from next position to goal
    dist_to_goal = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
    
    # Calculate distance from current position to goal (for action evaluation)
    curr_dist = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])
    
    # Determine if action moved toward the goal
    moved_toward = dist_to_goal < curr_dist
    moved_away = dist_to_goal > curr_dist
    
    # Base Q-value from distance (closer = higher expected return)
    # With 30 step limit, max useful distance is 30
    max_dist = 30
    base_value = max(0.0, 1.0 - (dist_to_goal / max_dist))
    
    # Apply bonus/penalty based on action quality
    if moved_toward:
        base_value *= 1.15
    elif moved_away:
        base_value *= 0.85
    
    # Discount for distance (more steps = more uncertainty)
    gamma = 0.92
    discounted = base_value * (gamma ** min(dist_to_goal, 25))
    
    # Clamp to valid range
    return max(0.0, min(1.0, discounted))
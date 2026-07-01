def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        """Parse ASCII grid to find positions"""
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
        
        return agent_pos, goal_pos, hole_positions, len(lines)
    
    # Parse states
    agent_pos, goal_pos, hole_positions, grid_size = parse_grid(state)
    next_agent_pos, _, _, _ = parse_grid(next_state)
    
    # Check terminal conditions in next_state
    if next_agent_pos == goal_pos:
        return 1.0
    
    if next_agent_pos in hole_positions:
        return 0.0
    
    if agent_pos is None or goal_pos is None or next_agent_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, goal_pos)
    
    # Check if action moved toward goal
    moved_closer = next_dist < current_dist
    moved_away = next_dist > current_dist
    
    # Base Q-value from distance (closer = higher value)
    # Max distance in 8x8 grid is 14 (0,0 to 7,7)
    max_dist = 14
    distance_score = 1.0 - (next_dist / max_dist)
    
    # Action improvement bonus
    if moved_closer:
        action_bonus = 0.15
    elif moved_away:
        action_bonus = -0.1
    else:
        action_bonus = 0.0
    
    # Safety check - holes nearby
    hole_risk = 0.0
    for hole in hole_positions:
        hole_dist = manhattan_distance(next_agent_pos, hole)
        if hole_dist == 1:  # Adjacent to hole
            hole_risk += 0.25
        elif hole_dist == 2:  # Near hole
            hole_risk += 0.08
    
    # Cap hole risk
    hole_risk = min(hole_risk, 0.5)
    
    # Combine factors
    q_value = distance_score + action_bonus - hole_risk
    
    # Ensure reasonable bounds
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value
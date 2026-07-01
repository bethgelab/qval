def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the grid states to extract positions
    def parse_grid(grid_str):
        lines = [line for line in grid_str.strip().split('\n') if line.strip()]
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
    
    # Get positions from states
    agent_pos, goal_pos, _ = parse_grid(state)
    next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    # Check if agent fell into a hole in next_state
    if next_agent_pos in next_holes:
        return -1.0
    
    # Check if agent reached the goal
    if next_agent_pos == goal_pos:
        return 1.0
    
    # If no goal found or agent position missing, return neutral
    if goal_pos is None or next_agent_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_distance = manhattan_distance(agent_pos, goal_pos)
    next_distance = manhattan_distance(next_agent_pos, goal_pos)
    
    # Estimate Q-value based on distance and progress
    max_distance = 14  # Max Manhattan distance on 8x8 grid
    
    # Base value from distance (closer = higher)
    base_value = 1.0 - (next_distance / max_distance)
    
    # Bonus for making progress toward goal
    progress_bonus = 0.15 if next_distance < current_distance else 0.0
    
    # Penalty for moving away from goal
    setback_penalty = -0.15 if next_distance > current_distance else 0.0
    
    # Q-value combines distance-based value with progress indicators
    q_value = base_value + progress_bonus + setback_penalty
    
    return q_value
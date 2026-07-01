def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.strip().split('\n')
    
    # Find agent position, goal position, and hole positions
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
    
    # Check if agent has fallen in a hole (game over)
    if agent_pos in hole_positions:
        return 0.0
    
    # Check if agent is on goal
    if agent_pos == goal_pos:
        return 1.0
    
    # If agent or goal not found, invalid state
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Base value decreases with distance (max dist in 8x8 is 14)
    max_dist = 14
    base_value = 1.0 - (manhattan_dist / max_dist)
    
    # Penalize for holes that might block the path
    # Count holes in the bounding box between agent and goal
    hole_penalty = 0.0
    min_row = min(agent_pos[0], goal_pos[0])
    max_row = max(agent_pos[0], goal_pos[0])
    min_col = min(agent_pos[1], goal_pos[1])
    max_col = max(agent_pos[1], goal_pos[1])
    
    for hole in hole_positions:
        if min_row <= hole[0] <= max_row and min_col <= hole[1] <= max_col:
            hole_penalty += 0.03
    
    # Cap the penalty to avoid negative values
    hole_penalty = min(hole_penalty, 0.4)
    
    # Also check if agent is very close to a hole (adjacent)
    adjacent_holes = 0
    for hole in hole_positions:
        if abs(hole[0] - agent_pos[0]) + abs(hole[1] - agent_pos[1]) <= 1:
            adjacent_holes += 1
    
    adjacent_penalty = min(adjacent_holes * 0.1, 0.3)
    
    final_value = max(0.0, base_value - hole_penalty - adjacent_penalty)
    
    return final_value
def signal_function(state: str) -> float:
    # Parse the state to extract grid information
    lines = state.strip().split('\n')
    
    # Find agent, goal, and hole positions
    agent_pos = None
    goal_pos = None
    holes = set()
    grid_width = 0
    grid_height = len(lines)
    
    for y, line in enumerate(lines):
        grid_width = max(grid_width, len(line))
        for x, char in enumerate(line):
            if char == '@':
                agent_pos = (x, y)
            elif char == 'G':
                goal_pos = (x, y)
            elif char == 'H':
                holes.add((x, y))
    
    # If no agent found, return 0
    if agent_pos is None:
        return 0.0
    
    # If at goal, return 1.0 (episode complete with reward)
    if agent_pos == goal_pos:
        return 1.0
    
    # If on a hole, return 0.0 (episode failed)
    if agent_pos in holes:
        return 0.0
    
    # Calculate Manhattan distance to goal
    if goal_pos is None:
        return 0.0
    
    dx = abs(agent_pos[0] - goal_pos[0])
    dy = abs(agent_pos[1] - goal_pos[1])
    manhattan_dist = dx + dy
    
    # Count holes in the bounding box between agent and goal
    min_x = min(agent_pos[0], goal_pos[0])
    max_x = max(agent_pos[0], goal_pos[0])
    min_y = min(agent_pos[1], goal_pos[1])
    max_y = max(agent_pos[1], goal_pos[1])
    
    holes_in_path = 0
    for hx, hy in holes:
        if min_x <= hx <= max_x and min_y <= hy <= max_y:
            holes_in_path += 1
    
    # Calculate value based on distance and obstacles
    # With 30 step limit, paths longer than 30 are essentially worthless
    max_useful_dist = 30
    
    # Base value from distance (closer = higher value)
    distance_value = 1.0 - (manhattan_dist / max_useful_dist)
    distance_value = max(0.0, distance_value)
    
    # Risk factor based on holes in potential path
    # More holes = higher risk of falling
    hole_penalty = holes_in_path * 0.05
    risk_factor = 1.0 - hole_penalty
    risk_factor = max(0.3, risk_factor)
    
    # Final value
    value = distance_value * risk_factor
    
    return value
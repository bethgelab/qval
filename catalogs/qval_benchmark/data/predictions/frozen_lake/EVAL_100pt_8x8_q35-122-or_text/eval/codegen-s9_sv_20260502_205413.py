def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == '@':
                agent_pos = (x, y)
            elif char == 'G':
                goal_pos = (x, y)
            elif char == 'H':
                hole_positions.append((x, y))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 1.0
    
    agent_x, agent_y = agent_pos
    goal_x, goal_y = goal_pos
    
    manhattan_distance = abs(agent_x - goal_x) + abs(agent_y - goal_y)
    
    if manhattan_distance > 30:
        return 0.0
    
    base_value = 1.0 - (manhattan_distance / 30.0)
    
    holes_in_bounding_box = 0
    min_x = min(agent_x, goal_x)
    max_x = max(agent_x, goal_x)
    min_y = min(agent_y, goal_y)
    max_y = max(agent_y, goal_y)
    
    for hole_x, hole_y in hole_positions:
        if min_x <= hole_x <= max_x and min_y <= hole_y <= max_y:
            holes_in_bounding_box += 1
    
    obstacle_penalty = min(0.5, holes_in_bounding_box * 0.1)
    
    final_value = max(0.0, base_value - obstacle_penalty)
    
    return final_value
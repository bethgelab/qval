def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = set()
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.add((r, c))
        
        return agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    state_agent, state_goal, state_holes = parse_grid(state)
    next_agent, next_goal, next_holes = parse_grid(next_state)
    
    # Immediate failure: agent fell into a hole
    if next_agent in next_holes:
        return 0.0
    
    # Immediate success: agent reached the goal
    if next_agent == state_goal:
        return 1.0
    
    # Calculate distance from next state to goal
    dist_to_goal = manhattan_distance(next_agent, state_goal)
    
    if dist_to_goal == float('inf'):
        return 0.0
    
    # Base value inversely related to distance (closer = higher Q)
    base_value = math.exp(-0.25 * dist_to_goal)
    
    # Adjust based on whether action moved agent closer to goal
    if state_agent is not None and next_agent is not None:
        old_dist = manhattan_distance(state_agent, state_goal)
        if old_dist != float('inf'):
            if dist_to_goal < old_dist:
                base_value *= 1.3  # Moved closer
            elif dist_to_goal > old_dist:
                base_value *= 0.6  # Moved away
            else:
                base_value *= 0.9  # Same distance
    
    # Penalty for nearby holes (risk of future failure)
    hole_penalty = 0.0
    if next_agent is not None:
        for hole in next_holes:
            if manhattan_distance(next_agent, hole) <= 2:
                hole_penalty += 0.05
    
    base_value -= hole_penalty
    
    # Clip to valid Q-value range
    return max(0.0, min(1.0, base_value))
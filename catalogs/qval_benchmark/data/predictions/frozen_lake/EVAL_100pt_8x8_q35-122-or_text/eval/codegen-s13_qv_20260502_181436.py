def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))
        
        return agent_pos, goal_pos, holes
    
    # Parse the next state to evaluate the outcome of the action
    agent_pos, goal_pos, holes = parse_grid(next_state)
    
    # Check for invalid state
    if agent_pos is None:
        return 0.0
    
    # Terminal state: fell into hole
    if agent_pos in holes:
        return -1.0
    
    # Terminal state: reached goal
    if goal_pos is not None and agent_pos == goal_pos:
        return 1.0
    
    # Non-terminal state: estimate based on distance to goal
    if goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distance to goal
    distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Maximum possible Manhattan distance on 8x8 grid is 14 (0,0 to 7,7)
    max_distance = 14
    
    # Base Q-value decreases with distance from goal
    # Closer to goal = higher expected return
    base_q = 1.0 - (distance / max_distance)
    
    # Check if action made progress compared to previous state
    prev_agent_pos, _, _ = parse_grid(state)
    if prev_agent_pos is not None:
        prev_distance = abs(prev_agent_pos[0] - goal_pos[0]) + abs(prev_agent_pos[1] - goal_pos[1])
        
        # If action moved away from goal, penalize
        if distance > prev_distance:
            base_q *= 0.6
        # If action stayed in same place (blocked or wrong direction)
        elif distance == prev_distance:
            base_q *= 0.8
    
    # Ensure Q-value stays in reasonable range
    return max(-1.0, min(1.0, base_q))
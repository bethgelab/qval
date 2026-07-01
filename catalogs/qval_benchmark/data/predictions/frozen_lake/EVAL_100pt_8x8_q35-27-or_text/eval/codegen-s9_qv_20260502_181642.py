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
    
    curr_agent, curr_goal, curr_holes = parse_grid(state)
    next_agent, next_goal, next_holes = parse_grid(next_state)
    
    # Goal reached in next state - immediate success
    if next_agent and next_goal and next_agent == next_goal:
        return 1.0
    
    # Fell into hole in next state - immediate failure
    if next_agent and next_agent in next_holes:
        return 0.0
    
    # Cannot compute meaningful estimate without valid positions
    if not next_agent or not next_goal:
        return 0.5
    
    # Manhattan distance to goal
    dist = abs(next_agent[0] - next_goal[0]) + abs(next_agent[1] - next_goal[1])
    
    # Maximum Manhattan distance on 8x8 grid is 14 (0,0 to 7,7)
    max_dist = 14
    
    # Base Q-value: inverse relationship with distance to goal
    # Closer states have higher probability of reaching goal within step limit
    base_q = max(0.0, 1.0 - (dist / max_dist))
    
    # Adjust based on whether action moved towards or away from goal
    if curr_agent and curr_goal:
        curr_dist = abs(curr_agent[0] - curr_goal[0]) + abs(curr_agent[1] - curr_goal[1])
        if dist < curr_dist:
            # Moved closer to goal - positive signal
            base_q = min(1.0, base_q + 0.15)
        elif dist > curr_dist:
            # Moved away from goal - negative signal
            base_q = max(0.0, base_q - 0.15)
    
    # Penalty for holes nearby (increased risk of failure)
    hole_penalty = 0.0
    for hole in next_holes:
        hole_dist = abs(next_agent[0] - hole[0]) + abs(next_agent[1] - hole[1])
        if hole_dist <= 2:
            hole_penalty += 0.08
        elif hole_dist <= 4:
            hole_penalty += 0.03
    
    base_q = max(0.0, base_q - hole_penalty)
    
    # Bonus for very close proximity to goal (high confidence of success)
    if dist <= 2:
        base_q = min(1.0, base_q + 0.2)
    
    # Clamp to valid Q-value range
    return max(0.0, min(1.0, base_q))
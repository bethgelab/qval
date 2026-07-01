def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        goal_pos = None
        holes = []
        
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (r, c)
                elif cell == 'G':
                    goal_pos = (r, c)
                elif cell == 'H':
                    holes.append((r, c))
        
        return agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # Parse current and next states
    agent_pos, goal_pos, holes = parse_grid(state)
    next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    # Check if goal was reached in next state
    if next_goal_pos is not None and next_agent_pos == next_goal_pos:
        return 1.0
    
    # Check if agent fell into a hole in next state
    if next_agent_pos in next_holes:
        return 0.0
    
    # Calculate distances
    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, goal_pos)
    
    # Handle edge cases
    if next_dist == float('inf') or next_dist == 0:
        return 0.0
    
    # Base Q-value inversely proportional to distance to goal
    base_q = 1.0 / (next_dist + 1)
    
    # Bonus for moving closer to goal (efficiency matters)
    if next_dist < current_dist:
        improvement = current_dist - next_dist
        q_value = base_q * (1.0 + 0.15 * improvement)
    else:
        # No progress or moving away - lower value
        q_value = base_q * 0.7
    
    # Penalize if next state is near holes
    if next_agent_pos:
        for hole_pos in next_holes:
            hole_dist = manhattan_distance(next_agent_pos, hole_pos)
            if hole_dist <= 2:
                q_value *= 0.8
    
    return q_value
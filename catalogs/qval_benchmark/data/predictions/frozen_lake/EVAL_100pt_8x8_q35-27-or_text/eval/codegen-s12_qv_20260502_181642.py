def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        return [list(line) for line in lines]
    
    def find_positions(grid):
        agent_pos = None
        goal_pos = None
        holes = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (i, j)
                elif cell == 'G':
                    goal_pos = (i, j)
                elif cell == 'H':
                    holes.append((i, j))
        return agent_pos, goal_pos, holes
    
    # Parse both states
    grid = parse_grid(state)
    agent_pos, goal_pos, holes = find_positions(grid)
    
    next_grid = parse_grid(next_state)
    next_agent_pos, next_goal_pos, next_holes = find_positions(next_grid)
    
    # Check if goal reached in next state
    if next_agent_pos == next_goal_pos:
        return 1.0
    
    # Check if fell into hole
    if next_agent_pos in next_holes:
        return 0.0
    
    # Use goal from next_state for distance calculation
    goal_to_use = next_goal_pos if next_goal_pos else goal_pos
    
    # Calculate distances
    if goal_to_use and next_agent_pos:
        dist_to_goal = abs(next_agent_pos[0] - goal_to_use[0]) + abs(next_agent_pos[1] - goal_to_use[1])
    else:
        dist_to_goal = 16  # Max possible Manhattan distance on 8x8
    
    # Check if action moved toward goal
    if agent_pos and next_agent_pos and goal_to_use:
        old_dist = abs(agent_pos[0] - goal_to_use[0]) + abs(agent_pos[1] - goal_to_use[1])
        if dist_to_goal < old_dist:
            progress_bonus = 0.3
        elif dist_to_goal > old_dist:
            progress_bonus = -0.3
        else:
            progress_bonus = 0.0
    else:
        progress_bonus = 0.0
    
    # Base Q-value from distance (closer = better)
    # Normalize: dist 0 -> 1.0, dist 16 -> ~0.1
    base_value = 1.0 / (1.0 + dist_to_goal * 0.2)
    
    # Apply progress adjustment
    q_value = base_value + progress_bonus
    
    # Cap between 0 and 1
    return max(0.0, min(1.0, q_value))
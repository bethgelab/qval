def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines if line]
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
        
        return grid, agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def action_direction(action):
        if action == 'up':
            return (-1, 0)
        elif action == 'down':
            return (1, 0)
        elif action == 'left':
            return (0, -1)
        elif action == 'right':
            return (0, 1)
        return (0, 0)
    
    # Parse states
    _, agent_pos, goal_pos, holes = parse_grid(state)
    _, next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    # Calculate distances
    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    next_dist_to_goal = manhattan_distance(next_agent_pos, next_goal_pos) if next_agent_pos else float('inf')
    
    # Check immediate outcomes
    reached_goal = next_agent_pos == next_goal_pos
    fell_in_hole = next_agent_pos in next_holes
    
    # Calculate distance change
    if dist_to_goal == float('inf'):
        dist_change = 0
    elif next_dist_to_goal == float('inf'):
        dist_change = 0
    else:
        dist_change = dist_to_goal - next_dist_to_goal
    
    # Base Q-value estimation
    # Discount factor approximation (steps remaining in episode)
    discount = 0.99
    
    # Immediate reward component
    if reached_goal:
        immediate_reward = 1.0
    elif fell_in_hole:
        immediate_reward = 0.0
    else:
        immediate_reward = 0.0
    
    # Value from distance reduction (fewer steps = higher value)
    # Normalize by max possible distance (14 for 8x8 grid)
    max_dist = 14
    distance_bonus = max(0, dist_change) / max_dist * 0.5
    
    # Penalty for holes nearby
    hole_penalty = 0.0
    if next_agent_pos:
        for hole in next_holes:
            hole_dist = manhattan_distance(next_agent_pos, hole)
            if hole_dist <= 2:
                hole_penalty += 0.1
    
    # Action direction bonus (moving toward goal is better)
    direction_bonus = 0.0
    if agent_pos and goal_pos and next_agent_pos:
        action_dir = action_direction(action)
        goal_dir = (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])
        if goal_dir != (0, 0):
            goal_dir_norm = (goal_dir[0] / abs(goal_dir[0] + goal_dir[1]), 
                           goal_dir[1] / abs(goal_dir[0] + goal_dir[1]))
            dot_product = action_dir[0] * goal_dir_norm[0] + action_dir[1] * goal_dir_norm[1]
            direction_bonus = max(0, dot_product) * 0.3
    
    # Combine components
    q_value = immediate_reward + distance_bonus + direction_bonus - hole_penalty
    
    # Apply discount based on remaining steps (estimate)
    if not reached_goal and not fell_in_hole:
        q_value *= discount
    
    # Ensure value is in reasonable range [0, 1]
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value
def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse grid from string representation
    def parse_grid(grid_str):
        lines = [line for line in grid_str.strip().split('\n') if line.strip()]
        return lines
    
    # Find key positions in grid
    def find_positions(grid):
        agent = None
        goal = None
        holes = []
        
        for r, row in enumerate(grid):
            for c, char in enumerate(row):
                if char == '@':
                    agent = (r, c)
                elif char == 'G':
                    goal = (r, c)
                elif char == 'H':
                    holes.append((r, c))
        
        return agent, goal, holes
    
    # Parse both states
    state_grid = parse_grid(state)
    next_grid = parse_grid(next_state)
    
    state_agent, state_goal, _ = find_positions(state_grid)
    next_agent, _, next_holes = find_positions(next_grid)
    
    # Terminal condition: reached goal
    if next_agent == state_goal:
        return 1.0
    
    # Terminal condition: fell in hole
    if next_agent in next_holes:
        return -1.0
    
    # Calculate Manhattan distance to goal
    if next_agent and state_goal:
        distance = abs(next_agent[0] - state_goal[0]) + abs(next_agent[1] - state_goal[1])
    else:
        distance = 14  # Max possible on 8x8 grid
    
    # Action direction mapping
    action_map = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }
    
    # Estimate if action moves toward goal
    moves_toward = False
    if action in action_map and next_agent and state_goal:
        # Check if action direction aligns with goal direction
        action_dir = action_map[action]
        goal_dir = (state_goal[0] - next_agent[0], state_goal[1] - next_agent[1])
        # If action component matches goal component direction
        if action_dir[0] * goal_dir[0] >= 0 and action_dir[1] * goal_dir[1] >= 0:
            moves_toward = True
    
    # Q-value estimation
    # Closer to goal = higher probability of success
    # Max Manhattan distance on 8x8 is 14 (from corner to corner)
    distance_factor = max(0, 1.0 - distance / 14.0)
    
    # Small bonus for taking valid action
    action_bonus = 0.05
    
    # Bonus for moving toward goal
    progress_bonus = 0.1 if moves_toward else 0.0
    
    # Step limit consideration (30 steps max)
    # With 30 steps, we have room to maneuver
    step_factor = 0.9
    
    q_value = (distance_factor + action_bonus + progress_bonus) * step_factor
    
    # Clamp to valid range
    return max(-1.0, min(1.0, q_value))
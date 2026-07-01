def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == '@':
                    agent_pos = (i, j)
                elif char == 'G':
                    goal_pos = (i, j)
                elif char == 'H':
                    holes.append((i, j))
        
        return agent_pos, goal_pos, holes, len(lines), len(lines[0]) if lines else 0
    
    agent_pos, goal_pos, holes, rows, cols = parse_grid(state)
    next_agent_pos, next_goal_pos, next_holes, next_rows, next_cols = parse_grid(next_state)
    
    # Agent disappeared - likely fell in hole
    if next_agent_pos is None:
        return 0.0
    
    # Check if goal reached
    if next_agent_pos == next_goal_pos:
        return 1.0
    
    # Check if fell in hole
    if next_agent_pos in next_holes:
        return 0.0
    
    # No goal found
    if next_goal_pos is None:
        return 0.0
    
    # Calculate Manhattan distances
    current_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    next_dist = abs(next_agent_pos[0] - next_goal_pos[0]) + abs(next_agent_pos[1] - next_goal_pos[1])
    
    max_dist = rows + cols
    
    # Base score from proximity to goal
    dist_score = 1.0 - (next_dist / max_dist)
    
    # Direction bonus/penalty
    if next_dist < current_dist:
        direction_bonus = 0.2
    elif next_dist > current_dist:
        direction_bonus = -0.15
    else:
        direction_bonus = 0.0
    
    # Hole proximity penalty
    hole_penalty = 0.0
    for hole in next_holes:
        hole_dist = abs(next_agent_pos[0] - hole[0]) + abs(next_agent_pos[1] - hole[1])
        if hole_dist == 1:
            hole_penalty += 0.25
        elif hole_dist == 2:
            hole_penalty += 0.1
    
    # Action effectiveness based on action type
    action_bonus = 0.0
    if action == 'left' and next_agent_pos[1] < agent_pos[1]:
        action_bonus = 0.05
    elif action == 'right' and next_agent_pos[1] > agent_pos[1]:
        action_bonus = 0.05
    elif action == 'up' and next_agent_pos[0] < agent_pos[0]:
        action_bonus = 0.05
    elif action == 'down' and next_agent_pos[0] > agent_pos[0]:
        action_bonus = 0.05
    
    # Combine all factors
    q_value = dist_score + direction_bonus - hole_penalty + action_bonus
    
    # Apply step limit consideration (assume ~15 steps used on average)
    step_factor = 0.9
    
    q_value = q_value * step_factor
    
    # Clamp to valid range
    return max(0.0, min(1.0, q_value))
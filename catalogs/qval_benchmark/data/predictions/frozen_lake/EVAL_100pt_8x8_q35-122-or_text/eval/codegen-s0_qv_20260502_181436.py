def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse state and next_state to extract grid information
    state_lines = [line for line in state.strip().split('\n') if line.strip()]
    next_lines = [line for line in next_state.strip().split('\n') if line.strip()]
    
    # Find agent position, goal position in current state
    current_pos = None
    goal_pos = None
    
    for r, line in enumerate(state_lines):
        for c, char in enumerate(line):
            if char == '@':
                current_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # Find agent position and check for hole/goal in next state
    next_pos = None
    next_is_goal = False
    next_is_hole = False
    
    for r, line in enumerate(next_lines):
        for c, char in enumerate(line):
            if char == '@':
                next_pos = (r, c)
            elif char == 'G':
                next_is_goal = True
            elif char == 'H':
                next_is_hole = True
    
    # If agent fell into a hole, Q-value is 0 (episode ends with 0 reward)
    if next_is_hole:
        return 0.0
    
    # If agent reached the goal, Q-value is 1.0 (episode ends with 1.0 reward)
    if next_is_goal:
        return 1.0
    
    # If we couldn't parse positions, return neutral estimate
    if current_pos is None or next_pos is None or goal_pos is None:
        return 0.5
    
    # Calculate Manhattan distance to goal
    current_dist = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
    next_dist = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
    
    # Maximum possible Manhattan distance on 8x8 grid is 14
    max_dist = 14
    
    # Base probability decreases with distance to goal
    # Closer to goal = higher probability of reaching it
    base_prob = 1.0 - (next_dist / max_dist)
    
    # Bonus for moving toward goal
    progress = current_dist - next_dist
    if progress > 0:
        base_prob += 0.15  # Moving closer increases probability
    elif progress < 0:
        base_prob -= 0.15  # Moving away decreases probability
    
    # Check if action direction makes sense
    action_map = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }
    
    if action in action_map:
        dr, dc = action_map[action]
        # Check if action was valid (agent moved)
        if next_pos != current_pos:
            # Action resulted in movement, validate it was toward goal
            goal_dr = goal_pos[0] - next_pos[0]
            goal_dc = goal_pos[1] - next_pos[1]
            
            # If action moved agent in the general direction of goal, boost Q-value
            if (dr == 0 and dc * goal_dc > 0) or (dc == 0 and dr * goal_dr > 0):
                base_prob += 0.1
            # If action moved away from goal, reduce Q-value
            elif (dr == 0 and dc * goal_dc < 0) or (dc == 0 and dr * goal_dr < 0):
                base_prob -= 0.1
    
    # Account for remaining steps (30 step limit)
    # If very close to goal, high probability; if far, lower probability
    # Discount slightly based on distance (more steps needed = more risk)
    step_penalty = next_dist * 0.02
    base_prob -= step_penalty
    
    # Clamp to valid probability range
    return max(0.0, min(1.0, base_prob))
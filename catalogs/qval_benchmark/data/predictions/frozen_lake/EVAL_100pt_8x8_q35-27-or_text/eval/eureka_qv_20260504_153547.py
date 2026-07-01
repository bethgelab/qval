def signal_function(state: str, action: str, next_state: str):
    # Parse the state grid
    lines = state.strip().split('\n')
    height = len(lines)
    width = len(lines[0]) if lines else 0
    
    # Find agent, goal, and holes
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
        return 0.0, {"base": 0.0}
    
    agent_x, agent_y = agent_pos
    goal_x, goal_y = goal_pos
    
    # Check if already at goal
    if agent_pos == goal_pos:
        return 1.0, {"goal_reached": 1.0}
    
    # Manhattan distance to goal
    dist_to_goal = abs(goal_x - agent_x) + abs(goal_y - agent_y)
    
    # Minimum distance to any hole
    min_dist_to_hole = float('inf')
    for hx, hy in hole_positions:
        dist = abs(hx - agent_x) + abs(hy - agent_y)
        min_dist_to_hole = min(min_dist_to_hole, dist)
    if min_dist_to_hole == float('inf'):
        min_dist_to_hole = 100
    
    # Parse next state
    next_lines = next_state.strip().split('\n')
    next_agent_pos = None
    for y, line in enumerate(next_lines):
        for x, char in enumerate(line):
            if char == '@':
                next_agent_pos = (x, y)
                break
        if next_agent_pos:
            break
    
    # Check if we fell into a hole
    fell_into_hole = False
    if next_agent_pos and next_agent_pos != agent_pos:
        if next_agent_pos in hole_positions:
            fell_into_hole = True
    
    # Calculate distance change
    if next_agent_pos:
        next_dist_to_goal = abs(goal_x - next_agent_pos[0]) + abs(goal_y - next_agent_pos[1])
        dist_change = dist_to_goal - next_dist_to_goal
        made_progress = dist_change > 0
        stuck = next_agent_pos == agent_pos
    else:
        dist_change = 0
        made_progress = False
        stuck = True
    
    # Determine action direction
    action_dx, action_dy = 0, 0
    if action == 'left':
        action_dx = -1
    elif action == 'right':
        action_dx = 1
    elif action == 'up':
        action_dy = -1
    elif action == 'down':
        action_dy = 1
    
    # Check if action is toward goal
    goal_dx = 1 if goal_x > agent_x else (-1 if goal_x < agent_x else 0)
    goal_dy = 1 if goal_y > agent_y else (-1 if goal_y < agent_y else 0)
    
    align_with_goal = (action_dx == goal_dx and goal_dx != 0) or (action_dy == goal_dy and goal_dy != 0)
    
    # Check if moving toward any hole
    moving_toward_hole = False
    if action_dx != 0 or action_dy != 0:
        for hx, hy in hole_positions:
            hole_dx = 1 if hx > agent_x else (-1 if hx < agent_x else 0)
            hole_dy = 1 if hy > agent_y else (-1 if hy < agent_y else 0)
            if (action_dx == hole_dx and hole_dx != 0) or (action_dy == hole_dy and hole_dy != 0):
                moving_toward_hole = True
                break
    
    # Check if goal is blocked (nearby holes around goal)
    goal_blocked = False
    if dist_to_goal <= 4:
        hole_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = goal_x + dx, goal_y + dy
            if 0 <= nx < width and 0 <= ny < height:
                cell = lines[ny][nx]
                if cell == 'H':
                    hole_neighbors += 1
        if hole_neighbors >= 1:
            goal_blocked = True
    
    # Q-value components (calibrated to prevent overestimation)
    
    # Proximity to goal (0 to 0.40) - more aggressive decay for far distances
    if dist_to_goal <= 3:
        proximity_score = 0.40 * (1.0 - dist_to_goal / 6.0)
    else:
        proximity_score = 0.40 * max(0.0, 1.0 - dist_to_goal / 12.0)
    
    # Direction bonus for moving toward goal (0 to 0.15)
    direction_bonus = 0.15 if align_with_goal else 0.0
    
    # Movement bonus for actual progress (0 to 0.10)
    movement_bonus = max(0.0, dist_change) * 0.10
    
    # Safety score - positive for being away from holes (0 to 0.20)
    if min_dist_to_hole <= 1:
        safety_score = 0.0
    elif min_dist_to_hole <= 3:
        safety_score = 0.20 * (min_dist_to_hole - 1) / 2.0
    else:
        safety_score = 0.20
    
    # Step efficiency (0 to 0.15)
    estimated_steps_needed = dist_to_goal + 3
    steps_remaining = 30 - estimated_steps_needed
    if steps_remaining <= 0:
        step_efficiency = 0.0
    else:
        step_efficiency = 0.15 * min(1.0, steps_remaining / 30.0)
    
    # Penalties (negative values)
    
    # Hole penalty if action leads to hole
    hole_penalty = -0.50 if fell_into_hole else 0.0
    
    # Penalty for moving toward holes when close
    hole_direction_penalty = 0.0
    if moving_toward_hole and min_dist_to_hole <= 3:
        hole_direction_penalty = -0.15
    
    # Goal blocked penalty
    goal_blocked_penalty = -0.15 if goal_blocked else 0.0
    
    # Stuck penalty
    stuck_penalty = -0.10 if stuck and not fell_into_hole else 0.0
    
    total = proximity_score + direction_bonus + movement_bonus + safety_score + step_efficiency + hole_penalty + hole_direction_penalty + goal_blocked_penalty + stuck_penalty
    
    # Cap at 0.99 for non-terminal states
    if agent_pos != goal_pos:
        total = min(0.99, max(0.0, total))
    
    return total, {
        "proximity_score": proximity_score,
        "direction_bonus": direction_bonus,
        "movement_bonus": movement_bonus,
        "safety_score": safety_score,
        "step_efficiency": step_efficiency,
        "hole_penalty": hole_penalty,
        "hole_direction_penalty": hole_direction_penalty,
        "goal_blocked_penalty": goal_blocked_penalty,
        "stuck_penalty": stuck_penalty,
    }
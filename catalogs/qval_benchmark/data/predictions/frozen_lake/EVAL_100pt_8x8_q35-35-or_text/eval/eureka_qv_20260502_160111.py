def signal_function(state: str, action: str, next_state: str):
    import math
    
    def parse_grid(s):
        lines = s.strip().split('\n')
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
        return grid, agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def action_direction(action):
        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        return directions.get(action, (0, 0))
    
    # Parse states
    grid, agent_pos, goal_pos, holes = parse_grid(state)
    next_grid, next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    # Check if goal reached in next state
    goal_reached = next_goal_pos is not None and next_agent_pos == next_goal_pos
    
    # Check if action hits a hole
    hits_hole = next_agent_pos is not None and next_agent_pos in next_holes
    
    # Calculate distance to goal
    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    
    # Calculate distance after action
    if next_agent_pos is not None and goal_pos is not None:
        dist_after = manhattan_distance(next_agent_pos, goal_pos)
        dist_improvement = dist_to_goal - dist_after
    else:
        dist_improvement = 0
    
    # Check if action moves toward goal
    moves_toward = False
    moves_away = False
    if agent_pos is not None and goal_pos is not None and next_agent_pos is not None:
        direction = action_direction(action)
        expected_pos = (agent_pos[0] + direction[0], agent_pos[1] + direction[1])
        if expected_pos == next_agent_pos:
            dist_before = manhattan_distance(agent_pos, goal_pos)
            dist_after = manhattan_distance(next_agent_pos, goal_pos)
            if dist_after < dist_before:
                moves_toward = True
            elif dist_after > dist_before:
                moves_away = True
    
    # Safety assessment: check proximity to holes
    safety_penalty = 0.0
    safety_bonus = 0.0
    if agent_pos is not None and holes:
        min_dist_to_hole = float('inf')
        for hole in holes:
            dist_to_hole = manhattan_distance(agent_pos, hole)
            min_dist_to_hole = min(min_dist_to_hole, dist_to_hole)
        
        # Check if action moves toward nearest hole
        moves_toward_hole = False
        moves_away_from_hole = False
        if next_agent_pos is not None:
            dist_before_hole = min_dist_to_hole
            next_nearest_hole = min(holes, key=lambda h: manhattan_distance(next_agent_pos, h))
            dist_after_hole = manhattan_distance(next_agent_pos, next_nearest_hole)
            if dist_after_hole < dist_before_hole:
                moves_toward_hole = True
            elif dist_after_hole > dist_before_hole:
                moves_away_from_hole = True
        
        # Calibrated safety penalty based on proximity (reduced from previous version)
        if min_dist_to_hole == 1:
            safety_penalty -= 0.25
        elif min_dist_to_hole == 2:
            safety_penalty -= 0.12
        elif min_dist_to_hole == 3:
            safety_penalty -= 0.05
        elif min_dist_to_hole == 4:
            safety_penalty -= 0.02
        else:
            safety_penalty -= 0.005
        
        # Additional penalty for moving toward a hole
        if moves_toward_hole:
            safety_penalty -= 0.15
        # Bonus for moving away from a hole
        elif moves_away_from_hole:
            safety_bonus += 0.04
    
    # Discount factor: exponential decay based on estimated steps to goal
    # Using gamma=0.95 per step for sparse reward alignment
    gamma_per_step = 0.95
    if dist_to_goal > 0:
        estimated_steps = dist_to_goal
        discount_factor = gamma_per_step ** estimated_steps
        discount_factor = max(0.01, discount_factor)
    else:
        discount_factor = 1.0
    
    # Check if goal reached in next state
    if goal_reached:
        return 0.99, {
            "goal_reward": 0.99,
            "discount_factor": discount_factor,
            "dist_to_goal": dist_to_goal,
        }
    
    # Penalty for hitting hole
    if hits_hole:
        return -0.90, {
            "hole_penalty": -0.90,
            "discount_factor": discount_factor,
            "dist_to_goal": dist_to_goal,
        }
    
    # Base value from exponential discount of goal reward
    base_value = discount_factor
    
    # Progress-based adjustment
    progress_multiplier = 1.0
    if dist_improvement > 0:
        progress_multiplier = 1.0 + 0.10 * dist_improvement
    elif dist_improvement < 0:
        progress_multiplier = 1.0 - 0.08 * abs(dist_improvement)
    
    # Directional bonus/penalty
    if moves_toward:
        progress_multiplier *= 1.08
    elif moves_away:
        progress_multiplier *= 0.92
    
    # Step-level improvement bonus
    if dist_improvement > 0:
        progress_multiplier += 0.03 * dist_improvement
    
    # Apply safety adjustments
    total = base_value * progress_multiplier + safety_penalty + safety_bonus
    
    # Ensure minimum floor for exploration
    total = max(0.01, total)
    
    # Clamp to valid range
    total = min(0.99, total)
    
    return total, {
        "base_value": base_value,
        "progress_multiplier": progress_multiplier,
        "safety_penalty": safety_penalty,
        "safety_bonus": safety_bonus,
        "discount_factor": discount_factor,
        "dist_to_goal": dist_to_goal,
        "dist_improvement": dist_improvement,
        "moves_toward": 1.0 if moves_toward else 0.0,
        "moves_away": 1.0 if moves_away else 0.0,
        "moves_toward_hole": 1.0 if moves_toward_hole else 0.0,
        "moves_away_from_hole": 1.0 if moves_away_from_hole else 0.0,
        "goal_reached": 1.0 if goal_reached else 0.0,
        "hits_hole": 1.0 if hits_hole else 0.0,
    }
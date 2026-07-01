def signal_function(state: str, action: str, next_state: str) -> float:
    import re
    
    # Parse state to find agent and goal positions
    def parse_positions(s):
        lines = s.strip().split('\n')
        agent_pos = None
        goal_pos = None
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
        return agent_pos, goal_pos
    
    # Parse next_state to check if agent fell in hole or reached goal
    def check_next_state(s):
        lines = s.strip().split('\n')
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    # Check if this position was a hole in original state
                    return 'agent', (row_idx, col_idx)
                elif char == 'H' and '@' not in s:
                    # Agent fell in hole (hole visible, no agent)
                    return 'hole', None
        return 'goal', None
    
    # Calculate Manhattan distance
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # Get action delta
    def get_action_delta(a):
        if a == 'up':
            return (-1, 0)
        elif a == 'down':
            return (1, 0)
        elif a == 'left':
            return (0, -1)
        elif a == 'right':
            return (0, 1)
        return (0, 0)
    
    # Parse positions from states
    agent_pos, goal_pos = parse_positions(state)
    next_agent_pos, _ = parse_positions(next_state)
    
    # Check next state outcome
    next_type, _ = check_next_state(next_state)
    
    # If agent fell in hole, Q-value is 0 (terminal failure)
    if next_type == 'hole':
        return 0.0
    
    # If agent reached goal, Q-value is 1.0 (terminal success)
    if next_type == 'goal' or (next_agent_pos and next_agent_pos == goal_pos):
        return 1.0
    
    # Calculate distances
    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, goal_pos)
    
    # If no valid positions found, return 0
    if current_dist == float('inf') or next_dist == float('inf'):
        return 0.0
    
    # Calculate action quality
    # Positive if moving toward goal, negative if moving away
    action_delta = get_action_delta(action)
    goal_delta = (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])
    
    # Dot product to measure if action aligns with goal direction
    if goal_delta[0] == 0 and goal_delta[1] == 0:
        action_alignment = 1.0
    else:
        norm = (goal_delta[0]**2 + goal_delta[1]**2) ** 0.5
        action_alignment = (action_delta[0] * goal_delta[0] + action_delta[1] * goal_delta[1]) / norm
    
    # Base Q-value from distance (closer = higher value)
    # Use exponential decay with distance
    gamma = 0.95  # Discount factor per step
    base_value = gamma ** next_dist
    
    # Adjust for action quality
    if action_alignment > 0:
        # Moving toward goal - boost value
        value = base_value * (1 + 0.1 * action_alignment)
    else:
        # Moving away from goal - reduce value
        value = base_value * (1 - 0.2 * abs(action_alignment))
    
    # Penalize if distance increased (bad action)
    if next_dist > current_dist:
        value *= 0.7
    elif next_dist < current_dist:
        value *= 1.1
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, value))
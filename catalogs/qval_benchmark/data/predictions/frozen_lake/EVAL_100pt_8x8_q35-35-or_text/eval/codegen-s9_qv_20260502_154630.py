import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Parse the grid from the next_state string
    rows = next_state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    # Scan the grid to find positions
    for r_idx, row in enumerate(rows):
        if not row:
            continue
        for c_idx, char in enumerate(row):
            if char == '@':
                agent_pos = (r_idx, c_idx)
            elif char == 'G':
                goal_pos = (r_idx, c_idx)
            elif char == 'H':
                hole_positions.append((r_idx, c_idx))
    
    # If agent position is not found, assume invalid state
    if agent_pos is None:
        return 0.0
    
    # If goal is not found, assume no reward possible
    if goal_pos is None:
        return 0.0
        
    # Check immediate outcome based on next_state
    # Reaching the goal yields 1.0 reward (episode terminates)
    if agent_pos == goal_pos:
        return 1.0
    
    # Falling into a hole yields 0.0 reward (episode terminates)
    for h_pos in hole_positions:
        if agent_pos == h_pos:
            return 0.0
            
    # Estimate value based on distance to goal (Manhattan distance)
    dist_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Calculate distance to nearest hole for safety assessment
    dist_hole = float('inf')
    for h_pos in hole_positions:
        d = abs(agent_pos[0] - h_pos[0]) + abs(agent_pos[1] - h_pos[1])
        if d < dist_hole:
            dist_hole = d
            
    # Base value decays with distance to goal
    # Max distance on 8x8 is 14. Value is 1.0 at dist 0, decreasing thereafter.
    # Using inverse relationship to approximate probability of reaching goal.
    base_value = 1.0 / (1.0 + dist_goal)
    
    # Safety penalty: proximity to holes reduces expected value
    safety_factor = 1.0
    if dist_hole < 2:
        safety_factor = 0.1
    elif dist_hole < 4:
        safety_factor = 0.5
    elif dist_hole < dist_goal:
        # If a hole is closer than the goal, path is risky
        safety_factor = 0.6
        
    estimated_q = base_value * safety_factor
    
    # Ensure result is within valid range [0.0, 1.0]
    return max(0.0, min(1.0, estimated_q))
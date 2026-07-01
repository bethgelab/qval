import math

def signal_function(state: str) -> float:
    # Clean the input string to handle potential spaces or carriage returns
    clean_state = state.replace(' ', '').replace('\r', '').strip()
    lines = clean_state.split('\n')
    
    # Flatten the grid to a list of characters and find positions
    # We assume an 8x8 grid, but we'll parse dynamically based on content
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Iterate through lines to find positions
    for r_idx, line in enumerate(lines):
        for c_idx, char in enumerate(line):
            if char == '@':
                agent_pos = (r_idx, c_idx)
            elif char == 'G':
                goal_pos = (r_idx, c_idx)
            elif char == 'H':
                holes.append((r_idx, c_idx))
    
    # If agent or goal not found, return 0.0 (invalid or end state)
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Check if agent is on the goal
    if agent_pos == goal_pos:
        return 1.0
    
    # Check if agent is on a hole
    for hole in holes:
        if agent_pos == hole:
            return 0.0
            
    # Calculate Manhattan distance to goal
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Calculate minimum Manhattan distance to any hole
    min_dist_to_hole = float('inf')
    for hole in holes:
        d = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
        if d < min_dist_to_hole:
            min_dist_to_hole = d
            
    # Base value decreases with distance to goal
    # Using 1/(1+d) to normalize between 0 and 1
    base_value = 1.0 / (1.0 + dist_to_goal)
    
    # Safety penalty for proximity to holes
    # If a hole is very close, reduce value significantly
    safety_factor = 1.0
    if min_dist_to_hole != float('inf'):
        # If hole is adjacent (dist 1), factor is 0.5
        # If hole is dist 2, factor is 0.66
        # If hole is dist >= 3, factor is 1.0
        penalty = max(0.0, 3.0 - min_dist_to_hole)
        safety_factor = 1.0 / (1.0 + penalty)
    
    estimated_value = base_value * safety_factor
    
    # Clamp between 0.0 and 1.0
    return max(0.0, min(1.0, estimated_value))
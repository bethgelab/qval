def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    # Parse the grid to find positions
    # We assume a grid structure where columns are aligned.
    # We will track column index based on valid grid characters to handle spacing.
    for r, line in enumerate(lines):
        c = 0
        for char in line:
            if char in ['@', 'G', 'H', 'S', '.']:
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))
                c += 1
    
    # If state is invalid or missing key components
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # If agent is on goal, value is 1.0 (terminal success)
    if agent_pos == goal_pos:
        return 1.0
    
    # If agent is on a hole, value is 0.0 (terminal failure)
    for h in holes:
        if agent_pos == h:
            return 0.0
            
    # Calculate Manhattan distance to goal
    dist_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Calculate minimum Manhattan distance to any hole
    min_dist_hole = float('inf')
    for h in holes:
        d = abs(agent_pos[0] - h[0]) + abs(agent_pos[1] - h[1])
        if d < min_dist_hole:
            min_dist_hole = d
            
    # Heuristic estimation
    # Base value decreases with distance to goal (efficiency preference)
    # Value is 1.0 at dist 0, decays towards 0 as dist increases
    base_value = 1.0 / (1.0 + dist_goal)
    
    # Apply penalty for proximity to holes
    if min_dist_hole == float('inf'):
        hole_factor = 1.0
    else:
        # If hole is close, factor is smaller. If far, factor approaches 1.
        hole_factor = min_dist_hole / (1.0 + min_dist_hole)
        
    estimated_value = base_value * hole_factor
    
    # Clamp to valid probability range [0.0, 1.0]
    return max(0.0, min(1.0, estimated_value))
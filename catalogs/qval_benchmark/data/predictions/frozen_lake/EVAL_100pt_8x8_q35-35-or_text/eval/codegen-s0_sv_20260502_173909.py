import math

def signal_function(state: str) -> float:
    lines = state.splitlines()
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    # Parse the grid to find agent, goal, and holes
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_positions.append((r, c))
    
    # Validate state presence
    if agent_pos is None or goal_pos is None:
        return 0.0
        
    ra, ca = agent_pos
    rg, cg = goal_pos
    
    # Calculate Manhattan distance to goal
    dist_goal = abs(ra - rg) + abs(ca - cg)
    
    # If agent is at goal, value is maximum
    if dist_goal == 0:
        return 1.0
    
    # Calculate minimum Manhattan distance to any hole
    min_dist_hole = float('inf')
    for hr, hc in hole_positions:
        d = abs(ra - hr) + abs(ca - hc)
        if d < min_dist_hole:
            min_dist_hole = d
    
    # Determine safety factor based on proximity to holes
    # If no holes exist, safety is 1.0
    if min_dist_hole == float('inf'):
        safety = 1.0
    else:
        # Heuristic: probability of success is roughly proportional to 
        # distance to safety relative to distance to goal
        safety = min_dist_hole / (min_dist_hole + dist_goal)
    
    # Discount factor based on steps required (Manhattan distance)
    # Using gamma = 0.95 to reflect preference for fewer steps
    gamma = 0.95
    discount = math.pow(gamma, dist_goal)
    
    # Estimated state value
    value = safety * discount
    return value
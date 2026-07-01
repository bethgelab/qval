import math

def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a given 8x8 Frozen Lake grid state.
    V(s) is approximated based on the distance to the goal and the density
    of holes (H) in the bounding box between the agent (@) and the goal (G).
    """
    # 1. Parse the grid from the string representation
    rows = []
    for line in state.strip().split('\n'):
        # Filter out whitespace to handle various grid formats
        row = [c for c in line if not c.isspace()]
        if row:
            rows.append(row)
    
    if not rows:
        return 0.0
            
    agent_pos = None
    goal_pos = None
    holes = []
    
    # 2. Locate the agent, the goal, and all holes
    for r in range(len(rows)):
        for c in range(len(rows[r])):
            char = rows[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
    
    # If vital components are missing, we cannot estimate value
    if agent_pos is None or goal_pos is None:
        return 0.0
            
    r1, c1 = agent_pos
    r2, c2 = goal_pos
    
    # 3. Calculate Manhattan distance as the primary driver for value (discounting)
    dist = abs(r1 - r2) + abs(c1 - c2)
    
    # If distance exceeds the step limit, the goal is unreachable
    if dist > 30:
        return 0.0
    if dist == 0:
        return 1.0
            
    # 4. Analyze hole density in the rectangular area between agent and goal
    # This serves as a heuristic for the probability of success (avoiding holes)
    r_min, r_max = min(r1, r2), max(r1, r2)
    c_min, c_max = min(c1, c2), max(c1, c2)
    area = (r_max - r_min + 1) * (c_max - c_min + 1)
    
    hole_count = 0
    for hr, hc in holes:
        if r_min <= hr <= r_max and c_min <= hc <= c_max:
            hole_count += 1
    
    density = hole_count / area
    
    # 5. Compute estimated value
    # V(s) is modeled as: (gamma^dist) * (safety_factor^dist)
    # where gamma is the discount factor and safety_factor represents the 
    # likelihood of a step being safe.
    # Using an exponential dampening factor for density to avoid being overly punitive.
    gamma = 0.95
    safety_multiplier = math.exp(-0.5 * density)
    
    try:
        # Combined factor accounts for both discounting and risk of failure per step
        v = math.pow(gamma * safety_multiplier, dist)
    except (OverflowError, ValueError):
        v = 0.0
            
    return float(v)
def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake grid.
    V(s) is approximated based on Manhattan distance to the goal, 
    the presence of holes in the direct path (bounding box), 
    and the immediate proximity of hazards.
    """
    # Parse the ASCII grid
    lines = [line for line in state.strip().split('\n') if line.strip()]
    if not lines:
        return 0.0
    
    # Initialize positions
    ar, ac = -1, -1  # Agent
    gr, gc = -1, -1  # Goal
    
    # Extract coordinates of the agent and the goal
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                ar, ac = r, c
            elif char == 'G':
                gr, gc = r, c
    
    # If agent or goal is missing from the state, value is 0.0
    if ar == -1 or gr == -1:
        return 0.0
    
    # Calculate Manhattan distance
    dist = abs(ar - gr) + abs(ac - gc)
    
    # Immediate success
    if dist == 0:
        return 1.0
    
    # Heuristic 1: Count holes in the bounding box between agent and goal.
    # This acts as a proxy for obstacles that might force a detour or cause failure.
    holes_in_box = 0
    r_min, r_max = min(ar, gr), max(ar, gr)
    c_min, c_max = min(ac, gc), max(ac, gc)
    for r in range(r_min, r_max + 1):
        if r < len(lines):
            for c in range(c_min, c_max + 1):
                if c < len(lines[r]):
                    if lines[r][c] == 'H':
                        holes_in_box += 1
    
    # Heuristic 2: Check for holes immediately adjacent to the agent.
    # This accounts for immediate risk.
    adjacent_holes = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < len(lines) and 0 <= nc < len(lines[nr]):
            if lines[nr][nc] == 'H':
                adjacent_holes += 1
    
    # Combine features into a value estimate
    # Distance decay: Higher distance reduces the probability of successful reaching.
    # Bounding box holes: Each hole in the way reduces the estimated value.
    # Local risk: Adjacent holes further penalize the current state.
    # Base coefficients are chosen to keep the value within [0, 1].
    dist_factor = 0.95 ** dist
    box_factor = 0.80 ** holes_in_box
    risk_factor = 0.90 ** adjacent_holes
    
    value = dist_factor * box_factor * risk_factor
    
    return float(max(0.0, min(1.0, value)))
def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake 8x8 grid.
    V(s) represents the expected discounted cumulative reward (1.0 at goal).
    """
    # 1. Parse the ASCII grid from the state string
    lines = [line.strip() for line in state.strip().split('\n') if line.strip()]
    grid = []
    for line in lines:
        # Extract valid grid characters from the line
        row = [char for char in line if char in ('S', 'F', 'H', 'G', '@')]
        if row:
            grid.append(row)
    
    if not grid:
        return 0.0

    agent_pos = None
    goal_pos = None
    holes = []

    # Locate agent, goal, and holes
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))
                
    # Fallback: If '@' is not found, it might be that the goal is reached or agent is on 'S'
    # However, per prompt, '@' marks the agent.
    if agent_pos is None:
        return 0.0
    if goal_pos is None:
        return 0.0

    # Distance to goal
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If the agent is already on the goal (dist 0)
    if dist_to_goal == 0:
        return 1.0

    # 2. Calculate Risk Factor
    # Risk is based on the proximity of holes to the agent and the goal.
    risk = 0.0
    for hr, hc in holes:
        d_h = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if d_h == 1:
            # Immediate danger: one wrong move leads to a hole
            risk += 0.35
        elif d_h == 2:
            # Moderate danger
            risk += 0.15
            
    # Also consider if the goal itself is difficult to reach (surrounded by holes)
    for hr, hc in holes:
        if abs(hr - goal_pos[0]) + abs(hc - goal_pos[1]) == 1:
            risk += 0.1

    # 3. Estimate State Value
    # We use a discount factor (gamma) to represent the expected discounted reward.
    # V(s) approx gamma^dist * (1 - risk_penalty)
    gamma = 0.93
    # Ensure risk doesn't nullify the value unless it's high, and ensure value is non-negative.
    safety_factor = max(0.0, 1.0 - risk)
    value = (gamma ** dist_to_goal) * safety_factor

    return float(value)
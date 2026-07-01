def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The estimate is based on the proximity to the goal and the proximity to holes in the next state.
    """
    def parse_grid(s: str):
        """Parses the ASCII grid representation from the string."""
        lines = s.split('\n')
        grid = []
        for line in lines:
            # Remove trailing whitespace and newlines but preserve leading spaces
            row = list(line.rstrip('\r\n'))
            # A Frozen Lake grid is 8x8, so we look for lines that are at least 8 characters
            if len(row) >= 8:
                grid.append(row)
        return grid

    grid_s = parse_grid(state)
    grid_ns = parse_grid(next_state)

    if not grid_s or not grid_ns:
        return 0.0

    # Collect all goal and hole positions from both grids to account for the agent
    # potentially covering them in the 'next_state' (where they are replaced by '@').
    goals = set()
    holes = set()
    for g in [grid_s, grid_ns]:
        for r in range(len(g)):
            for c in range(len(g[r])):
                char = g[r][c]
                if char == 'G':
                    goals.add((r, c))
                elif char == 'H':
                    holes.add((r, c))

    # Find the agent's current position in the next_state
    agent_pos = None
    for r in range(len(grid_ns)):
        for c in range(len(grid_ns[r])):
            if grid_ns[r][c] == '@':
                agent_pos = (r, c)
                break
        if agent_pos:
            break
    
    # If @ is not in next_state (should not happen in valid transitions), check state
    if not agent_pos:
        for r in range(len(grid_s)):
            for c in range(len(grid_s[r])):
                if grid_s[r][c] == '@':
                    agent_pos = (r, c)
                    break
            if agent_pos:
                break
                
    if not agent_pos:
        return 0.0

    # Determine if the agent has reached a terminal state
    if agent_pos in goals:
        return 1.0
    if agent_pos in holes:
        return 0.0

    # If no goal is found on the map, the value is zero
    if not goals:
        return 0.0
    
    # The goal is the first one found in the set
    goal_pos = next(iter(goals))
    
    # Manhattan distance to the goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Base value estimate using a discount factor relative to the distance
    # A distance of 0 gives 1.0, distance of 14 gives ~0.22
    val = 0.9 ** dist
    
    # Risk assessment: proximity to holes
    # If the agent is adjacent to one or more holes, reduce the estimated value
    risk_count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
        if (nr, nc) in holes:
            risk_count += 1
            
    if risk_count > 0:
        # Use a penalty factor for proximity to holes
        val *= (0.6 ** risk_count)
        
    return float(val)
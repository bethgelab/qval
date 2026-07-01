def signal_function(state: str) -> float:
    # Parse the grid from the state string
    lines = state.splitlines()
    # Remove empty lines that might result from trailing newlines
    lines = [line for line in lines if line]
    
    if not lines:
        return 0.0

    agent_pos = None
    goal_pos = None
    
    # Scan grid to find Agent (@) and Goal (G) positions
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    
    # If agent or goal is missing, value is 0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # If agent is already at the goal, value is 1.0
    if agent_pos == goal_pos:
        return 1.0

    # Calculate Manhattan distance as a proxy for optimal path length
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # If distance exceeds step limit, value is 0
    if dist > 30:
        return 0.0

    # Base value decreases exponentially with distance (discount factor ~0.95)
    # This reflects the preference for fewer steps and the sparse reward structure
    value = 0.95 ** dist

    # Check for immediate danger (holes adjacent to agent)
    # Neighbors: Up, Down, Left, Right
    neighbors = [
        (agent_pos[0] - 1, agent_pos[1]),
        (agent_pos[0] + 1, agent_pos[1]),
        (agent_pos[0], agent_pos[1] - 1),
        (agent_pos[0], agent_pos[1] + 1)
    ]

    hole_adj_count = 0
    for nr, nc in neighbors:
        # Check bounds safely
        if 0 <= nr < len(lines):
            row_len = len(lines[nr])
            if 0 <= nc < row_len:
                if lines[nr][nc] == 'H':
                    hole_adj_count += 1
        # Moving off-grid keeps agent in place, so out-of-bounds is safe (not a hole)

    # Apply penalty for proximity to holes
    # Each adjacent hole reduces the estimated value
    if hole_adj_count > 0:
        value *= (1.0 - 0.1 * hole_adj_count)

    # Clamp value to valid range [0.0, 1.0]
    return max(0.0, min(1.0, value))
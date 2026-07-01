def signal_function(state: str, action: str, next_state: str) -> float:
    # Helper to find positions
    def find_positions(grid_str):
        rows = grid_str.split('\n')
        agent = None
        goal = None
        holes = []
        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                if char == '@':
                    agent = (r, c)
                elif char == 'G':
                    goal = (r, c)
                elif char == 'H':
                    holes.append((r, c))
        return agent, goal, holes

    # Parse state and next_state
    agent_pos, goal_pos, holes = find_positions(state)
    next_agent_pos, _, _ = find_positions(next_state)

    # Fallback for invalid states
    if agent_pos is None or goal_pos is None:
        return 0.0

    # Check terminal conditions in next_state
    # Case 1: Agent marker present in next_state
    if next_agent_pos is not None:
        if next_agent_pos == goal_pos:
            return 1.0
        if next_agent_pos in holes:
            return 0.0
    # Case 2: Agent marker missing in next_state (Terminal state where agent removed)
    else:
        # Check the tile where the agent was in state
        r, c = agent_pos
        rows_next = next_state.split('\n')
        if r < len(rows_next) and c < len(rows_next[r]):
            tile = rows_next[r][c]
            if tile == 'G':
                return 1.0
            if tile == 'H':
                return 0.0
        return 0.0

    # Calculate distances
    dist_curr = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    dist_next = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])

    # Base Value: Inverse relationship with distance
    base_val = 1.0 / (1.0 + dist_curr)

    # Move Value: Reward progress, penalize regress or wall
    move_val = 0.0
    if dist_next < dist_curr:
        move_val = 0.25
    elif dist_next > dist_curr:
        move_val = -0.15
    elif next_agent_pos != agent_pos:
        move_val = -0.05  # Moved but distance unchanged (sideways)
    else:
        move_val = -0.10  # Hit wall (no position change)

    # Hole Risk: Penalize if adjacent to hole in current state
    hole_risk = 0.0
    for h in holes:
        if abs(agent_pos[0] - h[0]) + abs(agent_pos[1] - h[1]) == 1:
            hole_risk = -0.20
            break

    # Combine and clamp
    total = base_val + move_val + hole_risk
    return max(0.0, min(1.0, total))
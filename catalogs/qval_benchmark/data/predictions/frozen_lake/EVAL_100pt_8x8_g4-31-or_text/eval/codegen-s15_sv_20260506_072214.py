def signal_function(state: str) -> float:
    """
    Estimates the state-value V(s) for a Frozen Lake 8x8 grid.
    The value is approximated based on the Manhattan distance to the goal
    and the proximity of hazards (holes).
    """
    # Robustly parse the grid into an 8x8 structure.
    # We look for standard Frozen Lake characters: Agent (@), Goal (G), Hole (H), and Empty (.) or Start (S).
    all_chars = [char for char in state if char in 'SGH@.']
    if len(all_chars) >= 64:
        grid = ["".join(all_chars[i:i+8]) for i in range(0, 64, 8)]
    else:
        # Handle cases where the grid might be provided as 8 lines with potential spaces.
        lines = [line.strip() for line in state.strip().split('\n') if line.strip()]
        grid = []
        for line in lines:
            # Normalize the line to 8 characters, treating spaces as empty cells.
            line = line.replace(' ', '.')
            line = line[:8].ljust(8, '.')
            grid.append(line)
        # Pad if there are fewer than 8 lines.
        while len(grid) < 8:
            grid.append('.' * 8)
        grid = grid[:8]

    agent_pos = None
    goal_pos = None
    holes = []
    
    # Identify key positions on the map.
    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.append((r, c))

    # If essential elements are missing, the state value is undefined/zero.
    if not agent_pos or not goal_pos:
        return 0.0
    
    # Agent is already at the goal.
    if agent_pos == goal_pos:
        return 1.0

    # Manhattan distance to the goal.
    d = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Check if the agent or goal is completely trapped by holes or walls.
    def is_blocked(pos):
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if grid[nr][nc] != 'H':
                    return False
        return True

    if is_blocked(agent_pos) or is_blocked(goal_pos):
        return 0.0

    # Base value estimate based on distance.
    # V(s) should be higher for states closer to the goal.
    value = 1.0 / (1.0 + 0.1 * d)
    
    # Apply multiplicative penalties for holes nearby the agent (immediate danger)
    # or nearby the goal (bottlenecks).
    for hr, hc in holes:
        dist_agent_hole = abs(agent_pos[0] - hr) + abs(agent_pos[1] - hc)
        if dist_agent_hole == 1:
            value *= 0.80
        elif dist_agent_hole == 2:
            value *= 0.95
            
        dist_goal_hole = abs(goal_pos[0] - hr) + abs(goal_pos[1] - hc)
        if dist_goal_hole == 1:
            value *= 0.90
        elif dist_goal_hole == 2:
            value *= 0.98

    # Ensure the value is clamped between 0 and 1.
    return max(0.0, min(1.0, value))
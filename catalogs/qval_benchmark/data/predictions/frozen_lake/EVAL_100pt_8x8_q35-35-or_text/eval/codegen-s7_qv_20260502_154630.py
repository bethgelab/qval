def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        for r, row in enumerate(lines):
            for c, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (r, c)
                elif cell == 'G':
                    goal_pos = (r, c)
                elif cell == 'H':
                    holes.append((r, c))
        return agent_pos, goal_pos, holes

    agent_pos, goal_pos, holes = parse_grid(state)
    next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)

    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # If goal reached in next state, return high Q-value
    if next_goal_pos and next_agent_pos == next_goal_pos:
        return 0.95

    # If agent is in a hole, return very low Q-value
    if next_agent_pos in next_holes:
        return 0.0

    # Calculate distances
    if goal_pos and next_agent_pos:
        current_dist = manhattan_distance(agent_pos, goal_pos)
        next_dist = manhattan_distance(next_agent_pos, goal_pos)

        # Base Q-value inversely proportional to distance
        max_dist = 14  # Maximum Manhattan distance in 8x8 grid
        base_q = max(0.0, 1.0 - (next_dist / max_dist))

        # Adjust based on whether we're getting closer
        if next_dist < current_dist:
            base_q *= 1.15  # Bonus for moving closer
        elif next_dist > current_dist:
            base_q *= 0.6  # Penalty for moving away

        # Small bonus for being in a safe position (not adjacent to holes)
        if next_agent_pos:
            adjacent_holes = sum(1 for h in next_holes 
                                if manhattan_distance(next_agent_pos, h) == 1)
            if adjacent_holes == 0:
                base_q = min(1.0, base_q * 1.05)

        return max(0.0, min(1.0, base_q))

    return 0.5
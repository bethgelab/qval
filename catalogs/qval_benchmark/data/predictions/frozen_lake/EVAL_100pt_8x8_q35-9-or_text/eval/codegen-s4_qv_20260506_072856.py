def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        if not lines:
            return None, None, [], 0, 0
        grid = [list(line) for line in lines]
        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        agent_pos = None
        goal_pos = None
        holes = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '@':
                    agent_pos = (r, c)
                elif grid[r][c] == 'G':
                    goal_pos = (r, c)
                elif grid[r][c] == 'H':
                    holes.append((r, c))
        return agent_pos, goal_pos, holes, rows, cols

    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    agent_pos, goal_pos, holes, rows, cols = parse_grid(state)
    next_agent_pos, next_goal_pos, next_holes, next_rows, next_cols = parse_grid(next_state)

    if agent_pos is None or goal_pos is None or next_agent_pos is None or next_goal_pos is None:
        return 0.0

    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, next_goal_pos)

    dist_change = current_dist - next_dist
    dist_score = min(dist_change, 2.0) / 2.0

    min_hole_dist = float('inf')
    for hole in holes:
        hole_dist = manhattan_distance(agent_pos, hole)
        min_hole_dist = min(min_hole_dist, hole_dist)

    hole_risk = 0.0
    if min_hole_dist < 3:
        hole_risk = 1.0
    elif min_hole_dist < 5:
        hole_risk = 0.5
    elif min_hole_dist < 8:
        hole_risk = 0.2
    elif min_hole_dist < 12:
        hole_risk = 0.1
    else:
        hole_risk = 0.0

    safety_score = 1.0 - hole_risk

    if next_dist == 0:
        q_value = 0.95
    elif dist_change > 0:
        q_value = 0.6 + dist_score * 0.2 + safety_score * 0.2
    elif dist_change == 0:
        q_value = 0.3 + safety_score * 0.2
    else:
        q_value = 0.1

    q_value = max(0.0, min(1.0, q_value))
    return q_value
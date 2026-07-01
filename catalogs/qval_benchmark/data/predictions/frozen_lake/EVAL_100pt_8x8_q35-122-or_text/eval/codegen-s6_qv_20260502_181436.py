def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        hole_positions = []
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                elif char == 'H':
                    hole_positions.append((row_idx, col_idx))
        return agent_pos, goal_pos, hole_positions

    curr_agent, goal, holes = parse_grid(state)
    next_agent, _, _ = parse_grid(next_state)

    if goal is None or curr_agent is None:
        return 0.0

    if next_agent == goal:
        return 1.0

    if next_agent in holes:
        return -1.0

    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    curr_dist = manhattan_distance(curr_agent, goal)
    next_dist = manhattan_distance(next_agent, goal)
    progress = curr_dist - next_dist

    if progress > 0:
        progress_bonus = 0.4 + 0.3 * (progress / max(curr_dist, 1))
    elif progress < 0:
        progress_bonus = 0.3 - 0.2 * (abs(progress) / max(curr_dist, 1))
    else:
        progress_bonus = 0.35

    remaining_steps = 30 - next_dist
    if remaining_steps <= 0:
        time_factor = 0.1
    else:
        time_factor = 0.3 + 0.5 * (remaining_steps / 30)

    safety_score = 0.0
    for hole in holes:
        if manhattan_distance(next_agent, hole) == 1:
            safety_score -= 0.2
        if manhattan_distance(next_agent, hole) == 0:
            safety_score -= 0.5

    q_value = progress_bonus * 0.5 + time_factor * 0.5 + safety_score
    return max(-1.0, min(1.0, q_value))
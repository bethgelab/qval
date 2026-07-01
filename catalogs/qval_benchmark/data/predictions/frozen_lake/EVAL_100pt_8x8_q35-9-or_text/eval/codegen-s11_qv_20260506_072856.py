def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines if line.strip()]
        agent_pos = None
        goal_pos = None
        holes = []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (r, c)
                elif cell == 'G':
                    goal_pos = (r, c)
                elif cell == 'H':
                    holes.append((r, c))
        return agent_pos, goal_pos, holes, grid

    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def distance_to_hole(pos, holes):
        if not holes or pos is None:
            return float('inf')
        return min(manhattan_distance(pos, hole) for hole in holes)

    agent_pos, goal_pos, holes, _ = parse_grid(state)
    next_agent_pos, _, _, _ = parse_grid(next_state)

    if agent_pos is None or goal_pos is None or next_agent_pos is None:
        return 0.0

    dist_to_goal = manhattan_distance(agent_pos, goal_pos)
    next_dist_to_goal = manhattan_distance(next_agent_pos, goal_pos)
    dist_to_hole = distance_to_hole(agent_pos, holes)
    next_dist_to_hole = distance_to_hole(next_agent_pos, holes)

    q_value = 0.0

    if dist_to_goal > 0 and next_dist_to_goal < dist_to_goal:
        q_value += 0.5
    if dist_to_goal > 0 and next_dist_to_goal < dist_to_goal / 2:
        q_value += 0.3
    if dist_to_hole > 0 and next_dist_to_hole > dist_to_hole:
        q_value += 0.3
    if dist_to_hole > 0 and next_dist_to_hole < dist_to_hole:
        q_value -= 0.2
    if next_dist_to_goal == 0:
        q_value += 1.0

    return q_value
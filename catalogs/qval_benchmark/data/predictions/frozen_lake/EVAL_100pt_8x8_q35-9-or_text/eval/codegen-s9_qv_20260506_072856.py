import math

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        positions = {}
        lines = grid_str.strip().split('\n')
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char in ['@', 'G', 'H', 'S', '.']:
                    positions[char] = (x, y)
        return positions

    current_positions = parse_grid(state)
    next_positions = parse_grid(next_state)

    agent_pos = current_positions.get('@', (0, 0))
    next_agent_pos = next_positions.get('@', (0, 0))
    goal_pos = current_positions.get('G', (7, 7))

    def manhattan(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    dist_to_goal = manhattan(agent_pos, goal_pos)
    dist_to_goal_next = manhattan(next_agent_pos, goal_pos)
    dist_moved = manhattan(agent_pos, next_agent_pos)

    nearest_hole_dist = float('inf')
    for char, char_pos in current_positions.items():
        if char == 'H':
            d = manhattan(agent_pos, char_pos)
            nearest_hole_dist = min(nearest_hole_dist, d)

    next_hole_dist = float('inf')
    for char, char_pos in next_positions.items():
        if char == 'H':
            d = manhattan(next_agent_pos, char_pos)
            next_hole_dist = min(next_hole_dist, d)

    step_limit = 30
    steps_remaining = step_limit - dist_to_goal
    if steps_remaining < 0:
        steps_remaining = 0

    base_q = 0.0
    if dist_moved == 0:
        base_q = 0.0
    elif dist_to_goal_next < dist_to_goal:
        progress = 1.0 - (dist_to_goal_next / max(1, dist_to_goal))
        base_q = min(1.0, progress * (steps_remaining / step_limit))
    else:
        base_q = min(1.0, steps_remaining / step_limit)

    hole_penalty = 0.0
    if nearest_hole_dist <= 2:
        hole_penalty = 0.4
    elif nearest_hole_dist <= 4:
        hole_penalty = 0.2
    elif nearest_hole_dist <= 6:
        hole_penalty = 0.1

    if next_hole_dist == 0:
        q_value = 0.0
    else:
        q_value = base_q * (1.0 - hole_penalty)

    if goal_pos in next_positions:
        q_value = 1.0

    return float(q_value)
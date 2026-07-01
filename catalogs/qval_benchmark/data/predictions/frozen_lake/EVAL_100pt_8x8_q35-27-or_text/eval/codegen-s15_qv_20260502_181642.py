def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        rows = len(lines)
        cols = len(lines[0]) if lines else 0
        agent_pos = None
        goal_pos = None
        holes = []
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))
        return rows, cols, agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_hole_at(grid_str, r, c):
        lines = grid_str.strip().split('\n')
        if 0 <= r < len(lines) and 0 <= c < len(lines[r]):
            return lines[r][c] == 'H'
        return False
    
    rows, cols, agent_pos, goal_pos, holes = parse_grid(state)
    next_rows, next_cols, next_agent_pos, next_goal_pos, next_holes = parse_grid(next_state)
    
    if goal_pos is None:
        return 0.0
    
    if next_agent_pos is None:
        return 0.0
    
    if next_agent_pos == next_goal_pos:
        return 0.95
    
    if is_hole_at(next_state, next_agent_pos[0], next_agent_pos[1]):
        return 0.01
    
    current_dist = manhattan_distance(agent_pos, goal_pos)
    next_dist = manhattan_distance(next_agent_pos, goal_pos)
    
    if current_dist == float('inf') or next_dist == float('inf'):
        return 0.0
    
    progress = current_dist - next_dist
    
    safe_margin = 1.0
    for r in range(max(0, next_agent_pos[0] - 1), min(rows, next_agent_pos[0] + 2)):
        for c in range(max(0, next_agent_pos[1] - 1), min(cols, next_agent_pos[1] + 2)):
            if (r, c) != next_agent_pos and is_hole_at(next_state, r, c):
                safe_margin *= 0.85
    
    distance_factor = max(0.0, 1.0 - (next_dist / (rows + cols)))
    
    if progress > 0:
        action_bonus = 0.3
    elif progress == 0:
        action_bonus = 0.0
    else:
        action_bonus = -0.2
    
    q_value = (0.4 * distance_factor + 0.3 * action_bonus + 0.3 * safe_margin)
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value
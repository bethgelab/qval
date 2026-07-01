def signal_function(state: str, action: str, next_state: str) -> float:
    def find_positions(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        for row_idx, line in enumerate(lines):
            for col_idx, char in enumerate(line):
                if char == '@':
                    agent_pos = (row_idx, col_idx)
                elif char == 'G':
                    goal_pos = (row_idx, col_idx)
                elif char == 'H':
                    holes.append((row_idx, col_idx))
        return agent_pos, goal_pos, holes
    
    def manhattan_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    agent_pos, goal_pos, holes = find_positions(state)
    next_agent_pos, next_goal_pos, next_holes = find_positions(next_state)
    
    if next_agent_pos is None or next_goal_pos is None:
        return 0.5
    
    if next_agent_pos == next_goal_pos:
        return 1.0
    
    lines = next_state.strip().split('\n')
    row, col = next_agent_pos
    if row < len(lines) and col < len(lines[row]) and lines[row][col] == 'H':
        return 0.0
    
    dist_before = manhattan_distance(agent_pos, goal_pos)
    dist_after = manhattan_distance(next_agent_pos, next_goal_pos)
    
    if dist_before == float('inf') or dist_after == float('inf'):
        return 0.5
    
    max_dist = 14
    dist_score = max(0, 1 - dist_after / max_dist)
    
    progress = dist_before - dist_after
    progress_bonus = max(0, min(0.25, progress * 0.1))
    
    safety_score = 1.0
    for hole in next_holes:
        hole_dist = manhattan_distance(next_agent_pos, hole)
        if hole_dist <= 2:
            safety_score -= 0.15 * (2 - hole_dist + 1)
    safety_score = max(0, safety_score)
    
    if next_agent_pos != agent_pos:
        moved_successfully = 1.0
    else:
        moved_successfully = 0.7
    
    q_value = 0.5 * dist_score + 0.2 * progress_bonus + 0.2 * safety_score + 0.1 * moved_successfully
    q_value = min(1.0, max(0.0, q_value))
    
    return q_value
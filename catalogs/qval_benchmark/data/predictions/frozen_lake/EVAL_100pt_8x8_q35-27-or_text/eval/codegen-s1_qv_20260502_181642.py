def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        rows = len(lines)
        cols = len(lines[0]) if lines else 0
        
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == '@':
                    agent_pos = (i, j)
                elif char == 'G':
                    goal_pos = (i, j)
                elif char == 'H':
                    holes.append((i, j))
        return agent_pos, goal_pos, holes, rows, cols
    
    def manhattan_dist(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # Parse both states
    agent_pos, goal_pos, holes, rows, cols = parse_grid(state)
    next_agent_pos, _, _, _, _ = parse_grid(next_state)
    
    # Handle invalid states
    if agent_pos is None or goal_pos is None or next_agent_pos is None:
        return 0.0
    
    # Check immediate outcomes
    if next_agent_pos == goal_pos:
        return 0.95  # High value for reaching goal
    
    if next_agent_pos in holes:
        return 0.01  # Very low value for falling in hole
    
    # Calculate distances
    current_dist = manhattan_dist(agent_pos, goal_pos)
    next_dist = manhattan_dist(next_agent_pos, goal_pos)
    
    # Check if action was beneficial
    moved_closer = next_dist < current_dist
    moved_further = next_dist > current_dist
    stayed_in_place = agent_pos == next_agent_pos
    
    # Base score from distance to goal (normalized)
    max_dist = rows + cols - 2  # Maximum possible Manhattan distance
    distance_score = 1.0 - (next_dist / max_dist) if max_dist > 0 else 0.5
    
    # Start with distance-based value
    q_value = distance_score
    
    # Adjust based on action effectiveness
    if moved_closer:
        q_value *= 1.25  # Bonus for progress
    elif moved_further:
        q_value *= 0.65  # Penalty for moving away
    elif stayed_in_place:
        q_value *= 0.75  # Penalty for wasted step
    
    # Safety assessment - holes nearby reduce value
    nearby_holes = 0
    for hole in holes:
        if manhattan_dist(next_agent_pos, hole) <= 2:
            nearby_holes += 1
    
    if nearby_holes > 0:
        q_value *= (1.0 - nearby_holes * 0.08)
    
    # Bonus if very close to goal
    if next_dist <= 2:
        q_value += 0.15
    
    # Penalty if very far from goal
    if next_dist >= 10:
        q_value *= 0.85
    
    # Clamp to valid range
    q_value = max(0.0, min(1.0, q_value))
    
    return q_value
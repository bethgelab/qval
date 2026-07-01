def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    
    agent_pos = None
    goal_pos = None
    hole_count = 0
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '@':
                agent_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                hole_count += 1
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    ar, ac = agent_pos
    gr, gc = goal_pos
    
    manhattan_dist = abs(ar - gr) + abs(ac - gc)
    
    # If distance exceeds step limit, impossible to reach goal
    if manhattan_dist >= 30:
        return 0.0
    
    # Base probability based on distance to goal (closer = higher chance)
    base_value = 1.0 - (manhattan_dist / 30.0)
    
    # Hole density penalty - more holes = harder to navigate safely
    total_cells = height * width
    hole_density = hole_count / total_cells if total_cells > 0 else 0
    hole_penalty = hole_density * 0.3
    
    # Nearby holes are more dangerous - count holes in 3x3 neighborhood
    nearby_holes = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < height and 0 <= nc < width:
                if nr < len(lines) and nc < len(lines[nr]):
                    if lines[nr][nc] == 'H':
                        nearby_holes += 1
    
    nearby_penalty = nearby_holes * 0.05
    
    # Calculate final value with bounds
    value = base_value - hole_penalty - nearby_penalty
    value = max(0.0, min(1.0, value))
    
    return value
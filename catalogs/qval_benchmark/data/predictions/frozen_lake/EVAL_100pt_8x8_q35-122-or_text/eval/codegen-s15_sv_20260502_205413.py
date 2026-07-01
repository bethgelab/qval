def signal_function(state: str) -> float:
    # Parse the grid from state string
    lines = state.strip().split('\n')
    
    # Find agent, goal, and holes
    agent_pos = None
    goal_pos = None
    holes = []
    grid_size = len(lines)
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '@':
                agent_pos = (i, j)
            elif char == 'G':
                goal_pos = (i, j)
            elif char == 'H':
                holes.append((i, j))
    
    # Check terminal states
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    # Check if agent is on goal (terminal success)
    if agent_pos == goal_pos:
        return 1.0
    
    # Check if agent is on hole (terminal failure)
    if agent_pos in holes:
        return 0.0
    
    # Calculate Manhattan distance to goal
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    # Maximum reasonable distance (accounting for 30 step limit)
    max_dist = 30
    
    # Base value from distance (closer is better)
    # With discount factor ~0.99, value decreases with distance
    base_value = max(0.0, 1.0 - (dist / max_dist))
    
    # Check for holes nearby (adjacent cells)
    adjacent_holes = 0
    for hole in holes:
        if abs(hole[0] - agent_pos[0]) + abs(hole[1] - agent_pos[1]) == 1:
            adjacent_holes += 1
    
    # Safety penalty for nearby holes (increased risk of falling)
    safety_penalty = 0.15 * adjacent_holes
    
    # Also penalize if holes are very close (within 2 cells)
    nearby_holes = 0
    for hole in holes:
        hole_dist = abs(hole[0] - agent_pos[0]) + abs(hole[1] - agent_pos[1])
        if hole_dist == 2:
            nearby_holes += 1
    
    safety_penalty += 0.05 * nearby_holes
    
    # Final value (bounded between 0 and 1)
    value = max(0.0, min(1.0, base_value - safety_penalty))
    
    return value
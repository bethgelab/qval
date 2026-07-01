def signal_function(state: str, action: str, next_state: str) -> float:
    lines = next_state.strip().split('\n')
    if not lines:
        return 0.0
    rows = len(lines)
    cols = max(len(line) for line in lines) if lines else 0
    
    agent_pos = None
    goal_pos = None
    hole_positions = []
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '@':
                agent_pos = (i, j)
            elif char == 'G':
                goal_pos = (i, j)
            elif char == 'H':
                hole_positions.append((i, j))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    if agent_pos == goal_pos:
        return 0.95
    
    if agent_pos in hole_positions:
        return -0.5
    
    manhattan_dist = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
    
    if manhattan_dist <= 3:
        return 0.85
    elif manhattan_dist <= 7:
        return 0.6
    elif manhattan_dist <= 12:
        return 0.35
    elif manhattan_dist <= 18:
        return 0.15
    else:
        return 0.05
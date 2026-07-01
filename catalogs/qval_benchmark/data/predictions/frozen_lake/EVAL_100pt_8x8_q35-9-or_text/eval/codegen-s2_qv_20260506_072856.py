import math

def signal_function(state: str, action: str, next_state: str) -> float:
    lines = next_state.split('\n')
    
    agent_pos = None
    goal_pos = None
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '@':
                agent_pos = (i, j)
            elif char == 'G':
                goal_pos = (i, j)
    
    if goal_pos is None:
        return 0.1
    
    if agent_pos == goal_pos:
        return 1.0
    
    if agent_pos:
        i, j = agent_pos
        if 0 <= i < len(lines) and 0 <= j < len(lines[i]):
            if lines[i][j] == 'H':
                return 0.0
    
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    return math.exp(-dist / 10)
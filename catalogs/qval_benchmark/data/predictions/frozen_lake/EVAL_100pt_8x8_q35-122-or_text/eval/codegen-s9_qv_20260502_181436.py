def signal_function(state: str, action: str, next_state: str) -> float:
    import math
    
    def parse_grid(s):
        lines = [line for line in s.strip().split('\n') if line.strip()]
        agent_pos = None
        goal_pos = None
        holes = set()
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.add((r, c))
        
        return agent_pos, goal_pos, holes
    
    next_agent_pos, goal_pos, next_holes = parse_grid(next_state)
    prev_agent_pos, _, _ = parse_grid(state)
    
    if next_agent_pos == goal_pos:
        return 1.0
    
    if next_agent_pos in next_holes:
        return 0.0
    
    if goal_pos is None or next_agent_pos is None:
        return 0.0
    
    dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
    
    max_dist = 14
    dist_score = 1.0 - (dist / max_dist)
    
    action_dirs = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }
    
    if prev_agent_pos and next_agent_pos:
        prev_dist = abs(prev_agent_pos[0] - goal_pos[0]) + abs(prev_agent_pos[1] - goal_pos[1])
        
        if dist < prev_dist:
            action_score = 1.0
        elif dist == prev_dist:
            action_score = 0.5
        else:
            action_score = 0.3
    else:
        action_score = 0.5
    
    discount = 0.95
    q_value = dist_score * 0.7 + action_score * 0.3
    q_value *= (discount ** dist)
    
    return max(0.0, min(1.0, q_value))
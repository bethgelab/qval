import math

def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        agent_pos = None
        goal_pos = None
        holes = []
        grid_rows = []
        
        for r, line in enumerate(lines):
            row_chars = []
            for c, char in enumerate(line):
                row_chars.append(char)
                if char == '@':
                    agent_pos = (r, c)
                elif char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.append((r, c))
            grid_rows.append(row_chars)
        
        return grid_rows, agent_pos, goal_pos, holes
    
    def manhattan_dist(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    current_grid, current_agent, goal_pos, holes = parse_grid(state)
    next_grid, next_agent, _, _ = parse_grid(next_state)
    
    if current_agent is None or goal_pos is None or next_agent is None:
        return 0.0
    
    current_dist = manhattan_dist(current_agent, goal_pos)
    next_dist = manhattan_dist(next_agent, goal_pos)
    
    is_goal = (next_agent == goal_pos)
    
    next_is_hole = False
    if (next_agent[0] < len(next_grid) and 
        next_agent[1] < len(next_grid[next_agent[0]])):
        next_is_hole = next_grid[next_agent[0]][next_agent[1]] == 'H'
    
    min_hole_dist = float('inf')
    for hole in holes:
        dist = manhattan_dist(next_agent, hole)
        if dist < min_hole_dist:
            min_hole_dist = dist
    
    if is_goal:
        return 1.0
    
    if next_is_hole:
        return -0.5
    
    progress = 0.0
    if next_dist < current_dist:
        progress = 0.3 * (1.0 - next_dist / 14.0)
    elif next_dist > current_dist:
        progress = -0.2 * (1.0 - current_dist / 14.0)
    
    hole_penalty = 0.0
    if min_hole_dist <= 1:
        hole_penalty = -0.4
    elif min_hole_dist <= 2:
        hole_penalty = -0.2
    elif min_hole_dist <= 3:
        hole_penalty = -0.1
    
    distance_value = max(0.0, 1.0 - current_dist / 12.0) * 0.5
    
    action_direction = action.lower()
    action_bonus = 0.0
    if current_agent[0] < goal_pos[0] and action_direction == 'down':
        action_bonus = 0.1
    elif current_agent[0] > goal_pos[0] and action_direction == 'up':
        action_bonus = 0.1
    elif current_agent[1] < goal_pos[1] and action_direction == 'right':
        action_bonus = 0.1
    elif current_agent[1] > goal_pos[1] and action_direction == 'left':
        action_bonus = 0.1
    
    q_value = distance_value + progress + hole_penalty + action_bonus
    q_value = max(-0.5, min(0.9, q_value))
    
    return q_value
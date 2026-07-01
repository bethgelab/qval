def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = grid_str.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        goal_pos = None
        holes = []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == '@':
                    agent_pos = (r, c)
                elif cell == 'G':
                    goal_pos = (r, c)
                elif cell == 'H':
                    holes.append((r, c))
        return {'agent': agent_pos, 'goal': goal_pos, 'holes': holes}
    
    def manhattan(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_next_in_holes(next_state_info):
        agent = next_state_info['agent']
        holes = next_state_info['holes']
        return agent is not None and agent in holes
    
    def count_nearby_holes(next_state_info, agent_pos, max_dist=2):
        if agent_pos is None:
            return 0
        holes = next_state_info['holes']
        count = 0
        for hole in holes:
            if manhattan(agent_pos, hole) <= max_dist:
                count += 1
        return count
    
    state_info = parse_grid(state)
    next_state_info = parse_grid(next_state)
    
    next_agent = next_state_info['agent']
    goal_pos = state_info['goal']
    current_agent = state_info['agent']
    
    if next_agent is None or goal_pos is None:
        return 0.0
    
    if next_agent == goal_pos:
        return 1.0
    
    if is_next_in_holes(next_state_info):
        return 0.0
    
    current_dist = manhattan(current_agent, goal_pos)
    next_dist = manhattan(next_agent, goal_pos)
    
    if current_dist == 0:
        return 0.0
    
    if current_dist == float('inf'):
        return 0.0
    
    base_value = 1.0 / (current_dist + 1)
    
    if next_dist < current_dist:
        base_value += 0.3
    elif next_dist > current_dist:
        base_value -= 0.2
    
    nearby_holes = count_nearby_holes(next_state_info, next_agent)
    base_value -= nearby_holes * 0.15
    
    return max(0.0, min(1.0, base_value))
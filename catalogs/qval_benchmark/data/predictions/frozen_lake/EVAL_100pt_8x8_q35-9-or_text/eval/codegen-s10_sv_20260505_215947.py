def signal_function(state: str) -> float:
    lines = state.split('\n')
    grid = [line for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for y, line in enumerate(grid):
        for x, char in enumerate(line):
            if char == '@':
                agent_pos = (x, y)
            elif char == 'G':
                goal_pos = (x, y)
            elif char == 'H':
                holes.append((x, y))
    
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    distance_to_goal = manhattan(agent_pos, goal_pos)
    
    if distance_to_goal > 30:
        return 0.0
    
    min_distance_to_holes = float('inf')
    if holes:
        for hole in holes:
            min_distance_to_holes = min(min_distance_to_holes, manhattan(agent_pos, hole))
    
    if min_distance_to_holes == float('inf'):
        min_distance_to_holes = 100
    
    base_value = max(0.0, 1.0 - distance_to_goal / 30)
    hole_penalty = min(1.0, (30 - min_distance_to_holes) / 30)
    
    return base_value * (1.0 - hole_penalty * 0.5)
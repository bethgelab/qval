import math

def signal_function(state: str) -> float:
    lines = state.strip().split('\n')
    grid = [list(line) for line in lines if line.strip()]
    
    agent_pos = None
    goal_pos = None
    holes = []
    grid_height = len(grid)
    grid_width = len(grid[0]) if grid else 0
    
    for r in range(grid_height):
        for c in range(grid_width):
            cell = grid[r][c]
            if cell == '@':
                agent_pos = (r, c)
            elif cell == 'G':
                goal_pos = (r, c)
            elif cell == 'H':
                holes.append((r, c))
    
    if agent_pos == goal_pos:
        return 1.0
    if agent_pos is None or goal_pos is None:
        return 0.0
    
    dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    
    if dist_to_goal > 30:
        return 0.0
    
    discount = 0.99
    base_value = discount ** dist_to_goal
    
    min_dist_to_hole = float('inf')
    if holes:
        for hole in holes:
            d = abs(agent_pos[0] - hole[0]) + abs(agent_pos[1] - hole[1])
            min_dist_to_hole = min(min_dist_to_hole, d)
    
    if min_dist_to_hole != float('inf'):
        hole_penalty = 1.0 / (1.0 + min_dist_to_hole * 0.1)
        base_value *= hole_penalty
    
    num_holes = len(holes)
    if num_holes > 10:
        base_value *= 0.6
    elif num_holes > 5:
        base_value *= 0.8
    elif num_holes > 2:
        base_value *= 0.9
    
    return max(0.0, min(1.0, base_value))